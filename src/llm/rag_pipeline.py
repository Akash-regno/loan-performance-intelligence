"""
src/llm/rag_pipeline.py
------------------------
RAG (Retrieval-Augmented Generation) pipeline over the organizer's
data_dictionary.md and validation_rules.json.

Uses ChromaDB as the vector store. Embeddings via OpenAI or a local
Ollama model (configured in config.yaml).

Usage:
    from src.llm.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    rag.build_index()                              # one-time setup
    chunks = rag.retrieve("What is days_past_due?", top_k=3)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 500          # characters per document chunk
CHUNK_OVERLAP = 100       # overlap between adjacent chunks
COLLECTION_NAME = "loan_performance_kb"


class _InMemoryTFIDFCollection:

    """Zero-dependency vector search fallback using scikit-learn TF-IDF."""

    def __init__(self, documents: list[str], ids: list[str], metadatas: list[dict]) -> None:
        self.documents = documents
        self.ids = ids
        self.metadatas = metadatas
        if documents:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def query(self, query_texts: list[str], n_results: int = 3) -> dict[str, list]:
        if not self.documents or self.vectorizer is None or self.tfidf_matrix is None:
            return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform(query_texts)
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_k = min(n_results, len(self.documents))
        top_indices = sims.argsort()[::-1][:top_k]
        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "ids": [[self.ids[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[float(1.0 - sims[i]) for i in top_indices]],
        }

    def count(self) -> int:
        return len(self.documents)


class RAGPipeline:

    """Build and query a ChromaDB vector index over the knowledge base.

    Knowledge base documents:
      - data_dictionary.md
      - validation_rules.json (converted to plain text)
      - model_card.md
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        self.llm_cfg = self.cfg["llm"]
        self._client: Any = None
        self._collection: Any = None
        self._embedding_fn: Any = None
        self._is_built: bool = False

    # ──────────────────────────────────────────────────────────
    # Index Building
    # ──────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> "RAGPipeline":
        """Build the vector index from source documents.
        Uses ChromaDB when installed, with seamless in-memory TF-IDF fallback.
        """
        # Load all documents
        documents, ids, metadatas = [], [], []
        for doc_text, doc_id, meta in self._load_all_documents():
            chunks = self._chunk_text(doc_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append({**meta, "chunk_index": i, "chunk_id": chunk_id})

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=".chromadb_cache")
            self._embedding_fn = self._get_embedding_function()

            if force_rebuild:
                try:
                    self._client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass

            existing = [c.name for c in self._client.list_collections()]
            if COLLECTION_NAME in existing and not force_rebuild:
                log.info("RAG index already exists — loading from cache")
                self._collection = self._client.get_collection(
                    COLLECTION_NAME, embedding_function=self._embedding_fn
                )
                self._is_built = True
                return self

            self._collection = self._client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            if documents:
                batch_size = 100
                for start in range(0, len(documents), batch_size):
                    self._collection.add(
                        documents=documents[start:start + batch_size],
                        ids=ids[start:start + batch_size],
                        metadatas=metadatas[start:start + batch_size],
                    )
                log.info("ChromaDB RAG index built: %d chunks from %d documents", len(documents), len(set(m["source"] for m in metadatas)))
            self._is_built = True
            return self
        except Exception as exc:
            log.info("ChromaDB unavailable (%s) — using high-performance in-memory TF-IDF vector index fallback", exc)
            self._collection = _InMemoryTFIDFCollection(documents, ids, metadatas)
            self._is_built = True
            log.info("In-memory RAG index built: %d chunks", len(documents))
            return self


    # ──────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-K most relevant chunks for a query.

        Returns
        -------
        list of dicts with keys: chunk_id, text, source, distance
        """
        if not self._is_built:
            log.warning("Index not built — call build_index() first")
            return []

        top_k = top_k or self.llm_cfg["top_k_chunks"]

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
            )
        except Exception as exc:
            log.error("RAG retrieval failed: %s", exc)
            return []

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": doc,
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i] if "distances" in results else None,
            })

        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM prompt."""
        parts = []
        for chunk in chunks:
            parts.append(
                f"[SOURCE: {chunk['source']} | ID: {chunk['chunk_id']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # Document loading
    # ──────────────────────────────────────────────────────────

    def _load_all_documents(self):
        """Yield (text, doc_id, metadata) for each knowledge base document."""
        paths_cfg = self.cfg["paths"]

        # Data dictionary
        dd_path = Path(paths_cfg.get("data_dictionary", "data/raw/data_dictionary.md"))
        if dd_path.exists():
            text = dd_path.read_text(encoding="utf-8")
            yield text, "data_dictionary", {"source": "data_dictionary.md", "type": "schema"}
        else:
            log.warning("data_dictionary.md not found at %s", dd_path)

        # Validation rules (converted to text)
        vr_path = Path(paths_cfg.get("validation_rules", "data/raw/validation_rules.json"))
        if vr_path.exists():
            text = self._rules_to_text(vr_path)
            yield text, "validation_rules", {"source": "validation_rules.json", "type": "rules"}

        # Model card
        mc_path = Path("model_card.md")
        if mc_path.exists():
            text = mc_path.read_text(encoding="utf-8")
            yield text, "model_card", {"source": "model_card.md", "type": "documentation"}

        # Inline regulatory / CECL context (static)
        yield self._get_static_context(), "regulatory_context", {
            "source": "regulatory_context", "type": "regulatory"
        }

    @staticmethod
    def _rules_to_text(path: Path) -> str:
        """Convert validation_rules.json to plain English text."""
        with path.open("r", encoding="utf-8") as fh:
            rules = json.load(fh)

        if not isinstance(rules, list):
            return str(rules)

        lines = ["Validation Rules:\n"]
        for rule in rules:
            rule_id = rule.get("rule_id", "UNKNOWN")
            desc = rule.get("description", "No description")
            lines.append(f"- Rule {rule_id}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _get_static_context() -> str:
        return """
Loan Performance Intelligence Engine — Regulatory Context:

CECL (Current Expected Credit Loss, FASB ASC 326):
- Requires forward-looking expected credit loss estimation over the life of a loan.
- Probability of Default (PD) × Loss Given Default (LGD) × Exposure at Default (EAD) framework.

Days Past Due (DPD):
- 30 DPD: First stage delinquency. Borrower missed 1 payment.
- 60 DPD: Second stage. Typically triggers servicer outreach.
- 90 DPD: Third stage. Loan is often classified as non-performing.
- 120+ DPD: Often triggers charge-off or default classification.

Prepayment:
- Voluntary: Borrower pays off loan early (e.g., refinancing when rates drop).
- Involuntary: Property sold or loan paid off due to default resolution.
- Key drivers: interest rate spread, home price appreciation, loan age, LTV.

Loan-to-Value (LTV):
- LTV > 80%: Higher credit risk, may require PMI.
- LTV > 100%: Underwater loan — borrower owes more than property value.
"""

    # ──────────────────────────────────────────────────────────
    # Text chunking
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        """Split text into overlapping chunks by character count."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to break at a sentence boundary
            if end < len(text):
                boundary = text.rfind(". ", start, end)
                if boundary > start:
                    end = boundary + 1
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if len(c) > 20]

    # ──────────────────────────────────────────────────────────
    # Embedding function factory
    # ──────────────────────────────────────────────────────────

    def _get_embedding_function(self) -> Any:
        """Return the appropriate ChromaDB embedding function."""
        provider = self.llm_cfg.get("provider", "openai")

        if provider == "openai":
            try:
                from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

                return OpenAIEmbeddingFunction(
                    api_key=self._get_openai_key(),
                    model_name="text-embedding-3-small",
                )
            except Exception:
                pass

        # Fallback: default ChromaDB embedding (sentence-transformers)
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            log.info("Using DefaultEmbeddingFunction (sentence-transformers)")
            return DefaultEmbeddingFunction()
        except Exception as exc:
            log.error("No embedding function available: %s", exc)
            return None

    @staticmethod
    def _get_openai_key() -> str:
        import os
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            log.warning("OPENAI_API_KEY not set — embedding may fail")
        return key
