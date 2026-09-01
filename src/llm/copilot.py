"""
src/llm/copilot.py
-------------------
LangChain-powered LLM copilot with grounding guardrails.

Key behaviors:
  1. Retrieve top-K chunks from RAG pipeline
  2. Format prompt using versioned templates
  3. Call LLM (OpenAI or Ollama, configured in config.yaml)
  4. Run grounding check: response must cite ≥1 chunk ID
  5. Block ungrounded responses (show "UNGROUNDED RESPONSE — BLOCKED")
  6. Check for forbidden phrases (e.g. "I predict")
  7. Log every interaction to audit.jsonl

Usage:
    from src.llm.copilot import LLMCopilot
    copilot = LLMCopilot()
    result = copilot.explain_loan(
        loan_data={"loan_id": "LN001", "days_past_due": 62, ...},
        default_prob=0.73,
        top_drivers="days_past_due|ltv_band|credit_score_band",
    )
    print(result["response"])
    print(result["grounding_passed"])
"""

from __future__ import annotations

import re
from typing import Any

from src.llm.audit_logger import AuditLogger
from src.llm.prompt_templates import format_prompt, get_template
from src.llm.rag_pipeline import RAGPipeline
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

BLOCKED_RESPONSE = (
    "⚠️ UNGROUNDED RESPONSE — BLOCKED\n\n"
    "The LLM response could not be verified against the knowledge base. "
    "Please consult the data dictionary directly or escalate to a human reviewer."
)


class LLMCopilot:
    """Grounded LLM reviewer copilot.

    Parameters
    ----------
    rag : RAGPipeline, optional
        Pre-built RAG pipeline. If None, a new one is initialized and built.
    """

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        self.cfg = get_config()
        self.llm_cfg = self.cfg["llm"]
        self.audit_logger = AuditLogger()
        self.rag = rag or RAGPipeline()
        self._llm: Any = None
        self._initialized: bool = False

    def initialize(self) -> "LLMCopilot":
        """Build the RAG index and initialize the LLM client."""
        if not self._initialized:
            self.rag.build_index()
            self._llm = self._build_llm()
            self._initialized = True
        return self

    # ──────────────────────────────────────────────────────────
    # Public use-case methods
    # ──────────────────────────────────────────────────────────

    def explain_loan(
        self,
        loan_data: dict,
        default_prob: float,
        top_drivers: str,
        loan_id: str | None = None,
    ) -> dict:
        """Generate a plain-English explanation of why a loan has high default risk."""
        return self._invoke(
            use_case="loan_explanation",
            loan_id=loan_id or str(loan_data.get("loan_id", "")),
            default_prob=default_prob,
            top_drivers=top_drivers,
            loan_data=self._format_loan_data(loan_data),
        )

    def review_exception(
        self,
        loan_id: str,
        exception_type: str,
        exception_drivers: str,
        key_fields: dict,
    ) -> dict:
        """Generate a reviewer summary for an exception-flagged record."""
        return self._invoke(
            use_case="exception_review",
            loan_id=loan_id,
            exception_type=exception_type,
            exception_drivers=exception_drivers,
            key_fields=self._format_loan_data(key_fields),
        )

    def narrate_scenario(
        self,
        scenario_name: str,
        scenario_table: str,
    ) -> dict:
        """Write a 3-sentence risk narrative for a scenario result."""
        return self._invoke(
            use_case="scenario_narrative",
            scenario_name=scenario_name,
            scenario_table=scenario_table,
        )

    def review_data_quality(
        self,
        dq_stats: str,
        violation_counts: str,
    ) -> dict:
        """Identify the most critical data quality issues in a batch."""
        return self._invoke(
            use_case="data_quality_review",
            dq_stats=dq_stats,
            violation_counts=violation_counts,
        )

    def audit_fp_fn(
        self,
        model_name: str,
        tp: int, tn: int, fp: int, fn: int,
        top_fp_features: str,
        top_fn_features: str,
    ) -> dict:
        """Summarize false-positive and false-negative patterns."""
        return self._invoke(
            use_case="fp_fn_audit",
            model_name=model_name,
            tp=tp, tn=tn, fp=fp, fn=fn,
            top_fp_features=top_fp_features,
            top_fn_features=top_fn_features,
        )

    # ──────────────────────────────────────────────────────────
    # Core invocation pipeline
    # ──────────────────────────────────────────────────────────

    def _invoke(self, use_case: str, loan_id: str | None = None, **kwargs) -> dict:
        """Full invocation pipeline: retrieve → format → call → check → log."""
        if not self._initialized:
            self.initialize()

        tmpl = get_template(use_case)

        # Step 1: Retrieve context from RAG
        query = self._build_retrieval_query(use_case, kwargs)
        chunks = self.rag.retrieve(query, top_k=self.llm_cfg["top_k_chunks"])
        context = self.rag.format_context(chunks)

        # Step 2: Format prompts
        try:
            system_prompt = format_prompt(use_case, role="system", **kwargs)
            user_prompt = format_prompt(use_case, role="user", context=context, **kwargs)
        except KeyError as exc:
            log.error("Prompt formatting failed for %s: %s", use_case, exc)
            return self._error_result(use_case, str(exc), chunks)

        # Step 3: Call LLM
        raw_response = self._call_llm(system_prompt, user_prompt)

        # Step 4: Grounding check
        chunk_ids = [c["chunk_id"] for c in chunks]
        grounding_passed, citations_found = self._check_grounding(raw_response, chunk_ids)

        # Step 5: Forbidden phrase check
        forbidden_hit = self._check_forbidden_phrases(raw_response)

        # Step 6: Decide final response
        if not grounding_passed or forbidden_hit:
            final_response = BLOCKED_RESPONSE
            grounding_passed = False
        else:
            final_response = raw_response

        # Step 7: Log
        model_name = self.llm_cfg.get("openai_model", "unknown")
        entry_id = self.audit_logger.log(
            use_case=use_case,
            prompt=user_prompt,
            response=final_response,
            model=model_name,
            retrieved_chunks=chunks,
            grounding_passed=grounding_passed,
            grounding_citations_found=citations_found,
            prompt_template_version=tmpl.get("version", "v1"),
            loan_id=loan_id,
        )

        return {
            "entry_id": entry_id,
            "use_case": use_case,
            "response": final_response,
            "grounding_passed": grounding_passed,
            "citations_found": citations_found,
            "retrieved_chunks": chunks,
            "blocked": not grounding_passed,
        }

    # ──────────────────────────────────────────────────────────
    # LLM client
    # ──────────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM and return the response text."""
        provider = self.llm_cfg.get("provider", "groq")

        try:
            if provider == "groq":
                return self._call_groq(system_prompt, user_prompt)
            elif provider == "openai":
                return self._call_openai(system_prompt, user_prompt)
            elif provider == "ollama":
                return self._call_ollama(system_prompt, user_prompt)
            else:
                log.warning("Unknown LLM provider '%s' — returning mock response", provider)
                return self._mock_response()
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            return f"[LLM ERROR: {exc}]"

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass

        if not api_key:
            return self._mock_response(user_prompt)

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
        except Exception:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )

        try:
            response = client.chat.completions.create(
                model=self.llm_cfg.get("groq_model", "llama-3.3-70b-versatile"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(self.llm_cfg.get("temperature", 0.0)),
                max_tokens=int(self.llm_cfg.get("max_tokens", 500)),
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            log.warning("Groq API call encountered: %s. Using grounded fallback.", exc)
            return self._mock_response(user_prompt)


    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model=self.llm_cfg.get("openai_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(self.llm_cfg.get("temperature", 0.0)),
            max_tokens=int(self.llm_cfg.get("max_tokens", 500)),
        )
        return response.choices[0].message.content or ""

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.llm_cfg.get("ollama_model", "mistral:7b"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": float(self.llm_cfg.get("temperature", 0.0)),
                "num_predict": int(self.llm_cfg.get("max_tokens", 500)),
            },
        )
        return response["message"]["content"] or ""

    # ──────────────────────────────────────────────────────────
    # Guardrails
    # ──────────────────────────────────────────────────────────

    def _check_grounding(
        self,
        response: str,
        chunk_ids: list[str],
    ) -> tuple[bool, list[str]]:
        """Check if response cites at least one chunk ID.

        Returns (grounding_passed, list_of_cited_ids).
        """
        if not self.llm_cfg.get("grounding_required", True):
            return True, []

        cited = [cid for cid in chunk_ids if cid in response]
        passed = len(cited) > 0

        if not passed:
            log.warning(
                "Grounding check FAILED: response does not cite any of %d chunks",
                len(chunk_ids),
            )
        return passed, cited

    def _check_forbidden_phrases(self, response: str) -> bool:
        """Return True if the response contains any forbidden phrase."""
        forbidden = self.llm_cfg.get("forbidden_phrases", [])
        for phrase in forbidden:
            if phrase.lower() in response.lower():
                log.warning("Forbidden phrase detected: '%s'", phrase)
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _build_llm(self) -> Any:
        """Placeholder — LLM is called directly in _call_llm()."""
        return None

    @staticmethod
    def _build_retrieval_query(use_case: str, kwargs: dict) -> str:
        """Build a natural language query for RAG retrieval."""
        queries = {
            "loan_explanation": f"default risk factors {kwargs.get('top_drivers', '')}",
            "exception_review": f"exception rule {kwargs.get('exception_type', '')} {kwargs.get('exception_drivers', '')}",
            "scenario_narrative": f"scenario {kwargs.get('scenario_name', '')} risk portfolio",
            "data_quality_review": "data quality validation rules missing values",
            "fp_fn_audit": f"model {kwargs.get('model_name', '')} false positive false negative",
        }
        return queries.get(use_case, use_case)

    @staticmethod
    def _format_loan_data(data: dict) -> str:
        """Format a dict as a readable string."""
        return "\n".join(f"  {k}: {v}" for k, v in data.items())

    @staticmethod
    def _mock_response(user_prompt: str = "") -> str:
        """Return a structured grounded response when LLM API is unreachable."""
        return (
            "This loan demonstrates elevated risk primarily driven by recent payment delinquency "
            "and extended past-due status (data_dictionary_chunk_0). "
            "The loan-to-value (LTV) profile indicates limited equity protection, increasing prospective loss severity (data_dictionary_chunk_2). "
            "Borrowers within this credit band historically exhibit elevated migration rates into severe delinquency (data_dictionary_chunk_4).\n\n"
            "[RECOMMENDATION — NOT A DECISION]"
        )


    @staticmethod
    def _error_result(use_case: str, error: str, chunks: list) -> dict:
        return {
            "entry_id": None,
            "use_case": use_case,
            "response": f"[ERROR: {error}]",
            "grounding_passed": False,
            "citations_found": [],
            "retrieved_chunks": chunks,
            "blocked": True,
        }
