"""
src/llm/audit_logger.py
------------------------
JSONL audit logger for all LLM prompt/response pairs.

Every LLM invocation is logged with:
  - timestamp, use_case, loan_id (if applicable)
  - prompt_template_version
  - retrieved_chunks (chunk IDs used for grounding)
  - prompt (full text sent to LLM)
  - model (LLM model name)
  - response (full LLM output)
  - grounding_passed (bool)
  - grounding_citations_found (list of cited chunk IDs)
  - reviewer_action (filled by HITL panel later)
  - reviewer_correction (human correction if rejected)

Usage:
    from src.llm.audit_logger import AuditLogger
    logger = AuditLogger()
    entry_id = logger.log(use_case="loan_explanation", ...)
    logger.update_reviewer_action(entry_id, action="approved")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class AuditLogger:
    """Append-only JSONL audit log for all LLM prompts and responses."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        cfg = get_config()
        self.log_path = Path(log_path or cfg["paths"]["llm_audit_log"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────

    def log(
        self,
        use_case: str,
        prompt: str,
        response: str,
        model: str,
        retrieved_chunks: list[dict] | None = None,
        grounding_passed: bool = True,
        grounding_citations_found: list[str] | None = None,
        prompt_template_version: str = "v1",
        loan_id: str | None = None,
        extra: dict | None = None,
    ) -> str:
        """Log one LLM interaction. Returns the entry_id for later updates.

        Parameters
        ----------
        use_case : str
            One of: loan_explanation, exception_review, scenario_narrative,
                    data_quality_review, fp_fn_audit
        prompt : str
            Full prompt sent to the LLM.
        response : str
            Full LLM output (or error message if blocked).
        model : str
            LLM model name (e.g. 'gpt-4o-mini', 'mistral:7b').
        retrieved_chunks : list of dicts
            Chunks retrieved from RAG (each has 'chunk_id' and 'text').
        grounding_passed : bool
            True if response cited at least one retrieved chunk ID.
        grounding_citations_found : list of str
            Chunk IDs actually cited in the response.
        loan_id : str, optional
            Loan identifier (if loan-level use case).
        extra : dict, optional
            Any additional metadata.
        """
        entry_id = str(uuid.uuid4())[:8]
        chunk_ids = [c.get("chunk_id", "") for c in (retrieved_chunks or [])]

        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "use_case": use_case,
            "loan_id": loan_id,
            "prompt_template_version": prompt_template_version,
            "model": model,
            "retrieved_chunk_ids": chunk_ids,
            "prompt": prompt,
            "response": response,
            "grounding_passed": grounding_passed,
            "grounding_citations_found": grounding_citations_found or [],
            "reviewer_action": None,
            "reviewer_correction": None,
        }
        if extra:
            entry.update(extra)

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if not grounding_passed:
            log.warning(
                "UNGROUNDED LLM RESPONSE logged (entry_id=%s, use_case=%s)",
                entry_id, use_case,
            )

        return entry_id

    def update_reviewer_action(
        self,
        entry_id: str,
        action: str,
        correction: str | None = None,
    ) -> None:
        """Update reviewer_action for a logged entry (HITL decision).

        This re-writes the specific line in the JSONL file.
        For large logs, consider a DB-backed approach.

        Parameters
        ----------
        action : str
            One of: 'approved', 'rejected', 'corrected'
        correction : str, optional
            Human correction text (if action == 'corrected').
        """
        if not self.log_path.exists():
            log.warning("Audit log not found — cannot update entry %s", entry_id)
            return

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []
        for line in lines:
            try:
                entry = json.loads(line)
                if entry.get("entry_id") == entry_id:
                    entry["reviewer_action"] = action
                    entry["reviewer_correction"] = correction
                    updated = True
                new_lines.append(json.dumps(entry, ensure_ascii=False))
            except json.JSONDecodeError:
                new_lines.append(line)

        if updated:
            self.log_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            log.info("Audit log updated: entry=%s action=%s", entry_id, action)
        else:
            log.warning("Entry %s not found in audit log", entry_id)

    # ──────────────────────────────────────────────────────────
    # Reading / Querying
    # ──────────────────────────────────────────────────────────

    def read_all(self) -> list[dict]:
        """Return all log entries as a list of dicts."""
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_ungrounded_examples(self) -> list[dict]:
        """Return all entries where grounding_passed = False.

        Required by the judging criteria: 'examples where the LLM was wrong,
        vague, or overconfident'.
        """
        return [e for e in self.read_all() if not e.get("grounding_passed", True)]

    def get_rejected_examples(self) -> list[dict]:
        """Return all entries where reviewer_action = 'rejected'."""
        return [e for e in self.read_all() if e.get("reviewer_action") == "rejected"]

    def summary_stats(self) -> dict[str, Any]:
        """Return summary statistics for the audit log."""
        entries = self.read_all()
        if not entries:
            return {"n_entries": 0}

        return {
            "n_entries": len(entries),
            "n_grounded": sum(1 for e in entries if e.get("grounding_passed")),
            "n_ungrounded": sum(1 for e in entries if not e.get("grounding_passed")),
            "n_approved": sum(1 for e in entries if e.get("reviewer_action") == "approved"),
            "n_rejected": sum(1 for e in entries if e.get("reviewer_action") == "rejected"),
            "n_corrected": sum(1 for e in entries if e.get("reviewer_action") == "corrected"),
            "n_pending": sum(1 for e in entries if e.get("reviewer_action") is None),
            "use_case_counts": {
                k: sum(1 for e in entries if e.get("use_case") == k)
                for k in ["loan_explanation", "exception_review", "scenario_narrative",
                          "data_quality_review", "fp_fn_audit"]
            },
        }
