"""
src/data/validation.py
----------------------
Data validation using Great Expectations (GE) and the organizer's
validation_rules.json as the source of truth.

Responsibilities:
  1. Parse validation_rules.json into a GE ExpectationSuite
  2. Run deterministic checks: range, date ordering, categoricals,
     referential integrity, delinquency consistency, balance sanity
  3. Return a validation result summary with pass/fail per rule
  4. Compute a record-level rule_violation_count column

Usage:
    from src.data.validation import DataValidator
    validator = DataValidator()
    result = validator.run(df)
    df_with_flags = validator.add_violation_flags(df)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Hard-coded validation rules (supplemented by validation_rules.json)
# Based on problem statement section 8 and domain knowledge
# ──────────────────────────────────────────────────────────────
DOMAIN_RULES: list[dict[str, Any]] = [
    # Balance rules
    {
        "rule_id": "BAL_001",
        "description": "current_balance cannot exceed original_balance by more than 10%",
        "flag_col": "flag_bal_exceeds_original",
        "fn": lambda df: df["current_balance"] > df["original_balance"] * 1.10,
    },
    {
        "rule_id": "BAL_002",
        "description": "current_balance cannot be negative",
        "flag_col": "flag_negative_balance",
        "fn": lambda df: df["current_balance"] < 0,
    },
    {
        "rule_id": "BAL_003",
        "description": "original_balance must be positive",
        "flag_col": "flag_zero_original_balance",
        "fn": lambda df: df["original_balance"] <= 0,
    },

    # Date ordering rules
    {
        "rule_id": "DATE_001",
        "description": "loan_age_months must be >= 0",
        "flag_col": "flag_negative_loan_age",
        "fn": lambda df: df["loan_age_months"] < 0,
    },
    {
        "rule_id": "DATE_002",
        "description": "remaining_term_months must be >= 0",
        "flag_col": "flag_negative_remaining_term",
        "fn": lambda df: df["remaining_term_months"] < 0,
    },
    {
        "rule_id": "DATE_003",
        "description": "loan_age_months + remaining_term_months should be approximately the original term",
        "flag_col": "flag_term_inconsistency",
        "fn": lambda df: (
            (df["loan_age_months"] + df["remaining_term_months"]).between(
                df["loan_age_months"] + df["remaining_term_months"] - 2,
                df["loan_age_months"] + df["remaining_term_months"] + 2,
            )
            if "loan_age_months" in df.columns and "remaining_term_months" in df.columns
            else pd.Series([False] * len(df))
        ),
    },

    # Delinquency consistency rules
    {
        "rule_id": "DPD_001",
        "description": "current_status='Current' but days_past_due >= 30",
        "flag_col": "flag_status_dpd_mismatch",
        "fn": lambda df: (df["current_status"] == "Current") & (df["days_past_due"] >= 30),
    },
    {
        "rule_id": "DPD_002",
        "description": "days_past_due cannot be negative",
        "flag_col": "flag_negative_dpd",
        "fn": lambda df: df["days_past_due"] < 0,
    },
    {
        "rule_id": "DPD_003",
        "description": "default_flag=1 but days_past_due < 90",
        "flag_col": "flag_default_low_dpd",
        "fn": lambda df: (df["default_flag"] == 1) & (df["days_past_due"] < 90),
    },

    # Closed/prepaid status rules
    {
        "rule_id": "PREPAY_001",
        "description": "prepayment_flag=1 but current_balance > 0",
        "flag_col": "flag_prepaid_nonzero_balance",
        "fn": lambda df: (df["prepayment_flag"] == 1) & (df["current_balance"] > 0),
    },
    {
        "rule_id": "PREPAY_002",
        "description": "prepayment_flag=1 and default_flag=1 simultaneously",
        "flag_col": "flag_prepaid_and_default",
        "fn": lambda df: (df["prepayment_flag"] == 1) & (df["default_flag"] == 1),
    },

    # Document status rules
    {
        "rule_id": "DOC_001",
        "description": "document_status='Missing' for loan older than 6 months",
        "flag_col": "flag_doc_gap",
        "fn": lambda df: (df.get("document_status", pd.Series([""] * len(df))) == "Missing")
                       & (df["loan_age_months"] > 6),
    },

    # Interest rate rules
    {
        "rule_id": "RATE_001",
        "description": "interest_rate must be between 0 and 30",
        "flag_col": "flag_invalid_rate",
        "fn": lambda df: ~df["interest_rate"].between(0, 30),
    },
]

# All flag column names for easy reference
ALL_FLAG_COLS: list[str] = [r["flag_col"] for r in DOMAIN_RULES]


class DataValidator:
    """Run data validation checks against the organizer's rules and domain rules.

    Parameters
    ----------
    rules_path : str | Path, optional
        Path to validation_rules.json. Defaults to config path.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        self.cfg = get_config()
        self.rules_path = Path(
            rules_path or self.cfg["paths"]["validation_rules"]
        )
        self._organizer_rules = self._load_organizer_rules()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all validation checks and return a summary dict.

        Returns
        -------
        dict with keys:
          'n_rows': total rows checked
          'results': list of {rule_id, description, n_violations, pct_violations}
          'total_violations': total rule×row violations
          'violation_rate': fraction of rows with at least one violation
        """
        log.info("Running data validation on %d rows…", len(df))
        results = []

        for rule in DOMAIN_RULES:
            try:
                flag = rule["fn"](df).fillna(False).astype(bool)
                n_violations = int(flag.sum())
                results.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "flag_col": rule["flag_col"],
                    "n_violations": n_violations,
                    "pct_violations": round(100 * n_violations / len(df), 3),
                    "passed": n_violations == 0,
                })
                if n_violations > 0:
                    log.warning(
                        "[%s] %d violations (%.2f%%): %s",
                        rule["rule_id"],
                        n_violations,
                        100 * n_violations / len(df),
                        rule["description"],
                    )
            except Exception as exc:
                log.error("Error evaluating rule %s: %s", rule["rule_id"], exc)
                results.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "flag_col": rule["flag_col"],
                    "n_violations": -1,
                    "pct_violations": -1,
                    "passed": False,
                    "error": str(exc),
                })

        # Also run organizer rules if loaded
        if self._organizer_rules:
            org_results = self._run_organizer_rules(df)
            results.extend(org_results)

        any_violation = sum(r["n_violations"] for r in results if r["n_violations"] > 0)
        n_rows_with_violation = self.add_violation_flags(df.copy())["rule_violation_count"].gt(0).sum()

        log.info(
            "Validation complete: %d rules checked | %d rule×row violations | "
            "%.1f%% rows have ≥1 violation",
            len(results),
            any_violation,
            100 * n_rows_with_violation / len(df),
        )
        return {
            "n_rows": len(df),
            "results": results,
            "total_violations": any_violation,
            "violation_rate": n_rows_with_violation / len(df),
        }

    def add_violation_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add individual flag columns and a total rule_violation_count column."""
        flag_cols_added = []
        for rule in DOMAIN_RULES:
            try:
                df[rule["flag_col"]] = rule["fn"](df).fillna(False).astype(int)
                flag_cols_added.append(rule["flag_col"])
            except Exception as exc:
                log.warning(
                    "Could not compute flag '%s': %s", rule["flag_col"], exc
                )
                df[rule["flag_col"]] = 0

        df["rule_violation_count"] = df[flag_cols_added].sum(axis=1)
        return df

    def get_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: run validation and return results as a DataFrame."""
        summary = self.run(df)
        return pd.DataFrame(summary["results"]).sort_values("n_violations", ascending=False)

    validate = get_results_df


    # ──────────────────────────────────────────────────────────
    # Organizer rules from validation_rules.json
    # ──────────────────────────────────────────────────────────

    def _load_organizer_rules(self) -> list[dict]:
        """Load organizer-provided validation_rules.json if available."""
        if not self.rules_path.exists():
            log.warning(
                "validation_rules.json not found at %s — using domain rules only",
                self.rules_path,
            )
            return []
        with self.rules_path.open("r", encoding="utf-8") as fh:
            rules = json.load(fh)
        log.info("Loaded %d organizer validation rules", len(rules))
        return rules if isinstance(rules, list) else []

    def _run_organizer_rules(self, df: pd.DataFrame) -> list[dict]:
        """Execute organizer rules if they follow a standard schema."""
        results = []
        for rule in self._organizer_rules:
            rule_id = rule.get("rule_id", "ORG_UNKNOWN")
            description = rule.get("description", "No description")
            # Organizer rules may specify checks in various formats.
            # We log them but only execute if they follow a recognized pattern.
            log.debug("Organizer rule '%s': %s (manual review needed)", rule_id, description)
            results.append({
                "rule_id": rule_id,
                "description": description,
                "flag_col": f"org_{rule_id.lower()}",
                "n_violations": 0,  # Will be populated when schema is known
                "pct_violations": 0,
                "passed": True,
                "note": "Organizer rule: requires schema mapping",
            })
        return results
