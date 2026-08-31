"""
src/anomaly/exception.py
-------------------------
Rule-based exception detection combined with ML exception prediction.

Rule-based exceptions come directly from validation_rules.json + domain rules
(already computed in src/data/validation.py as flag columns).

This module:
  1. Aggregates all flag columns into a single exception summary per loan-period
  2. Generates exception_type labels from the triggered rules
  3. Produces 20+ reviewer-ready exception examples for the dashboard
  4. Provides SHAP-based driver explanation per exception

Usage:
    from src.anomaly.exception import ExceptionEngine
    engine = ExceptionEngine()
    df_with_exceptions = engine.run(df)
    top20 = engine.get_reviewer_examples(df_with_exceptions, n=20)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# Maps flag column → exception type label
FLAG_TO_EXCEPTION_TYPE = {
    "flag_bal_exceeds_original":    "balance_error",
    "flag_negative_balance":        "balance_error",
    "flag_zero_original_balance":   "balance_error",
    "flag_negative_loan_age":       "date_violation",
    "flag_negative_remaining_term": "date_violation",
    "flag_status_dpd_mismatch":     "status_conflict",
    "flag_negative_dpd":            "date_violation",
    "flag_default_low_dpd":         "status_conflict",
    "flag_prepaid_nonzero_balance":  "status_conflict",
    "flag_prepaid_and_default":     "status_conflict",
    "flag_doc_gap":                 "doc_missing",
    "flag_invalid_rate":            "data_quality",
    "servicer_conflict_flag":       "servicer_dispute",
    "stale_record_flag":            "stale_record",
}

ALL_FLAG_COLS = list(FLAG_TO_EXCEPTION_TYPE.keys())

# Priority ordering: which exception type to report when multiple flags triggered
EXCEPTION_PRIORITY = [
    "status_conflict", "balance_error", "servicer_dispute",
    "doc_missing", "date_violation", "stale_record", "data_quality",
]


class ExceptionEngine:
    """Aggregate rule-based exception flags and generate reviewer summaries."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exception_required and exception_type columns to df.

        Parameters
        ----------
        df : DataFrame
            Must have been processed by DataValidator.add_violation_flags() first.

        Returns
        -------
        DataFrame with added columns:
          exception_required  : int (0/1)
          exception_type      : str (e.g. 'status_conflict', '' if none)
          n_exceptions        : int (total rules triggered)
          exception_drivers   : str (pipe-separated list of triggered rule IDs)
        """
        df = df.copy()
        available_flags = [c for c in ALL_FLAG_COLS if c in df.columns]

        if not available_flags:
            log.warning("No exception flag columns found — have you run DataValidator first?")
            df["exception_required"] = 0
            df["exception_type"] = ""
            df["n_exceptions"] = 0
            df["exception_drivers"] = ""
            return df

        # Count triggered rules per row
        df["n_exceptions"] = df[available_flags].sum(axis=1)
        df["exception_required"] = (df["n_exceptions"] > 0).astype(int)

        # Determine primary exception type (highest priority triggered)
        df["exception_type"] = df.apply(
            lambda row: self._get_exception_type(row, available_flags), axis=1
        )

        # List of all triggered drivers
        df["exception_drivers"] = df.apply(
            lambda row: self._get_exception_drivers(row, available_flags), axis=1
        )

        n_exceptions = int(df["exception_required"].sum())
        log.info(
            "Exception engine: %d rows flagged (%.1f%%) | type distribution: %s",
            n_exceptions,
            100 * n_exceptions / len(df),
            df.loc[df["exception_required"] == 1, "exception_type"]
            .value_counts().to_dict(),
        )
        return df

    def get_reviewer_examples(
        self,
        df: pd.DataFrame,
        n: int = 20,
        shap_values: np.ndarray | None = None,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return top-N exception examples ready for human review.

        Selects the most complex exceptions (highest n_exceptions) and
        enriches with SHAP driver explanation if available.

        Returns a DataFrame with all key columns a reviewer needs.
        """
        if "exception_required" not in df.columns:
            log.warning("Run engine.run(df) before get_reviewer_examples()")
            return pd.DataFrame()

        # Filter to exception rows
        exc_df = df[df["exception_required"] == 1].copy()
        if len(exc_df) == 0:
            log.warning("No exceptions found in DataFrame.")
            return pd.DataFrame()

        # Sort by severity: number of exceptions desc, then anomaly_score desc
        sort_cols = ["n_exceptions"]
        if "anomaly_score" in exc_df.columns:
            sort_cols.append("anomaly_score")
        exc_df = exc_df.sort_values(sort_cols, ascending=False).head(n)

        # Build reviewer-ready output
        show_cols = [
            "loan_id", "month_index", "reporting_month",
            "current_status", "days_past_due", "current_balance",
            "original_balance", "credit_score_band", "ltv_band",
            "servicer_name", "exception_type", "n_exceptions",
            "exception_drivers",
        ]
        if "anomaly_score" in exc_df.columns:
            show_cols.append("anomaly_score")

        available = [c for c in show_cols if c in exc_df.columns]
        result = exc_df[available].reset_index(drop=True)

        # Add SHAP top drivers if provided
        if shap_values is not None and feature_cols is not None:
            exc_idx = exc_df.index.tolist()
            exc_shap = shap_values[exc_idx] if len(shap_values) > max(exc_idx) else None
            if exc_shap is not None:
                top_shap_features = [
                    "|".join(
                        np.array(feature_cols)[np.argsort(np.abs(row))[::-1][:3]].tolist()
                    )
                    for row in exc_shap
                ]
                result["shap_top_drivers"] = top_shap_features

        log.info("Reviewer examples generated: %d rows", len(result))
        return result

    def export_top_examples(
        self, df: pd.DataFrame, output_path: str = "outputs/anomalies/top20_examples.csv"
    ) -> None:
        """Export top-20 exception examples to CSV."""
        from pathlib import Path

        examples = self.get_reviewer_examples(df, n=20)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        examples.to_csv(out, index=False)
        log.info("Top-20 exception examples exported → %s", out)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_exception_type(row: pd.Series, flag_cols: list[str]) -> str:
        """Return the highest-priority exception type for a row."""
        triggered_types = set()
        for col in flag_cols:
            if row.get(col, 0) == 1:
                exc_type = FLAG_TO_EXCEPTION_TYPE.get(col, "data_quality")
                triggered_types.add(exc_type)

        if not triggered_types:
            return ""

        for priority_type in EXCEPTION_PRIORITY:
            if priority_type in triggered_types:
                return priority_type

        return list(triggered_types)[0]

    @staticmethod
    def _get_exception_drivers(row: pd.Series, flag_cols: list[str]) -> str:
        """Return pipe-separated list of triggered flag column names."""
        triggered = [col for col in flag_cols if row.get(col, 0) == 1]
        return "|".join(triggered)
