"""
src/utils/submission.py
-----------------------
Build, validate, and export the final submission.csv.

Responsibilities:
  1. Merge predictions from all models onto the test DataFrame
  2. Validate against submission_template.csv column schema
  3. Ensure probabilities are within [0, 1]
  4. Compute top_drivers from SHAP values
  5. Compute confidence = 1 - entropy of the probability distribution
  6. Write submission.csv with SHA-256 checksum

Usage:
    from src.utils.submission import SubmissionBuilder
    builder = SubmissionBuilder()
    submission_df = builder.build(test_df, predictions_dict, shap_values)
    builder.export(submission_df)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = [
    "loan_id",
    "month_index",
    "prob_next_3m_delinquency",
    "prob_next_6m_delinquency",
    "prob_next_12m_default",
    "prob_next_12m_prepayment",
    "next_state",
    "exception_required",
    "exception_type",
    "anomaly_score",
    "top_drivers",
    "action",
    "confidence",
]


class SubmissionBuilder:
    """Build and validate the final submission DataFrame."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.out_path = Path(self.cfg["submission"]["output_file"])

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def build(
        self,
        test_df: pd.DataFrame,
        predictions: dict[str, np.ndarray],
        shap_values: np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        hitl_decisions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        """Assemble the submission DataFrame.

        Parameters
        ----------
        test_df : DataFrame
            Test feature matrix (must contain loan_id, month_index).
        predictions : dict
            Keys: model name → values: predicted probabilities or class labels.
            Expected keys:
              'prob_next_3m_delinquency', 'prob_next_6m_delinquency',
              'prob_next_12m_default', 'prob_next_12m_prepayment',
              'next_state', 'exception_required', 'exception_type',
              'anomaly_score'
        shap_values : ndarray, optional
            SHAP values matrix (n_samples × n_features) for the default model.
        feature_cols : list, optional
            Feature column names matching shap_values columns.
        hitl_decisions : DataFrame, optional
            Human review decisions (loan_id → action).

        Returns
        -------
        DataFrame
            Validated submission DataFrame.
        """
        log.info("Building submission DataFrame (%d rows)…", len(test_df))

        sub = test_df[["loan_id", "month_index"]].copy()

        # Probabilities
        for col in [
            "prob_next_3m_delinquency",
            "prob_next_6m_delinquency",
            "prob_next_12m_default",
            "prob_next_12m_prepayment",
        ]:
            sub[col] = self._clip_prob(predictions.get(col, np.zeros(len(test_df))))

        # Class predictions
        sub["next_state"] = predictions.get("next_state", "Unknown")
        sub["exception_required"] = predictions.get(
            "exception_required", np.zeros(len(test_df), dtype=int)
        ).astype(int)
        sub["exception_type"] = predictions.get("exception_type", "")

        # Anomaly score
        sub["anomaly_score"] = self._clip_prob(
            predictions.get("anomaly_score", np.zeros(len(test_df)))
        )

        # Top drivers from SHAP
        sub["top_drivers"] = self._compute_top_drivers(
            shap_values, feature_cols, n_top=3
        )

        # HITL action
        sub["action"] = self._merge_hitl_action(sub["loan_id"], hitl_decisions)

        # Confidence = 1 - normalized entropy across the 4 prob columns
        prob_cols = [
            "prob_next_3m_delinquency",
            "prob_next_6m_delinquency",
            "prob_next_12m_default",
            "prob_next_12m_prepayment",
        ]
        sub["confidence"] = self._compute_confidence(sub[prob_cols].values)

        return sub

    def validate(self, sub: pd.DataFrame) -> None:
        """Validate submission DataFrame against required schema.

        Raises
        ------
        ValueError
            If required columns are missing or probabilities are out of range.
        """
        log.info("Validating submission schema…")
        missing = [c for c in REQUIRED_COLUMNS if c not in sub.columns]
        if missing:
            raise ValueError(f"Submission missing required columns: {missing}")

        prob_cols = [c for c in REQUIRED_COLUMNS if c.startswith("prob_") or c == "anomaly_score"]
        for col in prob_cols:
            if sub[col].isna().any():
                raise ValueError(f"NaN values found in column '{col}'")
            if (sub[col] < 0).any() or (sub[col] > 1).any():
                raise ValueError(
                    f"Probabilities out of [0,1] range in column '{col}'"
                )

        if sub["loan_id"].isna().any():
            raise ValueError("NaN values found in 'loan_id'")

        log.info("✓ Submission validation passed (%d rows, %d cols)", *sub.shape)

    def export(self, sub: pd.DataFrame) -> Path:
        """Write submission.csv and print SHA-256 checksum.

        Returns
        -------
        Path
            Path to the written file.
        """
        self.validate(sub)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(self.out_path, index=False)

        checksum = self._sha256(self.out_path)
        log.info("submission.csv written → %s", self.out_path.resolve())
        log.info("SHA-256: %s", checksum)
        return self.out_path

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _clip_prob(arr: np.ndarray) -> np.ndarray:
        """Clip values to [0, 1] and fill NaN with 0."""
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isnan(arr), 0.0, arr)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _compute_top_drivers(
        shap_values: np.ndarray | None,
        feature_cols: list[str] | None,
        n_top: int = 3,
    ) -> pd.Series:
        """Return pipe-separated top-N SHAP feature names per row."""
        if shap_values is None or feature_cols is None:
            return pd.Series(["" for _ in range(
                len(shap_values) if shap_values is not None else 0
            )])

        top_idx = np.argsort(np.abs(shap_values), axis=1)[:, ::-1][:, :n_top]
        cols = np.array(feature_cols)
        drivers = ["|".join(cols[row].tolist()) for row in top_idx]
        return pd.Series(drivers)

    @staticmethod
    def _merge_hitl_action(
        loan_ids: pd.Series,
        hitl_decisions: pd.DataFrame | None,
    ) -> pd.Series:
        """Merge human reviewer decisions; default to 'pending'."""
        if hitl_decisions is None or hitl_decisions.empty:
            return pd.Series(["pending"] * len(loan_ids))

        decision_map = hitl_decisions.set_index("loan_id")["action"].to_dict()
        return loan_ids.map(decision_map).fillna("pending")

    @staticmethod
    def _compute_confidence(prob_matrix: np.ndarray) -> np.ndarray:
        """Compute per-row confidence as 1 - normalized Shannon entropy.

        Higher = more confident prediction.
        """
        # Clip to avoid log(0)
        p = np.clip(prob_matrix, 1e-9, 1 - 1e-9)
        entropy = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p), axis=1)
        max_entropy = np.log(2) * prob_matrix.shape[1]
        normalized_entropy = entropy / max_entropy
        return np.clip(1.0 - normalized_entropy, 0.0, 1.0)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


SubmissionExporter = SubmissionBuilder

