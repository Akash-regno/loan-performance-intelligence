"""
src/anomaly/detector.py
------------------------
Unsupervised anomaly detection ensemble:
  - Isolation Forest (global outliers)
  - Local Outlier Factor (local density anomalies)
  - HBOS — Histogram-Based Outlier Score (fast univariate)

Final anomaly_score ∈ [0, 1] is a weighted average of normalised
rank scores from all three detectors.

Features used: same as ML models (no target columns).

Usage:
    from src.anomaly.detector import AnomalyDetector
    detector = AnomalyDetector()
    detector.fit(train_df, feature_cols)
    scores = detector.predict(test_df)   # Returns Series: anomaly_score
    top20 = detector.get_top_examples(test_df, scores, n=20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class AnomalyDetector:
    """Ensemble anomaly detector: IF + LOF + HBOS.

    Parameters
    ----------
    contamination : float
        Expected fraction of outliers in training data (default 0.05 = 5%).
    weights : dict
        Ensemble weights for each detector.
    """

    def __init__(
        self,
        contamination: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        cfg = get_config()["anomaly"]
        self.contamination = contamination or cfg["contamination"]
        self.weights = weights or cfg["ensemble_weights"]
        self.n_top_examples = cfg["n_top_examples"]

        self._if_model: Any = None
        self._lof_model: Any = None
        self._hbos_model: Any = None
        self.feature_cols: list[str] = []
        self._is_fitted = False

    # ──────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> "AnomalyDetector":
        """Fit all three detectors on training data.

        Parameters
        ----------
        feature_cols : list of str
            Numeric features to use. If None, all numeric columns (except IDs) are used.
        """
        self.feature_cols = feature_cols or self._auto_select_features(df)
        X = self._prepare_features(df)

        log.info(
            "Fitting anomaly ensemble on %d rows × %d features…",
            X.shape[0], X.shape[1],
        )

        try:
            from pyod.models.iforest import IForest
            from pyod.models.lof import LOF
            from pyod.models.hbos import HBOS

            self._if_model = IForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
            self._lof_model = LOF(
                contamination=self.contamination, n_neighbors=20, n_jobs=-1
            )
            self._hbos_model = HBOS(contamination=self.contamination, n_bins=10)

            self._if_model.fit(X)
            log.info("Isolation Forest fitted.")
            self._lof_model.fit(X)
            log.info("LOF fitted.")
            self._hbos_model.fit(X)
            log.info("HBOS fitted.")

        except ImportError:
            log.warning("PyOD not installed — using sklearn IsolationForest fallback")
            from sklearn.ensemble import IsolationForest

            self._if_model = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
            self._if_model.fit(X)
            self._lof_model = None
            self._hbos_model = None

        self._is_fitted = True
        log.info("Anomaly ensemble fitted.")
        return self

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Compute anomaly_score ∈ [0, 1] for each row.

        Higher score = more anomalous.
        """
        self._check_fitted()
        X = self._prepare_features(df)

        scores_list = []
        weight_list = []

        # Isolation Forest score
        if self._if_model is not None:
            try:
                # PyOD models have decision_scores_; sklearn has decision_function
                if hasattr(self._if_model, "decision_function_"):
                    raw = self._if_model.decision_function(X)
                elif hasattr(self._if_model, "decision_function"):
                    raw = -self._if_model.decision_function(X)  # sklearn: negate
                else:
                    raw = self._if_model.predict_proba(X)[:, 1]
                scores_list.append(self._rank_normalize(raw))
                weight_list.append(self.weights.get("isolation_forest", 0.4))
            except Exception as exc:
                log.warning("IF scoring failed: %s", exc)

        # LOF score
        if self._lof_model is not None:
            try:
                raw = self._lof_model.predict_proba(X)[:, 1]
                scores_list.append(self._rank_normalize(raw))
                weight_list.append(self.weights.get("lof", 0.3))
            except Exception as exc:
                log.warning("LOF scoring failed: %s", exc)

        # HBOS score
        if self._hbos_model is not None:
            try:
                raw = self._hbos_model.predict_proba(X)[:, 1]
                scores_list.append(self._rank_normalize(raw))
                weight_list.append(self.weights.get("hbos", 0.3))
            except Exception as exc:
                log.warning("HBOS scoring failed: %s", exc)

        if not scores_list:
            log.warning("All detectors failed — returning zeros")
            return pd.Series(0.0, index=df.index)

        # Weighted average
        total_weight = sum(weight_list)
        weights_norm = [w / total_weight for w in weight_list]
        ensemble_score = sum(s * w for s, w in zip(scores_list, weights_norm))

        return pd.Series(np.clip(ensemble_score, 0, 1), index=df.index, name="anomaly_score")

    def predict_binary(self, df: pd.DataFrame, threshold: float = None) -> pd.Series:
        """Return binary anomaly flags (1 = anomaly)."""
        scores = self.predict(df)
        t = threshold or self._percentile_threshold(scores)
        return (scores >= t).astype(int)

    # ──────────────────────────────────────────────────────────
    # Top examples
    # ──────────────────────────────────────────────────────────

    def get_top_examples(
        self,
        df: pd.DataFrame,
        scores: pd.Series | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Return top-N anomalous loans with their scores and key features.

        Minimum required output: 20 reviewer-ready examples (per problem statement).
        """
        n = n or self.n_top_examples
        if scores is None:
            scores = self.predict(df)

        top_idx = scores.nlargest(n).index
        top_df = df.loc[top_idx].copy()
        top_df["anomaly_score"] = scores.loc[top_idx].values

        # Include key contextual columns
        show_cols = [
            "loan_id", "month_index", "current_status", "days_past_due",
            "current_balance", "original_balance", "credit_score_band",
            "ltv_band", "servicer_name", "anomaly_score",
        ]
        available = ["anomaly_score"] + [c for c in show_cols if c in top_df.columns and c != "anomaly_score"]
        return top_df[available].sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, directory: str | Path = "models/anomaly") -> Path:
        import pickle

        self._check_fitted()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "anomaly_detector.pkl"
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("AnomalyDetector saved → %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        import pickle

        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numeric feature matrix, imputing NaN with column medians."""
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].copy().astype(float)
        X = X.fillna(X.median())
        return X.values

    def _auto_select_features(self, df: pd.DataFrame) -> list[str]:
        """Select all numeric columns except identifiers and targets."""
        exclude = {
            "loan_id", "month_index", "reporting_month", "origination_month",
            "last_updated_at", "next_3m_delinquency_flag", "next_6m_delinquency_flag",
            "next_12m_default_flag", "next_12m_prepayment_flag",
            "next_state", "exception_required", "exception_type",
        }
        return [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]

    @staticmethod
    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize raw scores to [0, 1] via rank-based normalization."""
        from scipy.stats import rankdata
        ranks = rankdata(scores)
        return (ranks - 1) / max(len(ranks) - 1, 1)

    def _percentile_threshold(self, scores: pd.Series) -> float:
        """Threshold at the (1 - contamination) percentile."""
        return float(np.percentile(scores, (1 - self.contamination) * 100))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector must be fit() before predict().")
