"""
src/features/imbalance.py
--------------------------
Class imbalance handling utilities.

Strategies implemented:
  1. SMOTE-NC (Synthetic Minority Oversampling for Nominal + Continuous)
     - Applied ONLY to training data, never to test/validation
  2. Class weight computation (scale_pos_weight for XGBoost / LightGBM)
  3. Optimal threshold tuning from validation set (maximize F1 or PR-AUC)

Usage:
    from src.features.imbalance import ImbalanceHandler
    handler = ImbalanceHandler()

    # 1. Compute class weights (used in model params)
    weights = handler.compute_class_weights(y_train)

    # 2. Oversample training data (optional, use with caution on large datasets)
    X_resampled, y_resampled = handler.oversample(X_train, y_train, cat_cols)

    # 3. Tune threshold on validation
    threshold = handler.tune_threshold(y_val, y_prob_val, metric='f1')
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

from src.utils.logger import get_logger

log = get_logger(__name__)


class ImbalanceHandler:
    """Handle class imbalance in binary classification tasks."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # ──────────────────────────────────────────────────────────
    # Class weights
    # ──────────────────────────────────────────────────────────

    def compute_class_weights(self, y: pd.Series | np.ndarray) -> dict[int, float]:
        """Compute balanced class weights (n_samples / (n_classes × n_class_i)).

        Returns a dict {0: weight_negative, 1: weight_positive} compatible
        with sklearn's class_weight parameter.
        """
        from sklearn.utils.class_weight import compute_class_weight

        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        log.info("Class weights: %s", weight_dict)
        return weight_dict

    def compute_scale_pos_weight(self, y: pd.Series | np.ndarray) -> float:
        """Compute scale_pos_weight for XGBoost/LightGBM binary tasks.

        scale_pos_weight = n_negative / n_positive

        Returns
        -------
        float
            Recommended scale_pos_weight value.
        """
        y = np.asarray(y).astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())

        if n_pos == 0:
            log.warning("No positive samples found — scale_pos_weight = 1.0")
            return 1.0

        spw = n_neg / n_pos
        log.info(
            "scale_pos_weight = %.2f (n_neg=%d, n_pos=%d, ratio=%.1f:1)",
            spw, n_neg, n_pos, spw,
        )
        return spw

    # ──────────────────────────────────────────────────────────
    # Oversampling
    # ──────────────────────────────────────────────────────────

    def oversample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_cols: list[str] | None = None,
        strategy: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-NC (handles mixed numeric/categorical features).

        Parameters
        ----------
        X : array-like
            Feature matrix (n_samples × n_features).
        y : array-like
            Binary target vector.
        categorical_cols : list of str, optional
            Column names (if X is DataFrame) or integer indices that are categorical.
            Required for SMOTE-NC; if None, falls back to SMOTE.
        strategy : float
            Desired ratio of minority / majority after resampling.
            Default 0.2 = oversample minority to 20% of majority.

        Returns
        -------
        X_res, y_res : ndarray
            Resampled feature matrix and target.
        """
        try:
            from imblearn.over_sampling import SMOTENC, SMOTE

            if isinstance(X, pd.DataFrame) and categorical_cols:
                cat_indices = [
                    X.columns.tolist().index(c)
                    for c in categorical_cols
                    if c in X.columns
                ]
                if cat_indices:
                    sampler = SMOTENC(
                        categorical_features=cat_indices,
                        sampling_strategy=strategy,
                        random_state=self.random_state,
                    )
                else:
                    sampler = SMOTE(
                        sampling_strategy=strategy,
                        random_state=self.random_state,
                    )
            else:
                sampler = SMOTE(
                    sampling_strategy=strategy,
                    random_state=self.random_state,
                )

            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            y_arr = np.asarray(y).astype(int)

            X_res, y_res = sampler.fit_resample(X_arr, y_arr)

            log.info(
                "SMOTE resampling: %d → %d samples (minority: %d → %d)",
                len(y_arr), len(y_res),
                int((y_arr == 1).sum()), int((y_res == 1).sum()),
            )
            return X_res, y_res

        except ImportError:
            log.warning("imbalanced-learn not installed — returning original data")
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            return X_arr, np.asarray(y).astype(int)

    # ──────────────────────────────────────────────────────────
    # Threshold tuning
    # ──────────────────────────────────────────────────────────

    def tune_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        metric: Literal["f1", "recall_at_precision"] = "f1",
        min_precision: float = 0.5,
        threshold_range: tuple[float, float] = (0.05, 0.95),
        n_steps: int = 100,
    ) -> float:
        """Find the optimal decision threshold on a validation set.

        Parameters
        ----------
        metric : {'f1', 'recall_at_precision'}
            Optimization objective.
        min_precision : float
            Minimum precision required (used if metric='recall_at_precision').
        threshold_range : tuple
            (min_threshold, max_threshold) search space.
        n_steps : int
            Number of threshold values to try.

        Returns
        -------
        float
            Optimal threshold value.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)

        thresholds = np.linspace(*threshold_range, n_steps)
        best_score = -1.0
        best_threshold = 0.5

        if metric == "f1":
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = t

        elif metric == "recall_at_precision":
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
            valid = precisions >= min_precision
            if valid.any():
                best_recall = recalls[valid].max()
                best_threshold = float(
                    pr_thresholds[valid[:-1]][
                        np.argmax(recalls[valid[:-1]])
                    ] if len(pr_thresholds[valid[:-1]]) > 0 else 0.5
                )
                best_score = best_recall
            else:
                log.warning(
                    "No threshold achieves precision >= %.2f — defaulting to 0.5",
                    min_precision,
                )
                best_threshold = 0.5

        log.info(
            "Optimal threshold (metric=%s): %.4f → score=%.4f",
            metric, best_threshold, best_score,
        )
        return float(best_threshold)

    def class_imbalance_summary(self, y: pd.Series | np.ndarray) -> dict:
        """Return a summary dict of class distribution."""
        y = np.asarray(y).astype(int)
        n_total = len(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        return {
            "n_total": n_total,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": round(n_pos / n_total, 5) if n_total > 0 else 0,
            "imbalance_ratio": round(n_neg / n_pos, 2) if n_pos > 0 else float("inf"),
        }
