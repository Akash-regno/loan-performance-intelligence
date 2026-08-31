"""
src/utils/metrics.py
--------------------
Shared evaluation metric functions used across all models.
All functions are stateless and accept numpy arrays or pandas Series.

Metrics implemented:
  - roc_auc_score_safe       : AUC-ROC with guard for single-class edge case
  - pr_auc_score             : Area under Precision-Recall curve
  - brier_score              : Probabilistic calibration quality
  - ece                      : Expected Calibration Error (10-bin)
  - ks_statistic             : Kolmogorov–Smirnov separation
  - lift_at_k                : Lift at top-K% of scores
  - classification_report_df : Full report as a tidy DataFrame
  - harrell_c_index          : Concordance index for survival models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Binary classification metrics
# ──────────────────────────────────────────────────────────────

def roc_auc_score_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC with guard for degenerate single-class input."""
    unique = np.unique(y_true)
    if len(unique) < 2:
        log.warning("AUC-ROC undefined: only class %s present. Returning 0.5.", unique)
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the Precision-Recall curve (preferred for imbalanced data)."""
    unique = np.unique(y_true)
    if len(unique) < 2:
        log.warning("PR-AUC undefined: only class %s present. Returning 0.0.", unique)
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score: mean squared error of predicted probabilities. Lower is better."""
    return float(brier_score_loss(y_true, y_prob))


def ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (weighted mean absolute calibration gap).

    Target: ECE < 0.05 for well-calibrated models.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n_samples = max(len(y_prob), 1)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == bin_edges[i + 1])
        bin_size = int(bin_mask.sum())
        if bin_size > 0:
            bin_acc = float(y_true[bin_mask].mean())
            bin_conf = float(y_prob[bin_mask].mean())
            ece_val += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece_val)



def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic: max separation between score distributions."""
    pos_scores = np.sort(y_prob[y_true == 1])
    neg_scores = np.sort(y_prob[y_true == 0])

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    all_scores = np.sort(np.unique(y_prob))
    cdf_pos = np.searchsorted(pos_scores, all_scores, side="right") / len(pos_scores)
    cdf_neg = np.searchsorted(neg_scores, all_scores, side="right") / len(neg_scores)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Lift at top-K% of predicted scores vs. random baseline.

    Parameters
    ----------
    k : float
        Top fraction (0–1) of records to inspect. Default 0.10 = top 10%.
    """
    n = len(y_true)
    top_k = max(1, int(np.ceil(k * n)))
    idx = np.argsort(y_prob)[::-1][:top_k]
    base_rate = y_true.mean()
    if base_rate == 0:
        return 1.0
    captured_rate = y_true[idx].mean()
    return float(captured_rate / base_rate)


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.5,
) -> float:
    """Maximum recall achievable at a minimum precision threshold."""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    mask = precisions >= min_precision
    if not mask.any():
        return 0.0
    return float(recalls[mask].max())


def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return sklearn classification_report as a tidy DataFrame."""
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).transpose().round(4)


# ──────────────────────────────────────────────────────────────
# Multi-class metrics
# ──────────────────────────────────────────────────────────────

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score for multi-class targets."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ──────────────────────────────────────────────────────────────
# Survival metrics
# ──────────────────────────────────────────────────────────────

def harrell_c_index(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
) -> float:
    """Harrell's concordance index for survival models.

    Target: C-index > 0.65 for a meaningful survival model.
    """
    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(event_times, -predicted_risk, event_observed))
    except ImportError:
        log.warning("lifelines not installed. Returning placeholder C-index.")
        return float("nan")



def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score for multi-class classification."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared probability error). Lower is better."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


# ──────────────────────────────────────────────────────────────
# Composite evaluation report
# ──────────────────────────────────────────────────────────────

def binary_eval_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
) -> dict[str, float]:
    """Compute all standard binary classification metrics in one call.

    Returns a flat dict suitable for logging, MLflow, or display.
    """
    y_pred = (y_prob >= threshold).astype(int)
    report = {
        f"{label}_roc_auc": roc_auc_score_safe(y_true, y_prob),
        f"{label}_pr_auc": pr_auc_score(y_true, y_prob),
        f"{label}_brier": brier_score(y_true, y_prob),
        f"{label}_ece": ece(y_true, y_prob),
        f"{label}_ks": ks_statistic(y_true, y_prob),
        f"{label}_lift10": lift_at_k(y_true, y_prob, k=0.10),
        f"{label}_recall_at_prec50": recall_at_precision(y_true, y_prob, 0.5),
    }

    # F1, precision, recall at chosen threshold
    from sklearn.metrics import f1_score, precision_score, recall_score

    report[f"{label}_f1"] = float(
        f1_score(y_true, y_pred, zero_division=0)
    )
    report[f"{label}_precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    report[f"{label}_recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    report[f"{label}_threshold"] = threshold

    return report
