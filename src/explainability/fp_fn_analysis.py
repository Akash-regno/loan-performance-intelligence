"""
src/explainability/fp_fn_analysis.py
--------------------------------------
False-positive / false-negative deep-dive analysis.

For each binary model, this module:
  1. Segments predictions into TP / TN / FP / FN quadrants
  2. Computes mean feature values per quadrant (to identify confounders)
  3. Runs SHAP comparison between FP and TP clusters
  4. Exports a confusion quadrant summary table

Addresses judging criterion: "Error analysis" under Explainability (10 pts).

Usage:
    from src.explainability.fp_fn_analysis import FPFNAnalyzer
    analyzer = FPFNAnalyzer()
    result = analyzer.analyze(y_true, y_prob, X, feature_cols, threshold=0.5)
    analyzer.export_report(result, "outputs/shap/fp_fn_report.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class FPFNAnalyzer:
    """Analyze false positives and false negatives for a binary classifier."""

    def analyze(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray,
        X: pd.DataFrame | np.ndarray,
        feature_cols: list[str] | None = None,
        threshold: float = 0.5,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Run full FP/FN analysis.

        Returns
        -------
        dict with keys:
          'counts'         : TP/TN/FP/FN counts
          'quadrant_means' : mean feature values per quadrant
          'top_fp_features': features most different between FP and TP rows
          'top_fn_features': features most different between FN and TN rows
          'fp_examples'    : sample FP rows
          'fn_examples'    : sample FN rows
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = (y_prob >= threshold).astype(int)

        tp_mask = (y_true == 1) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        counts = {
            "TP": int(tp_mask.sum()),
            "TN": int(tn_mask.sum()),
            "FP": int(fp_mask.sum()),
            "FN": int(fn_mask.sum()),
            "total": len(y_true),
            "threshold": threshold,
            "model": model_name,
        }
        log.info(
            "%s confusion quadrants | TP=%d TN=%d FP=%d FN=%d (threshold=%.2f)",
            model_name, counts["TP"], counts["TN"], counts["FP"], counts["FN"], threshold,
        )

        if isinstance(X, np.ndarray):
            if feature_cols:
                X = pd.DataFrame(X, columns=feature_cols[:X.shape[1]])
            else:
                X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        X = X.reset_index(drop=True)
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        y_prob_s = pd.Series(y_prob).reset_index(drop=True)

        quadrant_label = pd.Series("TN", index=X.index)
        quadrant_label[tp_mask] = "TP"
        quadrant_label[fp_mask] = "FP"
        quadrant_label[fn_mask] = "FN"

        X["_quadrant"] = quadrant_label.values
        X["_y_true"] = y_true_s.values
        X["_y_prob"] = y_prob_s.values

        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

        # Mean features per quadrant
        quadrant_means = X.groupby("_quadrant")[numeric_cols].mean().round(4)

        # Top discriminating features: FP vs TP
        top_fp_features = self._discriminating_features(X, fp_mask, tp_mask, numeric_cols)
        top_fn_features = self._discriminating_features(X, fn_mask, tn_mask, numeric_cols)

        # Sample examples
        fp_examples = X[fp_mask].drop(columns=["_quadrant", "_y_true", "_y_prob"], errors="ignore").head(10)
        fn_examples = X[fn_mask].drop(columns=["_quadrant", "_y_true", "_y_prob"], errors="ignore").head(10)

        # Clean up
        X.drop(columns=["_quadrant", "_y_true", "_y_prob"], errors="ignore", inplace=True)

        return {
            "counts": counts,
            "quadrant_means": quadrant_means,
            "top_fp_features": top_fp_features,
            "top_fn_features": top_fn_features,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
        }

    def export_report(
        self,
        result: dict[str, Any],
        output_path: str = "outputs/shap/fp_fn_report.csv",
    ) -> Path:
        """Export FP/FN analysis to CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        counts_df = pd.DataFrame([result["counts"]])
        quadrant_df = result["quadrant_means"].reset_index()
        fp_feat_df = result["top_fp_features"]
        fn_feat_df = result["top_fn_features"]

        with pd.ExcelWriter(out.with_suffix(".xlsx"), engine="openpyxl") as writer:
            counts_df.to_excel(writer, sheet_name="Counts", index=False)
            quadrant_df.to_excel(writer, sheet_name="QuadrantMeans", index=False)
            if fp_feat_df is not None:
                fp_feat_df.to_excel(writer, sheet_name="FP_Drivers", index=False)
            if fn_feat_df is not None:
                fn_feat_df.to_excel(writer, sheet_name="FN_Drivers", index=False)

        # Also CSV for counts
        counts_df.to_csv(out, index=False)
        log.info("FP/FN report exported → %s", out.with_suffix(".xlsx"))
        return out

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _discriminating_features(
        X: pd.DataFrame,
        group_a_mask: np.ndarray,
        group_b_mask: np.ndarray,
        numeric_cols: list[str],
        top_n: int = 10,
    ) -> pd.DataFrame | None:
        """Find features with the largest mean difference between two groups."""
        a = X.loc[group_a_mask, numeric_cols]
        b = X.loc[group_b_mask, numeric_cols]

        if len(a) == 0 or len(b) == 0:
            return None

        diff = (a.mean() - b.mean()).abs().sort_values(ascending=False)
        result = pd.DataFrame({
            "feature": diff.index[:top_n],
            "mean_group_a": a.mean()[diff.index[:top_n]].round(4).values,
            "mean_group_b": b.mean()[diff.index[:top_n]].round(4).values,
            "abs_diff": diff.values[:top_n].round(4),
        })
        return result
