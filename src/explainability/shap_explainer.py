"""
src/explainability/shap_explainer.py
--------------------------------------
SHAP-based global and local explainability for all ML models.

Global: mean |SHAP| bar chart (top-20 features per model)
Local:  per-loan waterfall / force plot
Output: SHAP value CSVs + PNG plots saved to outputs/shap/

Usage:
    from src.explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(model, feature_cols)
    explainer.compute(X_test)
    explainer.plot_global(save_path="outputs/shap/default_global.png")
    explainer.plot_local(loan_idx=42, save_path="outputs/shap/loan_42.png")
    shap_df = explainer.get_shap_dataframe()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for LightGBM / XGBoost models.

    Parameters
    ----------
    model : fitted model object
        Must have a `.model` attribute (i.e. a BaseModel instance or the raw estimator).
    feature_cols : list of str
        Column names matching the feature matrix used at training time.
    model_name : str
        Label used in plot titles and filenames.
    """

    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.model_name = model_name
        self._shap_values: np.ndarray | None = None
        self._base_value: float | None = None
        self._explainer: Any = None

    # ──────────────────────────────────────────────────────────
    # Computing SHAP values
    # ──────────────────────────────────────────────────────────

    def compute(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int = 5000,
    ) -> np.ndarray:
        """Compute SHAP values for the given feature matrix.

        For large datasets, computes on a random subsample to save memory.

        Parameters
        ----------
        max_samples : int
            Maximum number of rows to compute SHAP for. Default 5000.

        Returns
        -------
        np.ndarray
            SHAP values array (n_samples × n_features).
        """
        try:
            import shap
        except ImportError:
            log.error("shap not installed. Run: pip install shap")
            raise

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Subsample for speed
        if max_samples is not None and len(X_arr) > max_samples:
            idx = np.random.choice(len(X_arr), max_samples, replace=False)
            X_sample = X_arr[idx]
            log.info(
                "SHAP: subsampling %d → %d rows for efficiency", len(X_arr), max_samples
            )
        else:
            X_sample = X_arr


        # Get raw estimator
        raw_model = getattr(self.model, "model", self.model)

        log.info("Computing SHAP values for %s (%d rows)…", self.model_name, len(X_sample))
        self._explainer = shap.TreeExplainer(raw_model)
        shap_output = self._explainer(X_sample)

        # Handle multi-class: use class 1 for binary, sum across classes otherwise
        if hasattr(shap_output, "values"):
            vals = shap_output.values
            if vals.ndim == 3:
                # Multi-class (n_samples, n_features, n_classes) — use argmax class
                vals = vals[:, :, 1] if vals.shape[2] == 2 else vals.mean(axis=2)
            self._shap_values = vals
            self._base_value = float(
                shap_output.base_values.mean()
                if hasattr(shap_output, "base_values") else 0.0
            )
        else:
            self._shap_values = np.array(shap_output)

        log.info("SHAP values computed: shape=%s", str(self._shap_values.shape))
        return self._shap_values

    # ──────────────────────────────────────────────────────────
    # Global importance
    # ──────────────────────────────────────────────────────────

    def get_global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return global feature importance as mean |SHAP| values."""
        self._check_computed()
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_cols[:len(mean_abs)],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    def plot_global(
        self,
        top_n: int = 20,
        save_path: str | None = None,
    ) -> None:
        """Plot and optionally save global SHAP summary bar chart."""
        self._check_computed()
        import shap
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            feature_names=self.feature_cols[:self._shap_values.shape[1]],
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        plt.title(f"{self.model_name} — Global Feature Importance (SHAP)", fontsize=14)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Global SHAP plot saved → %s", save_path)
        plt.close(fig)

    def plot_local(
        self,
        X: pd.DataFrame | np.ndarray,
        loan_idx: int = 0,
        save_path: str | None = None,
    ) -> None:
        """Plot SHAP waterfall (force plot) for a single loan."""
        self._check_computed()
        import shap
        import matplotlib.pyplot as plt

        sv = self._shap_values[loan_idx]
        base = self._base_value or 0.0

        exp = shap.Explanation(
            values=sv,
            base_values=base,
            feature_names=self.feature_cols[:len(sv)],
        )

        shap.waterfall_plot(exp, show=False)
        plt.title(f"{self.model_name} — Loan #{loan_idx} SHAP Waterfall", fontsize=12)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Local SHAP waterfall saved → %s", save_path)
        plt.close()

    # ──────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────

    def get_shap_dataframe(self) -> pd.DataFrame:
        """Return SHAP values as a DataFrame (n_samples × n_features)."""
        self._check_computed()
        cols = self.feature_cols[:self._shap_values.shape[1]]
        return pd.DataFrame(self._shap_values, columns=cols)

    def save_shap_csv(self, path: str = "outputs/shap/shap_values.csv") -> Path:
        """Save SHAP values to CSV."""
        df = self.get_shap_dataframe()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info("SHAP values CSV saved → %s", out)
        return out

    def get_top_drivers(self, n_top: int = 3) -> list[str]:
        """Return per-row pipe-separated top-N SHAP feature names.

        Used to populate the 'top_drivers' column in submission.csv.
        """
        self._check_computed()
        cols = np.array(self.feature_cols[:self._shap_values.shape[1]])
        top_idx = np.argsort(np.abs(self._shap_values), axis=1)[:, ::-1][:, :n_top]
        return ["|".join(cols[row].tolist()) for row in top_idx]

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _check_computed(self) -> None:
        if self._shap_values is None:
            raise RuntimeError("Call compute(X) before accessing SHAP values.")
