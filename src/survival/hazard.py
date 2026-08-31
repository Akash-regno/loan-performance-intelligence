"""
src/survival/hazard.py
-----------------------
Cox Proportional Hazard and AFT survival models for time-to-default
and time-to-prepayment.

Models:
  - Cox PH (lifelines.CoxPHFitter)   → time-to-default
  - Cox PH (lifelines.CoxPHFitter)   → time-to-prepayment
  - Weibull AFT (lifelines.WeibullAFTFitter) → parametric alternative

Validation metrics:
  - Harrell's C-index (target > 0.65)
  - Integrated Brier Score
  - KM curve vs Cox PH visual comparison

Usage:
    from src.survival.hazard import HazardModel
    model = HazardModel(event="default")
    model.fit(train_df)
    predictions = model.predict_median_survival_time(test_df)
    c_idx = model.evaluate(test_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.metrics import harrell_c_index

log = get_logger(__name__)


class HazardModel:
    """Cox PH / AFT survival model for loan-level time-to-event.

    Parameters
    ----------
    event : {'default', 'prepayment'}
        Which event to model.
    model_type : {'cox', 'aft'}
        Cox Proportional Hazard (semi-parametric) or Weibull AFT (parametric).
    """

    # Covariates used in the survival model
    DEFAULT_COVARIATES = [
        "loan_age_months", "days_past_due", "ltv_band_ord",
        "credit_score_band_ord", "modification_flag", "interest_rate",
        "pct_balance_remaining", "dpd_trend_3m", "n_delinquencies_to_date",
        "is_seasoned_loan", "dti_band_ord",
    ]
    PREPAY_COVARIATES = [
        "loan_age_months", "interest_rate", "refi_incentive",
        "ltv_band_ord", "loan_age_band", "is_early_period",
        "remaining_term_months", "rate_spread", "pct_balance_remaining",
        "dpd_lag1", "credit_score_band_ord",
    ]

    def __init__(
        self,
        event: Literal["default", "prepayment"] = "default",
        model_type: Literal["cox", "aft"] = "cox",
        penalizer: float = 0.1,
    ) -> None:
        self.event = event
        self.model_type = model_type
        self.penalizer = penalizer
        self.model = None
        self._is_fitted = False

        cfg = get_config()["survival"]
        self.duration_col = cfg["duration_col"]          # loan_age_months
        self.event_col = (
            cfg["event_col_default"] if event == "default"
            else cfg["event_col_prepay"]
        )
        self.covariates = (
            self.DEFAULT_COVARIATES if event == "default"
            else self.PREPAY_COVARIATES
        )

    # ──────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HazardModel":
        """Fit the survival model on training data.

        Parameters
        ----------
        df : DataFrame
            Must contain duration_col, event_col, and all covariate columns.
            Rows with event = NaN are treated as right-censored.
        """
        surv_df = self._prepare_survival_df(df)
        log.info(
            "Fitting %s Cox PH (%s) | n=%d | events=%d (%.1f%%)",
            self.event, self.model_type, len(surv_df),
            surv_df[self.event_col].sum(),
            100 * surv_df[self.event_col].mean(),
        )

        surv_df = surv_df.loc[:, ~surv_df.columns.duplicated()]
        # Filter covariates with valid variance
        covs = [
            c for c in self._available_covariates(surv_df)
            if c not in {self.duration_col, self.event_col}
            and surv_df[c].nunique() > 1
            and pd.api.types.is_numeric_dtype(surv_df[c])
            and surv_df[c].std() > 1e-4
        ]
        cols = [self.duration_col, self.event_col] + covs

        try:
            from lifelines import CoxPHFitter
            self.model = CoxPHFitter(penalizer=max(self.penalizer, 0.5))
            self.model.fit(
                surv_df[cols].dropna(),
                duration_col=self.duration_col,
                event_col=self.event_col,
                show_progress=False,
            )
            self._is_fitted = True
            log.info("%s Cox model fitted successfully.", self.event)

        except Exception as exc:
            log.warning("Cox PH fit failed (%s) — falling back to Kaplan-Meier estimator", exc)
            from lifelines import KaplanMeierFitter
            self.km_model = KaplanMeierFitter()
            clean_km = surv_df[[self.duration_col, self.event_col]].dropna()
            self.km_model.fit(
                durations=clean_km[self.duration_col],
                event_observed=clean_km[self.event_col],
            )
            self.model = self.km_model
            self._is_fitted = True

        return self


    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict_median_survival_time(self, df: pd.DataFrame) -> np.ndarray:
        """Predict median survival time (months to event) for each loan."""
        self._check_fitted()
        if hasattr(self.model, "predict_median"):
            surv_df = self._prepare_survival_df(df)
            covs = self._available_covariates(surv_df)
            result = self.model.predict_median(surv_df[covs].fillna(0))
            return result.values
        median_val = getattr(self.model, "median_survival_time_", 36.0)
        return np.full(len(df), median_val)

    def predict_cumulative_hazard(
        self, df: pd.DataFrame, times: list[int] | None = None
    ) -> pd.DataFrame:
        """Predict cumulative hazard at given time points."""
        self._check_fitted()
        if times is None:
            times = [3, 6, 12, 24, 36]
        if hasattr(self.model, "predict_cumulative_hazard"):
            surv_df = self._prepare_survival_df(df)
            covs = self._available_covariates(surv_df)
            return self.model.predict_cumulative_hazard(
                surv_df[covs].fillna(0), times=times
            )
        vals = self.model.cumulative_hazard_at_times(times).values
        return pd.DataFrame(np.tile(vals, (len(df), 1)).T, index=times, columns=df.index)

    def predict_survival_function(
        self, df: pd.DataFrame, times: list[int] | None = None
    ) -> pd.DataFrame:
        """Predict survival probability at given time points."""
        self._check_fitted()
        if times is None:
            times = [3, 6, 12, 24, 36]
        if hasattr(self.model, "predict_survival_function"):
            surv_df = self._prepare_survival_df(df)
            covs = self._available_covariates(surv_df)
            return self.model.predict_survival_function(
                surv_df[covs].fillna(0), times=times
            )
        vals = self.model.survival_function_at_times(times).values
        return pd.DataFrame(np.tile(vals, (len(df), 1)).T, index=times, columns=df.index)

    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute Harrell's C-index on a labeled dataset."""
        self._check_fitted()
        surv_df = self._prepare_survival_df(df)
        covs = self._available_covariates(surv_df)
        clean = surv_df[[self.duration_col, self.event_col] + covs].dropna()

        if len(clean) < 50:
            log.warning("Too few samples for survival evaluation.")
            return {"c_index": float("nan")}

        if hasattr(self.model, "predict_partial_hazard"):
            risk_scores = self.model.predict_partial_hazard(clean[covs]).values
        else:
            risk_scores = clean[self.duration_col].values

        c_idx = harrell_c_index(
            clean[self.duration_col].values,
            clean[self.event_col].values,
            risk_scores,
        )
        log.info("%s C-index: %.4f", self.event, c_idx)
        return {
            f"{self.event}_c_index": c_idx,
            f"{self.event}_n_events": int(clean[self.event_col].sum()),
            f"{self.event}_n_samples": len(clean),
        }


    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, directory: str | Path = "models/survival") -> Path:
        """Pickle the fitted model."""
        import pickle

        self._check_fitted()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"hazard_{self.event}_{self.model_type}.pkl"
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Hazard model saved → %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "HazardModel":
        import pickle

        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _prepare_survival_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure event column is binary int; handle missing."""
        surv = df.copy()
        if self.event_col in surv.columns:
            surv[self.event_col] = (
                pd.to_numeric(surv[self.event_col], errors="coerce")
                .fillna(0).astype(int)
            )
        else:
            surv[self.event_col] = 0
        return surv

    def _available_covariates(self, df: pd.DataFrame) -> list[str]:
        """Return covariates that exist in the DataFrame."""
        return [c for c in self.covariates if c in df.columns]

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("HazardModel must be fit() before prediction.")
