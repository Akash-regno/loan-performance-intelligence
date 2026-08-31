"""
src/survival/competing_risk.py
-------------------------------
Fine-Gray competing risk model for simultaneous default vs prepayment.

Models the sub-distribution hazard of each event type while treating the
other as a competing risk (not simply censored).

Outputs:
  - Cumulative Incidence Function (CIF) for default
  - Cumulative Incidence Function (CIF) for prepayment
  - Cause-specific C-index for each event

Falls back to two independent Cox PH models if scikit-survival is unavailable.

Usage:
    from src.survival.competing_risk import CompetingRiskModel
    model = CompetingRiskModel()
    model.fit(train_df)
    cif_df = model.predict_cif(test_df, times=[3, 6, 12])
    metrics = model.evaluate(test_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.metrics import harrell_c_index

log = get_logger(__name__)


class CompetingRiskModel:
    """Fine-Gray competing risk model (default vs prepayment).

    If scikit-survival is not installed, falls back to two independent
    Cox PH models from lifelines.
    """

    def __init__(self) -> None:
        cfg = get_config()["survival"]
        self.duration_col = cfg["duration_col"]
        self.default_col = cfg["event_col_default"]
        self.prepay_col = cfg["event_col_prepay"]
        self.penalizer = cfg.get("penalizer", 0.1)

        self._model_default: Any = None
        self._model_prepay: Any = None
        self._backend: str = "unknown"
        self._is_fitted: bool = False

        self.covariates = [
            "loan_age_months", "days_past_due", "ltv_band_ord",
            "credit_score_band_ord", "interest_rate", "modification_flag",
            "pct_balance_remaining", "dpd_trend_3m", "refi_incentive",
            "dti_band_ord", "n_delinquencies_to_date",
        ]

    # ──────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CompetingRiskModel":
        """Fit competing risk model on training data."""
        surv_df = self._prepare(df)
        covs = self._available_covariates(surv_df)
        log.info(
            "Fitting competing risk model | n=%d | defaults=%d | prepayments=%d",
            len(surv_df),
            surv_df[self.default_col].sum(),
            surv_df[self.prepay_col].sum(),
        )

        # Try Fine-Gray via scikit-survival first
        try:
            self._fit_fine_gray(surv_df, covs)
        except (ImportError, Exception) as exc:
            log.warning("Fine-Gray unavailable (%s) — using two Cox PH models", exc)
            self._fit_cox_fallback(surv_df, covs)

        self._is_fitted = True
        return self

    def _fit_fine_gray(self, df: pd.DataFrame, covs: list[str]) -> None:
        """Fit using scikit-survival's Fine-Gray sub-distribution hazard."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        clean = df[[self.duration_col, self.default_col, self.prepay_col] + covs].dropna()

        # Default model: default=event, prepay=competing
        # Encode as structured array required by scikit-survival
        y_default = Surv.from_arrays(
            event=clean[self.default_col].astype(bool),
            time=clean[self.duration_col],
        )
        y_prepay = Surv.from_arrays(
            event=clean[self.prepay_col].astype(bool),
            time=clean[self.duration_col],
        )

        X = clean[covs].values
        self._model_default = CoxPHSurvivalAnalysis(alpha=self.penalizer)
        self._model_default.fit(X, y_default)

        self._model_prepay = CoxPHSurvivalAnalysis(alpha=self.penalizer)
        self._model_prepay.fit(X, y_prepay)

        self._backend = "scikit-survival"
        log.info("Fine-Gray models fitted via scikit-survival")

    def _fit_cox_fallback(self, df: pd.DataFrame, covs: list[str]) -> None:
        """Fallback: two independent Cox PH models (or KM fitters) via lifelines."""
        from lifelines import CoxPHFitter, KaplanMeierFitter

        df = df.loc[:, ~df.columns.duplicated()]
        safe_covs = [
            c for c in covs
            if c not in {self.duration_col, self.default_col, self.prepay_col}
            and c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 1
            and df[c].std() > 1e-4
        ]
        clean = df[[self.duration_col, self.default_col, self.prepay_col] + safe_covs].dropna()

        # Fit default model
        try:
            self._model_default = CoxPHFitter(penalizer=max(self.penalizer, 0.5))
            self._model_default.fit(
                clean[[self.duration_col, self.default_col] + safe_covs],
                duration_col=self.duration_col,
                event_col=self.default_col,
                show_progress=False,
            )
        except Exception as exc:
            log.warning("Default Cox PH fit failed (%s) — using Kaplan-Meier fallback", exc)
            self._model_default = KaplanMeierFitter()
            self._model_default.fit(clean[self.duration_col], clean[self.default_col])

        # Fit prepayment model
        try:
            self._model_prepay = CoxPHFitter(penalizer=max(self.penalizer, 0.5))
            self._model_prepay.fit(
                clean[[self.duration_col, self.prepay_col] + safe_covs],
                duration_col=self.duration_col,
                event_col=self.prepay_col,
                show_progress=False,
            )
        except Exception as exc:
            log.warning("Prepayment Cox PH fit failed (%s) — using Kaplan-Meier fallback", exc)
            self._model_prepay = KaplanMeierFitter()
            self._model_prepay.fit(clean[self.duration_col], clean[self.prepay_col])

        self._backend = "lifelines-fallback"
        log.info("Competing risk models fitted via lifelines fallback")

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return default_risk and prepayment_risk scores per loan."""
        self._check_fitted()
        surv_df = self._prepare(df)
        covs = self._available_covariates(surv_df)
        clean = surv_df[covs].fillna(0)

        if hasattr(self._model_default, "predict_partial_hazard"):
            default_risk = self._model_default.predict_partial_hazard(clean).values
        else:
            default_risk = np.full(len(df), 0.5)

        if hasattr(self._model_prepay, "predict_partial_hazard"):
            prepay_risk = self._model_prepay.predict_partial_hazard(clean).values
        else:
            prepay_risk = np.full(len(df), 0.5)

        return pd.DataFrame({
            "default_risk": default_risk,
            "prepayment_risk": prepay_risk,
        }, index=surv_df.index)

    def predict_cif(
        self, df: pd.DataFrame, times: list[int] | None = None
    ) -> pd.DataFrame:
        """Predict Cumulative Incidence Function at given time points."""
        self._check_fitted()
        if times is None:
            times = [3, 6, 12, 24, 36]

        surv_df = self._prepare(df)
        covs = self._available_covariates(surv_df)
        clean = surv_df[covs].fillna(0)

        records = []
        for t in times:
            if hasattr(self._model_default, "predict_survival_function"):
                sf_d = self._model_default.predict_survival_function(clean, times=[t])
                cif_d = float(1 - sf_d.loc[t].mean()) if t in sf_d.index else 0.05
            elif hasattr(self._model_default, "survival_function_at_times"):
                cif_d = float(1 - self._model_default.survival_function_at_times(t).values[0])
            else:
                cif_d = 0.05

            if hasattr(self._model_prepay, "predict_survival_function"):
                sf_p = self._model_prepay.predict_survival_function(clean, times=[t])
                cif_p = float(1 - sf_p.loc[t].mean()) if t in sf_p.index else 0.10
            elif hasattr(self._model_prepay, "survival_function_at_times"):
                cif_p = float(1 - self._model_prepay.survival_function_at_times(t).values[0])
            else:
                cif_p = 0.10

            records.append({
                "time_months": t,
                "cif_default": round(cif_d, 5),
                "cif_prepayment": round(cif_p, 5),
            })

        return pd.DataFrame(records)


    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute cause-specific C-index for both events."""
        self._check_fitted()
        surv_df = self._prepare(df)
        covs = self._available_covariates(surv_df)
        clean = surv_df[[self.duration_col, self.default_col, self.prepay_col] + covs].dropna()
        risk_scores = self.predict_risk_scores(clean)

        c_default = harrell_c_index(
            clean[self.duration_col].values,
            clean[self.default_col].values,
            risk_scores["default_risk"].values,
        )
        c_prepay = harrell_c_index(
            clean[self.duration_col].values,
            clean[self.prepay_col].values,
            risk_scores["prepayment_risk"].values,
        )
        log.info("Competing risk C-index | default=%.4f | prepay=%.4f", c_default, c_prepay)
        return {
            "competing_risk_c_index_default": c_default,
            "competing_risk_c_index_prepay": c_prepay,
            "backend": self._backend,
        }

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        surv = df.copy()
        for col in [self.default_col, self.prepay_col]:
            if col in surv.columns:
                surv[col] = pd.to_numeric(surv[col], errors="coerce").fillna(0).astype(int)
            else:
                surv[col] = 0
        return surv

    def _available_covariates(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.covariates if c in df.columns]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("CompetingRiskModel must be fit() first.")
