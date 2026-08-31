"""
src/features/interactions.py
-----------------------------
Cross-feature interaction terms and ratio features.

Features created:
  - ltv_credit_risk        : ltv_band_ord × (max_credit_score - credit_score_band_ord)
  - age_dpd_interaction    : loan_age_months × dpd_trend_3m
  - balance_dpd_risk       : pct_balance_remaining × dpd_lag1
  - high_ltv_low_credit    : binary flag for LTV > 80 AND credit < 680
  - balance_to_payment     : current_balance / (scheduled_payment + 1)
  - dpd_acceleration       : dpd_lag1 - dpd_lag3 (rate of DPD change)
  - risk_composite         : weighted composite of DPD, LTV, credit signals

Usage:
    from src.features.interactions import InteractionFeatures
    fe = InteractionFeatures()
    df = fe.transform(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Ordinal band midpoint values for interaction math
CREDIT_SCORE_MIDPOINTS = {
    "<620": 610, "620-639": 630, "640-659": 650, "660-679": 670,
    "680-699": 690, "700-719": 710, "720-739": 730, "740-759": 750,
    "760-779": 770, "780+": 790,
}
LTV_MIDPOINTS = {
    "0-60": 50, "60-70": 65, "70-75": 73, "75-80": 78, "80-85": 83,
    "85-90": 88, "90-95": 93, "95-97": 96, "97+": 100,
}
DTI_MIDPOINTS = {
    "0-20": 15, "20-30": 25, "30-36": 33, "36-43": 40,
    "43-45": 44, "45+": 48,
}


class InteractionFeatures:
    """Generate cross-feature interaction terms."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features. Returns a copy."""
        df = df.copy()
        log.info("Generating interaction features…")

        df = self._map_band_midpoints(df)
        df = self._compute_interactions(df)

        return df

    def _map_band_midpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map band strings to numeric midpoint values."""
        if "credit_score_band" in df.columns:
            df["credit_score_mid"] = (
                df["credit_score_band"].map(CREDIT_SCORE_MIDPOINTS).fillna(660.0)
            )
        if "ltv_band" in df.columns:
            df["ltv_mid"] = (
                df["ltv_band"].map(LTV_MIDPOINTS).fillna(80.0)
            )
        if "dti_band" in df.columns:
            df["dti_mid"] = (
                df["dti_band"].map(DTI_MIDPOINTS).fillna(36.0)
            )
        return df

    def _compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all interaction terms."""

        # LTV × Credit Risk interaction
        # Higher LTV + lower credit = higher risk
        if "ltv_mid" in df.columns and "credit_score_mid" in df.columns:
            df["ltv_credit_risk"] = (
                df["ltv_mid"] / 100.0
                * (800.0 - df["credit_score_mid"]) / 180.0
            ).round(4)

        # Loan age × DPD trend (worsening trend on older loan = high risk)
        if "loan_age_months" in df.columns and "dpd_trend_3m" in df.columns:
            df["age_dpd_interaction"] = (
                df["loan_age_months"].astype(float) * df["dpd_trend_3m"]
            ).round(4)

        # Balance remaining × recent DPD (high remaining balance + DPD = high loss)
        if "pct_balance_remaining" in df.columns and "dpd_lag1" in df.columns:
            df["balance_dpd_risk"] = (
                df["pct_balance_remaining"] * df["dpd_lag1"]
            ).round(4)

        # High LTV + low credit binary flag
        if "ltv_mid" in df.columns and "credit_score_mid" in df.columns:
            df["high_ltv_low_credit"] = (
                (df["ltv_mid"] > 80) & (df["credit_score_mid"] < 680)
            ).astype(int)

        # DPD acceleration (positive = worsening, negative = improving)
        if "dpd_lag1" in df.columns and "dpd_lag3" in df.columns:
            df["dpd_acceleration"] = (
                df["dpd_lag1"] - df["dpd_lag3"]
            ).round(2)

        # Balance-to-interest burden
        if "current_balance" in df.columns and "interest_rate" in df.columns:
            df["annual_interest_burden"] = (
                df["current_balance"] * df["interest_rate"] / 100.0
            ).round(2)

        # DTI × DPD (high debt burden + delinquency = compounding stress)
        if "dti_mid" in df.columns and "dpd_lag1" in df.columns:
            df["dti_dpd_stress"] = (
                df["dti_mid"] / 50.0 * df["dpd_lag1"]
            ).round(4)

        # Composite risk score (weighted average of normalized risk signals)
        risk_components = []
        if "ltv_mid" in df.columns:
            risk_components.append(df["ltv_mid"].clip(0, 100) / 100.0 * 0.3)
        if "credit_score_mid" in df.columns:
            risk_components.append(
                (800 - df["credit_score_mid"].clip(500, 800)) / 300.0 * 0.3
            )
        if "dpd_lag1" in df.columns:
            risk_components.append(
                df["dpd_lag1"].clip(0, 90) / 90.0 * 0.25
            )
        if "dti_mid" in df.columns:
            risk_components.append(df["dti_mid"].clip(0, 50) / 50.0 * 0.15)

        if risk_components:
            df["risk_composite"] = sum(risk_components).round(4)

        return df
