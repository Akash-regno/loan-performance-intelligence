"""
src/features/temporal.py
------------------------
Temporal feature engineering for loan-level panel data.

Features created (all computable at or before the current period):
  - vintage_year          : origination year extracted from origination_month
  - vintage_quarter       : origination quarter (e.g., 2021Q3)
  - cohort_quarter        : vintage_year + 'Q' + quarter (e.g., 2021Q3)
  - months_to_maturity    : remaining_term_months (alias for clarity)
  - loan_age_band         : binned loan_age_months (0-12, 12-24, 24-36, 36-60, 60+)
  - is_early_period       : loan_age_months <= 12 (high-risk early-delinquency window)
  - is_seasoned_loan      : loan_age_months >= 36
  - reporting_year        : year from reporting_month
  - reporting_month_num   : month number from reporting_month (1–12)
  - reporting_quarter     : quarter from reporting_month

Usage:
    from src.features.temporal import TemporalFeatures
    fe = TemporalFeatures()
    df = fe.transform(df)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class TemporalFeatures:
    """Generate temporal features from loan age and date columns."""

    LOAN_AGE_BINS = [0, 12, 24, 36, 60, float("inf")]
    LOAN_AGE_LABELS = ["0-12m", "12-24m", "24-36m", "36-60m", "60m+"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all temporal features to df. Returns a copy."""
        df = df.copy()
        log.info("Generating temporal features…")

        df = self._origination_features(df)
        df = self._reporting_features(df)
        df = self._loan_age_features(df)

        new_cols = [
            c for c in [
                "vintage_year", "vintage_quarter", "cohort_quarter",
                "months_to_maturity", "loan_age_band",
                "is_early_period", "is_seasoned_loan",
                "reporting_year", "reporting_month_num", "reporting_quarter",
            ]
            if c in df.columns
        ]
        log.info("Temporal features added: %s", new_cols)
        return df

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _origination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract vintage year and quarter from origination_month."""
        if "origination_month" not in df.columns:
            log.warning("'origination_month' not found — skipping vintage features")
            return df

        orig_dt = pd.to_datetime(
            df["origination_month"].astype(str) + "-01", errors="coerce"
        )
        df["vintage_year"] = orig_dt.dt.year.astype("Int64")
        df["vintage_quarter"] = orig_dt.dt.quarter.astype("Int64")
        df["cohort_quarter"] = (
            orig_dt.dt.year.astype(str) + "Q" + orig_dt.dt.quarter.astype(str)
        ).where(orig_dt.notna(), other=pd.NA)
        return df

    def _reporting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, and quarter from reporting_month."""
        if "reporting_month" not in df.columns:
            return df

        rep_dt = pd.to_datetime(
            df["reporting_month"].astype(str) + "-01", errors="coerce"
        )
        df["reporting_year"] = rep_dt.dt.year.astype("Int64")
        df["reporting_month_num"] = rep_dt.dt.month.astype("Int64")
        df["reporting_quarter"] = rep_dt.dt.quarter.astype("Int64")
        return df

    def _loan_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create loan age bands and lifecycle flags."""
        if "loan_age_months" not in df.columns:
            return df

        age = df["loan_age_months"].astype(float)

        df["months_to_maturity"] = df.get(
            "remaining_term_months", pd.Series(np.nan, index=df.index)
        )

        df["loan_age_band"] = pd.cut(
            age,
            bins=self.LOAN_AGE_BINS,
            labels=self.LOAN_AGE_LABELS,
            right=False,
        ).astype(str)

        df["is_early_period"] = (age <= 12).astype(int)
        df["is_seasoned_loan"] = (age >= 36).astype(int)
        return df
