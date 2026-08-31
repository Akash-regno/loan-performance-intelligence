"""
src/features/lag_rolling.py
---------------------------
Lag and rolling window features for loan panel data.

All features are computed within each loan_id group, sorted by month_index,
ensuring NO future information leaks into past periods.

Features created:
  DPD lags:
    dpd_lag1, dpd_lag3, dpd_lag6
    dpd_max_3m, dpd_max_6m, dpd_max_12m
    dpd_mean_3m, dpd_mean_6m
    dpd_trend_3m  (slope of DPD over last 3 months)
    ever_30dpd    (any DPD >= 30 in loan history)
    ever_60dpd    (any DPD >= 60 in loan history)
    ever_90dpd    (any DPD >= 90 in loan history)
    n_times_30dpd (count of 30+ DPD months)

  Balance lags:
    balance_change_1m   (current_balance - 1m ago)
    balance_change_3m   (current_balance - 3m ago)
    pct_balance_remaining (current_balance / original_balance)
    balance_paid_to_date  (original_balance - current_balance)

  Status history:
    status_change_flag       (current_status changed vs last month)
    n_modifications_to_date  (cumulative modification_flag count)
    n_delinquencies_to_date  (cumulative DPD >= 30 count)

Usage:
    from src.features.lag_rolling import LagRollingFeatures
    fe = LagRollingFeatures()
    df = fe.transform(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class LagRollingFeatures:
    """Compute lag and rolling window features within each loan group."""

    def __init__(self) -> None:
        self.cfg = get_config()
        feat_cfg = self.cfg["features"]
        self.lag_periods: list[int] = feat_cfg["lag_periods"]       # [1, 3, 6]
        self.rolling_windows: list[int] = feat_cfg["rolling_windows"]  # [3, 6, 12]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag/rolling features. Input must be sorted by (loan_id, month_index)."""
        df = df.copy()
        log.info("Generating lag/rolling features for %d rows…", len(df))

        # Verify sort order
        df = df.sort_values(["loan_id", "month_index"]).reset_index(drop=True)

        df = self._dpd_features(df)
        df = self._balance_features(df)
        df = self._status_history(df)

        log.info("Lag/rolling features complete.")
        return df

    # ──────────────────────────────────────────────────────────
    # DPD features
    # ──────────────────────────────────────────────────────────

    def _dpd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "days_past_due" not in df.columns:
            log.warning("'days_past_due' missing — skipping DPD lag features")
            return df

        grp = df.groupby("loan_id", sort=False)["days_past_due"]

        # Lag features
        for lag in self.lag_periods:
            df[f"dpd_lag{lag}"] = grp.shift(lag).fillna(0)

        # Rolling max, mean
        for window in self.rolling_windows:
            df[f"dpd_max_{window}m"] = (
                grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).max())
                .fillna(0)
            )
            if window <= 6:  # Only compute mean for smaller windows
                df[f"dpd_mean_{window}m"] = (
                    grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
                    .fillna(0)
                    .round(2)
                )

        # DPD trend: slope of DPD over last 3 months (positive = worsening)
        def _trend_slope(s: pd.Series) -> pd.Series:
            def slope(x):
                if len(x) < 2:
                    return 0.0
                n = len(x)
                idx = np.arange(n)
                if idx.std() == 0:
                    return 0.0
                return float(np.polyfit(idx, x, 1)[0])
            return s.shift(1).rolling(3, min_periods=2).apply(slope, raw=True)

        df["dpd_trend_3m"] = grp.transform(_trend_slope).fillna(0).round(4)

        # Ever-delinquent flags (expanding window, using only past info via shift)
        dpd_shift = df.groupby("loan_id", sort=False)["days_past_due"].shift(1)
        df["ever_30dpd"] = (
            df.groupby("loan_id", sort=False)["days_past_due"]
            .transform(lambda s: s.shift(1).expanding().max())
            .ge(30).astype(int)
        )
        df["ever_60dpd"] = (
            df.groupby("loan_id", sort=False)["days_past_due"]
            .transform(lambda s: s.shift(1).expanding().max())
            .ge(60).astype(int)
        )
        df["ever_90dpd"] = (
            df.groupby("loan_id", sort=False)["days_past_due"]
            .transform(lambda s: s.shift(1).expanding().max())
            .ge(90).astype(int)
        )
        df["n_times_30dpd"] = (
            df.groupby("loan_id", sort=False)["days_past_due"]
            .transform(lambda s: (s.shift(1) >= 30).expanding().sum())
            .fillna(0).astype(int)
        )

        return df

    # ──────────────────────────────────────────────────────────
    # Balance features
    # ──────────────────────────────────────────────────────────

    def _balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "current_balance" not in df.columns:
            return df

        grp_bal = df.groupby("loan_id", sort=False)["current_balance"]

        # Balance changes (using shift to avoid leakage)
        df["balance_change_1m"] = (
            df["current_balance"] - grp_bal.shift(1)
        ).fillna(0).round(2)

        df["balance_change_3m"] = (
            df["current_balance"] - grp_bal.shift(3)
        ).fillna(0).round(2)

        # Pct balance remaining
        if "original_balance" in df.columns:
            df["pct_balance_remaining"] = (
                df["current_balance"] / (df["original_balance"] + 1e-9)
            ).clip(0, 1.5).round(4)

            df["balance_paid_to_date"] = (
                df["original_balance"] - df["current_balance"]
            ).clip(0, None).round(2)

        return df

    # ──────────────────────────────────────────────────────────
    # Status history
    # ──────────────────────────────────────────────────────────

    def _status_history(self, df: pd.DataFrame) -> pd.DataFrame:
        if "current_status" not in df.columns:
            return df

        grp_status = df.groupby("loan_id", sort=False)["current_status"]

        # Status change flag: 1 if status differs from previous month
        df["status_change_flag"] = (
            df["current_status"] != grp_status.shift(1)
        ).astype(int)

        # Cumulative modifications to date (using shift to avoid leakage)
        if "modification_flag" in df.columns:
            df["n_modifications_to_date"] = (
                df.groupby("loan_id", sort=False)["modification_flag"]
                .transform(lambda s: s.shift(1).expanding().sum())
                .fillna(0).astype(int)
            )

        # Cumulative DPD >= 30 events to date
        if "days_past_due" in df.columns:
            df["n_delinquencies_to_date"] = (
                df.groupby("loan_id", sort=False)["days_past_due"]
                .transform(lambda s: (s.shift(1) >= 30).expanding().sum())
                .fillna(0).astype(int)
            )

        return df
