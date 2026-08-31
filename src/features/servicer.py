"""
src/features/servicer.py
------------------------
Servicer-specific features for conflict detection, source reliability,
and staleness scoring.

Features created:
  - servicer_conflict_score   : fraction of fields conflicting with servicer_updates
  - n_servicer_conflicts      : raw count of conflicting fields (from cleaning)
  - source_system_enc         : ordinal encoding of source_system reliability
  - record_staleness_days     : days between last_updated_at and reporting_month
  - servicer_reliability      : per-servicer historical conflict rate (target-encoded)
  - source_mismatch_flag      : any source field differs from main panel

Usage:
    from src.features.servicer import ServicerFeatures
    fe = ServicerFeatures()
    fe.fit(train_df)
    df = fe.transform(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE_SYSTEM_RELIABILITY = {
    # Higher = more reliable. Values are assumptions; will be updated from data.
    "primary": 3,
    "secondary": 2,
    "manual": 1,
    "legacy": 1,
    "Unknown": 0,
}


class ServicerFeatures:
    """Build servicer-quality and source reliability features."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self._servicer_conflict_rates: dict[str, float] = {}
        self._global_conflict_rate: float = 0.0
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "ServicerFeatures":
        """Fit per-servicer conflict rate on training data."""
        if "servicer_name" not in df.columns or "servicer_conflict_flag" not in df.columns:
            log.warning("servicer_name or servicer_conflict_flag missing — using defaults")
            self._is_fitted = True
            return self

        self._global_conflict_rate = float(df["servicer_conflict_flag"].mean())
        self._servicer_conflict_rates = (
            df.groupby("servicer_name")["servicer_conflict_flag"]
            .mean()
            .to_dict()
        )
        log.info(
            "ServicerFeatures fitted: %d servicers | global conflict rate=%.3f",
            len(self._servicer_conflict_rates),
            self._global_conflict_rate,
        )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add servicer feature columns."""
        df = df.copy()

        # Source system reliability score
        if "source_system" in df.columns:
            df["source_system_enc"] = (
                df["source_system"].astype(str)
                .str.lower()
                .map(SOURCE_SYSTEM_RELIABILITY)
                .fillna(0)
                .astype(int)
            )

        # Record staleness in days
        if "last_updated_at" in df.columns and "reporting_month" in df.columns:
            try:
                last_updated = pd.to_datetime(df["last_updated_at"], errors="coerce")
                reporting = pd.to_datetime(
                    df["reporting_month"].astype(str) + "-01", errors="coerce"
                )
                df["record_staleness_days"] = (
                    (reporting - last_updated).dt.days.clip(0, None).fillna(0)
                )
            except Exception as exc:
                log.warning("Could not compute staleness: %s", exc)
                df["record_staleness_days"] = 0

        # Per-servicer historical conflict rate (target encoding from training)
        if "servicer_name" in df.columns and self._is_fitted:
            df["servicer_reliability"] = (
                df["servicer_name"].map(self._servicer_conflict_rates)
                .fillna(self._global_conflict_rate)
            )

        # Conflict score normalization
        if "n_servicer_conflicts" in df.columns:
            max_possible_conflicts = max(
                1, df["n_servicer_conflicts"].max() if len(df) > 0 else 1
            )
            df["servicer_conflict_score"] = (
                df["n_servicer_conflicts"] / max_possible_conflicts
            ).clip(0, 1).round(4)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
