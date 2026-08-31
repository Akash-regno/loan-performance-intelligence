"""
src/features/macro.py
---------------------
Join macro scenario data and create macro-derived features.

The macro_scenarios.csv provides base, adverse, and high-prepayment
scenario parameters. At feature engineering time, we join the BASE
scenario values by reporting_month to create macro context features.

Features created:
  - scenario_rate_level     : interest rate assumption for the period
  - scenario_hpi_level      : HPI index assumption
  - scenario_unemployment   : unemployment rate assumption
  - rate_spread             : loan interest_rate - scenario_rate_level
  - refi_incentive          : positive if loan rate > current market rate (refi opportunity)

Scenario-modified features are created at inference time by ScenarioEngine.

Usage:
    from src.features.macro import MacroFeatures
    fe = MacroFeatures()
    fe.fit(macro_df)
    df = fe.transform(df)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class MacroFeatures:
    """Join and derive macro-environment features."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self._macro_map: dict[str, dict] = {}  # reporting_month → macro row dict
        self._scenario_cols: list[str] = []
        self._is_fitted: bool = False

    def fit(self, macro_df: pd.DataFrame | None = None) -> "MacroFeatures":
        """Load macro data and build a lookup by reporting_month.

        Parameters
        ----------
        macro_df : DataFrame, optional
            Pre-loaded macro_scenarios.csv. If None, loads from config path.
        """
        if macro_df is None:
            path = Path(self.cfg["paths"]["macro_file"])
            if not path.exists():
                log.warning("Macro file not found at %s — skipping macro features", path)
                self._is_fitted = True
                return self
            macro_df = pd.read_csv(path)

        # Filter to base scenario if multiple scenarios are present
        if "scenario" in macro_df.columns:
            base = macro_df[macro_df["scenario"].str.lower() == "base"].copy()
            if len(base) == 0:
                base = macro_df.copy()
        else:
            base = macro_df.copy()

        # Build month → macro dict
        if "reporting_month" in base.columns:
            self._scenario_cols = [
                c for c in base.columns if c not in {"reporting_month", "scenario"}
            ]
            for _, row in base.iterrows():
                self._macro_map[str(row["reporting_month"])] = row.to_dict()

        log.info(
            "MacroFeatures fitted: %d reporting months | macro cols: %s",
            len(self._macro_map),
            self._scenario_cols,
        )
        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        scenario_overrides: dict | None = None,
    ) -> pd.DataFrame:
        """Add macro features by joining on reporting_month.

        Parameters
        ----------
        scenario_overrides : dict, optional
            Delta overrides for scenario simulation (used by ScenarioEngine).
            E.g. {'rate_delta': 3.0, 'hpi_delta': -0.15}
        """
        df = df.copy()

        if not self._macro_map or "reporting_month" not in df.columns:
            log.warning("No macro data available — adding zero-filled macro columns")
            for col in ["scenario_rate_level", "scenario_hpi_level", "scenario_unemployment"]:
                df[col] = 0.0
            return df

        # Vectorized join
        macro_lookup = pd.DataFrame.from_dict(self._macro_map, orient="index")
        macro_lookup.index.name = "reporting_month"
        macro_lookup = macro_lookup.reset_index()

        df = df.merge(macro_lookup, on="reporting_month", how="left")

        # Apply scenario overrides (for ScenarioEngine)
        if scenario_overrides:
            rate_delta = scenario_overrides.get("rate_delta", 0.0)
            hpi_delta = scenario_overrides.get("hpi_delta", 0.0)
            unemp_delta = scenario_overrides.get("unemployment_delta", 0.0)

            if "interest_rate" in df.columns:
                df["interest_rate"] = (df["interest_rate"] + rate_delta).clip(0, 30)
            for col in self._scenario_cols:
                if "rate" in col.lower() and col in df.columns:
                    df[col] = df[col] + rate_delta
                if "hpi" in col.lower() and col in df.columns:
                    df[col] = df[col] * (1 + hpi_delta)
                if "unemployment" in col.lower() and col in df.columns:
                    df[col] = df[col] + unemp_delta

        # Derived macro features
        if "interest_rate" in df.columns:
            # Try to find market rate from macro
            market_rate_col = next(
                (c for c in self._scenario_cols if "rate" in c.lower()), None
            )
            if market_rate_col and market_rate_col in df.columns:
                df["rate_spread"] = (
                    df["interest_rate"] - df[market_rate_col]
                ).round(4)
                # Positive spread = loan rate above market → refi incentive
                df["refi_incentive"] = (df["rate_spread"] > 0.5).astype(int)

        return df

    def fit_transform(
        self, df: pd.DataFrame, macro_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        return self.fit(macro_df).transform(df)
