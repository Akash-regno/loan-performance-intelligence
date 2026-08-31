"""
src/data/ingestion.py
---------------------
Load all organizer-provided CSV files, enforce schema, cast types,
join static attributes and servicer updates, and sort by loan_id + month_index.

No row is ever modified in data/raw/. Outputs a merged, type-cast
DataFrame ready for validation and profiling.

Usage:
    from src.data.ingestion import DataIngestion
    ingestion = DataIngestion()
    df = ingestion.run()  # returns full merged training DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Expected schema (column → pandas dtype)
# ──────────────────────────────────────────────────────────────
MONTHLY_SCHEMA: dict[str, str] = {
    "loan_id":              "str",
    "month_index":          "int64",
    "reporting_month":      "str",
    "origination_month":    "str",
    "loan_age_months":      "int64",
    "remaining_term_months":"int64",
    "original_balance":     "float64",
    "current_balance":      "float64",
    "interest_rate":        "float64",
    "credit_score_band":    "str",
    "ltv_band":             "str",
    "dti_band":             "str",
    "state":                "str",
    "loan_purpose":         "str",
    "occupancy_type":       "str",
    "property_type":        "str",
    "servicer_name":        "str",
    "current_status":       "str",
    "days_past_due":        "int64",
    "modification_flag":    "int64",
    "prepayment_flag":      "int64",
    "default_flag":         "int64",
    "loss_severity_band":   "str",
    "last_updated_at":      "str",
    "source_system":        "str",
    "document_status":      "str",
}

TARGET_COLUMNS: list[str] = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]


class DataIngestion:
    """Load, merge, and type-cast all organizer data files.

    Parameters
    ----------
    config_path : str, optional
        Override path to config.yaml.
    """

    def __init__(self, config_path: str | None = None) -> None:
        set_global_seed()
        self.cfg = get_config(config_path)
        self.paths = self.cfg["paths"]

    # ──────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────

    def run(self, split: str = "train") -> pd.DataFrame:
        """Full ingestion pipeline.

        Parameters
        ----------
        split : str
            'train' or 'test'.

        Returns
        -------
        DataFrame
            Merged, type-cast, sorted DataFrame.
        """
        log.info("Starting data ingestion — split='%s'", split)

        monthly = self._load_monthly(split)
        static = self._load_static()
        servicer = self._load_servicer()
        macro = self._load_macro()

        merged = self._merge(monthly, static, servicer, macro)
        merged = self._enforce_schema(merged)
        merged = self._deduplicate(merged)
        merged = self._sort(merged)

        log.info(
            "Ingestion complete: %d rows × %d cols", *merged.shape
        )
        return merged

    load = run


    # ──────────────────────────────────────────────────────────
    # Private loaders
    # ──────────────────────────────────────────────────────────

    def _load_monthly(self, split: str) -> pd.DataFrame:
        """Load the main monthly performance file."""
        key = "train_file" if split == "train" else "test_file"
        path = Path(self.paths[key])
        self._assert_exists(path)

        log.info("Loading monthly performance file: %s", path.name)
        df = pd.read_csv(path, low_memory=False)
        log.info("  → %d rows, %d cols", *df.shape)
        return df

    def _load_static(self) -> pd.DataFrame:
        """Load loan static attributes."""
        path = Path(self.paths["static_file"])
        if not path.exists():
            log.warning("Static attributes file not found: %s — skipping join", path)
            return pd.DataFrame(columns=["loan_id"])

        log.info("Loading static attributes: %s", path.name)
        df = pd.read_csv(path, low_memory=False)
        log.info("  → %d rows, %d cols", *df.shape)
        return df

    def _load_servicer(self) -> pd.DataFrame:
        """Load servicer updates for conflict detection."""
        path = Path(self.paths["servicer_file"])
        if not path.exists():
            log.warning("Servicer updates file not found: %s — skipping join", path)
            return pd.DataFrame(columns=["loan_id", "month_index"])

        log.info("Loading servicer updates: %s", path.name)
        df = pd.read_csv(path, low_memory=False)
        log.info("  → %d rows, %d cols", *df.shape)
        return df

    def _load_macro(self) -> pd.DataFrame:
        """Load macro scenario assumptions."""
        path = Path(self.paths["macro_file"])
        if not path.exists():
            log.warning("Macro scenarios file not found: %s — skipping join", path)
            return pd.DataFrame()

        log.info("Loading macro scenarios: %s", path.name)
        df = pd.read_csv(path, low_memory=False)
        log.info("  → %d rows, %d cols", *df.shape)
        return df

    # ──────────────────────────────────────────────────────────
    # Merge strategy
    # ──────────────────────────────────────────────────────────

    def _merge(
        self,
        monthly: pd.DataFrame,
        static: pd.DataFrame,
        servicer: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join static, servicer, and macro onto the monthly panel."""
        df = monthly.copy()

        # Join static attributes (one row per loan_id)
        if "loan_id" in static.columns and len(static) > 0:
            # Avoid column name collisions — suffix static-only cols
            static_only_cols = [
                c for c in static.columns
                if c != "loan_id" and c not in df.columns
            ]
            static_sub = static[["loan_id"] + static_only_cols]
            df = df.merge(static_sub, on="loan_id", how="left")
            log.info("Joined static attributes: +%d columns", len(static_only_cols))

        # Join servicer updates (one row per loan_id + month_index)
        if (
            "loan_id" in servicer.columns
            and "month_index" in servicer.columns
            and len(servicer) > 0
        ):
            # Prefix servicer columns to track conflicts
            svc_cols = [
                c for c in servicer.columns
                if c not in {"loan_id", "month_index"}
            ]
            servicer_renamed = servicer.rename(
                columns={c: f"svc_{c}" for c in svc_cols}
            )
            df = df.merge(
                servicer_renamed,
                on=["loan_id", "month_index"],
                how="left",
            )
            log.info("Joined servicer updates: +%d columns", len(svc_cols))

        # Join macro (by reporting_month, base scenario only at ingestion)
        if "reporting_month" in df.columns and len(macro) > 0:
            # If macro has a reporting_month column, join on it
            if "reporting_month" in macro.columns:
                macro_base = macro[macro.get("scenario", pd.Series(["base"])) == "base"] \
                    if "scenario" in macro.columns else macro
                macro_cols = [
                    c for c in macro_base.columns
                    if c != "reporting_month"
                ]
                macro_sub = macro_base[["reporting_month"] + macro_cols]
                df = df.merge(macro_sub, on="reporting_month", how="left")
                log.info("Joined macro data: +%d columns", len(macro_cols))

        return df

    # ──────────────────────────────────────────────────────────
    # Schema enforcement
    # ──────────────────────────────────────────────────────────

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to expected dtypes from MONTHLY_SCHEMA.

        Missing columns get a NaN column with the correct dtype.
        Extra columns beyond the schema are allowed (from static/servicer/macro joins).
        """
        for col, dtype in MONTHLY_SCHEMA.items():
            if col not in df.columns:
                log.warning("Schema column '%s' missing — adding NaN column", col)
                df[col] = pd.array([pd.NA] * len(df), dtype=object)
                continue

            try:
                if dtype == "str":
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
                elif dtype == "int64":
                    # Coerce to numeric, fill NaN with -1 sentinel
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            except Exception as exc:
                log.warning(
                    "Failed to cast column '%s' to %s: %s", col, dtype, exc
                )

        # Cast target columns if present (training split only)
        for col in TARGET_COLUMNS:
            if col in df.columns:
                if col in {"next_state", "exception_type"}:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df

    # ──────────────────────────────────────────────────────────
    # De-duplication
    # ──────────────────────────────────────────────────────────

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact-duplicate rows; keep last record per (loan_id, month_index)."""
        n_before = len(df)
        df = df.drop_duplicates(subset=["loan_id", "month_index"], keep="last")
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            log.warning("De-duplication removed %d duplicate rows", n_dropped)
        else:
            log.info("De-duplication: no duplicates found")
        return df

    # ──────────────────────────────────────────────────────────
    # Sorting
    # ──────────────────────────────────────────────────────────

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by loan_id then month_index for correct lag/rolling features."""
        return df.sort_values(["loan_id", "month_index"], ignore_index=True)

    # ──────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _assert_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Place organizer-provided files in data/raw/"
            )

    def get_schema_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame: column name, dtype, missing%, n_unique."""
        report = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "n_unique": df.nunique().values,
            "sample_value": [
                df[c].dropna().iloc[0] if df[c].notna().any() else None
                for c in df.columns
            ],
        })
        return report.sort_values("missing_pct", ascending=False)
