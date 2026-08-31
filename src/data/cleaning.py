"""
src/data/cleaning.py
--------------------
Data cleaning: missing-value imputation, IQR-based outlier capping,
servicer conflict resolution, and stale-record flagging.

Design principles:
  - All imputers are fit ONLY on training data, then applied to test
  - No information from the target columns influences imputation
  - Cleaning decisions are logged for the AI Development Log
  - Raw data in data/raw/ is NEVER modified

Usage:
    from src.data.cleaning import DataCleaner
    cleaner = DataCleaner()
    cleaner.fit(train_df)         # fit imputers on train
    clean_train = cleaner.transform(train_df)
    clean_test  = cleaner.transform(test_df)  # uses train-fitted imputers
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Imputation strategy per column
# ──────────────────────────────────────────────────────────────
# 'median'  → numeric, use median of training data
# 'mode'    → categorical/flag, use most-frequent value
# 'ffill'   → forward-fill within loan group (temporal order)
# 'zero'    → fill with 0 (for binary flags)
# 'const'   → fill with a constant string sentinel
IMPUTATION_STRATEGY: dict[str, str] = {
    # Numeric
    "current_balance":       "ffill",
    "original_balance":      "median",
    "interest_rate":         "median",
    "loan_age_months":       "ffill",
    "remaining_term_months": "ffill",
    "days_past_due":         "ffill",
    # Binary flags
    "modification_flag":     "zero",
    "prepayment_flag":       "zero",
    "default_flag":          "zero",
    # Categorical
    "credit_score_band":     "mode",
    "ltv_band":              "mode",
    "dti_band":              "mode",
    "state":                 "mode",
    "loan_purpose":          "mode",
    "occupancy_type":        "mode",
    "property_type":         "mode",
    "servicer_name":         "mode",
    "current_status":        "ffill",
    "loss_severity_band":    "const",
    "document_status":       "const",
    "source_system":         "const",
}

CONST_FILL_VALUE = "Unknown"
STALENESS_THRESHOLD_DAYS: int = 90


class DataCleaner:
    """Clean loan data with train-fitted imputers.

    Attributes
    ----------
    _fit_values : dict
        Stores fitted imputation values (median/mode) from training data.
    _iqr_bounds : dict
        Stores (lower, upper) IQR bounds for numeric column capping.
    _is_fitted : bool
        True after fit() has been called.
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        self._fit_values: dict[str, Any] = {}
        self._iqr_bounds: dict[str, tuple[float, float]] = {}
        self._is_fitted: bool = False

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Fit imputers and IQR bounds on training data.

        MUST be called before transform(). Must NOT be called on test data.
        """
        log.info("Fitting DataCleaner on %d training rows…", len(df))

        for col, strategy in IMPUTATION_STRATEGY.items():
            if col not in df.columns:
                continue

            if strategy == "median":
                val = df[col].median()
                self._fit_values[col] = val
                log.debug("  Fitted median for '%s': %.4f", col, val)

            elif strategy == "mode":
                mode_result = df[col].mode()
                val = mode_result.iloc[0] if len(mode_result) > 0 else CONST_FILL_VALUE
                self._fit_values[col] = val
                log.debug("  Fitted mode for '%s': %s", col, val)

            elif strategy in {"ffill", "zero", "const"}:
                # These don't need a pre-fit value from training
                self._fit_values[col] = strategy

        # Fit IQR bounds for numeric columns
        numeric_cols = [
            "current_balance", "original_balance", "interest_rate",
            "days_past_due", "remaining_term_months", "loan_age_months",
        ]
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna().astype(float)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            self._iqr_bounds[col] = (q1 - 3 * iqr, q3 + 3 * iqr)

        self._is_fitted = True
        log.info("DataCleaner fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning to a DataFrame using train-fitted parameters.

        Steps:
          1. Missing-value imputation (column-by-column)
          2. IQR-based outlier capping (numeric cols, domain bounds applied)
          3. Servicer conflict resolution
          4. Staleness flag
          5. Date parsing / reformatting
        """
        if not self._is_fitted:
            raise RuntimeError("DataCleaner must be fit() before transform(). "
                               "Call fit(train_df) first.")

        df = df.copy()
        log.info("Cleaning DataFrame (%d rows)…", len(df))

        df = self._impute(df)
        df = self._cap_outliers(df)
        df = self._resolve_servicer_conflicts(df)
        df = self._flag_staleness(df)
        df = self._apply_domain_bounds(df)

        log.info("Cleaning complete.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data in one call."""
        return self.fit(df).transform(df)

    # ──────────────────────────────────────────────────────────
    # Private cleaning steps
    # ──────────────────────────────────────────────────────────

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation strategy column by column."""
        for col, strategy in IMPUTATION_STRATEGY.items():
            if col not in df.columns:
                continue

            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            if strategy in ("median", "mode"):
                fill_val = self._fit_values.get(col, CONST_FILL_VALUE)
                df[col] = df[col].fillna(fill_val)

            elif strategy == "ffill":
                # Forward-fill within each loan group
                if "loan_id" in df.columns:
                    df[col] = (
                        df.groupby("loan_id")[col]
                        .transform(lambda s: s.ffill().bfill())
                    )
                else:
                    df[col] = df[col].ffill().bfill()
                # If still missing after ffill, use median/mode fallback
                if df[col].isna().any():
                    fallback = self._fit_values.get(col)
                    if fallback not in (None, "ffill", "zero", "const"):
                        df[col] = df[col].fillna(fallback)

            elif strategy == "zero":
                df[col] = df[col].fillna(0)

            elif strategy == "const":
                df[col] = df[col].fillna(CONST_FILL_VALUE)

            remaining = int(df[col].isna().sum())
            if n_missing > 0:
                log.debug(
                    "Imputed '%s': %d → %d remaining missing", col, n_missing, remaining
                )

        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap numeric columns at IQR-derived bounds (fitted on train)."""
        for col, (lower, upper) in self._iqr_bounds.items():
            if col not in df.columns:
                continue
            n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_capped > 0:
                df[col] = df[col].clip(lower, upper)
                log.debug("Capped %d outliers in '%s' [%.2f, %.2f]", n_capped, col, lower, upper)
        return df

    def _apply_domain_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hard domain-knowledge bounds regardless of IQR."""
        # Interest rate: physically impossible outside [0, 30]
        if "interest_rate" in df.columns:
            df["interest_rate"] = df["interest_rate"].clip(0, 30)

        # DPD: cannot be negative
        if "days_past_due" in df.columns:
            df["days_past_due"] = df["days_past_due"].clip(0, None)

        # Balances: cannot be negative
        for col in ["current_balance", "original_balance"]:
            if col in df.columns:
                df[col] = df[col].clip(0, None)

        # Loan age: cannot be negative
        if "loan_age_months" in df.columns:
            df["loan_age_months"] = df["loan_age_months"].clip(0, None)

        return df

    def _resolve_servicer_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect conflicts between main panel and servicer_updates.

        Fields prefixed with 'svc_' come from the servicer join.
        For each svc_ field, compare to the main field and set a conflict flag.
        """
        svc_cols = [c for c in df.columns if c.startswith("svc_")]
        if not svc_cols:
            df["servicer_conflict_flag"] = 0
            df["n_servicer_conflicts"] = 0
            return df

        conflict_flags = []
        for svc_col in svc_cols:
            base_col = svc_col.replace("svc_", "", 1)
            if base_col not in df.columns:
                continue

            # For numeric columns, flag if absolute difference > 1% of main value
            if pd.api.types.is_numeric_dtype(df[base_col]) and pd.api.types.is_numeric_dtype(df[svc_col]):
                pct_diff = (df[base_col] - df[svc_col]).abs() / (df[base_col].abs() + 1e-9)
                flag = (pct_diff > 0.01).astype(int)
            else:
                # For categorical: exact mismatch (ignoring NaN)
                flag = (
                    df[base_col].astype(str) != df[svc_col].astype(str)
                ).astype(int)
                # Don't flag if either side is NaN
                flag = flag & df[base_col].notna().astype(int) & df[svc_col].notna().astype(int)

            conflict_flags.append(flag)

        if conflict_flags:
            n_conflicts = pd.concat(conflict_flags, axis=1).sum(axis=1)
            df["n_servicer_conflicts"] = n_conflicts
            df["servicer_conflict_flag"] = (n_conflicts > 0).astype(int)
        else:
            df["n_servicer_conflicts"] = 0
            df["servicer_conflict_flag"] = 0

        n_conflict_rows = int(df["servicer_conflict_flag"].sum())
        if n_conflict_rows > 0:
            log.warning(
                "Servicer conflicts detected in %d rows (%.1f%%)",
                n_conflict_rows,
                100 * n_conflict_rows / len(df),
            )

        return df

    def _flag_staleness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag records where last_updated_at is more than N days before reporting_month."""
        if "last_updated_at" not in df.columns or "reporting_month" not in df.columns:
            df["stale_record_flag"] = 0
            return df

        try:
            last_updated = pd.to_datetime(df["last_updated_at"], errors="coerce")
            reporting = pd.to_datetime(
                df["reporting_month"].astype(str) + "-01", errors="coerce"
            )
            staleness_days = (reporting - last_updated).dt.days
            threshold = self.cfg["features"].get(
                "staleness_threshold_days", STALENESS_THRESHOLD_DAYS
            )
            df["stale_record_flag"] = (staleness_days > threshold).astype(int)
            n_stale = int(df["stale_record_flag"].sum())
            if n_stale > 0:
                log.warning(
                    "Stale records (last_updated > %d days before reporting): %d (%.1f%%)",
                    threshold,
                    n_stale,
                    100 * n_stale / len(df),
                )
        except Exception as exc:
            log.warning("Could not compute staleness flag: %s", exc)
            df["stale_record_flag"] = 0

        return df
