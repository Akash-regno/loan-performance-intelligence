"""
src/features/encoding.py
------------------------
Categorical encoding for loan features.

Encoders implemented:
  - OrdinalEncoder      : credit_score_band, ltv_band, dti_band (ordered bands)
  - TargetEncoder       : servicer_name, state (high-cardinality, fit on train)
  - BinaryEncoder       : loan_purpose, occupancy_type, property_type, document_status
  - LabelEncoder        : current_status → numeric index

All encoders follow the fit-on-train / transform-on-test discipline.

Usage:
    from src.features.encoding import CategoricalEncoder
    enc = CategoricalEncoder()
    enc.fit(train_df, target_col='next_12m_default_flag')
    train_enc = enc.transform(train_df)
    test_enc = enc.transform(test_df)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class CategoricalEncoder:
    """Encode categorical columns using appropriate strategies.

    Attributes
    ----------
    _ordinal_maps : dict
        Mapping from category string → integer rank.
    _target_enc_means : dict
        Per-category target mean (for target encoding).
    _status_map : dict
        current_status → integer.
    _is_fitted : bool
    """

    # Current status ordered by risk level
    STATUS_ORDER = [
        "Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Liquidated",
        "Unknown",
    ]

    def __init__(self) -> None:
        self.cfg = get_config()
        feat_cfg = self.cfg["features"]

        self._credit_score_order: list[str] = feat_cfg["credit_score_band_order"]
        self._ltv_order: list[str] = feat_cfg["ltv_band_order"]
        self._dti_order: list[str] = feat_cfg["dti_band_order"]
        self._target_encode_cols: list[str] = feat_cfg["target_encode_cols"]

        self._ordinal_maps: dict[str, dict[str, int]] = {}
        self._target_enc_means: dict[str, dict[str, float]] = {}
        self._global_target_mean: dict[str, float] = {}
        self._status_map: dict[str, int] = {}
        self._binary_dummies: dict[str, list[str]] = {}
        self._is_fitted: bool = False

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "next_12m_default_flag",
    ) -> "CategoricalEncoder":
        """Fit all encoders on training data.

        Parameters
        ----------
        df : DataFrame
            Training DataFrame.
        target_col : str
            Target column used for target encoding. Must be present in df.
        """
        log.info("Fitting CategoricalEncoder on %d training rows…", len(df))

        # Ordinal band encodings (deterministic from config — no fitting needed)
        self._ordinal_maps["credit_score_band"] = {
            v: i for i, v in enumerate(self._credit_score_order)
        }
        self._ordinal_maps["ltv_band"] = {
            v: i for i, v in enumerate(self._ltv_order)
        }
        self._ordinal_maps["dti_band"] = {
            v: i for i, v in enumerate(self._dti_order)
        }

        # Status encoding
        self._status_map = {s: i for i, s in enumerate(self.STATUS_ORDER)}

        # Target encoding (fit on training data only)
        if target_col in df.columns:
            global_mean = df[target_col].dropna().astype(float).mean()
            for col in self._target_encode_cols:
                if col not in df.columns:
                    continue
                self._global_target_mean[col] = global_mean
                means = df.groupby(col)[target_col].mean().to_dict()
                self._target_enc_means[col] = means
                log.debug("Target-encoded '%s' (%d categories)", col, len(means))

        # Binary (one-hot) columns — record which dummies were created on train
        for col in ["loan_purpose", "occupancy_type", "property_type", "document_status"]:
            if col in df.columns:
                cats = sorted(df[col].dropna().astype(str).unique().tolist())
                self._binary_dummies[col] = cats

        self._is_fitted = True
        log.info("CategoricalEncoder fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all encoders to df. Returns a copy with encoded columns added."""
        if not self._is_fitted:
            raise RuntimeError("Fit encoder before calling transform().")

        df = df.copy()

        df = self._encode_ordinal(df)
        df = self._encode_status(df)
        df = self._encode_target(df)
        df = self._encode_binary(df)

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "next_12m_default_flag",
    ) -> pd.DataFrame:
        """Fit and transform in one call (for training data only)."""
        return self.fit(df, target_col).transform(df)

    # ──────────────────────────────────────────────────────────
    # Private encoders
    # ──────────────────────────────────────────────────────────

    def _encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map ordered band strings to integers."""
        for col, mapping in self._ordinal_maps.items():
            if col not in df.columns:
                continue
            # Unknown categories get -1
            df[f"{col}_ord"] = (
                df[col].astype(str).map(mapping).fillna(-1).astype(int)
            )
        return df

    def _encode_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode current_status as ordered integer."""
        if "current_status" not in df.columns:
            return df
        df["current_status_enc"] = (
            df["current_status"].astype(str).map(self._status_map).fillna(
                len(self.STATUS_ORDER) - 1  # Unknown
            ).astype(int)
        )
        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Target-encode high-cardinality columns using train-fitted means."""
        for col in self._target_encode_cols:
            if col not in df.columns:
                continue
            if col not in self._target_enc_means:
                continue
            global_mean = self._global_target_mean.get(col, 0.0)
            df[f"{col}_te"] = (
                df[col].astype(str).map(self._target_enc_means[col]).fillna(global_mean)
            )
        return df

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode low-cardinality columns using train-observed categories."""
        for col, train_cats in self._binary_dummies.items():
            if col not in df.columns:
                continue
            for cat in train_cats:
                # Safe column name: replace spaces and special chars
                safe_cat = str(cat).replace(" ", "_").replace("/", "_")
                df[f"{col}_{safe_cat}"] = (df[col].astype(str) == cat).astype(int)
        return df
