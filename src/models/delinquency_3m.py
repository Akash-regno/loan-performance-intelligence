"""
src/models/delinquency_3m.py  /  delinquency_6m.py
---------------------------------------------------
3-month and 6-month delinquency prediction models.

Target: next_3m_delinquency_flag / next_6m_delinquency_flag (binary)
Algorithm: LightGBM (binary)

This file implements the 3-month model; the 6-month model is a subclass
with a different model_name — both share identical architecture.

Usage:
    from src.models.delinquency_3m import Delinquency3mModel
    from src.models.delinquency_6m import Delinquency6mModel
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class _DelinquencyBase(BaseModel):
    """Shared LightGBM delinquency classifier."""

    model_name: str = "_delinquency_base"
    _config_key: str = "delinquency_3m"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"].get(self._config_key, {})
        default_params = {
            "n_estimators": cfg.get("n_estimators", 500),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 6),
            "num_leaves": cfg.get("num_leaves", 63),
            "min_child_samples": cfg.get("min_child_samples", 50),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        super().__init__(params or default_params)

    def _build_model(self) -> Any:
        from lightgbm import LGBMClassifier

        params = {k: v for k, v in self.params.items() if v != "auto"}
        return LGBMClassifier(**params)

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        scale_pos_weight: float | None = None,
    ) -> "_DelinquencyBase":
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
        return super().train(X_train, y_train, X_val, y_val, feature_cols)

    def _supports_eval_set(self) -> bool:
        return True


class Delinquency3mModel(_DelinquencyBase):
    """LightGBM 3-month delinquency classifier."""
    model_name = "delinquency_3m"
    _config_key = "delinquency_3m"

    def __init__(self, params: dict | None = None) -> None:
        self.model_name = "delinquency_3m"
        self._config_key = "delinquency_3m"
        super().__init__(params)


class Delinquency6mModel(_DelinquencyBase):
    """LightGBM 6-month delinquency classifier."""
    model_name = "delinquency_6m"
    _config_key = "delinquency_6m"

    def __init__(self, params: dict | None = None) -> None:
        self.model_name = "delinquency_6m"
        self._config_key = "delinquency_6m"
        super().__init__(params)

