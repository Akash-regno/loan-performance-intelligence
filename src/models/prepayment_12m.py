"""
src/models/prepayment_12m.py
-----------------------------
12-month prepayment prediction model.

Target: next_12m_prepayment_flag (binary: 1 = prepay within 12 months)
Algorithm: LightGBM (binary)

Key drivers: interest rate spread, loan age, LTV, loan_purpose, refi_incentive.

Usage:
    from src.models.prepayment_12m import PrepaymentModel
    model = PrepaymentModel()
    model.train(X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class PrepaymentModel(BaseModel):
    """LightGBM binary classifier for 12-month prepayment prediction."""

    model_name = "prepayment_12m"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"]["prepayment_12m"]
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
    ) -> "PrepaymentModel":
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
        return super().train(X_train, y_train, X_val, y_val, feature_cols)

    def _supports_eval_set(self) -> bool:
        return True
