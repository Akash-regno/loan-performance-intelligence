"""
src/models/default_12m.py
--------------------------
12-month default prediction model.

Target: next_12m_default_flag (binary: 1 = default within 12 months)
Algorithm: XGBoost (binary:logistic)

This is the highest-stakes model in the pipeline.
Judging criteria: AUC-PR, AUC-ROC, KS statistic, Lift@10%, Brier Score.

Usage:
    from src.models.default_12m import DefaultModel
    model = DefaultModel()
    model.train(X_train, y_train, X_val, y_val)
    prob = model.predict_proba(X_test)
    metrics = model.evaluate(X_val, y_val)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class DefaultModel(BaseModel):
    """XGBoost binary classifier for 12-month default prediction."""

    model_name = "default_12m"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"]["default_12m"]
        default_params = {
            "n_estimators": cfg.get("n_estimators", 500),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 6),
            "min_child_weight": cfg.get("min_child_weight", 10),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        super().__init__(params or default_params)

    def _build_model(self) -> Any:
        from xgboost import XGBClassifier

        # Pop scale_pos_weight if it's 'auto' (computed separately)
        params = {k: v for k, v in self.params.items() if v != "auto"}
        return XGBClassifier(**params)

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        scale_pos_weight: float | None = None,
    ) -> "DefaultModel":
        """Train with optional scale_pos_weight injection."""
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
            log.info("Using scale_pos_weight = %.2f", scale_pos_weight)

        return super().train(X_train, y_train, X_val, y_val, feature_cols)

    def _supports_eval_set(self) -> bool:
        return True

    def _get_early_stopping_callback(self) -> list:
        try:
            from xgboost.callback import EarlyStopping
            return [EarlyStopping(rounds=50, save_best=True)]
        except ImportError:
            return []
