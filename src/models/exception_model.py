"""
src/models/exception_model.py
------------------------------
Exception detection model: predicts exception_required (binary)
and exception_type (multi-class).

Trained on historically flagged rows from the validation rules engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.base_model import BaseModel
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class ExceptionRequiredModel(BaseModel):
    """LightGBM binary classifier: exception_required."""

    model_name = "exception_required"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"]["exception_model"]
        default_params = {
            "n_estimators": cfg.get("n_estimators", 300),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 5),
            "num_leaves": cfg.get("num_leaves", 31),
            "min_child_samples": cfg.get("min_child_samples", 30),
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

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_cols=None,
              scale_pos_weight=None, **kwargs):
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
        return super().train(X_train, y_train, X_val, y_val, feature_cols)

    def _supports_eval_set(self) -> bool:
        return True


class ExceptionTypeModel(BaseModel):
    """LightGBM multi-class classifier: exception_type."""

    model_name = "exception_type"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"]["exception_model"]
        default_params = {
            "objective": "multiclass",
            "n_estimators": cfg.get("n_estimators", 300),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 5),
            "num_leaves": cfg.get("num_leaves", 31),
            "min_child_samples": cfg.get("min_child_samples", 30),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        super().__init__(params or default_params)
        self._label_encoder = LabelEncoder()
        self.classes_: list[str] = []

    def _build_model(self) -> Any:
        from lightgbm import LGBMClassifier
        params = {k: v for k, v in self.params.items() if v != "auto"}
        return LGBMClassifier(**params)

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_cols=None, **kwargs):
        y_arr = np.asarray(y_train).astype(str)
        y_enc = self._label_encoder.fit_transform(y_arr)
        self.classes_ = list(self._label_encoder.classes_)
        self.params["num_class"] = len(self.classes_)
        y_val_enc = None
        if y_val is not None:
            y_val_enc = self._label_encoder.transform(np.asarray(y_val).astype(str))
        return super().train(X_train, y_enc, X_val, y_val_enc, feature_cols)

    def predict(self, X, threshold=None):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.classes_[i] for i in idx]) if self.classes_ else idx

    def _supports_eval_set(self) -> bool:
        return True
