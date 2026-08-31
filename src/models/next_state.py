"""
src/models/next_state.py
-------------------------
Next-state multi-class prediction model.

Target: next_state (multi-class)
Expected states: Current, 30DPD, 60DPD, 90DPD, Default, Prepaid, Liquidated
Algorithm: LightGBM (multiclass softmax)

Outputs per-class probability for each loan → feeds into the Markov
transition matrix and scenario engine.

Usage:
    from src.models.next_state import NextStateModel
    model = NextStateModel()
    model.train(X_train, y_train, X_val, y_val)
    proba_matrix = model.predict_proba(X_test)  # shape: (n, n_classes)
    states = model.predict(X_test)              # argmax class labels
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.base_model import BaseModel
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.metrics import macro_f1

log = get_logger(__name__)

STATE_ORDER = [
    "Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Liquidated"
]


class NextStateModel(BaseModel):
    """LightGBM multi-class next-state classifier."""

    model_name = "next_state"

    def __init__(self, params: dict | None = None) -> None:
        cfg = get_config()["models"]["next_state"]
        default_params = {
            "objective": "multiclass",
            "n_estimators": cfg.get("n_estimators", 500),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 6),
            "num_leaves": cfg.get("num_leaves", 63),
            "min_child_samples": cfg.get("min_child_samples", 50),
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
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
        )



    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        **kwargs,
    ) -> "NextStateModel":
        # Encode string labels to integers
        y_arr = np.asarray(y_train).astype(str)
        y_enc = self._label_encoder.fit_transform(y_arr)
        self.classes_ = list(self._label_encoder.classes_)
        self.params["num_class"] = len(self.classes_)
        log.info("NextStateModel classes: %s", self.classes_)

        y_val_enc = None
        if y_val is not None:
            y_val_enc = self._label_encoder.transform(
                np.asarray(y_val).astype(str)
            )

        return super().train(X_train, y_enc, X_val, y_val_enc, feature_cols)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability matrix (n_samples × n_classes)."""
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_arr)  # Already (n, n_classes)

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        """Return predicted state labels (decoded)."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        if self.classes_:
            return np.array([self.classes_[i] for i in idx])
        return idx

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, float]:
        y_pred = self.predict(X)
        y_true = np.asarray(y).astype(str)
        f1 = macro_f1(y_true, y_pred)
        return {f"{self.model_name}_macro_f1": f1}

    def _supports_eval_set(self) -> bool:
        return True
