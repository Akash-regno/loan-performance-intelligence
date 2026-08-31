"""
src/models/calibration.py
--------------------------
Probability calibration wrapper for all binary models.

Uses isotonic regression (non-parametric) which outperforms Platt scaling
for large datasets (N > 10,000).

Post-calibration validation:
  - Reliability diagram (10-bin)
  - Expected Calibration Error (ECE) target: < 0.05
  - Brier score improvement check

Usage:
    from src.models.calibration import ModelCalibrator
    calibrator = ModelCalibrator(base_model)
    calibrator.fit(X_cal, y_cal)
    cal_probs = calibrator.predict_proba(X_test)
    ece_val = calibrator.get_ece(X_val, y_val)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.metrics import brier_score, ece

log = get_logger(__name__)


class ModelCalibrator:
    """Isotonic regression probability calibrator.

    Parameters
    ----------
    base_model : BaseModel
        A fitted BaseModel instance whose probabilities to calibrate.
    method : {'isotonic', 'sigmoid'}
        Calibration method. Default: 'isotonic'.
    """

    def __init__(self, base_model: BaseModel, method: str = "isotonic") -> None:
        self.base_model = base_model
        self.method = method
        self.model_name = f"{base_model.model_name}_calibrated"
        self._calibrated_model: Any = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.Series | np.ndarray,
        cv: str = "prefit",
    ) -> "ModelCalibrator":
        """Fit the calibrator on a hold-out calibration set.

        Parameters
        ----------
        X_cal, y_cal : calibration features and labels
            Must NOT overlap with the training data used to fit base_model.
        cv : 'prefit'
            Use pre-fitted base model (recommended for pipeline).
        """
        log.info(
            "Fitting %s calibrator (%s) on %d calibration samples…",
            self.model_name, self.method, len(y_cal),
        )
        X_arr = X_cal.values if isinstance(X_cal, pd.DataFrame) else np.asarray(X_cal)
        y_arr = np.asarray(y_cal).astype(int)

        self._calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model.model,
            method=self.method,
            cv=cv,
        )
        self._calibrated_model.fit(X_arr, y_arr)
        self._is_fitted = True

        # Report calibration quality
        cal_probs = self.predict_proba(X_cal)
        before_brier = brier_score(y_arr, self.base_model.predict_proba(X_cal))
        after_brier = brier_score(y_arr, cal_probs)
        after_ece = ece(y_arr, cal_probs)

        log.info(
            "Calibration complete | Brier: %.4f → %.4f | ECE: %.4f",
            before_brier, after_brier, after_ece,
        )
        if after_ece > 0.05:
            log.warning(
                "ECE %.4f > 0.05 target — consider larger calibration set or sigmoid method",
                after_ece,
            )
        return self

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Return calibrated probabilities."""
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        proba = self._calibrated_model.predict_proba(X_arr)
        # Binary: return positive class
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def get_ece(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> float:
        """Compute ECE on a labeled dataset."""
        probs = self.predict_proba(X)
        return ece(np.asarray(y).astype(int), probs)

    def save(self, directory: str | Path | None = None) -> Path:
        """Save calibrated model."""
        self._check_fitted()
        base_dir = Path(directory or f"models/calibrated")
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{self.model_name}.pkl"
        with path.open("wb") as fh:
            pickle.dump(
                {"calibrated_model": self._calibrated_model, "method": self.method},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.info("Calibrated model saved → %s", path)
        return path

    def load(self, path: str | Path) -> "ModelCalibrator":
        """Load a saved calibrated model."""
        path = Path(path)
        with path.open("rb") as fh:
            data = pickle.load(fh)
        self._calibrated_model = data["calibrated_model"]
        self.method = data["method"]
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ModelCalibrator must be fit() before predict_proba().")
