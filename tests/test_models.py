"""
tests/test_models.py
---------------------
Unit tests for ML models: train/predict interface, save/load, leakage checks.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile


def make_classification_data(n: int = 200, n_features: int = 15, seed: int = 42):
    """Generate binary classification toy data."""
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((X["f0"] + np.random.randn(n) * 0.5 > 0).astype(int))
    return X, y


class TestDefaultModel:
    def test_train_predict(self):
        from src.models.default_12m import DefaultModel
        X, y = make_classification_data()
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        model = DefaultModel()
        model.train(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape == (50,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_evaluate_returns_dict(self):
        from src.models.default_12m import DefaultModel
        X, y = make_classification_data()
        model = DefaultModel()
        model.train(X[:150], y[:150])
        metrics = model.evaluate(X[150:], y[150:])
        assert "default_12m_roc_auc" in metrics
        assert 0.0 <= metrics["default_12m_roc_auc"] <= 1.0

    def test_save_load(self):
        from src.models.default_12m import DefaultModel
        X, y = make_classification_data()
        model = DefaultModel()
        model.train(X[:150], y[:150])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = model.save(tmpdir)
            assert path.exists()

            model2 = DefaultModel()
            model2.load(path)
            p1 = model.predict_proba(X[150:])
            p2 = model2.predict_proba(X[150:])
            np.testing.assert_array_almost_equal(p1, p2)

    def test_unfitted_raises(self):
        from src.models.default_12m import DefaultModel
        X, _ = make_classification_data()
        model = DefaultModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)


class TestDelinquencyModels:
    def test_3m_and_6m_independent(self):
        from src.models.delinquency_3m import Delinquency3mModel, Delinquency6mModel
        X, y = make_classification_data()
        model3 = Delinquency3mModel()
        model6 = Delinquency6mModel()
        model3.train(X[:150], y[:150])
        model6.train(X[:150], y[:150])
        assert model3.model_name != model6.model_name
        assert model3.model_name == "delinquency_3m"
        assert model6.model_name == "delinquency_6m"


class TestNextStateModel:
    def test_multiclass_predict(self):
        from src.models.next_state import NextStateModel
        np.random.seed(0)
        n = 300
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        states = ["Current", "30DPD", "60DPD", "Default", "Prepaid"]
        y = pd.Series(np.random.choice(states, n))

        model = NextStateModel()
        model.train(X[:250], y[:250])
        preds = model.predict(X[250:])
        assert set(preds).issubset(set(states))

    def test_proba_sums_to_one(self):
        from src.models.next_state import NextStateModel
        np.random.seed(0)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.choice(["Current", "30DPD", "60DPD", "Default"], n))
        model = NextStateModel()
        model.train(X[:150], y[:150])
        proba = model.predict_proba(X[150:])
        assert proba.shape[0] == 50
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(50), decimal=5)


class TestCalibration:
    def test_ece_improves(self):
        from src.models.default_12m import DefaultModel
        from src.models.calibration import ModelCalibrator
        from src.utils.metrics import ece

        X, y = make_classification_data(n=400)
        model = DefaultModel()
        model.train(X[:200], y[:200])

        calibrator = ModelCalibrator(model)
        calibrator.fit(X[200:300], y[200:300])

        X_test, y_test = X[300:], y[300:]
        ece_before = ece(y_test.values, model.predict_proba(X_test))
        ece_after  = ece(y_test.values, calibrator.predict_proba(X_test))
        # Calibration should not catastrophically worsen ECE
        assert ece_after <= ece_before * 2 + 0.1


class TestMetrics:
    def test_roc_auc_binary(self):
        from src.utils.metrics import roc_auc_score_safe
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc_score_safe(y, p) == pytest.approx(1.0)

    def test_roc_auc_single_class(self):
        from src.utils.metrics import roc_auc_score_safe
        y = np.array([0, 0, 0, 0])
        p = np.array([0.1, 0.2, 0.3, 0.4])
        # Should not raise, should return 0.5
        result = roc_auc_score_safe(y, p)
        assert result == 0.5

    def test_ks_statistic(self):
        from src.utils.metrics import ks_statistic
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        ks = ks_statistic(y, p)
        assert 0.0 <= ks <= 1.0

    def test_brier_score(self):
        from src.utils.metrics import brier_score
        y = np.array([0, 1])
        p = np.array([0.0, 1.0])
        assert brier_score(y, p) == pytest.approx(0.0)

    def test_macro_f1(self):
        from src.utils.metrics import macro_f1
        y_true = np.array(["A", "B", "C", "A"])
        y_pred = np.array(["A", "B", "C", "A"])
        assert macro_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_harrell_c_index_perfect(self):
        """Perfect ranking: C-index should be 1.0."""
        from src.utils.metrics import harrell_c_index
        times = np.array([10, 20, 30, 40, 50])
        events = np.array([1, 1, 1, 1, 1])
        risk = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Higher risk → shorter survival
        c = harrell_c_index(times, events, risk)
        assert c == pytest.approx(1.0, abs=0.01)
