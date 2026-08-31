"""
tests/test_survival.py
-----------------------
Unit tests for Markov transition model and basic survival plumbing.
"""

import numpy as np
import pandas as pd
import pytest


def make_panel(n_loans: int = 20, n_months: int = 12) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    states = ["Current", "30DPD", "60DPD", "Default", "Prepaid"]
    for loan in range(n_loans):
        for m in range(n_months):
            rows.append({
                "loan_id": f"LN{loan:04d}",
                "month_index": m + 1,
                "current_status": np.random.choice(states, p=[0.7, 0.15, 0.07, 0.04, 0.04]),
                "loan_age_months": m + 6,
                "days_past_due": max(0, int(np.random.exponential(3))),
                "next_12m_default_flag": np.random.binomial(1, 0.08),
                "next_12m_prepayment_flag": np.random.binomial(1, 0.10),
            })
    return pd.DataFrame(rows)


class TestMarkovTransitionModel:
    def test_fit_builds_matrix(self):
        from src.survival.transition import MarkovTransitionModel, STATES
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        P = model.get_transition_matrix()
        assert P.shape == (len(STATES), len(STATES))

    def test_rows_sum_to_one(self):
        from src.survival.transition import MarkovTransitionModel
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        P = model.get_transition_matrix().values
        np.testing.assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[0]), decimal=5)

    def test_absorbing_states(self):
        from src.survival.transition import MarkovTransitionModel, ABSORBING_STATES, STATE_IDX
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        P = model.get_transition_matrix().values
        for state in ABSORBING_STATES:
            idx = STATE_IDX[state]
            assert P[idx, idx] == pytest.approx(1.0)

    def test_project_shape(self):
        from src.survival.transition import MarkovTransitionModel, STATES
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        proj = model.project(n_months=12)
        assert len(proj) == 13  # month 0 through 12
        assert "Current" in proj.columns

    def test_scenario_shift_rows_sum_to_one(self):
        from src.survival.transition import MarkovTransitionModel
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        P_shifted = model.apply_scenario_shift(rate_delta=3.0, hpi_delta=-0.15)
        np.testing.assert_array_almost_equal(
            P_shifted.sum(axis=1), np.ones(P_shifted.shape[0]), decimal=4
        )

    def test_scenario_adverse_increases_default(self):
        from src.survival.transition import MarkovTransitionModel, STATE_IDX
        df = make_panel()
        model = MarkovTransitionModel()
        model.fit(df)
        P_base = model._global_matrix
        P_adverse = model.apply_scenario_shift(rate_delta=3.0, hpi_delta=-0.15, unemployment_delta=4.0)
        cur_idx = STATE_IDX["Current"]
        dpd30_idx = STATE_IDX["30DPD"]
        assert P_adverse[cur_idx, dpd30_idx] >= P_base[cur_idx, dpd30_idx]


class TestAnomalyDetector:
    def test_scores_in_range(self):
        from src.anomaly.detector import AnomalyDetector
        np.random.seed(1)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(X, list(X.columns))
        scores = detector.predict(X)
        assert len(scores) == 100
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_top_examples(self):
        from src.anomaly.detector import AnomalyDetector
        np.random.seed(2)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        X["loan_id"] = [f"LN{i}" for i in range(100)]
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(X[[c for c in X.columns if c != "loan_id"]])
        scores = detector.predict(X[[c for c in X.columns if c != "loan_id"]])
        top = detector.get_top_examples(X, scores, n=5)
        assert len(top) == 5


class TestExceptionEngine:
    def test_no_exceptions_when_no_flags(self):
        from src.anomaly.exception import ExceptionEngine
        df = pd.DataFrame({"loan_id": ["LN001", "LN002"], "current_status": ["Current", "30DPD"]})
        engine = ExceptionEngine()
        result = engine.run(df)
        assert "exception_required" in result.columns
        assert result["exception_required"].sum() == 0

    def test_flags_detected(self):
        from src.anomaly.exception import ExceptionEngine
        df = pd.DataFrame({
            "loan_id": ["LN001", "LN002"],
            "current_status": ["Current", "30DPD"],
            "flag_status_dpd_mismatch": [1, 0],
            "flag_negative_balance": [0, 1],
        })
        engine = ExceptionEngine()
        result = engine.run(df)
        assert result["exception_required"].sum() == 2
        assert result.loc[0, "exception_type"] == "status_conflict"
        assert result.loc[1, "exception_type"] == "balance_error"
