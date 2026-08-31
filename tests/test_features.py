"""
tests/test_features.py
-----------------------
Unit tests for feature engineering modules.
"""

import numpy as np
import pandas as pd
import pytest


def make_loan_panel(n_loans: int = 10, n_months: int = 6) -> pd.DataFrame:
    """Create a minimal loan panel DataFrame for testing."""
    np.random.seed(42)
    rows = []
    for loan_id in range(n_loans):
        for month in range(n_months):
            rows.append({
                "loan_id": f"LN{loan_id:04d}",
                "month_index": month + 1,
                "reporting_month": f"2022-{month+1:02d}",
                "origination_month": "2021-01",
                "loan_age_months": month + 12,
                "remaining_term_months": 360 - month - 12,
                "original_balance": 300000.0,
                "current_balance": 300000.0 - month * 500,
                "interest_rate": 3.5 + np.random.normal(0, 0.5),
                "days_past_due": max(0, int(np.random.exponential(5))),
                "credit_score_band": "680-699",
                "ltv_band": "80-85",
                "dti_band": "30-36",
                "state": "CA",
                "loan_purpose": "Purchase",
                "occupancy_type": "Primary",
                "property_type": "Single Family",
                "servicer_name": "ServicerA",
                "current_status": "Current",
                "modification_flag": 0,
                "prepayment_flag": 0,
                "default_flag": 0,
                "loss_severity_band": "30-40",
                "source_system": "primary",
                "document_status": "Complete",
                "last_updated_at": f"2022-{month+1:02d}-15",
            })
    return pd.DataFrame(rows)


class TestTemporalFeatures:
    def test_vintage_year(self):
        from src.features.temporal import TemporalFeatures
        df = make_loan_panel()
        out = TemporalFeatures().transform(df)
        assert "vintage_year" in out.columns
        assert out["vintage_year"].dropna().iloc[0] == 2021

    def test_loan_age_band(self):
        from src.features.temporal import TemporalFeatures
        df = make_loan_panel()
        out = TemporalFeatures().transform(df)
        assert "loan_age_band" in out.columns
        assert set(out["loan_age_band"]).issubset({"0-12m", "12-24m", "24-36m", "36-60m", "60m+", "nan"})

    def test_is_seasoned(self):
        from src.features.temporal import TemporalFeatures
        df = make_loan_panel()
        out = TemporalFeatures().transform(df)
        assert "is_seasoned_loan" in out.columns
        # loan_age_months starts at 12 → none are >= 36 in first 6 months
        assert out["is_seasoned_loan"].sum() == 0


class TestLagRollingFeatures:
    def test_dpd_lag1(self):
        from src.features.lag_rolling import LagRollingFeatures
        df = make_loan_panel()
        out = LagRollingFeatures().transform(df)
        assert "dpd_lag1" in out.columns

    def test_no_future_leakage(self):
        """dpd_lag1 at month_index=1 should be 0 (first observation, no prior data)."""
        from src.features.lag_rolling import LagRollingFeatures
        df = make_loan_panel()
        out = LagRollingFeatures().transform(df)
        first_month = out[out["month_index"] == 1]
        assert (first_month["dpd_lag1"] == 0).all()

    def test_balance_features(self):
        from src.features.lag_rolling import LagRollingFeatures
        df = make_loan_panel()
        out = LagRollingFeatures().transform(df)
        assert "pct_balance_remaining" in out.columns
        assert (out["pct_balance_remaining"] >= 0).all()

    def test_ever_30dpd(self):
        from src.features.lag_rolling import LagRollingFeatures
        df = make_loan_panel()
        df.loc[df.index[5], "days_past_due"] = 35  # inject delinquency
        out = LagRollingFeatures().transform(df)
        assert "ever_30dpd" in out.columns


class TestCategoricalEncoder:
    def test_ordinal_encoding(self):
        from src.features.encoding import CategoricalEncoder
        df = make_loan_panel()
        df["next_12m_default_flag"] = 0
        enc = CategoricalEncoder()
        enc.fit(df, target_col="next_12m_default_flag")
        out = enc.transform(df)
        assert "credit_score_band_ord" in out.columns
        assert "ltv_band_ord" in out.columns

    def test_target_encoding_no_leakage(self):
        """Target encoding fitted on train must not use test target."""
        from src.features.encoding import CategoricalEncoder
        df = make_loan_panel()
        train = df.iloc[:40].copy()
        test = df.iloc[40:].copy()
        train["next_12m_default_flag"] = 0
        test["next_12m_default_flag"] = 1  # would leak if encoding uses this

        enc = CategoricalEncoder()
        enc.fit(train, target_col="next_12m_default_flag")
        out_test = enc.transform(test)
        # Target mean should be 0.0 (from train), not 1.0
        if "servicer_name_te" in out_test.columns:
            assert out_test["servicer_name_te"].iloc[0] == pytest.approx(0.0, abs=0.01)


class TestInteractionFeatures:
    def test_risk_composite(self):
        from src.features.interactions import InteractionFeatures
        df = make_loan_panel()
        # Add band midpoints
        df["credit_score_band"] = "680-699"
        df["ltv_band"] = "80-85"
        df["dti_band"] = "30-36"
        out = InteractionFeatures().transform(df)
        assert "risk_composite" in out.columns
        assert (out["risk_composite"] >= 0).all()
        assert (out["risk_composite"] <= 1.01).all()


class TestImbalanceHandler:
    def test_class_weights(self):
        from src.features.imbalance import ImbalanceHandler
        np.random.seed(0)
        y = np.array([0] * 900 + [1] * 100)
        handler = ImbalanceHandler()
        weights = handler.compute_class_weights(y)
        assert weights[1] > weights[0]  # Minority class gets higher weight

    def test_scale_pos_weight(self):
        from src.features.imbalance import ImbalanceHandler
        y = np.array([0] * 900 + [1] * 100)
        handler = ImbalanceHandler()
        spw = handler.compute_scale_pos_weight(y)
        assert abs(spw - 9.0) < 0.1

    def test_threshold_tuning(self):
        from src.features.imbalance import ImbalanceHandler
        np.random.seed(1)
        y_true = np.array([0] * 700 + [1] * 300)
        y_prob = np.where(y_true == 1, np.random.uniform(0.4, 0.9, 1000), np.random.uniform(0.1, 0.5, 1000))
        handler = ImbalanceHandler()
        t = handler.tune_threshold(y_true, y_prob, metric="f1")
        assert 0.1 <= t <= 0.9
