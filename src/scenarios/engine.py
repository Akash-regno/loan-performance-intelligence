"""
src/scenarios/engine.py
------------------------
Scenario Engine: re-scores all ML models under different macro conditions.

Three built-in scenarios (from macro_scenarios.csv):
  - base          : current macro conditions (no changes)
  - adverse       : rate +300bp, HPI -15%, unemployment +4pp
  - high_prepayment: rate -150bp, HPI +10%

For each scenario:
  1. Apply macro delta to feature matrix
  2. Re-score: default_12m, prepayment_12m, delinquency_3m, delinquency_6m
  3. Re-project Markov transition distribution
  4. Compute portfolio Expected Loss and segment-level breakdowns
  5. Return ScenarioResult

Usage:
    from src.scenarios.engine import ScenarioEngine
    engine = ScenarioEngine(models_dict, transition_model, feature_cols)
    result = engine.run(test_df, scenario="adverse")
    print(result.portfolio_el)
    segment_df = result.segment_breakdown("credit_score_band")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Loss Given Default (LGD) assumptions by loss_severity_band
LGD_MAP = {
    "0-10": 0.05, "10-20": 0.15, "20-30": 0.25, "30-40": 0.35,
    "40-50": 0.45, "50-60": 0.55, "60-70": 0.65, "70-80": 0.75,
    "80+": 0.85, "Unknown": 0.40,
}
DEFAULT_LGD = 0.40


@dataclass
class ScenarioResult:
    """Container for a single scenario's output."""

    scenario_name: str
    n_loans: int
    portfolio_el: float                        # Expected Loss ($)
    portfolio_el_rate: float                   # EL / total_balance
    delinquency_rate_3m: float
    delinquency_rate_6m: float
    default_rate_12m: float
    prepayment_rate_12m: float
    state_distribution: pd.DataFrame          # T+1 to T+12 projection
    segment_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    top_risk_loans: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


class ScenarioEngine:
    """Run base / adverse / high-prepayment scenario simulations.

    Parameters
    ----------
    models : dict
        Keys: model name → fitted model or calibrated model.
        Expected keys: 'delinquency_3m', 'delinquency_6m', 'default_12m',
                       'prepayment_12m'
    transition_model : MarkovTransitionModel, optional
        Fitted transition model for state distribution projection.
    feature_cols : list of str
        Features used by the ML models.
    """

    SCENARIO_PARAMS = {
        "base":             {"rate_delta": 0.0,  "hpi_delta": 0.0,   "unemployment_delta": 0.0},
        "adverse":          {"rate_delta": 3.0,  "hpi_delta": -0.15, "unemployment_delta": 4.0},
        "high_prepayment":  {"rate_delta": -1.5, "hpi_delta": 0.10,  "unemployment_delta": 0.0},
    }

    def __init__(
        self,
        models: dict[str, Any],
        transition_model: Any = None,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.models = models
        self.transition_model = transition_model
        self.feature_cols = feature_cols or []
        self.cfg = get_config()

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(
        self,
        test_df: pd.DataFrame,
        scenario: str = "base",
        custom_params: dict | None = None,
    ) -> ScenarioResult:
        """Run a scenario and return ScenarioResult.

        Parameters
        ----------
        test_df : DataFrame
            Feature matrix (post feature-engineering, test split).
        scenario : str
            One of: 'base', 'adverse', 'high_prepayment'
        custom_params : dict, optional
            Override scenario parameters (rate_delta, hpi_delta, unemployment_delta).
        """
        params = custom_params or self.SCENARIO_PARAMS.get(scenario, {})
        log.info(
            "Running scenario '%s' | params=%s | n=%d loans",
            scenario, params, len(test_df),
        )

        # Step 1: Apply macro feature delta
        df_modified = self._apply_macro_delta(test_df, params)

        # Step 2: Re-score all models
        predictions = self._score_all_models(df_modified)

        # Step 3: Compute portfolio metrics
        el, el_rate = self._compute_expected_loss(
            test_df, predictions.get("prob_next_12m_default", np.zeros(len(test_df)))
        )

        # Step 4: Segment breakdowns
        segment_tables = self._compute_segments(test_df, predictions)

        # Step 5: State distribution projection
        state_dist = self._project_state_distribution(params, n_months=12)

        # Step 6: Top-20 stress loans
        top_risk = self._get_top_risk_loans(test_df, predictions)

        # Step 7: Assemble result
        result = ScenarioResult(
            scenario_name=scenario,
            n_loans=len(test_df),
            portfolio_el=round(el, 2),
            portfolio_el_rate=round(el_rate, 5),
            delinquency_rate_3m=round(
                float(predictions.get("prob_next_3m_delinquency", np.array([0])).mean()), 5
            ),
            delinquency_rate_6m=round(
                float(predictions.get("prob_next_6m_delinquency", np.array([0])).mean()), 5
            ),
            default_rate_12m=round(
                float(predictions.get("prob_next_12m_default", np.array([0])).mean()), 5
            ),
            prepayment_rate_12m=round(
                float(predictions.get("prob_next_12m_prepayment", np.array([0])).mean()), 5
            ),
            state_distribution=state_dist,
            segment_tables=segment_tables,
            top_risk_loans=top_risk,
            raw_predictions=self._build_predictions_df(test_df, predictions),
        )

        log.info(
            "Scenario '%s' complete | EL=$%.2fM (%.3f%%) | default_rate=%.3f%% | prepay_rate=%.3f%%",
            scenario,
            el / 1e6,
            el_rate * 100,
            result.default_rate_12m * 100,
            result.prepayment_rate_12m * 100,
        )
        return result

    def run_all(self, test_df: pd.DataFrame) -> dict[str, ScenarioResult]:
        """Run all three scenarios and return comparison dict."""
        return {name: self.run(test_df, scenario=name) for name in self.SCENARIO_PARAMS}

    def compare(self, results: dict[str, ScenarioResult]) -> pd.DataFrame:
        """Return a comparison table across scenarios."""
        rows = []
        for name, r in results.items():
            rows.append({
                "scenario": name,
                "portfolio_el_usd": r.portfolio_el,
                "portfolio_el_rate_pct": round(r.portfolio_el_rate * 100, 3),
                "delinquency_rate_3m_pct": round(r.delinquency_rate_3m * 100, 3),
                "delinquency_rate_6m_pct": round(r.delinquency_rate_6m * 100, 3),
                "default_rate_12m_pct": round(r.default_rate_12m * 100, 3),
                "prepayment_rate_12m_pct": round(r.prepayment_rate_12m * 100, 3),
            })
        return pd.DataFrame(rows).set_index("scenario")

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _apply_macro_delta(
        self, df: pd.DataFrame, params: dict
    ) -> pd.DataFrame:
        """Apply macro parameter deltas to the feature matrix."""
        df = df.copy()
        rate_delta = params.get("rate_delta", 0.0)
        hpi_delta = params.get("hpi_delta", 0.0)

        if rate_delta != 0 and "interest_rate" in df.columns:
            df["interest_rate"] = (df["interest_rate"] + rate_delta).clip(0, 30)

        # Update any macro-derived features
        if "rate_spread" in df.columns and rate_delta != 0:
            df["rate_spread"] = df["rate_spread"] + rate_delta

        if "refi_incentive" in df.columns and "rate_spread" in df.columns:
            df["refi_incentive"] = (df["rate_spread"] > 0.5).astype(int)

        # HPI affects LTV (roughly: LTV increases when HPI falls)
        if hpi_delta != 0 and "ltv_mid" in df.columns:
            df["ltv_mid"] = (df["ltv_mid"] / (1 + hpi_delta)).clip(0, 200)

        return df

    def _score_all_models(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Score all available models and return probability arrays."""
        predictions = {}
        feat_arr = self._get_feature_array(df)

        for model_key, col_name in [
            ("delinquency_3m",  "prob_next_3m_delinquency"),
            ("delinquency_6m",  "prob_next_6m_delinquency"),
            ("default_12m",     "prob_next_12m_default"),
            ("prepayment_12m",  "prob_next_12m_prepayment"),
        ]:
            model = self.models.get(model_key)
            if model is None:
                log.warning("Model '%s' not found — using zeros", model_key)
                predictions[col_name] = np.zeros(len(df))
                continue
            try:
                predictions[col_name] = np.clip(model.predict_proba(feat_arr), 0, 1)
            except Exception as exc:
                log.error("Scoring '%s' failed: %s", model_key, exc)
                predictions[col_name] = np.zeros(len(df))

        return predictions

    def _get_feature_array(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns available in df."""
        available = [c for c in self.feature_cols if c in df.columns]
        return df[available]

    def _compute_expected_loss(
        self, df: pd.DataFrame, pd_scores: np.ndarray
    ) -> tuple[float, float]:
        """EL = PD × LGD × EAD summed over portfolio."""
        ead = df.get("current_balance", pd.Series(np.zeros(len(df)))).fillna(0).values

        # LGD from loss_severity_band
        if "loss_severity_band" in df.columns:
            lgd = df["loss_severity_band"].map(LGD_MAP).fillna(DEFAULT_LGD).values
        else:
            lgd = np.full(len(df), DEFAULT_LGD)

        el_per_loan = pd_scores * lgd * ead
        total_el = float(el_per_loan.sum())
        total_balance = float(ead.sum())
        el_rate = total_el / total_balance if total_balance > 0 else 0.0
        return total_el, el_rate

    def _compute_segments(
        self, df: pd.DataFrame, predictions: dict[str, np.ndarray]
    ) -> dict[str, pd.DataFrame]:
        """Compute segment-level default/delinquency rates."""
        df_pred = df.copy()
        for col, arr in predictions.items():
            df_pred[col] = arr

        segment_tables = {}
        seg_cols = ["credit_score_band", "vintage_year", "state", "servicer_name"]

        for seg_col in seg_cols:
            if seg_col not in df_pred.columns:
                continue
            agg = df_pred.groupby(seg_col).agg(
                n_loans=(seg_col, "count"),
                mean_default_rate=("prob_next_12m_default", "mean"),
                mean_prepay_rate=("prob_next_12m_prepayment", "mean"),
                mean_delinquency_3m=("prob_next_3m_delinquency", "mean"),
            ).round(4).reset_index()
            segment_tables[seg_col] = agg

        return segment_tables

    def _project_state_distribution(
        self, params: dict, n_months: int = 12
    ) -> pd.DataFrame:
        """Project portfolio state distribution using the transition model."""
        if self.transition_model is None:
            return pd.DataFrame(columns=["month"])

        try:
            matrix = self.transition_model.apply_scenario_shift(
                rate_delta=params.get("rate_delta", 0.0),
                hpi_delta=params.get("hpi_delta", 0.0),
                unemployment_delta=params.get("unemployment_delta", 0.0),
            )
            return self.transition_model.project(
                initial_distribution=None, n_months=n_months, matrix=matrix
            )
        except Exception as exc:
            log.error("State projection failed: %s", exc)
            return pd.DataFrame(columns=["month"])

    def _get_top_risk_loans(
        self, df: pd.DataFrame, predictions: dict[str, np.ndarray], n: int = 20
    ) -> pd.DataFrame:
        """Return top-N highest-risk loans under this scenario."""
        df_pred = df.copy()
        pd_col = "prob_next_12m_default"
        df_pred[pd_col] = predictions.get(pd_col, np.zeros(len(df)))

        show_cols = [
            "loan_id", "month_index", "current_status", "days_past_due",
            "current_balance", "credit_score_band", "ltv_band",
            "servicer_name", pd_col,
        ]
        available = [c for c in show_cols if c in df_pred.columns]
        return (
            df_pred[available]
            .nlargest(n, pd_col)
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_predictions_df(
        df: pd.DataFrame, predictions: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Build a slim prediction DataFrame with loan_id + scores."""
        result = pd.DataFrame()
        if "loan_id" in df.columns:
            result["loan_id"] = df["loan_id"].values
        if "month_index" in df.columns:
            result["month_index"] = df["month_index"].values
        for col, arr in predictions.items():
            result[col] = arr
        return result
