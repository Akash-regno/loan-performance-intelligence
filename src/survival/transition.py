"""
src/survival/transition.py
---------------------------
Multi-state Markov monthly transition model.

States: {Current, 30DPD, 60DPD, 90DPD, Default, Prepaid, Liquidated}

Computes:
  1. Empirical monthly transition probability matrix from training data
  2. Covariate-adjusted matrices per credit_score_band × vintage_year
  3. T+1, T+3, T+6, T+12 portfolio state distribution projections

Used by the Scenario Engine to shift transition probabilities under
adverse / high-prepayment macro conditions.

Usage:
    from src.survival.transition import MarkovTransitionModel
    model = MarkovTransitionModel()
    model.fit(train_df)
    projection = model.project(current_state_dist, n_months=12)
    model.get_transition_matrix()   # → DataFrame heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Canonical state ordering
STATES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Liquidated"]
N_STATES = len(STATES)
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Absorbing states: once in these, probability of staying = 1
ABSORBING_STATES = {"Default", "Prepaid", "Liquidated"}


class MarkovTransitionModel:
    """Monthly multi-state Markov transition model.

    Parameters
    ----------
    segment_cols : list of str
        Columns to segment transition matrices (e.g. credit_score_band).
        If empty, a single global matrix is estimated.
    """

    def __init__(self, segment_cols: list[str] | None = None) -> None:
        self.segment_cols = segment_cols or []
        self._global_matrix: np.ndarray | None = None          # (N_STATES × N_STATES)
        self._segment_matrices: dict[str, np.ndarray] = {}     # segment_key → matrix
        self._is_fitted: bool = False

    # ──────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "MarkovTransitionModel":
        """Estimate transition matrices from training panel data.

        Requires columns: loan_id, month_index, current_status
        """
        log.info("Fitting Markov transition model on %d rows…", len(df))
        self._validate_cols(df)

        # Build (from_state, to_state) transition pairs
        transitions = self._extract_transitions(df)
        log.info("Extracted %d state transitions.", len(transitions))

        # Global matrix
        self._global_matrix = self._estimate_matrix(transitions)
        log.info("Global transition matrix estimated.")

        # Segment matrices
        if self.segment_cols:
            for seg_col in self.segment_cols:
                if seg_col not in df.columns:
                    continue
                for seg_val, grp in transitions.groupby(f"seg_{seg_col}"):
                    if len(grp) < 100:
                        continue
                    key = f"{seg_col}={seg_val}"
                    self._segment_matrices[key] = self._estimate_matrix(grp)
            log.info(
                "Segment matrices computed: %d segments", len(self._segment_matrices)
            )

        self._is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────
    # Projection
    # ──────────────────────────────────────────────────────────

    def project(
        self,
        initial_distribution: np.ndarray | None = None,
        n_months: int = 12,
        matrix: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Project portfolio state distribution forward N months.

        Parameters
        ----------
        initial_distribution : array of shape (N_STATES,)
            Fraction of portfolio in each state at t=0.
            If None, defaults to 80% Current, 10% 30DPD, 5% 60DPD, 5% 90DPD.
        n_months : int
            Number of monthly steps to project.
        matrix : ndarray, optional
            Override transition matrix (for scenario simulations).

        Returns
        -------
        DataFrame
            Columns: month (0..n_months), + one col per state.
        """
        self._check_fitted()
        P = matrix if matrix is not None else self._global_matrix

        if initial_distribution is None:
            initial_distribution = np.array([0.80, 0.10, 0.05, 0.05, 0.0, 0.0, 0.0])

        dist = np.asarray(initial_distribution, dtype=float)
        dist = dist / dist.sum()  # Normalize

        records = [{"month": 0, **{s: round(float(dist[i]), 5) for i, s in enumerate(STATES)}}]

        for m in range(1, n_months + 1):
            dist = dist @ P
            records.append({
                "month": m,
                **{s: round(float(dist[i]), 5) for i, s in enumerate(STATES)},
            })

        return pd.DataFrame(records)

    def apply_scenario_shift(
        self,
        rate_delta: float = 0.0,
        hpi_delta: float = 0.0,
        unemployment_delta: float = 0.0,
    ) -> np.ndarray:
        """Return a scenario-shifted transition matrix.

        Rule-based shifts:
          - Adverse (higher rates, higher unemployment, lower HPI):
            ↑ P(Current→30DPD), ↑ P(30DPD→60DPD), ↑ P(60DPD→90DPD)
          - High-prepayment (lower rates):
            ↑ P(Current→Prepaid), ↑ P(30DPD→Prepaid)
        """
        self._check_fitted()
        P = self._global_matrix.copy()

        # Compute a stress factor proportional to the macro delta
        stress = (
            0.2 * max(rate_delta, 0) / 3.0
            + 0.1 * max(-hpi_delta, 0) / 0.15
            + 0.1 * max(unemployment_delta, 0) / 4.0
        )
        refi_boost = 0.3 * max(-rate_delta, 0) / 1.5

        # Apply shifts (clipped so matrix rows stay ~1)
        cur, dpd30, dpd60, dpd90 = (
            STATE_IDX["Current"], STATE_IDX["30DPD"],
            STATE_IDX["60DPD"], STATE_IDX["90DPD"],
        )
        default_idx = STATE_IDX["Default"]
        prepay_idx = STATE_IDX["Prepaid"]

        def _shift_row(row: np.ndarray, from_idx: int, to_idx: int, delta: float) -> np.ndarray:
            """Shift probability mass from from_idx to to_idx."""
            delta = min(delta, row[from_idx] * 0.5)  # Never shift more than 50%
            row[from_idx] -= delta
            row[to_idx] += delta
            return row

        if stress > 0:
            P[cur] = _shift_row(P[cur], cur, dpd30, stress * 0.05)
            P[dpd30] = _shift_row(P[dpd30], dpd30, dpd60, stress * 0.08)
            P[dpd60] = _shift_row(P[dpd60], dpd60, dpd90, stress * 0.10)
            P[dpd90] = _shift_row(P[dpd90], dpd90, default_idx, stress * 0.12)

        if refi_boost > 0:
            P[cur] = _shift_row(P[cur], cur, prepay_idx, refi_boost * 0.04)
            P[dpd30] = _shift_row(P[dpd30], dpd30, prepay_idx, refi_boost * 0.02)

        # Re-normalize rows
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        P = P / row_sums

        return P

    # ──────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────

    def get_transition_matrix(self, segment: str | None = None) -> pd.DataFrame:
        """Return the transition matrix as a labeled DataFrame (for heatmap)."""
        self._check_fitted()
        P = (
            self._segment_matrices.get(segment, self._global_matrix)
            if segment else self._global_matrix
        )
        return pd.DataFrame(P, index=STATES, columns=STATES).round(5)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _extract_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a DataFrame of (from_state, to_state) pairs."""
        df = df.sort_values(["loan_id", "month_index"])
        grp = df.groupby("loan_id")

        from_states = df["current_status"].astype(str)
        to_states = grp["current_status"].shift(-1).astype(str)

        transitions = pd.DataFrame({
            "from_state": from_states,
            "to_state": to_states,
        }).dropna(subset=["to_state"])

        # Remove rows where to_state is NaN (last month of each loan)
        transitions = transitions[transitions["to_state"] != "nan"]

        # Add segment columns if requested
        for seg_col in self.segment_cols:
            if seg_col in df.columns:
                transitions[f"seg_{seg_col}"] = df[seg_col].values[: len(transitions)]

        return transitions

    @staticmethod
    def _estimate_matrix(transitions: pd.DataFrame) -> np.ndarray:
        """Compute row-normalized transition count matrix."""
        P = np.zeros((N_STATES, N_STATES), dtype=float)

        for _, row in transitions.iterrows():
            fs = STATE_IDX.get(row["from_state"], -1)
            ts = STATE_IDX.get(row["to_state"], -1)
            if fs >= 0 and ts >= 0:
                P[fs, ts] += 1

        # Force absorbing states
        for state in ABSORBING_STATES:
            idx = STATE_IDX[state]
            P[idx, :] = 0
            P[idx, idx] = 1.0

        # Ensure all rows have at least one transition (self-loop fallback)
        row_sums = P.sum(axis=1)
        for i in range(N_STATES):
            if row_sums[i] == 0:
                P[i, i] = 1.0

        row_sums = P.sum(axis=1, keepdims=True)
        return P / row_sums

    def _validate_cols(self, df: pd.DataFrame) -> None:
        required = ["loan_id", "month_index", "current_status"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Transition model requires columns: {missing}")

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._global_matrix is None:
            raise RuntimeError("Transition model must be fitted before projecting.")

