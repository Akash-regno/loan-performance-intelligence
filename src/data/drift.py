"""
src/data/drift.py
-----------------
Train/test distribution drift detection using Evidently.

Produces:
  - HTML drift report (reports/drift_report.html)
  - Per-feature PSI (Population Stability Index) values
  - Per-feature KS statistic (for numeric columns)
  - Summary drift DataFrame

Features with PSI > 0.2 or KS p-value < 0.05 are flagged as drifted.

Usage:
    from src.data.drift import DriftDetector
    detector = DriftDetector()
    drift_summary = detector.run(train_df, test_df, feature_cols)
    detector.export_html_report(train_df, test_df, "reports/drift_report.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger

log = get_logger(__name__)

# PSI thresholds (standard industry practice)
PSI_STABLE = 0.10       # < 0.10: no significant change
PSI_WARN = 0.20         # 0.10–0.20: slight population change
# > 0.20: significant shift → drift flagged


class DriftDetector:
    """Detect train/test distribution drift.

    Uses:
      - PSI (Population Stability Index) for numeric and categorical columns
      - KS test (Kolmogorov–Smirnov) for numeric columns
      - Evidently for rich HTML report (optional dependency)
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute PSI and KS for each feature column.

        Returns
        -------
        pd.DataFrame
            Columns: feature, psi, ks_stat, ks_pvalue, drift_flag, drift_severity
        """
        if feature_cols is None:
            # Default: all shared numeric + categorical columns except IDs
            exclude = {"loan_id", "month_index", "reporting_month"}
            feature_cols = [
                c for c in train_df.columns
                if c in test_df.columns and c not in exclude
            ]

        log.info(
            "Running drift detection on %d features (train=%d, test=%d rows)…",
            len(feature_cols),
            len(train_df),
            len(test_df),
        )

        records = []
        for col in feature_cols:
            rec = self._analyze_column(col, train_df[col], test_df[col])
            records.append(rec)

        drift_df = pd.DataFrame(records).sort_values("psi", ascending=False)

        n_drifted = int(drift_df["drift_flag"].sum())
        log.info(
            "Drift detection complete: %d/%d features drifted (PSI > %.2f)",
            n_drifted,
            len(feature_cols),
            PSI_WARN,
        )

        return drift_df

    def export_html_report(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: str = "reports/drift_report.html",
        feature_cols: list[str] | None = None,
    ) -> Path:
        """Export rich Evidently drift report to HTML.

        Falls back to a minimal HTML table if Evidently is not installed.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            log.info("Generating Evidently drift report → %s", path)

            # Limit to numeric + low-cardinality categorical columns
            if feature_cols is None:
                exclude = {"loan_id", "month_index", "reporting_month"}
                feature_cols = [
                    c for c in train_df.columns
                    if c in test_df.columns and c not in exclude
                ]

            train_sub = train_df[feature_cols].copy()
            test_sub = test_df[feature_cols].copy()

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=train_sub, current_data=test_sub)
            report.save_html(str(path))
            log.info("Evidently drift report written: %s", path.resolve())

        except ImportError:
            log.warning("Evidently not installed — writing minimal drift HTML")
            drift_df = self.run(train_df, test_df, feature_cols)
            self._write_minimal_drift_html(drift_df, path)
        except Exception as exc:
            log.error("Evidently report failed: %s", exc)
            drift_df = self.run(train_df, test_df, feature_cols)
            self._write_minimal_drift_html(drift_df, path)

        return path

    # ──────────────────────────────────────────────────────────
    # Per-column analysis
    # ──────────────────────────────────────────────────────────

    def _analyze_column(
        self,
        col: str,
        train_series: pd.Series,
        test_series: pd.Series,
    ) -> dict[str, Any]:
        """Compute PSI and KS for a single column."""
        is_numeric = pd.api.types.is_numeric_dtype(train_series)

        if is_numeric:
            psi = self._psi_numeric(train_series, test_series)
            ks_stat, ks_pvalue = self._ks_test(train_series, test_series)
        else:
            psi = self._psi_categorical(train_series, test_series)
            ks_stat, ks_pvalue = None, None

        drift_flag = psi > PSI_WARN or (ks_pvalue is not None and ks_pvalue < 0.05)
        drift_severity = (
            "stable" if psi < PSI_STABLE
            else "warning" if psi < PSI_WARN
            else "drifted"
        )

        return {
            "feature": col,
            "dtype": str(train_series.dtype),
            "psi": round(psi, 5),
            "ks_stat": round(ks_stat, 5) if ks_stat is not None else None,
            "ks_pvalue": round(ks_pvalue, 5) if ks_pvalue is not None else None,
            "drift_flag": drift_flag,
            "drift_severity": drift_severity,
            "train_mean": round(float(train_series.dropna().mean()), 4) if is_numeric else None,
            "test_mean": round(float(test_series.dropna().mean()), 4) if is_numeric else None,
        }

    def _psi_numeric(
        self,
        train: pd.Series,
        test: pd.Series,
        n_bins: int | None = None,
    ) -> float:
        """Compute PSI for a numeric column using histogram binning."""
        n_bins = n_bins or self.n_bins
        train = train.dropna().astype(float)
        test = test.dropna().astype(float)

        if len(train) == 0 or len(test) == 0:
            return 0.0

        # Use training data to define bin edges
        bins = np.percentile(train, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates
        if len(bins) < 2:
            return 0.0

        train_counts, _ = np.histogram(train, bins=bins)
        test_counts, _ = np.histogram(test, bins=bins)

        return self._psi_from_counts(train_counts, test_counts)

    def _psi_categorical(
        self,
        train: pd.Series,
        test: pd.Series,
    ) -> float:
        """Compute PSI for a categorical column."""
        train = train.dropna().astype(str)
        test = test.dropna().astype(str)

        if len(train) == 0 or len(test) == 0:
            return 0.0

        all_categories = set(train.unique()) | set(test.unique())

        train_counts = np.array([
            (train == cat).sum() for cat in all_categories
        ], dtype=float)
        test_counts = np.array([
            (test == cat).sum() for cat in all_categories
        ], dtype=float)

        return self._psi_from_counts(train_counts, test_counts)

    @staticmethod
    def _psi_from_counts(
        train_counts: np.ndarray,
        test_counts: np.ndarray,
    ) -> float:
        """Compute PSI from raw bin counts."""
        # Normalize to proportions
        train_pct = train_counts / (train_counts.sum() + 1e-9)
        test_pct = test_counts / (test_counts.sum() + 1e-9)

        # Replace zeros to avoid log(0)
        train_pct = np.where(train_pct == 0, 1e-6, train_pct)
        test_pct = np.where(test_pct == 0, 1e-6, test_pct)

        psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        return float(psi)

    @staticmethod
    def _ks_test(
        train: pd.Series,
        test: pd.Series,
    ) -> tuple[float, float]:
        """Two-sample KS test for numeric columns."""
        train = train.dropna().astype(float)
        test = test.dropna().astype(float)

        if len(train) < 5 or len(test) < 5:
            return 0.0, 1.0

        stat, pvalue = scipy_stats.ks_2samp(train, test)
        return float(stat), float(pvalue)

    # ──────────────────────────────────────────────────────────
    # Fallback HTML
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_minimal_drift_html(drift_df: pd.DataFrame, path: Path) -> None:
        """Write minimal HTML drift report."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Drift Report</title>
<style>
  body {{ font-family: sans-serif; }}
  .drifted {{ background: #ffe0e0; }}
  .warning {{ background: #fff3cd; }}
  .stable {{ background: #d4edda; }}
</style>
</head>
<body>
<h1>Train/Test Drift Report</h1>
<p>PSI thresholds: &lt;0.10 stable | 0.10–0.20 warning | &gt;0.20 drifted</p>
{drift_df.to_html(index=False, classes='table')}
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        log.info("Minimal drift HTML written: %s", path)
