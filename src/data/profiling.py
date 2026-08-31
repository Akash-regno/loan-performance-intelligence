"""
src/data/profiling.py
---------------------
Data profiling: distribution statistics, missingness analysis,
outlier detection, correlation analysis, and data-quality scoring.

Outputs:
  - HTML report (ydata-profiling)
  - Custom stats DataFrame (per column)
  - DQ score (per row)
  - Outlier flags (per column)
  - Correlation matrix

Usage:
    from src.data.profiling import DataProfiler
    profiler = DataProfiler()
    stats = profiler.run(df)
    profiler.export_html_report(df, "reports/data_quality_report.html")
    df_with_dq = profiler.add_dq_scores(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataProfiler:
    """Profile loan data: distributions, missingness, outliers, correlations, DQ scores."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.dq_cfg = self.cfg["dq_scoring"]

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run full profiling suite.

        Returns a dict with:
          'column_stats'   : per-column statistics DataFrame
          'missing_matrix' : missing-value pattern DataFrame
          'outlier_counts' : outlier count per numeric column
          'correlation'    : Pearson correlation matrix (numeric cols)
          'summary'        : overall dataset stats
        """
        log.info("Starting data profiling on %d rows × %d cols…", *df.shape)

        col_stats = self._column_statistics(df)
        missing_matrix = self._missing_patterns(df)
        outlier_counts = self._outlier_detection(df)
        correlation = self._correlation_matrix(df)
        summary = self._dataset_summary(df, col_stats, outlier_counts)

        log.info("Profiling complete.")
        return {
            "column_stats": col_stats,
            "missing_matrix": missing_matrix,
            "outlier_counts": outlier_counts,
            "correlation": correlation,
            "summary": summary,
        }

    def export_html_report(
        self,
        df: pd.DataFrame,
        output_path: str = "reports/data_quality_report.html",
        title: str = "Loan Performance — Data Quality Report",
    ) -> Path:
        """Generate a rich HTML profile report using ydata-profiling."""
        try:
            from ydata_profiling import ProfileReport

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            log.info("Generating ydata-profiling HTML report → %s", path)
            profile = ProfileReport(
                df,
                title=title,
                explorative=True,
                minimal=False,
                samples=None,  # Include samples for transparency
            )
            profile.to_file(path)
            log.info("HTML report written: %s", path.resolve())
            return path
        except ImportError:
            log.warning(
                "ydata-profiling not installed. Run: pip install ydata-profiling"
            )
            # Fallback: write a minimal HTML summary
            return self._write_minimal_html(df, output_path)

    def add_dq_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append a per-row dq_score column (0–100).

        Formula (from config):
          100
          - 15 × (n_missing_critical / n_critical)
          - 5  × (n_missing_optional / n_optional)
          - 10 × outlier_flag
          - 5  × cross_col_contradiction
          - 5  × servicer_conflict_flag
          - 5  × staleness_flag

        Adds columns:
          dq_score, dq_band (green/yellow/red)
        """
        df = df.copy()

        critical_cols = self.dq_cfg["critical_columns"]
        optional_cols = self.dq_cfg["optional_columns"]
        weights = self.dq_cfg["weights"]

        # Missing critical
        available_critical = [c for c in critical_cols if c in df.columns]
        if available_critical:
            missing_critical_rate = df[available_critical].isna().mean(axis=1)
        else:
            missing_critical_rate = pd.Series(0.0, index=df.index)

        # Missing optional
        available_optional = [c for c in optional_cols if c in df.columns]
        if available_optional:
            missing_optional_rate = df[available_optional].isna().mean(axis=1)
        else:
            missing_optional_rate = pd.Series(0.0, index=df.index)

        # Outlier flag
        outlier_flag = self._compute_row_outlier_flag(df)

        # Cross-column contradiction
        contradiction_flag = self._compute_contradiction_flag(df)

        # Servicer conflict
        servicer_conflict = df.get(
            "servicer_conflict_flag", pd.Series(0, index=df.index)
        ).fillna(0).astype(int)

        # Staleness
        staleness = df.get(
            "stale_record_flag", pd.Series(0, index=df.index)
        ).fillna(0).astype(int)

        dq_score = (
            100.0
            - weights["missing_critical"] * missing_critical_rate
            - weights["missing_optional"] * missing_optional_rate
            - weights["outlier_flag"] * outlier_flag
            - weights["cross_col_contradiction"] * contradiction_flag
            - weights["servicer_conflict"] * servicer_conflict
            - weights["staleness_flag"] * staleness
        ).clip(0, 100).round(1)

        df["dq_score"] = dq_score

        # DQ band
        red_t = self.dq_cfg["thresholds"]["red_flag"]
        yellow_t = self.dq_cfg["thresholds"]["yellow_flag"]
        df["dq_band"] = pd.cut(
            dq_score,
            bins=[-1, red_t, yellow_t, 101],
            labels=["red", "yellow", "green"],
        )

        log.info(
            "DQ scores computed: red=%d (%.1f%%) yellow=%d (%.1f%%) green=%d (%.1f%%)",
            (df["dq_band"] == "red").sum(),
            100 * (df["dq_band"] == "red").mean(),
            (df["dq_band"] == "yellow").sum(),
            100 * (df["dq_band"] == "yellow").mean(),
            (df["dq_band"] == "green").sum(),
            100 * (df["dq_band"] == "green").mean(),
        )
        return df

    # ──────────────────────────────────────────────────────────
    # Private profiling helpers
    # ──────────────────────────────────────────────────────────

    def _column_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-column statistics: dtype, missing%, mean, std, min, max, skew, kurtosis."""
        records = []
        for col in df.columns:
            series = df[col]
            missing_pct = round(100 * series.isna().mean(), 3)
            is_numeric = pd.api.types.is_numeric_dtype(series)

            rec: dict[str, Any] = {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "missing_pct": missing_pct,
                "n_unique": int(series.nunique()),
                "cardinality_pct": round(100 * series.nunique() / len(df), 3),
            }

            if is_numeric:
                s = series.dropna().astype(float)
                rec.update({
                    "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
                    "std": round(float(s.std()), 4) if len(s) > 0 else None,
                    "min": round(float(s.min()), 4) if len(s) > 0 else None,
                    "p25": round(float(s.quantile(0.25)), 4) if len(s) > 0 else None,
                    "p50": round(float(s.quantile(0.50)), 4) if len(s) > 0 else None,
                    "p75": round(float(s.quantile(0.75)), 4) if len(s) > 0 else None,
                    "max": round(float(s.max()), 4) if len(s) > 0 else None,
                    "skewness": round(float(scipy_stats.skew(s)), 4) if len(s) > 2 else None,
                    "kurtosis": round(float(scipy_stats.kurtosis(s)), 4) if len(s) > 3 else None,
                })
            else:
                top_values = series.value_counts().head(5).to_dict()
                rec["top_values"] = str(top_values)

            records.append(rec)

        return pd.DataFrame(records).sort_values("missing_pct", ascending=False)

    def _missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute missing-value matrix: n_missing and pct per column."""
        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "n_missing"]
        missing["pct_missing"] = (missing["n_missing"] / len(df) * 100).round(3)
        return missing.sort_values("n_missing", ascending=False)

    def _outlier_detection(
        self, df: pd.DataFrame, iqr_factor: float = 3.0
    ) -> pd.DataFrame:
        """IQR-based outlier detection for all numeric columns.

        Returns DataFrame: column, n_outliers, pct_outliers, lower_bound, upper_bound
        """
        records = []
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna().astype(float)
            if len(s) < 10:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            n_outliers = int(((s < lower) | (s > upper)).sum())
            records.append({
                "column": col,
                "n_outliers": n_outliers,
                "pct_outliers": round(100 * n_outliers / len(s), 3),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
            })
        return pd.DataFrame(records).sort_values("n_outliers", ascending=False)

    def _correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlation matrix of numeric columns."""
        num_df = df.select_dtypes(include="number")
        return num_df.corr(method="pearson").round(4)

    def _dataset_summary(
        self,
        df: pd.DataFrame,
        col_stats: pd.DataFrame,
        outlier_counts: pd.DataFrame,
    ) -> dict[str, Any]:
        """High-level dataset stats."""
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_unique_loans": df["loan_id"].nunique() if "loan_id" in df.columns else None,
            "month_range": (
                f"{df['month_index'].min()} – {df['month_index'].max()}"
                if "month_index" in df.columns else None
            ),
            "total_missing_pct": round(100 * df.isna().mean().mean(), 3),
            "cols_with_missing": int((df.isna().sum() > 0).sum()),
            "total_outlier_cells": int(outlier_counts["n_outliers"].sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    def _compute_row_outlier_flag(self, df: pd.DataFrame) -> pd.Series:
        """Binary flag: 1 if any numeric column has an IQR outlier on this row."""
        flag = pd.Series(0, index=df.index)
        for col in ["current_balance", "original_balance", "interest_rate", "days_past_due"]:
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            flag |= ((s < lower) | (s > upper)).astype(int)
        return flag

    def _compute_contradiction_flag(self, df: pd.DataFrame) -> pd.Series:
        """Flag rows with cross-column logical contradictions."""
        flag = pd.Series(0, index=df.index)

        # current_status = Current but DPD >= 30
        if "current_status" in df.columns and "days_past_due" in df.columns:
            flag |= (
                (df["current_status"] == "Current") & (df["days_past_due"] >= 30)
            ).astype(int)

        # prepayment_flag=1 and default_flag=1
        if "prepayment_flag" in df.columns and "default_flag" in df.columns:
            flag |= (
                (df["prepayment_flag"] == 1) & (df["default_flag"] == 1)
            ).astype(int)

        # prepayment_flag=1 but current_balance > 0
        if "prepayment_flag" in df.columns and "current_balance" in df.columns:
            flag |= (
                (df["prepayment_flag"] == 1) & (df["current_balance"] > 0)
            ).astype(int)

        return flag.clip(0, 1)

    def _write_minimal_html(self, df: pd.DataFrame, output_path: str) -> Path:
        """Write a minimal HTML summary when ydata-profiling is unavailable."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        col_stats = self._column_statistics(df)
        html = f"""<!DOCTYPE html>
<html>
<head><title>Data Quality Report</title></head>
<body>
<h1>Loan Performance — Data Quality Report (Minimal)</h1>
<p>Rows: {len(df)} | Cols: {len(df.columns)}</p>
{col_stats.to_html(index=False)}
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        log.info("Minimal HTML report written: %s", path)
        return path
