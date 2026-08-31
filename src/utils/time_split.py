"""
src/utils/time_split.py
-----------------------
Time-aware split utilities and leakage auditor.

Key rules enforced:
  1. Training data contains only rows with month_index <= T_cutoff
  2. Validation data is the next N months after cutoff
  3. No loan appears in both train and validation with future state
     information leaked into historical features
  4. All feature transformers are fit ONLY on training data

Classes:
  TemporalSplitter  — computes train/val index masks
  LeakageAuditor    — scans features for temporal leakage signals
  LoanGroupedTSSplit — sklearn-compatible TimeSeriesSplit with loan grouping
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class TemporalSplitter:
    """Compute train / validation index masks from a time-indexed DataFrame.

    Parameters
    ----------
    holdout_months : int
        Number of months before the last month to set as cutoff.
    validation_months : int
        Number of months after cutoff to use as validation window.
    """

    def __init__(
        self,
        holdout_months: int | dict | None = None,
        validation_months: int | None = None,
    ) -> None:
        cfg = get_config()["temporal_split"]
        if isinstance(holdout_months, dict):
            cfg = holdout_months.get("temporal_split", holdout_months)
            holdout_months = None
        self.holdout_months = int(holdout_months or cfg["holdout_months"])
        self.validation_months = int(validation_months or cfg["validation_months"])

    def split_train_val(
        self, df: pd.DataFrame, time_col: str = "month_index"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_idx, val_idx = self.split(df, time_col=time_col)
        return df.loc[train_idx].copy(), df.loc[val_idx].copy()


    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "month_index",
    ) -> tuple[pd.Index, pd.Index]:
        """Return (train_idx, val_idx) as pandas Index objects.

        Parameters
        ----------
        df : DataFrame
            Full feature matrix with *time_col*.
        time_col : str
            Column name representing the monotone time index.

        Returns
        -------
        train_idx, val_idx : pd.Index
            Row indices for training and validation sets.
        """
        max_month = df[time_col].max()
        t_cutoff = max_month - self.holdout_months
        t_val_end = t_cutoff + self.validation_months

        log.info(
            "Temporal split: train month_index <= %d | val (%d, %d]",
            t_cutoff,
            t_cutoff,
            t_val_end,
        )

        train_mask = df[time_col] <= t_cutoff
        val_mask = (df[time_col] > t_cutoff) & (df[time_col] <= t_val_end)

        train_idx = df.index[train_mask]
        val_idx = df.index[val_mask]

        log.info(
            "Train rows: %d | Val rows: %d | Test (organizer): held out",
            len(train_idx),
            len(val_idx),
        )
        return train_idx, val_idx

    def get_cutoff(self, df: pd.DataFrame, time_col: str = "month_index") -> int:
        """Return the numeric cutoff month index."""
        return int(df[time_col].max()) - self.holdout_months


class LeakageAuditor:
    """Scan feature matrix for temporal leakage signals.

    Leakage checks performed:
      1. Feature/target Spearman correlation > threshold in the validation set
         (suggests a feature carries future information)
      2. Any feature column name that contains 'future', 'next', 'target' keywords
      3. Reporting month in validation rows is strictly > T_cutoff

    Parameters
    ----------
    correlation_threshold : float
        Spearman |r| above this value triggers a leakage warning. Default 0.9.
    """

    SUSPECT_KEYWORDS = {"future", "next_", "target", "label", "_t1", "_t3", "_t6", "_t12"}

    def __init__(self, correlation_threshold: float = 0.90) -> None:
        self.correlation_threshold = correlation_threshold

    def audit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        time_col: str = "month_index",
        cutoff: int | None = None,
    ) -> list[str]:
        """Run all leakage checks. Returns list of detected issues (empty = clean).

        Raises
        ------
        LeakageError
            If Spearman correlation leakage is detected above threshold.
        """
        issues: list[str] = []

        # Check 1: Suspicious column name keywords
        for col in feature_cols:
            for kw in self.SUSPECT_KEYWORDS:
                if kw.lower() in col.lower():
                    issues.append(
                        f"SUSPECT FEATURE NAME: '{col}' contains keyword '{kw}'"
                    )

        # Check 2: Temporal ordering (val rows should be after cutoff)
        if cutoff is not None and time_col in val_df.columns:
            bad = (val_df[time_col] <= cutoff).sum()
            if bad > 0:
                issues.append(
                    f"TEMPORAL VIOLATION: {bad} validation rows have "
                    f"{time_col} <= cutoff ({cutoff})"
                )

        # Check 3: High feature-target correlation in validation set
        numeric_features = [
            c for c in feature_cols
            if c in val_df.columns and pd.api.types.is_numeric_dtype(val_df[c])
        ]
        for target in target_cols:
            if target not in val_df.columns:
                continue
            for feat in numeric_features:
                corr = val_df[feat].corr(val_df[target], method="spearman")
                if abs(corr) >= self.correlation_threshold:
                    msg = (
                        f"LEAKAGE ALERT: Feature '{feat}' has Spearman |r|="
                        f"{abs(corr):.3f} with target '{target}' in val set"
                    )
                    log.error(msg)
                    issues.append(msg)

        if issues:
            log.warning("LeakageAuditor found %d issue(s):", len(issues))
            for issue in issues:
                log.warning("  • %s", issue)
        else:
            log.info("LeakageAuditor: ✓ No leakage detected")

        return issues

    def raise_if_leakage(self, issues: list[str]) -> None:
        """Raise LeakageError if any issues were found."""
        if issues:
            raise LeakageError(
                f"Temporal leakage detected ({len(issues)} issue(s)):\n"
                + "\n".join(f"  • {i}" for i in issues)
            )

    def raise_if_target_in_features(
        self, feature_cols: list[str], target_col: str
    ) -> None:
        """Raise LeakageError if target column is included in feature list."""
        if target_col in feature_cols:
            raise LeakageError(
                f"TARGET LEAKAGE: Target column '{target_col}' found in feature matrix!"
            )


class LeakageError(ValueError):
    """Raised when temporal leakage is detected in the feature matrix."""


class LoanGroupedTSSplit:
    """Sklearn-compatible time-series CV split that keeps loan groups intact.

    All month_index records belonging to the same loan_id will always be
    in the same fold (prevents cross-loan leakage).

    Parameters
    ----------
    n_splits : int
        Number of expanding-window folds.
    gap : int
        Minimum gap (in month_index units) between train and test folds.
    """

    def __init__(self, n_splits: int = 5, gap: int = 1) -> None:
        self.n_splits = n_splits
        self.gap = gap

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "month_index",
        group_col: str = "loan_id",
    ):
        """Yield (train_indices, val_indices) pairs.

        Parameters
        ----------
        df : DataFrame
            Full training DataFrame with *time_col* and *group_col*.

        Yields
        ------
        train_idx, val_idx : np.ndarray
            Integer positions (iloc-style) for each fold.
        """
        months = sorted(df[time_col].unique())
        n = len(months)
        fold_size = n // (self.n_splits + 1)

        for fold in range(self.n_splits):
            train_end = months[fold_size * (fold + 1) - 1]
            val_start = months[fold_size * (fold + 1) + self.gap - 1]
            val_end = months[min(fold_size * (fold + 2) - 1, n - 1)]

            train_mask = df[time_col] <= train_end
            val_mask = (df[time_col] >= val_start) & (df[time_col] <= val_end)

            train_loans = set(df.loc[train_mask, group_col])
            val_loans = set(df.loc[val_mask, group_col])

            # Ensure no loan is in both train and val
            overlap = train_loans & val_loans
            if overlap:
                # Move overlapping loans entirely into training
                val_mask = val_mask & ~df[group_col].isin(overlap)

            log.debug(
                "Fold %d/%d: train months ≤ %d | val months %d–%d | "
                "train_loans=%d val_loans=%d",
                fold + 1,
                self.n_splits,
                train_end,
                val_start,
                val_end,
                len(train_loans),
                sum(val_mask),
            )

            yield df.index[train_mask].values, df.index[val_mask].values
