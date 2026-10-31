"""Pure-pandas data helpers (time-based splits). Kept free of GCP calls for unit tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    """Inclusive upper bounds of each split, on `week_start`."""

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def compute_split_dates(max_week, val_weeks: int, test_weeks: int) -> SplitDates:
    """Last `test_weeks` -> test, the `val_weeks` before that -> validation, rest -> train."""
    max_week = pd.Timestamp(max_week)
    test_start = max_week - pd.Timedelta(weeks=test_weeks - 1)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)
    return SplitDates(
        train_end=val_start - pd.Timedelta(weeks=1),
        val_end=test_start - pd.Timedelta(weeks=1),
        test_end=max_week,
    )


def time_split(
    df: pd.DataFrame, val_weeks: int, test_weeks: int, date_col: str = "week_start"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    """Chronological train/val/test split (no shuffling -> no look-ahead leakage)."""
    dates = pd.to_datetime(df[date_col])
    split = compute_split_dates(dates.max(), val_weeks, test_weeks)
    train = df[dates <= split.train_end]
    val = df[(dates > split.train_end) & (dates <= split.val_end)]
    test = df[(dates > split.val_end) & (dates <= split.test_end)]
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split: train={len(train)} val={len(val)} test={len(test)}; "
            "reduce val_weeks/test_weeks or ingest more years."
        )
    return train, val, test, split


def risk_level(predicted: pd.Series, baseline: pd.Series) -> pd.Series:
    """Bucket a forecast against the state's trailing-year weekly average."""
    ratio = predicted / baseline.clip(lower=0.5)
    return pd.cut(
        ratio,
        bins=[-float("inf"), 0.75, 1.25, 2.0, float("inf")],
        labels=["low", "normal", "elevated", "high"],
    ).astype(str)
