"""Pure-pandas data helpers: time-based splits and risk buckets.

Kept free of any GCP calls so they can be unit tested locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    """Inclusive upper bound of each split, compared against ``week_start``.

    Attributes:
        train_end: Last week included in the training split.
        val_end: Last week included in the validation split.
        test_end: Last week included in the test split (the latest labelled week).
    """

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def compute_split_dates(max_week, val_weeks: int, test_weeks: int) -> SplitDates:
    """Work out the split boundaries counting back from the latest week.

    The last ``test_weeks`` weeks form the test split, the ``val_weeks`` weeks
    before them form the validation split, and everything earlier is training.

    Args:
        max_week: Latest ``week_start`` with a known target.
        val_weeks: Number of weeks in the validation split.
        test_weeks: Number of weeks in the test split.

    Returns:
        The ``SplitDates`` boundaries.
    """
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
    """Split rows chronologically into train, validation and test.

    Rows are never shuffled: the model is always evaluated on weeks that come
    after every week it was trained on, which is how it is used in production.

    Args:
        df: Labelled feature rows.
        val_weeks: Number of weeks in the validation split.
        test_weeks: Number of weeks in the test split.
        date_col: Column holding the week start date.

    Returns:
        A tuple ``(train, val, test, split_dates)``.

    Raises:
        ValueError: If any split ends up empty (too little history for the
            requested split sizes).
    """
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
    """Label each forecast relative to the state's normal activity.

    The ratio ``predicted / trailing 52-week mean`` is bucketed as:
    ``low`` (< 0.75), ``normal`` (0.75-1.25), ``elevated`` (1.25-2.0) and ``high`` (> 2.0).

    Args:
        predicted: Forecast event counts.
        baseline: The state's trailing 52-week average weekly count. Values below
            0.5 are raised to 0.5 so quiet states don't produce huge ratios.

    Returns:
        A string Series of risk labels aligned with ``predicted``.
    """
    ratio = predicted / baseline.clip(lower=0.5)
    return pd.cut(
        ratio,
        bins=[-float("inf"), 0.75, 1.25, 2.0, float("inf")],
        labels=["low", "normal", "elevated", "high"],
    ).astype(str)
