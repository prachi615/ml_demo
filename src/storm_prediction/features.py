"""Feature contract between BigQuery and the models.

``sql/build_features.sql`` creates the columns listed here, and every model reads
exactly these columns. Keeping the list in one place guarantees that training and
batch inference use the same features in the same order.
"""

from __future__ import annotations

import pandas as pd

# Weekly number of storm events in a state: the value we forecast.
TARGET = "event_count"

# Columns that identify a row but are not used as model inputs.
ID_COLUMNS = ["state", "week_start"]

# One-hot encoded by the model pipeline.
CATEGORICAL_FEATURES = ["state"]

# Every numeric feature is computed only from weeks BEFORE the target week.
NUMERIC_FEATURES = [
    # seasonality
    "month",
    "week_of_year_sin",
    "week_of_year_cos",
    # lags of the target
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_52",
    "same_week_mean_3y",
    # rolling statistics of the target
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_mean_52",
    "state_expanding_mean",
    # event-type mix over the last 4 weeks
    "tornado_sum_4",
    "hail_sum_4",
    "wind_sum_4",
    "flood_sum_4",
    # impact over the last 12 weeks
    "injuries_sum_12",
    "deaths_sum_12",
    "log_damage_sum_12",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled feature frame into model inputs and target.

    Args:
        df: Rows from the BigQuery feature table, including the target column.

    Returns:
        A tuple ``(X, y)`` where ``X`` holds the columns in ``FEATURES`` and ``y``
        is the target as floats.

    Raises:
        ValueError: If any feature or the target column is missing.
    """
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    X = prepare_features(df)
    y = df[TARGET].astype(float)
    return X, y


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and convert them to plain dtypes.

    BigQuery returns nullable pandas types (``Int64``, ``dbdate``) that some
    scikit-learn and XGBoost versions reject, so numeric columns are cast to
    ``float`` and ``state`` to ``str``. Missing values become ``NaN`` and are
    imputed inside the model pipeline.

    Args:
        df: Rows from the feature table; extra columns are ignored.

    Returns:
        A new DataFrame with exactly the columns in ``FEATURES``.
    """
    X = df[FEATURES].copy()
    X["state"] = X["state"].astype(str)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    return X
