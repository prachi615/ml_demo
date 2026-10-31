"""Feature contract between BigQuery (build_features.sql) and the models."""

from __future__ import annotations

import pandas as pd

TARGET = "event_count"
ID_COLUMNS = ["state", "week_start"]

CATEGORICAL_FEATURES = ["state"]
NUMERIC_FEATURES = [
    "month",
    "week_of_year_sin",
    "week_of_year_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_52",
    "same_week_mean_3y",
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_mean_52",
    "state_expanding_mean",
    "tornado_sum_4",
    "hail_sum_4",
    "wind_sum_4",
    "flood_sum_4",
    "injuries_sum_12",
    "deaths_sum_12",
    "log_damage_sum_12",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model matrix and target with stable dtypes."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    X = prepare_features(df)
    y = df[TARGET].astype(float)
    return X, y


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns and coerce BigQuery dtypes (Int64/dbdate) to plain ones."""
    X = df[FEATURES].copy()
    X["state"] = X["state"].astype(str)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    return X
