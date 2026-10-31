"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from storm_prediction.features import NUMERIC_FEATURES, TARGET


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """Synthetic frame with the same schema as the BigQuery feature table.

    Four states x 208 weeks with a seasonal signal plus noise, so models have
    something learnable and time splits have enough history. No GCP needed.
    """
    rng = np.random.default_rng(0)
    states = ["Texas", "Kansas", "Ohio", "Florida"]
    weeks = pd.date_range("2018-01-01", periods=208, freq="W-MON")
    rows = []
    for state_idx, state in enumerate(states):
        for week in weeks:
            seasonal = 5 + 4 * np.sin(2 * np.pi * week.isocalendar().week / 52) + state_idx * 2
            rows.append({"state": state, "week_start": week.date(), "base": max(seasonal, 0)})
    df = pd.DataFrame(rows)
    for col in NUMERIC_FEATURES:
        df[col] = df["base"] + rng.normal(0, 1, len(df))
    df["month"] = pd.to_datetime(df["week_start"]).dt.month
    df[TARGET] = rng.poisson(df["base"])
    df["is_forecast_row"] = False
    return df.drop(columns="base")
