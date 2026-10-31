"""Tests for time splits, risk labels and the SQL templates (no BigQuery needed)."""

import pandas as pd
import pytest

from storm_prediction.data import compute_split_dates, risk_level, time_split
from storm_prediction.gcp import render_sql


def test_split_dates_are_contiguous():
    """Validation and test windows are back-to-back, each exactly 52 weeks long."""
    s = compute_split_dates("2024-12-30", val_weeks=52, test_weeks=52)
    assert s.test_end == pd.Timestamp("2024-12-30")
    assert s.val_end + pd.Timedelta(weeks=52) == s.test_end
    assert s.train_end + pd.Timedelta(weeks=52) == s.val_end


def test_time_split_is_chronological_and_disjoint(feature_frame):
    """Every row lands in exactly one split, and splits never overlap in time."""
    train, val, test, _ = time_split(feature_frame, val_weeks=26, test_weeks=26)
    assert len(train) + len(val) + len(test) == len(feature_frame)
    assert pd.to_datetime(train["week_start"]).max() < pd.to_datetime(val["week_start"]).min()
    assert pd.to_datetime(val["week_start"]).max() < pd.to_datetime(test["week_start"]).min()
    assert test["week_start"].nunique() == 26


def test_time_split_rejects_too_little_data(feature_frame):
    """Asking for more val/test weeks than the data has raises a clear error."""
    with pytest.raises(ValueError):
        time_split(feature_frame, val_weeks=150, test_weeks=100)


def test_risk_level_buckets():
    """Forecast / baseline ratios map to the documented risk labels."""
    out = risk_level(pd.Series([0.0, 10.0, 15.0, 40.0]), pd.Series([10.0, 10.0, 10.0, 10.0]))
    assert out.tolist() == ["low", "normal", "elevated", "high"]


@pytest.mark.parametrize("name", ["ingest_raw.sql", "build_features.sql"])
def test_sql_templates_render_completely(name):
    """Every $placeholder in the SQL templates gets filled in."""
    sql = render_sql(
        name,
        project="p",
        dataset="d",
        raw_table="raw",
        features_table="feat",
        start_year=2000,
        end_year=2024,
    )
    assert "$" not in sql
    assert "`p.d." in sql


def test_feature_sql_has_no_same_week_leakage():
    """No rolling window may include the current (target) week."""
    sql = render_sql("build_features.sql", project="p", dataset="d", raw_table="r", features_table="f")
    # every rolling window must stop at the previous week
    assert "AND CURRENT ROW" not in sql.upper()
