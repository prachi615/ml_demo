"""Data components: BigQuery ingestion, feature engineering and dataset extraction."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Output

from storm_prediction.config import PIPELINE_IMAGE


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def ingest_raw_storm_events(
    project: str,
    bq_location: str,
    bq_dataset: str,
    raw_table: str,
    start_year: int,
    end_year: int,
    enabled: bool = True,
) -> str:
    """Copy NOAA Storm Events from the public dataset into our own BigQuery table."""
    import logging

    from storm_prediction.gcp import render_sql, run_query

    logging.basicConfig(level=logging.INFO)
    table_id = f"{project}.{bq_dataset}.{raw_table}"
    if not enabled:
        logging.info("Raw ingestion disabled; reusing existing table %s", table_id)
        return table_id

    sql = render_sql(
        "ingest_raw.sql",
        project=project,
        dataset=bq_dataset,
        raw_table=raw_table,
        start_year=start_year,
        end_year=end_year,
    )
    run_query(sql, project=project, location=bq_location)
    return table_id


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def build_feature_table(
    project: str,
    bq_location: str,
    bq_dataset: str,
    raw_table_id: str,
    features_table: str,
) -> str:
    """Materialise the weekly, leak-free feature table in BigQuery."""
    import logging

    from storm_prediction.gcp import render_sql, run_query

    logging.basicConfig(level=logging.INFO)
    raw_table = raw_table_id.split(".")[-1]
    sql = render_sql(
        "build_features.sql",
        project=project,
        dataset=bq_dataset,
        raw_table=raw_table,
        features_table=features_table,
    )
    run_query(sql, project=project, location=bq_location)
    return f"{project}.{bq_dataset}.{features_table}"


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def extract_training_splits(
    project: str,
    bq_location: str,
    features_table_id: str,
    val_weeks: int,
    test_weeks: int,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
) -> NamedTuple("Outputs", [("train_end", str), ("val_end", str), ("test_end", str)]):
    """Read labelled rows from BigQuery and split them chronologically."""
    import logging
    from typing import NamedTuple

    from storm_prediction.data import time_split
    from storm_prediction.gcp import read_query

    logging.basicConfig(level=logging.INFO)
    df = read_query(
        f"SELECT * FROM `{features_table_id}` WHERE NOT is_forecast_row AND event_count IS NOT NULL",
        project=project,
        location=bq_location,
    )
    train, val, test, split = time_split(df, val_weeks=val_weeks, test_weeks=test_weeks)

    for frame, artifact in ((train, train_data), (val, val_data), (test, test_data)):
        frame.to_parquet(artifact.path, index=False)
        artifact.metadata.update(
            {
                "rows": len(frame),
                "min_week": str(frame["week_start"].min()),
                "max_week": str(frame["week_start"].max()),
            }
        )
    logging.info("rows: train=%d val=%d test=%d", len(train), len(val), len(test))

    outputs = NamedTuple("Outputs", [("train_end", str), ("val_end", str), ("test_end", str)])
    return outputs(
        str(split.train_end.date()), str(split.val_end.date()), str(split.test_end.date())
    )


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def extract_evaluation_window(
    project: str,
    bq_location: str,
    features_table_id: str,
    lookback_weeks: int,
    eval_data: Output[Dataset],
) -> int:
    """Most recent `lookback_weeks` of labelled rows, used to monitor the champion."""
    from storm_prediction.gcp import read_query

    df = read_query(
        f"""
        SELECT *
        FROM `{features_table_id}`
        WHERE NOT is_forecast_row
          AND event_count IS NOT NULL
          AND week_start > DATE_SUB(
                (SELECT MAX(week_start) FROM `{features_table_id}` WHERE NOT is_forecast_row),
                INTERVAL {int(lookback_weeks)} WEEK)
        """,
        project=project,
        location=bq_location,
    )
    if df.empty:
        raise ValueError("Evaluation window is empty")
    df.to_parquet(eval_data.path, index=False)
    eval_data.metadata["rows"] = len(df)
    return len(df)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def extract_scoring_rows(
    project: str,
    bq_location: str,
    features_table_id: str,
    scoring_data: Output[Dataset],
) -> str:
    """Forecast rows (next unseen week, one row per state) for batch inference."""
    from storm_prediction.gcp import read_query

    df = read_query(
        f"SELECT * FROM `{features_table_id}` WHERE is_forecast_row",
        project=project,
        location=bq_location,
    )
    if df.empty:
        raise ValueError("No forecast rows found; did build_feature_table run?")
    df.to_parquet(scoring_data.path, index=False)
    scoring_week = str(df["week_start"].max())
    scoring_data.metadata.update({"rows": len(df), "scoring_week": scoring_week})
    return scoring_week
