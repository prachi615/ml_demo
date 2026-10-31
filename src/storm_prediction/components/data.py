"""Data components: BigQuery ingestion, feature engineering and dataset extraction.

All heavy lifting happens in BigQuery. Only the compact weekly feature rows are
pulled into the component containers, saved as parquet KFP ``Dataset`` artifacts
and passed to the next steps.
"""

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
    """Copy NOAA Storm Events from the public dataset into our own BigQuery table.

    Runs ``sql/ingest_raw.sql`` (CREATE OR REPLACE TABLE ... AS SELECT) against
    ``bigquery-public-data.noaa_historic_severe_storms.storms_*``. The result is
    partitioned by month and clustered by state and event type.

    Args:
        project: GCP project that owns the target dataset.
        bq_location: BigQuery location; must be ``US`` because the public data is there.
        bq_dataset: Target dataset, e.g. ``storm_ml``.
        raw_table: Target table name, e.g. ``storm_events_raw``.
        start_year: First year to load (inclusive).
        end_year: Last year to load (inclusive).
        enabled: If False, skip the copy and reuse the existing table.

    Returns:
        The fully qualified raw table id, ``project.dataset.table``.
    """
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
    """Build the weekly, leak-free feature table in BigQuery.

    Runs ``sql/build_features.sql``, which aggregates raw events per state and
    week, fills empty weeks with zeros and computes lag / rolling features.

    Args:
        project: GCP project that owns the dataset.
        bq_location: BigQuery location, e.g. ``US``.
        bq_dataset: Dataset holding both tables.
        raw_table_id: Raw table id returned by ``ingest_raw_storm_events``. Passing
            it (rather than just the name) makes this step wait for ingestion.
        features_table: Name of the feature table to create.

    Returns:
        The fully qualified feature table id.
    """
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
    """Read labelled rows from the feature table and split them by time.

    Forecast rows (next unseen week) are excluded. The latest ``test_weeks`` weeks
    become the test split and the ``val_weeks`` before them the validation split.

    Args:
        project: GCP project used to run the query.
        bq_location: BigQuery location.
        features_table_id: Fully qualified feature table id.
        val_weeks: Weeks in the validation split (used to pick the best model).
        test_weeks: Weeks in the test split (used to report final performance).
        train_data: Output parquet with the training rows.
        val_data: Output parquet with the validation rows.
        test_data: Output parquet with the test rows.

    Returns:
        train_end: Last training week (YYYY-MM-DD).
        val_end: Last validation week.
        test_end: Last test week.
    """
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

    # Save each split as parquet and record its size and date range as artifact
    # metadata, so it is visible in the Vertex AI Pipelines UI.

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
    """Extract the most recent labelled weeks to check how the champion is doing.

    Args:
        project: GCP project used to run the query.
        bq_location: BigQuery location.
        features_table_id: Fully qualified feature table id.
        lookback_weeks: How many of the latest labelled weeks to include.
        eval_data: Output parquet with the evaluation rows.

    Returns:
        The number of rows extracted.

    Raises:
        ValueError: If no labelled rows fall in the window.
    """
    from storm_prediction.gcp import read_query

    df = read_query(
        f"""
        SELECT *
        FROM `{features_table_id}`
        WHERE NOT is_forecast_row
          AND event_count IS NOT NULL
          -- keep only the latest `lookback_weeks` labelled weeks
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
    """Extract the forecast rows that batch inference will score.

    The feature table holds one row per state for the week after the last
    complete week of data (``is_forecast_row = TRUE``, target unknown).

    Args:
        project: GCP project used to run the query.
        bq_location: BigQuery location.
        features_table_id: Fully qualified feature table id.
        scoring_data: Output parquet with the rows to score.

    Returns:
        The forecast week (``week_start``) as a string.

    Raises:
        ValueError: If the table has no forecast rows.
    """
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
