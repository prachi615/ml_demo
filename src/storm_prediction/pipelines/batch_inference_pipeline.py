"""Batch inference pipeline.

Refreshes the BigQuery feature table, loads the champion model from the
Vertex AI Model Registry, forecasts next week's storm event count for every
state and appends the forecasts to a BigQuery predictions table.
"""

from kfp import dsl

from storm_prediction.components.data import (
    build_feature_table,
    extract_scoring_rows,
    ingest_raw_storm_events,
)
from storm_prediction.components.inference import batch_predict
from storm_prediction.components.monitoring import load_registered_model

PIPELINE_NAME = "storm-forecast-batch-inference"


@dsl.pipeline(name=PIPELINE_NAME, description="Weekly batch forecast of storm events per US state.")
def batch_inference_pipeline(
    project: str,
    location: str = "us-central1",
    bq_location: str = "US",
    bq_dataset: str = "storm_ml",
    raw_table: str = "storm_events_raw",
    features_table: str = "storm_features_weekly",
    predictions_table: str = "storm_predictions",
    start_year: int = 2000,
    end_year: int = 2024,
    refresh_raw_data: bool = True,
    model_display_name: str = "storm-weekly-event-forecaster",
    model_alias: str = "champion",
):
    ingest = ingest_raw_storm_events(
        project=project,
        bq_location=bq_location,
        bq_dataset=bq_dataset,
        raw_table=raw_table,
        start_year=start_year,
        end_year=end_year,
        enabled=refresh_raw_data,
    ).set_display_name("ingest-noaa-to-bigquery")
    ingest.set_caching_options(False)

    features = build_feature_table(
        project=project,
        bq_location=bq_location,
        bq_dataset=bq_dataset,
        raw_table_id=ingest.output,
        features_table=features_table,
    ).set_display_name("build-feature-table")
    features.set_caching_options(False)

    scoring = extract_scoring_rows(
        project=project,
        bq_location=bq_location,
        features_table_id=features.output,
    ).set_display_name("extract-scoring-rows")
    scoring.set_caching_options(False)

    champion = load_registered_model(
        project=project,
        location=location,
        model_display_name=model_display_name,
        alias=model_alias,
    ).set_display_name("load-champion-model")
    champion.set_caching_options(False)

    batch_predict(
        project=project,
        bq_location=bq_location,
        predictions_table_id=f"{project}.{bq_dataset}.{predictions_table}",
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        model_version=champion.outputs["version_name"],
        model_type=champion.outputs["model_type"],
        model=champion.outputs["model"],
        scoring_data=scoring.outputs["scoring_data"],
    ).set_display_name("batch-predict-to-bigquery").set_caching_options(False)
