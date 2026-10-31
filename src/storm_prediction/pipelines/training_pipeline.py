"""Training pipeline.

BigQuery (public NOAA) -> raw table -> feature table -> time-based splits
  -> train 4 models in parallel (2 linear baselines + random forest + XGBoost)
  -> evaluate each on the held-out test split
  -> leaderboard (Vertex UI + BigQuery)
  -> if the winner beats the baselines: register in Vertex AI Model Registry
"""

import json

from kfp import dsl

from storm_prediction.components.data import (
    build_feature_table,
    extract_training_splits,
    ingest_raw_storm_events,
)
from storm_prediction.components.training import (
    evaluate_model,
    register_model,
    select_best_model,
    train_model,
    write_leaderboard,
)
from storm_prediction.config import PIPELINE_IMAGE
from storm_prediction.models import DEFAULT_HYPERPARAMETERS, MODEL_TYPES

PIPELINE_NAME = "storm-forecast-training"


@dsl.pipeline(name=PIPELINE_NAME, description="Train, compare and register storm event forecasters.")
def training_pipeline(
    project: str,
    location: str = "us-central1",
    bq_location: str = "US",
    bq_dataset: str = "storm_ml",
    raw_table: str = "storm_events_raw",
    features_table: str = "storm_features_weekly",
    leaderboard_table: str = "model_leaderboard",
    start_year: int = 2000,
    end_year: int = 2024,
    refresh_raw_data: bool = True,
    val_weeks: int = 52,
    test_weeks: int = 52,
    hyperparameters_json: str = json.dumps(DEFAULT_HYPERPARAMETERS),
    min_improvement_pct: float = 2.0,
    model_display_name: str = "storm-weekly-event-forecaster",
    serving_container_image_uri: str = PIPELINE_IMAGE,
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

    splits = extract_training_splits(
        project=project,
        bq_location=bq_location,
        features_table_id=features.output,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
    ).set_display_name("extract-train-val-test")
    splits.set_caching_options(False)

    trained, evaluated = {}, {}
    for model_type in MODEL_TYPES:
        train_task = (
            train_model(
                model_type=model_type,
                hyperparameters_json=hyperparameters_json,
                train_data=splits.outputs["train_data"],
                val_data=splits.outputs["val_data"],
            )
            .set_display_name(f"train-{model_type}")
            .set_cpu_limit("8")
            .set_memory_limit("32G")
        )
        eval_task = evaluate_model(
            model=train_task.outputs["model"],
            eval_data=splits.outputs["test_data"],
            model_type=model_type,
            split_name="test",
        ).set_display_name(f"evaluate-{model_type}")
        trained[model_type], evaluated[model_type] = train_task, eval_task

    select = select_best_model(
        linear_regression_model=trained["linear_regression"].outputs["model"],
        ridge_model=trained["ridge"].outputs["model"],
        random_forest_model=trained["random_forest"].outputs["model"],
        xgboost_model=trained["xgboost"].outputs["model"],
        linear_regression_val=trained["linear_regression"].outputs["val_metrics_json"],
        ridge_val=trained["ridge"].outputs["val_metrics_json"],
        random_forest_val=trained["random_forest"].outputs["val_metrics_json"],
        xgboost_val=trained["xgboost"].outputs["val_metrics_json"],
        linear_regression_test=evaluated["linear_regression"].outputs["metrics_json"],
        ridge_test=evaluated["ridge"].outputs["metrics_json"],
        random_forest_test=evaluated["random_forest"].outputs["metrics_json"],
        xgboost_test=evaluated["xgboost"].outputs["metrics_json"],
        min_improvement_pct=min_improvement_pct,
    ).set_display_name("select-best-model")

    write_leaderboard(
        project=project,
        bq_location=bq_location,
        table_id=f"{project}.{bq_dataset}.{leaderboard_table}",
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        train_end=splits.outputs["train_end"],
        val_end=splits.outputs["val_end"],
        test_end=splits.outputs["test_end"],
        leaderboard=select.outputs["leaderboard"],
    ).set_display_name("write-leaderboard-to-bigquery")

    with dsl.If(select.outputs["beats_baselines"] == True, name="beats-baselines"):  # noqa: E712
        register_model(
            project=project,
            location=location,
            model_display_name=model_display_name,
            model_type=select.outputs["best_model_type"],
            test_rmse=select.outputs["best_test_rmse"],
            serving_container_image_uri=serving_container_image_uri,
            pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            model=select.outputs["best_model"],
        ).set_display_name("register-model")
