"""Training pipeline (runs monthly on a schedule, and on every deploy).

Training stage:
    BigQuery (public NOAA) -> raw table -> feature table -> time-based splits
    -> train the 4 models in parallel (linear regression, ridge, random forest, XGBoost)
    -> score each on the test split
    -> pick the best of the 4 for this dataset (lowest validation RMSE)
    -> write the leaderboard to BigQuery

Evaluation stage (only after training has finished):
    -> score the current champion on the SAME test split
    -> promote the new best model only if it beats the champion
    -> log the decision to BigQuery
    -> if promoted: register it in the Vertex AI Model Registry as the new champion
"""

import json

from kfp import dsl

from storm_prediction.components.data import (
    build_feature_table,
    extract_training_splits,
    ingest_raw_storm_events,
)
from storm_prediction.components.evaluation import decide_promotion, evaluate_champion
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


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Monthly retraining: find the best of 4 models, promote it if it beats the champion.",
)
def training_pipeline(
    project: str,
    location: str = "us-central1",
    bq_location: str = "US",
    bq_dataset: str = "storm_ml",
    raw_table: str = "storm_events_raw",
    features_table: str = "storm_features_weekly",
    leaderboard_table: str = "model_leaderboard",
    evaluation_table: str = "model_evaluations",
    start_year: int = 2000,
    end_year: int = 0,
    refresh_raw_data: bool = True,
    val_weeks: int = 52,
    test_weeks: int = 52,
    hyperparameters_json: str = json.dumps(DEFAULT_HYPERPARAMETERS),
    min_improvement_pct: float = 0.0,
    model_display_name: str = "storm-weekly-event-forecaster",
    serving_container_image_uri: str = PIPELINE_IMAGE,
):
    """Retrain the four models, evaluate the best one against the champion, promote if better.

    Args:
        project: GCP project for BigQuery, Vertex AI and the registry.
        location: Vertex AI region.
        bq_location: BigQuery location (``US``, same as the public NOAA dataset).
        bq_dataset: BigQuery dataset holding all project tables.
        raw_table: Raw storm events table name.
        features_table: Weekly feature table name.
        leaderboard_table: Table receiving one row per model per run.
        evaluation_table: Table receiving one promotion decision per run.
        start_year: First year of NOAA data to load.
        end_year: Last year of NOAA data to load; 0 means the current year.
        refresh_raw_data: Re-copy the raw data from the public dataset.
        val_weeks: Weeks in the validation split (used to pick the best model).
        test_weeks: Weeks in the test split (used to compare with the champion).
        hyperparameters_json: Per-model hyperparameters as JSON (see models.py).
        min_improvement_pct: Required test-RMSE improvement over the champion, in percent.
        model_display_name: Model Registry entry name.
        serving_container_image_uri: Image recorded with the registered model.
    """
    # ------------------------------------------------------------------ data
    # 1. Copy NOAA data into our BigQuery dataset. Caching is off for data steps
    #    because the source tables change between runs.
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

    # 2. Rebuild the weekly feature table in BigQuery.
    features = build_feature_table(
        project=project,
        bq_location=bq_location,
        bq_dataset=bq_dataset,
        raw_table_id=ingest.output,
        features_table=features_table,
    ).set_display_name("build-feature-table")
    features.set_caching_options(False)

    # 3. Pull labelled rows and split them by time into train / val / test.
    splits = extract_training_splits(
        project=project,
        bq_location=bq_location,
        features_table_id=features.output,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
    ).set_display_name("extract-train-val-test")
    splits.set_caching_options(False)

    # -------------------------------------------------------------- training
    # 4. Train and test-score the four models. The loop runs at compile time and
    #    creates 4 independent branches, which Vertex AI runs in parallel.
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
        ).set_display_name(f"test-{model_type}")
        trained[model_type], evaluated[model_type] = train_task, eval_task

    # 5. Pick the best of the four for this dataset (lowest validation RMSE).
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
    ).set_display_name("select-best-of-4")

    # 6. Keep a history of every run's leaderboard in BigQuery.
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

    # ------------------------------------------------------------ evaluation
    # 7. Only after training: score the current champion on the same test weeks.
    champion = (
        evaluate_champion(
            project=project,
            location=location,
            model_display_name=model_display_name,
            test_data=splits.outputs["test_data"],
        )
        .set_display_name("evaluate-current-champion")
        .after(select)
    )
    champion.set_caching_options(False)  # the champion can change between runs

    # 8. Compare new best vs champion and log the decision to BigQuery.
    decision = decide_promotion(
        project=project,
        bq_location=bq_location,
        evaluation_table_id=f"{project}.{bq_dataset}.{evaluation_table}",
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        new_model_type=select.outputs["best_model_type"],
        new_test_metrics_json=select.outputs["best_test_metrics_json"],
        champion_status=champion.outputs["champion_status"],
        champion_version=champion.outputs["champion_version"],
        champion_model_type=champion.outputs["champion_model_type"],
        champion_test_rmse=champion.outputs["champion_test_rmse"],
        min_improvement_pct=min_improvement_pct,
        test_end=splits.outputs["test_end"],
    ).set_display_name("decide-promotion")
    decision.set_caching_options(False)

    # 9. Register the new model as champion only if it won.
    with dsl.If(decision.outputs["promote"] == True, name="new-model-is-better"):  # noqa: E712
        register_model(
            project=project,
            location=location,
            model_display_name=model_display_name,
            model_type=select.outputs["best_model_type"],
            test_rmse=select.outputs["best_test_rmse"],
            serving_container_image_uri=serving_container_image_uri,
            pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            model=select.outputs["best_model"],
        ).set_display_name("register-as-champion")
