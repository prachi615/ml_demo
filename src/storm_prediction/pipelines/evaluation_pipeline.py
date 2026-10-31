"""Evaluation (champion monitoring) pipeline.

Refreshes data, scores the registered champion on the most recent labelled
weeks, compares it with its training-time RMSE and with naive baselines,
logs a monitoring row to BigQuery and (optionally) triggers retraining.
"""

from kfp import dsl

from storm_prediction.components.data import (
    build_feature_table,
    extract_evaluation_window,
    ingest_raw_storm_events,
)
from storm_prediction.components.monitoring import (
    check_model_health,
    evaluate_naive_baselines,
    load_registered_model,
    trigger_training_pipeline,
)
from storm_prediction.components.training import evaluate_model

PIPELINE_NAME = "storm-forecast-evaluation"


@dsl.pipeline(name=PIPELINE_NAME, description="Monitor the champion storm forecaster on fresh data.")
def evaluation_pipeline(
    project: str,
    location: str = "us-central1",
    bq_location: str = "US",
    bq_dataset: str = "storm_ml",
    raw_table: str = "storm_events_raw",
    features_table: str = "storm_features_weekly",
    monitoring_table: str = "model_monitoring",
    start_year: int = 2000,
    end_year: int = 2024,
    refresh_raw_data: bool = True,
    lookback_weeks: int = 26,
    model_display_name: str = "storm-weekly-event-forecaster",
    model_alias: str = "champion",
    max_rmse_degradation_pct: float = 15.0,
    auto_retrain: bool = False,
    training_template_path: str = "",
    pipeline_root: str = "",
    service_account: str = "",
    training_parameter_values_json: str = "{}",
):
    """Check the champion on recent data and trigger retraining if needed.

    Args:
        project: GCP project for BigQuery, Vertex AI and the registry.
        location: Vertex AI region.
        bq_location: BigQuery location (``US``).
        bq_dataset: BigQuery dataset holding all project tables.
        raw_table: Raw storm events table name.
        features_table: Weekly feature table name.
        monitoring_table: Table receiving one row per evaluation run.
        start_year: First year of NOAA data to load.
        end_year: Last year of NOAA data to load.
        refresh_raw_data: Re-copy the raw data from the public dataset.
        lookback_weeks: Number of recent labelled weeks to evaluate on.
        model_display_name: Model Registry entry name.
        model_alias: Version alias to evaluate (normally ``champion``).
        max_rmse_degradation_pct: Allowed RMSE increase vs training before flagging.
        auto_retrain: Submit the training pipeline when the model is flagged.
        training_template_path: Compiled training pipeline to submit.
        pipeline_root: GCS folder for the retraining run's artifacts.
        service_account: Service account for the retraining run.
        training_parameter_values_json: Parameters for the retraining run, as JSON.
    """
    # 1-2. Refresh raw data and features so the newest labelled weeks are available.
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

    # 3. Take the most recent labelled weeks.
    window = extract_evaluation_window(
        project=project,
        bq_location=bq_location,
        features_table_id=features.output,
        lookback_weeks=lookback_weeks,
    ).set_display_name("extract-evaluation-window")
    window.set_caching_options(False)

    # 4. Load the live model from the registry.
    champion = load_registered_model(
        project=project,
        location=location,
        model_display_name=model_display_name,
        alias=model_alias,
    ).set_display_name("load-champion-model")
    champion.set_caching_options(False)

    # 5. Score the champion and the naive baselines on the same weeks.
    champion_eval = evaluate_model(
        model=champion.outputs["model"],
        eval_data=window.outputs["eval_data"],
        model_type=champion.outputs["model_type"],
        split_name="recent",
    ).set_display_name("evaluate-champion")

    naive = evaluate_naive_baselines(eval_data=window.outputs["eval_data"]).set_display_name(
        "evaluate-naive-baselines"
    )

    # 6. Compare, log a monitoring row to BigQuery, decide on retraining.
    health = check_model_health(
        project=project,
        bq_location=bq_location,
        monitoring_table_id=f"{project}.{bq_dataset}.{monitoring_table}",
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        model_version=champion.outputs["version_name"],
        model_type=champion.outputs["model_type"],
        training_test_rmse=champion.outputs["training_test_rmse"],
        champion_metrics_json=champion_eval.outputs["metrics_json"],
        naive_metrics_json=naive.outputs["Output"],
        max_rmse_degradation_pct=max_rmse_degradation_pct,
    ).set_display_name("check-model-health")
    health.set_caching_options(False)

    # 7. Retrain only if the model is flagged AND auto-retraining is switched on.
    with dsl.If(health.outputs["needs_retraining"] == True, name="needs-retraining"):  # noqa: E712
        with dsl.If(auto_retrain == True, name="auto-retrain-enabled"):  # noqa: E712
            trigger_training_pipeline(
                project=project,
                location=location,
                template_path=training_template_path,
                pipeline_root=pipeline_root,
                service_account=service_account,
                parameter_values_json=training_parameter_values_json,
                reason=health.outputs["reason"],
            ).set_display_name("trigger-retraining")
