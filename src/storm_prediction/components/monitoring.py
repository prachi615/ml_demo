"""Components used by the evaluation (champion monitoring) pipeline.

``load_registered_model`` is also reused by the batch inference pipeline.
As in all components, imports are inside the function bodies because KFP ships
only each function's source to the container.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from storm_prediction.config import PIPELINE_IMAGE


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def load_registered_model(
    project: str,
    location: str,
    model_display_name: str,
    alias: str,
    model: Output[Model],
) -> NamedTuple(
    "Outputs",
    [("model_type", str), ("version_name", str), ("training_test_rmse", float)],
):
    """Fetch a model version from the Vertex AI Model Registry by alias.

    Resolves ``<model_display_name>@<alias>`` (normally ``champion``) and
    downloads its artifact directory from Cloud Storage.

    Args:
        project: GCP project of the registry.
        location: Vertex AI region.
        model_display_name: Registry entry name.
        alias: Version alias to load, e.g. ``champion`` or ``challenger``.
        model: Output directory; receives ``model.joblib``.

    Returns:
        model_type: Model type read from the version's labels.
        version_name: ``<resource name>@<version id>``, recorded with predictions.
        training_test_rmse: Test RMSE measured when the version was trained.

    Raises:
        RuntimeError: If no model with that name is registered yet.
    """
    from typing import NamedTuple

    from google.cloud import aiplatform

    from storm_prediction.gcp import download_gcs_dir

    aiplatform.init(project=project, location=location)
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    if not models:
        raise RuntimeError(f"No model named {model_display_name!r} in the registry; run training first.")
    # "<resource name>@<alias>" resolves to whichever version currently holds the alias.
    registered = aiplatform.Model(model_name=f"{models[0].resource_name}@{alias}")

    download_gcs_dir(registered.uri, model.path)
    # register_model stored the model type and test RMSE as labels.
    # register_model writes the type with dashes ("random-forest"); convert it back.
    labels = registered.labels or {}
    model_type = labels.get("model_type", "unknown").replace("-", "_")
    rmse = int(labels.get("test_rmse_milli", "0")) / 1000.0
    version_name = f"{registered.resource_name}@{registered.version_id}"
    model.metadata.update({"model_type": model_type, "version": version_name, "alias": alias})

    outputs = NamedTuple(
        "Outputs", [("model_type", str), ("version_name", str), ("training_test_rmse", float)]
    )
    return outputs(model_type, version_name, rmse)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_naive_baselines(
    eval_data: Input[Dataset],
    metrics: Output[Metrics],
) -> str:
    """Score two model-free forecasts on the evaluation window.

    * ``persistence``: next week = this week (``lag_1``)
    * ``seasonal_naive``: next week = average of the same week in the last 3 years

    A useful model must beat these; they are the floor in the health check.

    Args:
        eval_data: Parquet with labelled feature rows.
        metrics: Output metrics shown in the Vertex AI UI.

    Returns:
        JSON string ``{"persistence": {...}, "seasonal_naive": {...}}`` with the
        full metric set for each baseline.
    """
    import json

    import pandas as pd

    from storm_prediction.features import TARGET
    from storm_prediction.metrics import regression_metrics

    df = pd.read_parquet(eval_data.path)
    results = {
        "persistence": regression_metrics(df[TARGET], df["lag_1"].astype(float)),
        "seasonal_naive": regression_metrics(df[TARGET], df["same_week_mean_3y"].astype(float)),
    }
    for name, values in results.items():
        metrics.log_metric(f"{name}_rmse", values["rmse"])
        metrics.log_metric(f"{name}_mae", values["mae"])
    return json.dumps(results)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def check_model_health(
    project: str,
    bq_location: str,
    monitoring_table_id: str,
    pipeline_job_name: str,
    model_version: str,
    model_type: str,
    training_test_rmse: float,
    champion_metrics_json: str,
    naive_metrics_json: str,
    max_rmse_degradation_pct: float,
) -> NamedTuple("Outputs", [("needs_retraining", bool), ("reason", str)]):
    """Decide whether the champion needs retraining, and log the check to BigQuery.

    The champion is flagged when either:

    * its RMSE on recent weeks is more than ``max_rmse_degradation_pct`` percent
      worse than its test RMSE at training time (performance drift), or
    * it is no better than the best naive baseline.

    Args:
        project: GCP project used to write the monitoring row.
        bq_location: BigQuery location.
        monitoring_table_id: Fully qualified monitoring table id.
        pipeline_job_name: Name of the current pipeline run.
        model_version: Version that was evaluated.
        model_type: Its model type.
        training_test_rmse: Its test RMSE at training time.
        champion_metrics_json: Metrics JSON from ``evaluate_model`` on recent weeks.
        naive_metrics_json: Metrics JSON from ``evaluate_naive_baselines``.
        max_rmse_degradation_pct: Allowed RMSE increase before flagging.

    Returns:
        needs_retraining: True if any check failed.
        reason: Human-readable explanation (``healthy`` if nothing failed).
    """
    import json
    import logging
    from datetime import datetime, timezone
    from typing import NamedTuple

    import pandas as pd

    from storm_prediction.gcp import write_dataframe

    logging.basicConfig(level=logging.INFO)
    champion = json.loads(champion_metrics_json)
    naive = json.loads(naive_metrics_json)
    best_naive_rmse = min(v["rmse"] for v in naive.values())

    # Collect every failed check; any failure means retraining is needed.
    reasons = []
    degradation = 0.0
    if training_test_rmse > 0:
        degradation = 100.0 * (champion["rmse"] - training_test_rmse) / training_test_rmse
        if degradation > max_rmse_degradation_pct:
            reasons.append(f"RMSE degraded {degradation:.1f}% vs training test RMSE")
    if champion["rmse"] >= best_naive_rmse:
        reasons.append("champion no better than naive baseline")

    needs_retraining = bool(reasons)
    reason = "; ".join(reasons) or "healthy"
    logging.info("needs_retraining=%s reason=%s", needs_retraining, reason)

    # One monitoring row per run, so performance can be tracked over time.
    row = pd.DataFrame(
        [
            {
                "evaluated_at": datetime.now(timezone.utc),
                "pipeline_job_name": pipeline_job_name,
                "model_version": model_version,
                "model_type": model_type,
                "training_test_rmse": training_test_rmse,
                "live_rmse": champion["rmse"],
                "live_mae": champion["mae"],
                "live_wape": champion["wape"],
                "live_bias": champion["bias"],
                "persistence_rmse": naive["persistence"]["rmse"],
                "seasonal_naive_rmse": naive["seasonal_naive"]["rmse"],
                "rmse_degradation_pct": degradation,
                "needs_retraining": needs_retraining,
                "reason": reason,
            }
        ]
    )
    write_dataframe(row, monitoring_table_id, project=project, location=bq_location, append=True)

    outputs = NamedTuple("Outputs", [("needs_retraining", bool), ("reason", str)])
    return outputs(needs_retraining, reason)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def trigger_training_pipeline(
    project: str,
    location: str,
    template_path: str,
    pipeline_root: str,
    service_account: str,
    parameter_values_json: str,
    reason: str,
) -> str:
    """Submit a new run of the training pipeline.

    Called by the evaluation pipeline when the champion is flagged and
    ``auto_retrain`` is enabled. The run is submitted and not awaited.

    Args:
        project: GCP project to run in.
        location: Vertex AI region.
        template_path: Compiled training pipeline, either an Artifact Registry
            URL (``https://<region>-kfp.pkg.dev/...``) or a GCS / local YAML path.
        pipeline_root: GCS folder for pipeline artifacts.
        service_account: Service account for the run (empty = default).
        parameter_values_json: Training pipeline parameters as JSON.
        reason: Why retraining was triggered; printed to the logs.

    Returns:
        The resource name of the submitted pipeline job.
    """
    import json

    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name="storm-training-auto-retrain",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=json.loads(parameter_values_json),
        labels={"trigger": "auto-retrain"},
        enable_caching=False,  # always recompute on fresh data
    )
    job.submit(service_account=service_account or None)
    print(f"Retraining submitted because: {reason}")
    return job.resource_name
