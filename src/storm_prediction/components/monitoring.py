"""Components used by the evaluation (champion monitoring) pipeline."""

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
    """Resolve `<model>@<alias>` in the Vertex AI Model Registry and download its artifact."""
    from typing import NamedTuple

    from google.cloud import aiplatform

    from storm_prediction.gcp import download_gcs_dir

    aiplatform.init(project=project, location=location)
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    if not models:
        raise RuntimeError(f"No model named {model_display_name!r} in the registry; run training first.")
    registered = aiplatform.Model(model_name=f"{models[0].resource_name}@{alias}")

    download_gcs_dir(registered.uri, model.path)
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
    """Model-free reference forecasts: last week (persistence) and 3-year same-week mean."""
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
    """Compare live performance against training-time RMSE and naive baselines; log to BigQuery."""
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
    """Submit the (pre-compiled) training pipeline when the champion has degraded."""
    import json

    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name="storm-training-auto-retrain",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=json.loads(parameter_values_json),
        labels={"trigger": "auto-retrain"},
        enable_caching=False,
    )
    job.submit(service_account=service_account or None)
    print(f"Retraining submitted because: {reason}")
    return job.resource_name
