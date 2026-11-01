"""Batch inference components: load the champion and forecast next week.

Scoring runs inside a pipeline component instead of Vertex AI's built-in batch
prediction job: the saved model is a scikit-learn pipeline that may contain
XGBoost, which the prebuilt scikit-learn serving container cannot load.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

from storm_prediction.config import PIPELINE_IMAGE


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def load_registered_model(
    project: str,
    location: str,
    model_display_name: str,
    alias: str,
    model: Output[Model],
) -> NamedTuple("Outputs", [("model_type", str), ("version_name", str)]):
    """Fetch a model version from the Vertex AI Model Registry by alias.

    Resolves ``<model_display_name>@<alias>`` (normally ``champion``) and
    downloads its artifact directory from Cloud Storage.

    Args:
        project: GCP project of the registry.
        location: Vertex AI region.
        model_display_name: Registry entry name.
        alias: Version alias to load, normally ``champion``.
        model: Output directory; receives ``model.joblib``.

    Returns:
        model_type: Model type read from the version's labels.
        version_name: ``<resource name>@<version id>``, recorded with predictions.

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
    # register_model writes the type with dashes ("random-forest"); convert it back.
    model_type = (registered.labels or {}).get("model_type", "unknown").replace("-", "_")
    version_name = f"{registered.resource_name}@{registered.version_id}"
    model.metadata.update({"model_type": model_type, "version": version_name, "alias": alias})

    outputs = NamedTuple("Outputs", [("model_type", str), ("version_name", str)])
    return outputs(model_type, version_name)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def batch_predict(
    project: str,
    bq_location: str,
    predictions_table_id: str,
    pipeline_job_name: str,
    model_version: str,
    model_type: str,
    model: Input[Model],
    scoring_data: Input[Dataset],
    predictions: Output[Dataset],
) -> int:
    """Forecast next week's storm events for every state and save to BigQuery.

    Each output row has the state, forecast week, predicted event count, the
    state's trailing 52-week average and a ``risk_level`` label (low / normal /
    elevated / high) comparing the two. Model version and pipeline run are
    stored with each prediction for lineage.

    Args:
        project: GCP project used to write the predictions.
        bq_location: BigQuery location.
        predictions_table_id: Fully qualified predictions table id (appended to).
        pipeline_job_name: Name of the current pipeline run.
        model_version: Registry version used, from ``load_registered_model``.
        model_type: Its model type.
        model: Model directory containing ``model.joblib``.
        scoring_data: Parquet with the forecast rows.
        predictions: Output parquet copy of the predictions.

    Returns:
        The number of predictions written.
    """
    import os
    from datetime import datetime, timezone

    import joblib
    import pandas as pd

    from storm_prediction.data import risk_level
    from storm_prediction.features import prepare_features
    from storm_prediction.gcp import write_dataframe
    from storm_prediction.models import predict_counts

    df = pd.read_parquet(scoring_data.path)
    estimator = joblib.load(os.path.join(model.path, "model.joblib"))

    # The saved estimator includes preprocessing, so raw feature columns go straight in.
    out = pd.DataFrame(
        {
            "state": df["state"].astype(str),
            "forecast_week_start": pd.to_datetime(df["week_start"]).dt.date,
            "predicted_event_count": predict_counts(estimator, prepare_features(df)),
            "trailing_52w_mean": df["rolling_mean_52"].astype(float),
        }
    )
    out["risk_level"] = risk_level(out["predicted_event_count"], out["trailing_52w_mean"])
    # Lineage: which model and which run produced each prediction.
    out["model_version"] = model_version
    out["model_type"] = model_type
    out["pipeline_job_name"] = pipeline_job_name
    out["scored_at"] = datetime.now(timezone.utc)

    out.to_parquet(predictions.path, index=False)
    predictions.metadata["rows"] = len(out)
    write_dataframe(out, predictions_table_id, project=project, location=bq_location, append=True)
    return len(out)
