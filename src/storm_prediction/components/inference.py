"""Batch inference component."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

from storm_prediction.config import PIPELINE_IMAGE


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
    """Score forecast rows with the champion and append results to BigQuery."""
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

    out = pd.DataFrame(
        {
            "state": df["state"].astype(str),
            "forecast_week_start": pd.to_datetime(df["week_start"]).dt.date,
            "predicted_event_count": predict_counts(estimator, prepare_features(df)),
            "trailing_52w_mean": df["rolling_mean_52"].astype(float),
        }
    )
    out["risk_level"] = risk_level(out["predicted_event_count"], out["trailing_52w_mean"])
    out["model_version"] = model_version
    out["model_type"] = model_type
    out["pipeline_job_name"] = pipeline_job_name
    out["scored_at"] = datetime.now(timezone.utc)

    out.to_parquet(predictions.path, index=False)
    predictions.metadata["rows"] = len(out)
    write_dataframe(out, predictions_table_id, project=project, location=bq_location, append=True)
    return len(out)
