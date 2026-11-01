"""Evaluation components: run after training to decide whether to promote the new model.

The newly trained best model and the current champion are scored on the SAME
test weeks (the newest labelled data), so the comparison is fair. The champion
was trained on data that ends before these weeks, so it has not seen them either.

As in all components, imports are inside the function bodies because KFP ships
only each function's source to the container.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Markdown, Metrics, Output

from storm_prediction.config import PIPELINE_IMAGE


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_champion(
    project: str,
    location: str,
    model_display_name: str,
    test_data: Input[Dataset],
    metrics: Output[Metrics],
) -> NamedTuple(
    "Outputs",
    [
        ("champion_status", str),
        ("champion_version", str),
        ("champion_model_type", str),
        ("champion_test_rmse", float),
        ("champion_metrics_json", str),
    ],
):
    """Score the current champion model on this run's test split.

    Args:
        project: GCP project of the Model Registry.
        location: Vertex AI region.
        model_display_name: Registry entry name.
        test_data: Parquet with this run's test rows.
        metrics: Output metrics shown in the Vertex AI UI.

    Returns:
        champion_status: ``ok`` if the champion was scored, ``none`` if no
            champion is registered yet (first run), or ``incompatible`` if it
            can no longer score the current features (e.g. the feature list changed).
        champion_version: ``<resource name>@<version id>``, empty if none.
        champion_model_type: Model type from the version's labels, empty if none.
        champion_test_rmse: RMSE on the test split, -1 if not scored.
        champion_metrics_json: All metrics as JSON, ``{}`` if not scored.
    """
    import json
    import logging
    import os
    import tempfile
    from typing import NamedTuple

    import joblib
    import pandas as pd
    from google.cloud import aiplatform

    from storm_prediction.features import TARGET, prepare_features
    from storm_prediction.gcp import download_gcs_dir
    from storm_prediction.metrics import regression_metrics
    from storm_prediction.models import predict_counts

    logging.basicConfig(level=logging.INFO)
    outputs = NamedTuple(
        "Outputs",
        [
            ("champion_status", str),
            ("champion_version", str),
            ("champion_model_type", str),
            ("champion_test_rmse", float),
            ("champion_metrics_json", str),
        ],
    )

    # Find the champion; on the very first run there is none.
    aiplatform.init(project=project, location=location)
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    if not models:
        logging.info("No model named %s registered yet", model_display_name)
        return outputs("none", "", "", -1.0, "{}")
    try:
        champion = aiplatform.Model(model_name=f"{models[0].resource_name}@champion")
    except Exception as exc:  # the registry entry exists but has no champion alias
        logging.info("No champion alias found: %s", exc)
        return outputs("none", "", "", -1.0, "{}")

    version = f"{champion.resource_name}@{champion.version_id}"
    # register_model writes the type with dashes ("random-forest"); convert it back.
    model_type = (champion.labels or {}).get("model_type", "unknown").replace("-", "_")

    model_dir = tempfile.mkdtemp()
    download_gcs_dir(champion.uri, model_dir)
    df = pd.read_parquet(test_data.path)
    try:
        estimator = joblib.load(os.path.join(model_dir, "model.joblib"))
        predictions = predict_counts(estimator, prepare_features(df))
    except Exception as exc:
        # An old champion built for a different feature list can't score current
        # data, and batch inference couldn't use it either, so it must be replaced.
        logging.warning("Champion %s cannot score the current features: %s", version, exc)
        return outputs("incompatible", version, model_type, -1.0, "{}")

    result = regression_metrics(df[TARGET], predictions)
    for name, value in result.items():
        metrics.log_metric(name, value)
    metrics.metadata.update({"model_type": model_type, "version": version, "split": "test"})
    return outputs("ok", version, model_type, result["rmse"], json.dumps(result))


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def decide_promotion(
    project: str,
    bq_location: str,
    evaluation_table_id: str,
    pipeline_job_name: str,
    new_model_type: str,
    new_test_metrics_json: str,
    champion_status: str,
    champion_version: str,
    champion_model_type: str,
    champion_test_rmse: float,
    min_improvement_pct: float,
    test_end: str,
    report: Output[Markdown],
) -> NamedTuple("Outputs", [("promote", bool), ("reason", str)]):
    """Compare the new best model with the champion and log the decision to BigQuery.

    Args:
        project: GCP project used to write the evaluation row.
        bq_location: BigQuery location.
        evaluation_table_id: Fully qualified evaluation table id (appended to).
        pipeline_job_name: Name of the current pipeline run.
        new_model_type: Type of the best of the four newly trained models.
        new_test_metrics_json: Its test metrics as JSON.
        champion_status: ``ok``, ``none`` or ``incompatible`` (from ``evaluate_champion``).
        champion_version: Champion version name (empty if none).
        champion_model_type: Champion model type (empty if none).
        champion_test_rmse: Champion RMSE on the same test split.
        min_improvement_pct: Required RMSE improvement over the champion, in percent.
        test_end: Last week of the test split, for the record.
        report: Output Markdown summary shown in the Vertex AI UI.

    Returns:
        promote: True if the new model should become the champion.
        reason: Human-readable explanation of the decision.
    """
    import json
    from datetime import datetime, timezone
    from typing import NamedTuple

    import pandas as pd

    from storm_prediction.gcp import write_dataframe
    from storm_prediction.metrics import should_promote

    new = json.loads(new_test_metrics_json)
    champion_rmse = champion_test_rmse if champion_status == "ok" else None
    promote, reason = should_promote(new["rmse"], champion_rmse, min_improvement_pct)

    # One row per training run: the history of every promotion decision.
    row = pd.DataFrame(
        [
            {
                "evaluated_at": datetime.now(timezone.utc),
                "pipeline_job_name": pipeline_job_name,
                "test_end": test_end,
                "new_model_type": new_model_type,
                "new_test_rmse": new["rmse"],
                "new_test_mae": new["mae"],
                "new_test_r2": new["r2"],
                "new_test_wape": new["wape"],
                "champion_status": champion_status,
                "champion_version": champion_version,
                "champion_model_type": champion_model_type,
                "champion_test_rmse": champion_rmse,
                "min_improvement_pct": min_improvement_pct,
                "promoted": promote,
                "reason": reason,
            }
        ]
    )
    write_dataframe(row, evaluation_table_id, project=project, location=bq_location, append=True)

    champion_text = (
        f"`{champion_model_type}` ({champion_version}), test RMSE {champion_rmse:.4f}"
        if champion_rmse is not None
        else f"none usable ({champion_status})"
    )
    with open(report.path, "w", encoding="utf-8") as fh:
        fh.write("# Promotion decision\n\n")
        fh.write(f"- New best model: `{new_model_type}`, test RMSE {new['rmse']:.4f}\n")
        fh.write(f"- Current champion: {champion_text}\n")
        fh.write(f"- **Promote: {promote}** ({reason})\n")

    outputs = NamedTuple("Outputs", [("promote", bool), ("reason", str)])
    return outputs(promote, reason)
