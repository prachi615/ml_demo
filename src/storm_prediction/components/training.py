"""Training components: fit a model, pick the best one, register it in Vertex AI."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Markdown, Metrics, Model, Output

from storm_prediction.config import PIPELINE_IMAGE


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def train_model(
    model_type: str,
    hyperparameters_json: str,
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    val_metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("val_rmse", float), ("val_metrics_json", str)]):
    """Fit one model type on the training split and score it on validation."""
    import json
    import logging
    import os
    import time
    from typing import NamedTuple

    import joblib
    import pandas as pd

    from storm_prediction.features import split_xy
    from storm_prediction.metrics import regression_metrics
    from storm_prediction.models import build_model, predict_counts

    logging.basicConfig(level=logging.INFO)
    all_params = json.loads(hyperparameters_json or "{}")
    params = all_params.get(model_type, {})

    X_train, y_train = split_xy(pd.read_parquet(train_data.path))
    X_val, y_val = split_xy(pd.read_parquet(val_data.path))

    estimator = build_model(model_type, params)
    started = time.time()
    estimator.fit(X_train, y_train)
    train_seconds = time.time() - started

    metrics = regression_metrics(y_val, predict_counts(estimator, X_val))
    metrics["train_seconds"] = train_seconds
    for name, value in metrics.items():
        val_metrics.log_metric(name, value)

    # Vertex AI prebuilt/custom containers expect <artifact_uri>/model.joblib
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(estimator, os.path.join(model.path, "model.joblib"))
    model.metadata.update(
        {"model_type": model_type, "hyperparameters": params, "framework": "scikit-learn"}
    )
    logging.info("%s validation metrics: %s", model_type, metrics)

    outputs = NamedTuple("Outputs", [("val_rmse", float), ("val_metrics_json", str)])
    return outputs(metrics["rmse"], json.dumps(metrics))


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_model(
    model: Input[Model],
    eval_data: Input[Dataset],
    model_type: str,
    split_name: str,
    metrics: Output[Metrics],
    predictions: Output[Dataset],
) -> NamedTuple("Outputs", [("rmse", float), ("metrics_json", str)]):
    """Score a model on a labelled dataset; emit metrics + per-row predictions."""
    import json
    import os
    from typing import NamedTuple

    import joblib
    import pandas as pd

    from storm_prediction.features import TARGET, prepare_features
    from storm_prediction.metrics import regression_metrics
    from storm_prediction.models import predict_counts

    df = pd.read_parquet(eval_data.path)
    estimator = joblib.load(os.path.join(model.path, "model.joblib"))
    df["prediction"] = predict_counts(estimator, prepare_features(df))

    result = regression_metrics(df[TARGET], df["prediction"])
    for name, value in result.items():
        metrics.log_metric(name, value)
    metrics.metadata.update({"model_type": model_type, "split": split_name})

    df[["state", "week_start", TARGET, "prediction"]].to_parquet(predictions.path, index=False)

    outputs = NamedTuple("Outputs", [("rmse", float), ("metrics_json", str)])
    return outputs(result["rmse"], json.dumps(result))


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def select_best_model(
    linear_regression_model: Input[Model],
    ridge_model: Input[Model],
    random_forest_model: Input[Model],
    xgboost_model: Input[Model],
    linear_regression_val: str,
    ridge_val: str,
    random_forest_val: str,
    xgboost_val: str,
    linear_regression_test: str,
    ridge_test: str,
    random_forest_test: str,
    xgboost_test: str,
    min_improvement_pct: float,
    best_model: Output[Model],
    leaderboard: Output[Dataset],
    report: Output[Markdown],
) -> NamedTuple(
    "Outputs",
    [
        ("best_model_type", str),
        ("best_test_rmse", float),
        ("baseline_test_rmse", float),
        ("improvement_pct", float),
        ("beats_baselines", bool),
    ],
):
    """Choose the winner on VALIDATION RMSE, report its held-out TEST RMSE.

    The winner must beat the best linear baseline on the test split by at
    least `min_improvement_pct` percent to be eligible for registration.
    """
    import json
    import shutil
    from typing import NamedTuple

    import pandas as pd

    from storm_prediction.metrics import improvement_pct as pct
    from storm_prediction.models import BASELINE_MODEL_TYPES

    candidates = {
        "linear_regression": (linear_regression_model, linear_regression_val, linear_regression_test),
        "ridge": (ridge_model, ridge_val, ridge_test),
        "random_forest": (random_forest_model, random_forest_val, random_forest_test),
        "xgboost": (xgboost_model, xgboost_val, xgboost_test),
    }

    rows = []
    for name, (_, val_json, test_json) in candidates.items():
        val, test = json.loads(val_json), json.loads(test_json)
        rows.append(
            {
                "model_type": name,
                "is_baseline": name in BASELINE_MODEL_TYPES,
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
        )
    board = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    board["rank"] = board.index + 1

    best = board.iloc[0]
    baseline_rmse = float(board.loc[board["is_baseline"], "test_rmse"].min())
    improvement = pct(float(best["test_rmse"]), baseline_rmse)
    beats = (not bool(best["is_baseline"])) and improvement >= min_improvement_pct

    shutil.copytree(candidates[best["model_type"]][0].path, best_model.path, dirs_exist_ok=True)
    best_model.metadata.update(
        {
            "model_type": best["model_type"],
            "test_rmse": float(best["test_rmse"]),
            "val_rmse": float(best["val_rmse"]),
            "improvement_over_baseline_pct": improvement,
        }
    )

    board.to_parquet(leaderboard.path, index=False)
    cols = [
        "rank", "model_type", "is_baseline", "val_rmse", "val_mae",
        "test_rmse", "test_mae", "test_r2", "test_wape",
    ]
    with open(report.path, "w", encoding="utf-8") as fh:
        fh.write("# Storm forecaster leaderboard\n\n")
        fh.write(board[cols].to_markdown(index=False, floatfmt=".4f"))
        fh.write(
            f"\n\n**Winner:** `{best['model_type']}` | test RMSE {best['test_rmse']:.4f} | "
            f"best baseline test RMSE {baseline_rmse:.4f} | improvement {improvement:.2f}% | "
            f"eligible for registration: **{beats}**\n"
        )

    outputs = NamedTuple(
        "Outputs",
        [
            ("best_model_type", str),
            ("best_test_rmse", float),
            ("baseline_test_rmse", float),
            ("improvement_pct", float),
            ("beats_baselines", bool),
        ],
    )
    return outputs(str(best["model_type"]), float(best["test_rmse"]), baseline_rmse, improvement, beats)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def write_leaderboard(
    project: str,
    bq_location: str,
    table_id: str,
    pipeline_job_name: str,
    train_end: str,
    val_end: str,
    test_end: str,
    leaderboard: Input[Dataset],
):
    """Append the leaderboard of this run to BigQuery for history / dashboards."""
    from datetime import datetime, timezone

    import pandas as pd

    from storm_prediction.gcp import write_dataframe

    board = pd.read_parquet(leaderboard.path)
    board["pipeline_job_name"] = pipeline_job_name
    board["train_end"] = train_end
    board["val_end"] = val_end
    board["test_end"] = test_end
    board["logged_at"] = datetime.now(timezone.utc)
    write_dataframe(board, table_id, project=project, location=bq_location, append=True)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def register_model(
    project: str,
    location: str,
    model_display_name: str,
    model_type: str,
    test_rmse: float,
    serving_container_image_uri: str,
    pipeline_job_name: str,
    model: Input[Model],
) -> NamedTuple("Outputs", [("model_resource_name", str), ("version_id", str), ("promoted", bool)]):
    """Upload a new model version; move the `champion` alias if it beats the current champion."""
    import logging
    from typing import NamedTuple

    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project, location=location)

    existing = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    parent = existing[0].resource_name if existing else None

    def version_with_alias(alias: str):
        if not parent:
            return None
        try:
            return aiplatform.Model(model_name=f"{parent}@{alias}")
        except Exception:  # alias not assigned yet
            return None

    champion = version_with_alias("champion")
    champion_rmse = None
    if champion is not None and "test_rmse_milli" in (champion.labels or {}):
        champion_rmse = int(champion.labels["test_rmse_milli"]) / 1000.0

    promote = champion_rmse is None or test_rmse < champion_rmse
    logging.info("challenger rmse=%.4f champion rmse=%s promote=%s", test_rmse, champion_rmse, promote)

    uploaded = aiplatform.Model.upload(
        display_name=model_display_name,
        parent_model=parent,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri,
        is_default_version=promote,
        version_description=f"{model_type} trained by {pipeline_job_name}; test RMSE {test_rmse:.4f}",
        labels={
            "model_type": model_type.replace("_", "-"),
            "test_rmse_milli": str(int(round(test_rmse * 1000))),
            "use_case": "storm-forecast",
        },
        sync=True,
    )

    # Aliases are unique per model: move them explicitly from the previous holder.
    alias = "champion" if promote else "challenger"
    previous = version_with_alias(alias)
    registry = aiplatform.models.ModelRegistry(uploaded.resource_name)
    if previous is not None and previous.version_id != uploaded.version_id:
        registry.remove_version_aliases([alias], previous.version_id)
    registry.add_version_aliases([alias], uploaded.version_id)

    outputs = NamedTuple("Outputs", [("model_resource_name", str), ("version_id", str), ("promoted", bool)])
    return outputs(uploaded.resource_name, str(uploaded.version_id), promote)
