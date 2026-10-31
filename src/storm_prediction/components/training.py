"""Training components: fit a model, evaluate it, pick the best one, register it.

Note on imports: KFP ships only the *source of each function* to the container,
so every import a component needs is written inside the function body. The
``storm_prediction`` package itself is installed in the image (see Dockerfile).
"""

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
    """Train one model type and score it on the validation split.

    The four model types run as four parallel tasks of this same component.

    Args:
        model_type: ``linear_regression``, ``ridge``, ``random_forest`` or ``xgboost``.
        hyperparameters_json: JSON object keyed by model type, e.g.
            ``{"xgboost": {"max_depth": 6}}``. Only this model's entry is used;
            values override ``models.DEFAULT_HYPERPARAMETERS``.
        train_data: Parquet with the training rows.
        val_data: Parquet with the validation rows.
        model: Output directory; receives ``model.joblib``.
        val_metrics: Output metrics shown in the Vertex AI UI.

    Returns:
        val_rmse: Validation RMSE, used to pick the best model.
        val_metrics_json: All validation metrics as a JSON string.
    """
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
    # One JSON parameter holds the settings of all four models; pick ours.
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
    # log_metric makes each value visible on the task in the Vertex AI UI.
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
    """Score a trained model on a labelled dataset.

    Used by the training pipeline (test split) and the evaluation pipeline
    (recent weeks, to monitor the champion).

    Args:
        model: Model directory containing ``model.joblib``.
        eval_data: Parquet with labelled feature rows.
        model_type: Model type, recorded in the metrics metadata.
        split_name: Label for the data, e.g. ``test`` or ``recent``.
        metrics: Output metrics shown in the Vertex AI UI.
        predictions: Output parquet with ``state, week_start, event_count, prediction``.

    Returns:
        rmse: RMSE on ``eval_data``.
        metrics_json: All metrics as a JSON string.
    """
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
    # The saved estimator includes preprocessing, so raw feature columns go straight in.
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
    """Pick the best of the four models and decide if it may be registered.

    The winner is the model with the lowest validation RMSE. Its test RMSE (on
    weeks no model was tuned or selected on) is then compared with the best linear
    baseline. The winner is eligible for registration only if it is a challenger
    (random forest or XGBoost) and beats that baseline by at least
    ``min_improvement_pct`` percent.

    Args:
        linear_regression_model: Trained linear regression model.
        ridge_model: Trained ridge model.
        random_forest_model: Trained random forest model.
        xgboost_model: Trained XGBoost model.
        linear_regression_val: Validation metrics JSON of the linear regression model.
        ridge_val: Validation metrics JSON of the ridge model.
        random_forest_val: Validation metrics JSON of the random forest model.
        xgboost_val: Validation metrics JSON of the XGBoost model.
        linear_regression_test: Test metrics JSON of the linear regression model.
        ridge_test: Test metrics JSON of the ridge model.
        random_forest_test: Test metrics JSON of the random forest model.
        xgboost_test: Test metrics JSON of the XGBoost model.
        min_improvement_pct: Required test-RMSE improvement over the best baseline.
        best_model: Output; a copy of the winning model directory.
        leaderboard: Output parquet with one row per model, ranked.
        report: Output Markdown leaderboard shown in the Vertex AI UI.

    Returns:
        best_model_type: Name of the winning model type.
        best_test_rmse: Test RMSE of the winner.
        baseline_test_rmse: Best test RMSE among the linear baselines.
        improvement_pct: Winner's test-RMSE improvement over that baseline, in percent.
        beats_baselines: True if the winner may be registered.
    """
    import json
    import shutil
    from typing import NamedTuple

    import pandas as pd

    from storm_prediction.metrics import improvement_pct as pct
    from storm_prediction.models import BASELINE_MODEL_TYPES

    # model type -> (model artifact, validation metrics JSON, test metrics JSON)
    candidates = {
        "linear_regression": (linear_regression_model, linear_regression_val, linear_regression_test),
        "ridge": (ridge_model, ridge_val, ridge_test),
        "random_forest": (random_forest_model, random_forest_val, random_forest_test),
        "xgboost": (xgboost_model, xgboost_val, xgboost_test),
    }

    # Build the leaderboard: one row per model, columns prefixed val_ / test_.
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

    # Winner = lowest validation RMSE. The gate uses test RMSE, which played no
    # part in the choice, so it is an unbiased check.
    best = board.iloc[0]
    baseline_rmse = float(board.loc[board["is_baseline"], "test_rmse"].min())
    improvement = pct(float(best["test_rmse"]), baseline_rmse)
    beats = (not bool(best["is_baseline"])) and improvement >= min_improvement_pct

    # Copy the winner into this component's output so later steps get a single model.
    shutil.copytree(candidates[best["model_type"]][0].path, best_model.path, dirs_exist_ok=True)
    best_model.metadata.update(
        {
            "model_type": best["model_type"],
            "test_rmse": float(best["test_rmse"]),
            "val_rmse": float(best["val_rmse"]),
            "improvement_over_baseline_pct": improvement,
        }
    )

    # Persist the leaderboard (parquet for BigQuery, Markdown for the Vertex AI UI).
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
    """Append this run's leaderboard to BigQuery.

    The table keeps the history of every training run, which is useful for
    dashboards and for spotting gradual performance changes.

    Args:
        project: GCP project used to run the load job.
        bq_location: BigQuery location.
        table_id: Fully qualified leaderboard table id.
        pipeline_job_name: Name of the current pipeline run, for lineage.
        train_end: Last training week.
        val_end: Last validation week.
        test_end: Last test week.
        leaderboard: Parquet produced by ``select_best_model``.
    """
    from datetime import datetime, timezone

    import pandas as pd

    from storm_prediction.gcp import write_dataframe

    board = pd.read_parquet(leaderboard.path)
    # Add run context so rows from different runs can be told apart.
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
    """Upload the winner to the Vertex AI Model Registry and manage its alias.

    Every registered model is a new *version* of one registry entry. Other
    pipelines always load ``<model>@champion``, so promotion is just moving that
    alias:

    * no champion yet, or the new test RMSE is lower: the new version becomes
      ``champion`` and the default version;
    * otherwise it is stored as ``challenger`` and the champion stays live.

    The test RMSE is stored as a label (``test_rmse_milli`` = RMSE x 1000, because
    label values cannot contain dots) so later runs and the evaluation pipeline
    can compare against it.

    Args:
        project: GCP project of the registry.
        location: Vertex AI region, e.g. ``us-central1``.
        model_display_name: Registry entry name; versions are grouped under it.
        model_type: Winning model type, stored as a label.
        test_rmse: Winner's test RMSE.
        serving_container_image_uri: Image recorded with the model (required by
            the registry; only used if the model is deployed to an endpoint).
        pipeline_job_name: Name of the current pipeline run, for lineage.
        model: Model directory containing ``model.joblib``.

    Returns:
        model_resource_name: Registry resource name of the model.
        version_id: The new version's id.
        promoted: True if the new version became ``champion``.
    """
    import logging
    from typing import NamedTuple

    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project, location=location)

    # If the registry entry already exists, upload as a new version of it.
    existing = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    parent = existing[0].resource_name if existing else None

    def version_with_alias(alias: str):
        """Return the model version holding ``alias``, or None if there is none."""
        if not parent:
            return None
        try:
            return aiplatform.Model(model_name=f"{parent}@{alias}")
        except Exception:  # alias not assigned yet
            return None

    # Read the current champion's test RMSE from its label (if there is one).
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
