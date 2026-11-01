"""Model zoo: two linear baselines and two non-linear models.

| Model             | Role       |
|-------------------|------------|
| LinearRegression  | baseline   |
| Ridge             | baseline   |
| RandomForest      | non-linear |
| XGBoost           | non-linear |

Every model is wrapped in the same scikit-learn pipeline (imputation, optional
scaling, one-hot encoding of ``state``) and trained on ``log1p(event_count)``,
so the comparison between the four is like-for-like. Every training run picks the
best of the four for the current data, which must then beat LinearRegression by
more than 2% to be promoted (see components/training.py). The fitted
object is a single estimator that can be saved with ``joblib`` and used for
batch inference without extra preprocessing code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from storm_prediction.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Model type names used throughout the pipelines, configs and BigQuery tables.
LINEAR_REGRESSION = "linear_regression"
RIDGE = "ridge"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"

BASELINE_MODEL_TYPES = (LINEAR_REGRESSION, RIDGE)
CHALLENGER_MODEL_TYPES = (RANDOM_FOREST, XGBOOST)
MODEL_TYPES = BASELINE_MODEL_TYPES + CHALLENGER_MODEL_TYPES

# Defaults used when the pipeline config does not override them.
DEFAULT_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    LINEAR_REGRESSION: {},
    RIDGE: {"alpha": 10.0},
    RANDOM_FOREST: {
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "n_jobs": -1,
        "random_state": 42,
    },
    XGBOOST: {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    },
}


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the shared feature preprocessing step.

    Args:
        scale_numeric: Standardise numeric features. Needed for the linear models
            (and Ridge's penalty); tree models are scale-invariant, so it is skipped.

    Returns:
        A ``ColumnTransformer`` that imputes numeric features with the median and
        one-hot encodes ``state``. States not seen in training encode as all zeros.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_model(model_type: str, hyperparameters: dict[str, Any] | None = None):
    """Create an unfitted model pipeline.

    Args:
        model_type: One of ``MODEL_TYPES``.
        hyperparameters: Overrides merged on top of ``DEFAULT_HYPERPARAMETERS``
            for this model type.

    Returns:
        A scikit-learn ``TransformedTargetRegressor`` that fits on ``log1p(y)`` and
        returns predictions on the original count scale.

    Raises:
        ValueError: If ``model_type`` is unknown.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")

    params = {**DEFAULT_HYPERPARAMETERS[model_type], **(hyperparameters or {})}

    if model_type == LINEAR_REGRESSION:
        regressor, scale = LinearRegression(**params), True
    elif model_type == RIDGE:
        regressor, scale = Ridge(**params), True
    elif model_type == RANDOM_FOREST:
        regressor, scale = RandomForestRegressor(**params), False
    else:
        regressor, scale = XGBRegressor(**params), False

    pipeline = Pipeline([("preprocess", _preprocessor(scale)), ("regressor", regressor)])

    # Weekly counts are heavy-tailed (a few outbreak weeks have hundreds of events).
    # Fitting on log1p(y) keeps those weeks from dominating the loss; expm1 maps
    # predictions back to counts.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def predict_counts(model, X) -> np.ndarray:
    """Predict event counts that are never negative.

    Linear models can produce slightly negative values after ``expm1``; a negative
    number of storms is meaningless, so predictions are clipped at 0.

    Args:
        model: A fitted estimator returned by ``build_model``.
        X: Feature frame produced by ``features.prepare_features``.

    Returns:
        A 1-D array of predicted weekly event counts.
    """
    return np.clip(model.predict(X), 0.0, None)
