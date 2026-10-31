"""Model zoo: two linear baselines + two non-linear challengers.

All models share the same preprocessing and are trained on log1p(count) so the
comparison between baselines and challengers is apples-to-apples.
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

LINEAR_REGRESSION = "linear_regression"
RIDGE = "ridge"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"

BASELINE_MODEL_TYPES = (LINEAR_REGRESSION, RIDGE)
CHALLENGER_MODEL_TYPES = (RANDOM_FOREST, XGBOOST)
MODEL_TYPES = BASELINE_MODEL_TYPES + CHALLENGER_MODEL_TYPES

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
    """Return an unfitted sklearn estimator for `model_type`."""
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
    # Counts are heavy-tailed: fit on log1p(y), predict back on the count scale.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def predict_counts(model, X) -> np.ndarray:
    """Predict event counts, clipped at 0 (linear models can go slightly negative)."""
    return np.clip(model.predict(X), 0.0, None)
