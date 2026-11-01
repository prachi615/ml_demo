"""Tests for the four models and the metrics."""

import numpy as np
import pytest

from storm_prediction.features import split_xy
from storm_prediction.metrics import improvement_pct, regression_metrics
from storm_prediction.models import (
    BASELINE_MODEL_TYPES,
    CHALLENGER_MODEL_TYPES,
    MODEL_TYPES,
    build_model,
    predict_counts,
)


def test_model_zoo_has_two_linear_baselines_and_xgboost():
    """The model zoo matches the project spec: 2 linear baselines + RF + XGBoost."""
    assert BASELINE_MODEL_TYPES == ("linear_regression", "ridge")
    assert "xgboost" in CHALLENGER_MODEL_TYPES
    assert len(MODEL_TYPES) == 4


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_fits_and_predicts_non_negative_counts(model_type, feature_frame):
    """Each model trains end to end and never predicts a negative count."""
    X, y = split_xy(feature_frame)
    params = {"n_estimators": 20} if model_type in CHALLENGER_MODEL_TYPES else None
    model = build_model(model_type, params).fit(X, y)
    preds = predict_counts(model, X)
    assert preds.shape == (len(X),)
    assert np.all(preds >= 0)
    assert regression_metrics(y, preds)["r2"] > 0


def test_unknown_states_are_handled(feature_frame):
    """A state not seen during training still gets a prediction instead of an error."""
    X, y = split_xy(feature_frame)
    model = build_model("ridge").fit(X, y)
    X_new = X.head(3).copy()
    X_new["state"] = "Atlantis"
    assert predict_counts(model, X_new).shape == (3,)


def test_unknown_model_type_raises():
    """A typo in the model type fails loudly."""
    with pytest.raises(ValueError):
        build_model("deep_magic")


def test_metrics():
    """Perfect predictions give zero error; improvement_pct handles a zero reference."""
    m = regression_metrics([0, 2, 4], [0, 2, 4])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["wape"] == 0
    assert improvement_pct(9.0, 10.0) == pytest.approx(10.0)
    assert improvement_pct(1.0, 0.0) == 0.0


def test_should_promote_rules():
    """New model wins only if it is strictly better (and by the margin, if one is set)."""
    from storm_prediction.metrics import should_promote

    assert should_promote(5.0, None)[0]  # first run: nothing to beat
    assert should_promote(4.9, 5.0)[0]  # better
    assert not should_promote(5.0, 5.0)[0]  # equal is not better
    assert not should_promote(5.5, 5.0)[0]  # worse
    assert not should_promote(4.95, 5.0, min_improvement_pct=2.0)[0]  # better, but < 2%
    assert should_promote(4.8, 5.0, min_improvement_pct=2.0)[0]  # 4% better
