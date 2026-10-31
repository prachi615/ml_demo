"""Regression metrics used to compare models and to monitor the champion."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the metrics logged for every model and evaluation run.

    Args:
        y_true: Observed weekly event counts.
        y_pred: Predicted counts (negative values are clipped to 0).

    Returns:
        A dictionary with:

        * ``rmse``: root mean squared error, the main selection metric
        * ``mae``: mean absolute error, in events per state-week
        * ``r2``: share of variance explained
        * ``wape``: total absolute error / total events; unlike MAPE it stays
          defined for weeks with zero events
        * ``poisson_deviance``: a natural loss for count data
        * ``bias``: mean(prediction - actual); positive means over-forecasting
        * ``n_rows``: number of rows evaluated
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    abs_err_sum = float(np.abs(y_true - y_pred).sum())
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "wape": abs_err_sum / max(float(y_true.sum()), 1e-9),
        # Poisson deviance needs strictly positive predictions.
        "poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))),
        "bias": float((y_pred - y_true).mean()),
        "n_rows": float(len(y_true)),
    }


def improvement_pct(candidate_rmse: float, reference_rmse: float) -> float:
    """Relative RMSE improvement of a candidate over a reference model.

    Args:
        candidate_rmse: RMSE of the model being judged (e.g. XGBoost).
        reference_rmse: RMSE of the model it is compared to (e.g. best baseline).

    Returns:
        Improvement in percent; positive means the candidate is better.
        Returns 0 if ``reference_rmse`` is not positive.
    """
    if reference_rmse <= 0:
        return 0.0
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse
