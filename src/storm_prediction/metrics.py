"""Regression metrics for count forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    abs_err_sum = float(np.abs(y_true - y_pred).sum())
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # weighted absolute percentage error: robust to zero-count weeks
        "wape": abs_err_sum / max(float(y_true.sum()), 1e-9),
        "poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))),
        "bias": float((y_pred - y_true).mean()),
        "n_rows": float(len(y_true)),
    }


def improvement_pct(candidate_rmse: float, reference_rmse: float) -> float:
    """Relative RMSE improvement of candidate over reference, in percent."""
    if reference_rmse <= 0:
        return 0.0
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse
