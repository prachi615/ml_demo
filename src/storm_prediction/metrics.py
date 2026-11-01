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


def should_promote(
    new_rmse: float, champion_rmse: float | None, min_improvement_pct: float = 0.0
) -> tuple[bool, str]:
    """Decide whether a newly trained model should replace the current champion.

    Both RMSEs must be measured on the same test weeks, otherwise the comparison
    is meaningless.

    Args:
        new_rmse: Test RMSE of the newly trained best model.
        champion_rmse: Test RMSE of the current champion on the same test weeks,
            or None if there is no usable champion (first run, or the champion
            can no longer score the current features).
        min_improvement_pct: How much better (in percent) the new model must be.
            0 means "any improvement".

    Returns:
        A tuple ``(promote, reason)`` with a human-readable reason.
    """
    if champion_rmse is None:
        return True, "no usable champion yet"
    gain = improvement_pct(new_rmse, champion_rmse)
    if new_rmse < champion_rmse and gain >= min_improvement_pct:
        return True, f"new model is {gain:.2f}% better than the champion"
    return False, f"new model is not better enough ({gain:.2f}% vs required {min_improvement_pct:.2f}%)"


def beats_baseline(
    candidate_rmse: float, baseline_rmse: float, min_improvement_pct: float = 2.0
) -> tuple[bool, str]:
    """Check that a model beats the linear regression baseline by MORE than a margin.

    Both RMSEs must come from the same test weeks.

    Args:
        candidate_rmse: Test RMSE of the best of the four models.
        baseline_rmse: Test RMSE of the linear regression baseline.
        min_improvement_pct: Margin in percent; the improvement must be strictly greater.

    Returns:
        A tuple ``(passed, reason)``. If the best model *is* linear regression,
        the improvement is 0% and the check fails.
    """
    gain = improvement_pct(candidate_rmse, baseline_rmse)
    if gain > min_improvement_pct:
        return True, f"beats linear regression by {gain:.2f}% (> {min_improvement_pct:.2f}%)"
    return False, f"beats linear regression by only {gain:.2f}% (needs > {min_improvement_pct:.2f}%)"
