"""Forecast evaluation metrics with strict input validation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

_EPSILON = 1e-12


class EvaluationError(ValueError):
    """Raised when actual and predicted values cannot be compared."""


def _validate_arrays(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)

    if y_true.size == 0 or y_pred.size == 0:
        raise EvaluationError("Actual and predicted arrays cannot be empty.")
    if y_true.size != y_pred.size:
        raise EvaluationError(
            f"Length mismatch: actual has {y_true.size} values, "
            f"predicted has {y_pred.size}."
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise EvaluationError("No finite pairs are available for evaluation.")
    return y_true[mask], y_pred[mask]


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error in price units."""

    y_true, y_pred = _validate_arrays(actual, predicted)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error in price units."""

    y_true, y_pred = _validate_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mean_absolute_percentage_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Mean absolute percentage error (percent units, 0-100).

    Observations whose actual value is effectively zero are excluded because
    percentage error is undefined for them.
    """

    y_true, y_pred = _validate_arrays(actual, predicted)
    mask = np.abs(y_true) > _EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Share of step-to-step directions the forecast gets right (percent)."""

    y_true, y_pred = _validate_arrays(actual, predicted)
    if y_true.size < 2:
        return float("nan")
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return float(np.mean(true_direction == pred_direction) * 100.0)


def evaluate_forecast(
    actual: Sequence[float],
    predicted: Sequence[float],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return MAE, RMSE, MAPE and supporting diagnostics in one dictionary."""

    y_true, y_pred = _validate_arrays(actual, predicted)
    result: dict[str, Any] = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "n_samples": int(y_true.size),
    }
    if model_name:
        result["model"] = model_name
    return result


def format_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    """Format a metrics dictionary for presentation."""

    mae = metrics.get("mae", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    mape = metrics.get("mape", float("nan"))
    return {
        "MAE": f"{mae:,.2f}" if np.isfinite(mae) else "N/A",
        "RMSE": f"{rmse:,.2f}" if np.isfinite(rmse) else "N/A",
        "MAPE": f"{mape:.2f}%" if np.isfinite(mape) else "N/A",
    }
