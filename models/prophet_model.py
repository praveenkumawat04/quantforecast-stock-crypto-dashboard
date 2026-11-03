"""Prophet-based forecasting with leak-free evaluation.

Prophet expects a two-column frame (``ds``, ``y``).  This module converts the
dashboard's price series, trains an evaluation model on the chronological train
partition, scores it on the held-out test partition, then refits on all history
to produce the future forecast and uncertainty intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.evaluation import evaluate_forecast
from utils.preprocessing import chronological_split, future_dates, to_close_series

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on the environment
    from prophet import Prophet

    PROPHET_AVAILABLE = True
    PROPHET_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    Prophet = None  # type: ignore[assignment,misc]
    PROPHET_AVAILABLE = False
    PROPHET_IMPORT_ERROR = str(exc)

# Prophet/cmdstanpy are noisy at import and fit time.  Keep Streamlit output clean.
# Set levels after importing Prophet because it configures its own loggers.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
try:
    import cmdstanpy

    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)
except Exception:  # pragma: no cover - cmdstanpy is a Prophet dependency
    pass

ProgressCallback = Callable[[str, float], None]


class ProphetUnavailableError(RuntimeError):
    """Raised when Prophet cannot be imported."""


class ProphetForecastError(RuntimeError):
    """Raised when Prophet training or prediction fails."""


@dataclass
class ProphetForecastResult:
    """Container returned by :func:`run_prophet_forecast`."""

    test_dates: pd.DatetimeIndex
    test_actual: np.ndarray
    test_predictions: np.ndarray
    future_dates: pd.DatetimeIndex
    future_predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    metrics: dict[str, Any]
    forecast_frame: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)
    model_name: str = "Prophet"


def _require_prophet() -> None:
    if not PROPHET_AVAILABLE:
        raise ProphetUnavailableError(
            "Prophet is not available in this environment. Install it with "
            "`pip install prophet` and restart the app. "
            f"Import error: {PROPHET_IMPORT_ERROR}"
        )


def prepare_prophet_frame(close_prices: pd.Series) -> pd.DataFrame:
    """Convert a close-price series to Prophet's ``ds``/``y`` format."""

    series = to_close_series(close_prices)
    frame = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})
    frame["ds"] = pd.to_datetime(frame["ds"]).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["ds", "y"])
    frame = frame[~frame["ds"].duplicated(keep="last")].sort_values("ds").reset_index(drop=True)
    if len(frame) < 20:
        raise ProphetForecastError(
            "Prophet needs at least 20 observations. Select a longer date range."
        )
    return frame


def _build_prophet(
    changepoint_prior_scale: float,
    seasonality_mode: str,
    interval_width: float,
    yearly_seasonality: str | bool,
    weekly_seasonality: str | bool,
    daily_seasonality: str | bool,
) -> Any:
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        uncertainty_samples=500,
        mcmc_samples=0,
    )
    return model


def _predict_dates(model: Any, dates: pd.DatetimeIndex) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    forecast = model.predict(future)
    required = {"yhat", "yhat_lower", "yhat_upper"}
    if not required.issubset(forecast.columns):
        raise ProphetForecastError("Prophet returned an unexpected forecast schema.")
    return forecast


def run_prophet_forecast(
    close_prices: pd.Series,
    horizon: int,
    test_size: float = 0.2,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
    interval_width: float = 0.8,
    yearly_seasonality: str | bool = "auto",
    weekly_seasonality: str | bool = "auto",
    daily_seasonality: str | bool = False,
    trading_days: bool = True,
    random_state: int = 42,
    progress_callback: ProgressCallback | None = None,
) -> ProphetForecastResult:
    """Train Prophet, evaluate it chronologically, and produce future forecasts."""

    _require_prophet()

    if horizon < 1:
        raise ProphetForecastError("Forecast horizon must be positive.")
    if not 0.05 <= test_size <= 0.5:
        raise ProphetForecastError("test_size must be between 0.05 and 0.5.")
    if seasonality_mode not in {"additive", "multiplicative"}:
        raise ProphetForecastError("seasonality_mode must be 'additive' or 'multiplicative'.")
    if not 0.5 <= interval_width < 1.0:
        raise ProphetForecastError("interval_width must be in [0.5, 1).")

    frame = prepare_prophet_frame(close_prices)
    train, test = chronological_split(frame["y"], test_size=test_size)
    if len(train) < 15 or len(test) < 2:
        raise ProphetForecastError(
            "Not enough observations for Prophet training/evaluation. "
            "Use a longer historical range."
        )

    # Prophet's uncertainty simulation does not expose a seed parameter.
    np.random.seed(int(random_state))

    train_frame = frame.iloc[: len(train)].copy()
    test_frame = frame.iloc[len(train) :].copy()

    def notify(message: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(message, fraction)

    try:
        notify("Fitting Prophet on the training window…", 0.15)
        eval_model = _build_prophet(
            changepoint_prior_scale,
            seasonality_mode,
            interval_width,
            yearly_seasonality,
            weekly_seasonality,
            daily_seasonality,
        )
        eval_model.fit(train_frame)

        notify("Evaluating on held-out history…", 0.50)
        test_forecast = _predict_dates(eval_model, pd.DatetimeIndex(test_frame["ds"]))
        test_predictions = np.clip(
            test_forecast["yhat"].to_numpy(dtype=float), 0.0, None
        )
        test_actual = test_frame["y"].to_numpy(dtype=float)
        metrics = evaluate_forecast(test_actual, test_predictions, model_name="Prophet")

        notify("Refitting Prophet on full history…", 0.65)
        final_model = _build_prophet(
            changepoint_prior_scale,
            seasonality_mode,
            interval_width,
            yearly_seasonality,
            weekly_seasonality,
            daily_seasonality,
        )
        final_model.fit(frame)

        notify("Generating future forecast…", 0.90)
        dates = future_dates(
            last_date=frame["ds"].iloc[-1],
            steps=horizon,
            trading_days=trading_days,
        )
        future_forecast = _predict_dates(final_model, dates)

    except (ProphetUnavailableError, ProphetForecastError):
        raise
    except Exception as exc:  # pragma: no cover - cmdstan/environment dependent
        LOGGER.exception("Prophet forecast failed")
        raise ProphetForecastError(f"Prophet forecast failed: {exc}") from exc

    future_predictions = np.clip(
        future_forecast["yhat"].to_numpy(dtype=float), 0.0, None
    )
    lower = np.clip(future_forecast["yhat_lower"].to_numpy(dtype=float), 0.0, None)
    upper = np.clip(future_forecast["yhat_upper"].to_numpy(dtype=float), 0.0, None)
    # Guard against numerical crossings at the zero boundary.
    lower = np.minimum(lower, future_predictions)
    upper = np.maximum(upper, future_predictions)

    forecast_frame = pd.DataFrame(
        {
            "Date": dates,
            "Forecast": future_predictions,
            "Lower Bound": lower,
            "Upper Bound": upper,
        }
    )

    notify("Forecast complete", 1.0)

    return ProphetForecastResult(
        test_dates=pd.DatetimeIndex(test_frame["ds"]),
        test_actual=test_actual,
        test_predictions=test_predictions,
        future_dates=dates,
        future_predictions=future_predictions,
        lower_bound=lower,
        upper_bound=upper,
        metrics=metrics,
        forecast_frame=forecast_frame,
        config={
            "test_size": test_size,
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_mode": seasonality_mode,
            "interval_width": interval_width,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "random_state": random_state,
        },
        model_name="Prophet",
    )
