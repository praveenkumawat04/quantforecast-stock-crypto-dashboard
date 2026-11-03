"""Reusable preprocessing utilities for price-series forecasting.

All transformations are intentionally side-effect free and framework agnostic
so they can be unit-tested without TensorFlow or Streamlit installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_MA_WINDOWS: tuple[int, ...] = (20, 50)


class PreprocessingError(ValueError):
    """Raised when input data cannot be transformed safely."""


@dataclass(frozen=True)
class LSTMData:
    """Arrays required to train and evaluate an LSTM model."""

    scaler: MinMaxScaler
    scaled_train: np.ndarray
    scaled_test: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_prices: np.ndarray
    test_prices: np.ndarray
    window: int


def to_close_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Extract a clean, sorted close-price series from a DataFrame."""

    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        if "Close" not in data.columns:
            raise PreprocessingError("Input DataFrame must contain a 'Close' column.")
        series = data["Close"].copy()
    else:
        raise PreprocessingError("Input must be a pandas Series or DataFrame.")

    series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 2:
        raise PreprocessingError("At least two valid price observations are required.")
    if (series <= 0).any():
        raise PreprocessingError("Close prices must be strictly positive.")
    return series.astype(float)


def add_technical_indicators(
    data: pd.DataFrame,
    ma_windows: Sequence[int] = DEFAULT_MA_WINDOWS,
) -> pd.DataFrame:
    """Return a copy of OHLCV data enriched with moving averages and returns.

    New columns are named ``MA20``, ``MA50`` (per window), ``Daily Return`` and
    ``Pct Change``.  Missing values caused by indicator warm-up are preserved so
    charts can decide how to render them.
    """

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise PreprocessingError("Cannot add indicators to an empty DataFrame.")
    if "Close" not in data.columns:
        raise PreprocessingError("Input DataFrame must contain a 'Close' column.")

    frame = data.copy().sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")

    for window in ma_windows:
        if not isinstance(window, int) or window < 2:
            raise PreprocessingError("Moving-average windows must be integers >= 2.")
        if window > len(frame):
            frame[f"MA{window}"] = np.nan
        else:
            frame[f"MA{window}"] = close.rolling(window=window, min_periods=window).mean()

    frame["Daily Return"] = close.pct_change(fill_method=None)
    frame["Pct Change"] = frame["Daily Return"] * 100.0
    return frame


def chronological_split(
    series: pd.Series | np.ndarray,
    test_size: float = 0.2,
) -> tuple[pd.Series, pd.Series] | tuple[np.ndarray, np.ndarray]:
    """Split a series chronologically into train and test partitions.

    No shuffling is performed, which preserves temporal ordering and prevents
    look-ahead leakage.  ``test_size`` is the fraction reserved for testing.
    """

    if not 0.05 <= float(test_size) <= 0.5:
        raise PreprocessingError("test_size must be between 0.05 and 0.5.")

    n = len(series)
    if n < 10:
        raise PreprocessingError("At least 10 observations are required for a split.")
    test_n = max(2, int(round(n * float(test_size))))
    test_n = min(test_n, n - 2)
    split_at = n - test_n
    return series[:split_at], series[split_at:]


def make_sequences(values: np.ndarray | Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for supervised sequence learning.

    Returns
    -------
    X : ndarray, shape ``(n_samples, window, 1)``
    y : ndarray, shape ``(n_samples,)``
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise PreprocessingError("Sequence values must be finite and non-empty.")
    if not isinstance(window, int) or window < 2:
        raise PreprocessingError("window must be an integer >= 2.")
    if array.size <= window:
        raise PreprocessingError(
            f"Need more than {window} observations to build sequences; got {array.size}."
        )

    n_samples = array.size - window
    X = np.empty((n_samples, window, 1), dtype=float)
    y = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        X[idx, :, 0] = array[idx : idx + window]
        y[idx] = array[idx + window]
    return X, y


def scale_train_test(
    train_prices: np.ndarray | pd.Series,
    test_prices: np.ndarray | pd.Series,
) -> tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """Fit a MinMaxScaler on train data only and transform both partitions.

    Fitting on the training partition avoids optimistic leakage from future
    prices into the normalization range.
    """

    train = np.asarray(train_prices, dtype=float).reshape(-1)
    test = np.asarray(test_prices, dtype=float).reshape(-1)
    if train.size < 2:
        raise PreprocessingError("Training data must contain at least two prices.")
    if test.size == 0:
        raise PreprocessingError("Test data cannot be empty.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise PreprocessingError("Price arrays must contain only finite values.")
    if (train <= 0).any() or (test <= 0).any():
        raise PreprocessingError("Prices must be strictly positive.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
    scaled_test = scaler.transform(test.reshape(-1, 1)).reshape(-1)
    return scaler, scaled_train, scaled_test


def inverse_scale(scaler: MinMaxScaler, values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map scaled values back to the original price range."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    return scaler.inverse_transform(array.reshape(-1, 1)).reshape(-1)


def prepare_lstm_data(
    close_prices: pd.Series | Sequence[float],
    window: int = 60,
    test_size: float = 0.2,
) -> LSTMData:
    """Prepare scaled sliding-window arrays for LSTM training and testing.

    The scaler is fitted on the training partition only.  Test targets are
    predicted from windows that may begin in the training tail, but they never
    include the target observation itself.
    """

    if isinstance(close_prices, pd.DataFrame):
        series = to_close_series(close_prices)
    elif isinstance(close_prices, pd.Series):
        series = to_close_series(close_prices)
    else:
        array = np.asarray(close_prices, dtype=float).reshape(-1)
        if array.size < 10:
            raise PreprocessingError("At least 10 prices are required.")
        series = pd.Series(array)

    train, test = chronological_split(series, test_size=test_size)
    train_values = np.asarray(train, dtype=float)
    test_values = np.asarray(test, dtype=float)

    if len(train_values) <= window:
        raise PreprocessingError(
            f"Training split has {len(train_values)} rows but window={window}. "
            "Use a shorter window, less test data, or a longer history."
        )

    scaler, scaled_train, scaled_test = scale_train_test(train_values, test_values)
    X_train, y_train = make_sequences(scaled_train, window)

    # Prefix test windows with the final `window` training observations so every
    # test point has a full lookback.  Only past values are used.
    context = np.concatenate([scaled_train[-window:], scaled_test])
    X_test, y_test = make_sequences(context, window)
    if len(X_test) != len(test_values):  # pragma: no cover - invariant
        raise PreprocessingError("Internal error while aligning test sequences.")

    return LSTMData(
        scaler=scaler,
        scaled_train=scaled_train,
        scaled_test=scaled_test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_prices=train_values,
        test_prices=test_values,
        window=window,
    )


def future_dates(
    last_date: str | pd.Timestamp,
    steps: int,
    trading_days: bool = True,
) -> pd.DatetimeIndex:
    """Create future forecast dates after ``last_date``.

    ``trading_days=True`` produces business days (stocks); otherwise calendar
    days are used (crypto markets trade every day).
    """

    if not isinstance(steps, int) or steps < 1:
        raise PreprocessingError("Forecast horizon must be a positive integer.")
    if steps > 3650:
        raise PreprocessingError("Forecast horizon cannot exceed 3650 days.")
    last = pd.Timestamp(last_date).tz_localize(None).normalize()
    if trading_days:
        return pd.bdate_range(start=last + pd.Timedelta(days=1), periods=steps)
    return pd.date_range(start=last + pd.Timedelta(days=1), periods=steps, freq="D")
