"""LSTM price forecaster built with TensorFlow/Keras.

The implementation follows a leak-free workflow:

1. Split prices chronologically.
2. Fit the MinMaxScaler on the training partition only.
3. Train on sliding windows from the training partition.
4. Evaluate one-step-ahead predictions on the held-out test partition.
5. Generate future prices recursively, feeding each prediction back in.

TensorFlow is imported lazily so the rest of the dashboard (and the test suite)
can run even when TensorFlow is unavailable.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.evaluation import evaluate_forecast
from utils.preprocessing import (
    future_dates,
    inverse_scale,
    prepare_lstm_data,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - import availability depends on the environment
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    tf = None  # type: ignore[assignment]
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(exc)

ProgressCallback = Callable[[int, int, dict[str, float]], None]


class LSTMUnavailableError(RuntimeError):
    """Raised when TensorFlow cannot be imported."""


class LSTMForecastError(RuntimeError):
    """Raised when training or forecasting fails."""


@dataclass
class LSTMForecastResult:
    """Container returned by :func:`run_lstm_forecast`."""

    test_dates: pd.DatetimeIndex
    test_actual: np.ndarray
    test_predictions: np.ndarray
    future_dates: pd.DatetimeIndex
    future_predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    metrics: dict[str, Any]
    training_history: dict[str, list[float]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    epochs_trained: int = 0
    model_name: str = "LSTM"


def _require_tensorflow() -> None:
    if not TF_AVAILABLE:
        raise LSTMUnavailableError(
            "TensorFlow is not available in this environment. Install it with "
            "`pip install tensorflow` (Python 3.11-3.13) and restart the app. "
            f"Import error: {TF_IMPORT_ERROR}"
        )


def _set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.keras.utils.set_random_seed(seed)  # type: ignore[union-attr]


def _keras_progress_hook(
    callback: ProgressCallback | None,
    total_epochs: int,
) -> Any:
    """Create a Keras callback that forwards epoch events to plain Python."""

    class _ProgressCallback(tf.keras.callbacks.Callback):  # type: ignore[union-attr]
        def on_epoch_end(
            self,
            epoch: int,
            logs: dict[str, float] | None = None,
        ) -> None:
            if callback is not None:
                callback(epoch + 1, total_epochs, logs or {})

    return _ProgressCallback()


class LSTMForecaster:
    """Small, configurable stacked-LSTM regressor for closing prices."""

    def __init__(
        self,
        window: int = 60,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 40,
        patience: int = 8,
        random_state: int = 42,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2.")
        if units < 8:
            raise ValueError("units must be >= 8.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")

        _require_tensorflow()
        self.window = window
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.model: Any = None
        self.training_history: dict[str, list[float]] = {}

    def _build_model(self) -> Any:
        _set_seeds(self.random_state)
        model = tf.keras.Sequential(  # type: ignore[union-attr]
            [
                tf.keras.layers.Input(shape=(self.window, 1)),  # type: ignore[union-attr]
                tf.keras.layers.LSTM(
                    self.units,
                    return_sequences=True,
                    name="lstm_1",
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.LSTM(
                    max(16, self.units // 2),
                    return_sequences=False,
                    name="lstm_2",
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, name="close_price"),
            ],
            name="price_lstm",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),  # type: ignore[union-attr]
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        progress_callback: ProgressCallback | None = None,
    ) -> "LSTMForecaster":
        """Train the network on pre-built sliding-window arrays."""

        if X_train.ndim != 3 or X_train.shape[2] != 1:
            raise LSTMForecastError("X_train must have shape (samples, window, 1).")
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise LSTMForecastError("y_train must be 1-D and match X_train length.")
        if len(X_train) < 20:
            raise LSTMForecastError(
                "Not enough training sequences. Provide a longer price history or "
                "use a smaller window/test split."
            )

        self.model = self._build_model()
        validation_split = 0.1 if len(X_train) >= 80 else 0.0
        monitor = "val_loss" if validation_split else "loss"
        callbacks: list[Any] = [
            tf.keras.callbacks.EarlyStopping(  # type: ignore[union-attr]
                monitor=monitor,
                patience=self.patience,
                min_delta=1e-6,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        if progress_callback is not None:
            callbacks.append(_keras_progress_hook(progress_callback, self.epochs))

        try:
            history = self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                shuffle=False,  # never shuffle time-ordered sequences
                verbose=0,
                callbacks=callbacks,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.exception("LSTM training failed")
            raise LSTMForecastError(f"LSTM training failed: {exc}") from exc

        self.training_history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        return self

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        """Predict scaled values for input sequences."""

        if self.model is None:
            raise LSTMForecastError("The LSTM model must be fitted before prediction.")
        if len(X) == 0:
            return np.array([], dtype=float)
        return np.asarray(self.model.predict(X, verbose=0)).reshape(-1)

    def forecast(
        self,
        seed_prices: np.ndarray | pd.Series,
        scaler: Any,
        steps: int,
    ) -> np.ndarray:
        """Recursively forecast ``steps`` future prices.

        Each predicted value is appended to the lookback window and used as
        input for the next prediction.  Predictions are converted back to the
        original price scale with ``scaler``.
        """

        if self.model is None:
            raise LSTMForecastError("The LSTM model must be fitted before forecasting.")
        if steps < 1:
            raise ValueError("steps must be positive.")

        seed = np.asarray(seed_prices, dtype=float).reshape(-1)
        if seed.size < self.window:
            raise LSTMForecastError(
                f"Forecasting needs at least {self.window} seed prices; got {seed.size}."
            )

        scaled = scaler.transform(seed[-self.window :].reshape(-1, 1)).reshape(-1)
        window_values = scaled.copy()
        predictions_scaled: list[float] = []

        for _ in range(steps):
            x = window_values.reshape(1, self.window, 1)
            next_scaled = float(np.asarray(self.model(x, training=False)).reshape(-1)[0])
            predictions_scaled.append(next_scaled)
            window_values = np.append(window_values[1:], next_scaled)

        return inverse_scale(scaler, np.asarray(predictions_scaled))


def run_lstm_forecast(
    close_prices: pd.Series,
    horizon: int,
    window: int = 60,
    test_size: float = 0.2,
    epochs: int = 40,
    batch_size: int = 32,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    patience: int = 8,
    trading_days: bool = True,
    random_state: int = 42,
    progress_callback: ProgressCallback | None = None,
) -> LSTMForecastResult:
    """Run the complete leak-free LSTM training/evaluation/forecast pipeline."""

    _require_tensorflow()

    data = prepare_lstm_data(close_prices, window=window, test_size=test_size)
    if len(data.X_test) < 2:
        raise LSTMForecastError("Held-out test partition is too small.")

    forecaster = LSTMForecaster(
        window=window,
        units=units,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        random_state=random_state,
    )

    forecaster.fit(data.X_train, data.y_train, progress_callback=progress_callback)

    test_scaled_predictions = forecaster.predict_scaled(data.X_test)
    test_predictions = inverse_scale(data.scaler, test_scaled_predictions)
    metrics = evaluate_forecast(data.test_prices, test_predictions, model_name="LSTM")

    # Forecast from the most recent actual observations.  The model was trained
    # on the train split only, but using recent actual prices as the seed is
    # standard and does not leak into evaluation.
    full_prices = np.concatenate([data.train_prices, data.test_prices])
    future_predictions = forecaster.forecast(full_prices, data.scaler, horizon)
    future_predictions = np.clip(future_predictions, 0.0, None)

    dates = future_dates(
        last_date=close_prices.index[-1],
        steps=horizon,
        trading_days=trading_days,
    )

    # Approximate uncertainty band: held-out RMSE widened by sqrt(step), a
    # common random-walk-style heuristic.  It communicates model error without
    # pretending to be a calibrated probabilistic interval.
    rmse = float(metrics.get("rmse", 0.0))
    step_scale = np.sqrt(np.arange(1, horizon + 1, dtype=float))
    band = 1.96 * rmse * step_scale
    lower = np.clip(future_predictions - band, 0.0, None)
    upper = future_predictions + band

    history = {
        key: value[: forecaster.epochs]
        for key, value in forecaster.training_history.items()
    }
    epochs_trained = len(history.get("loss", []))

    return LSTMForecastResult(
        test_dates=pd.DatetimeIndex(close_prices.index[-len(data.test_prices) :]),
        test_actual=data.test_prices,
        test_predictions=test_predictions,
        future_dates=dates,
        future_predictions=future_predictions,
        lower_bound=lower,
        upper_bound=upper,
        metrics=metrics,
        training_history=history,
        config={
            "window": window,
            "test_size": test_size,
            "epochs": epochs,
            "epochs_trained": epochs_trained,
            "batch_size": batch_size,
            "units": units,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "patience": patience,
            "random_state": random_state,
        },
        epochs_trained=epochs_trained,
        model_name="LSTM",
    )
