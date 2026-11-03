"""Unit tests for preprocessing, scaling and sequence construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    PreprocessingError,
    add_technical_indicators,
    chronological_split,
    future_dates,
    inverse_scale,
    make_sequences,
    prepare_lstm_data,
    scale_train_test,
    to_close_series,
)


def price_series(rows: int = 120) -> pd.Series:
    index = pd.bdate_range("2022-01-03", periods=rows)
    values = np.linspace(50.0, 150.0, rows) + np.sin(np.arange(rows) / 5.0) * 3
    return pd.Series(values, index=index, name="Close")


def ohlcv_frame(rows: int = 80) -> pd.DataFrame:
    close = price_series(rows)
    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000, 5_000, rows),
        },
        index=close.index,
    )


class TestCloseSeries:
    def test_extracts_and_sorts_series(self) -> None:
        frame = ohlcv_frame(20).iloc[::-1]
        series = to_close_series(frame)
        assert series.is_monotonic_increasing
        assert len(series) == 20

    def test_drops_nan_and_duplicate_dates(self) -> None:
        series = price_series(10)
        series.iloc[2] = np.nan
        duplicated = pd.concat([series, series.iloc[[0]]])
        clean = to_close_series(duplicated)
        assert clean.notna().all()
        assert not clean.index.duplicated().any()

    def test_rejects_missing_close(self) -> None:
        with pytest.raises(PreprocessingError, match="Close"):
            to_close_series(pd.DataFrame({"Open": [1, 2, 3]}))

    def test_rejects_non_positive_prices(self) -> None:
        series = price_series(10)
        series.iloc[0] = -1
        with pytest.raises(PreprocessingError, match="positive"):
            to_close_series(series)


class TestIndicators:
    def test_moving_average_values(self) -> None:
        frame = add_technical_indicators(ohlcv_frame(60), ma_windows=(20, 50))
        assert "MA20" in frame.columns and "MA50" in frame.columns
        expected_ma20 = frame["Close"].iloc[:20].mean()
        assert frame["MA20"].iloc[19] == pytest.approx(expected_ma20)
        assert frame["MA20"].iloc[:19].isna().all()
        assert frame["MA50"].iloc[:49].isna().all()
        assert np.isclose(frame["MA50"].iloc[49], frame["Close"].iloc[:50].mean())

    def test_pct_change(self) -> None:
        frame = add_technical_indicators(ohlcv_frame(5))
        expected = (frame["Close"].iloc[1] / frame["Close"].iloc[0] - 1) * 100
        assert frame["Pct Change"].iloc[1] == pytest.approx(expected)

    def test_rejects_empty(self) -> None:
        with pytest.raises(PreprocessingError):
            add_technical_indicators(pd.DataFrame())


class TestSplit:
    def test_chronological_fraction(self) -> None:
        series = price_series(100)
        train, test = chronological_split(series, test_size=0.2)
        assert len(train) == 80
        assert len(test) == 20
        assert train.index[-1] < test.index[0]
        assert pd.concat([train, test]).equals(series)

    @pytest.mark.parametrize("bad_size", [0.0, 0.51, -0.1, 1.5])
    def test_invalid_test_size(self, bad_size: float) -> None:
        with pytest.raises(PreprocessingError):
            chronological_split(price_series(50), test_size=bad_size)


class TestSequences:
    def test_shapes_and_content(self) -> None:
        values = np.arange(10, dtype=float)
        X, y = make_sequences(values, window=4)
        assert X.shape == (6, 4, 1)
        assert y.shape == (6,)
        assert np.array_equal(X[0, :, 0], [0, 1, 2, 3])
        assert y[0] == 4
        assert np.array_equal(X[-1, :, 0], [5, 6, 7, 8])
        assert y[-1] == 9

    def test_rejects_short_series(self) -> None:
        with pytest.raises(PreprocessingError, match="more than"):
            make_sequences(np.arange(5), window=5)

    def test_rejects_bad_window(self) -> None:
        with pytest.raises(PreprocessingError):
            make_sequences(np.arange(10), window=1)


class TestScaling:
    def test_scaler_fits_train_only(self) -> None:
        train = np.array([100.0, 110.0, 120.0, 130.0])
        test = np.array([140.0, 500.0])
        scaler, scaled_train, scaled_test = scale_train_test(train, test)
        assert scaler.data_min_[0] == pytest.approx(100.0)
        assert scaler.data_max_[0] == pytest.approx(130.0)
        assert scaled_train.min() == pytest.approx(0.0)
        assert scaled_train.max() == pytest.approx(1.0)
        # Test values beyond the train range exceed 1 by design.
        assert scaled_test[1] > 1.0

    def test_inverse_scale_roundtrip(self) -> None:
        scaler, scaled, _ = scale_train_test(np.array([10.0, 20.0, 30.0]), np.array([25.0]))
        restored = inverse_scale(scaler, scaled)
        assert np.allclose(restored, [10.0, 20.0, 30.0])

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(PreprocessingError, match="positive"):
            scale_train_test(np.array([0.0, 1.0]), np.array([2.0]))


class TestPrepareLSTM:
    def test_shapes_and_no_leakage(self) -> None:
        series = price_series(200)
        data = prepare_lstm_data(series, window=30, test_size=0.2)

        assert len(data.train_prices) == 160
        assert len(data.test_prices) == 40
        assert data.X_train.shape == (130, 30, 1)
        assert data.X_test.shape == (40, 30, 1)
        assert len(data.y_test) == 40
        assert data.scaler.data_min_[0] == pytest.approx(data.train_prices.min())
        assert data.scaler.data_max_[0] == pytest.approx(data.train_prices.max())

        # The first test window ends on the final training observation.
        assert np.isclose(
            data.X_test[0, -1, 0],
            data.scaled_train[-1],
        )

    def test_rejects_window_larger_than_train(self) -> None:
        with pytest.raises(PreprocessingError, match="window"):
            prepare_lstm_data(price_series(40), window=60, test_size=0.2)


class TestFutureDates:
    def test_business_days(self) -> None:
        dates = future_dates(pd.Timestamp("2024-01-05"), steps=5, trading_days=True)
        assert len(dates) == 5
        assert dates[0] == pd.Timestamp("2024-01-08")
        assert all(day.weekday() < 5 for day in dates)

    def test_calendar_days(self) -> None:
        dates = future_dates(last_date="2024-01-05", steps=5, trading_days=False)
        assert list(dates) == list(pd.date_range("2024-01-06", periods=5, freq="D"))

    def test_rejects_bad_horizon(self) -> None:
        with pytest.raises(PreprocessingError):
            future_dates("2024-01-05", steps=0)
