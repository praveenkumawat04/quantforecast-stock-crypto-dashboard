"""Unit tests for the yfinance data-acquisition layer.

No network calls are made: ``yfinance.download`` is patched with deterministic
DataFrames.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.data_fetcher import (
    DataFetchError,
    daily_change_pct,
    fetch_ohlcv,
    infer_asset_type,
    last_close,
    normalize_ohlcv,
    normalize_ticker,
    validate_date_range,
)


def make_ohlcv(rows: int = 80, start: str = "2024-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start=start, periods=rows)
    base = np.linspace(100.0, 140.0, rows)
    return pd.DataFrame(
        {
            "Open": base - 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Adj Close": base,
            "Volume": np.arange(rows, dtype=float) * 1_000 + 100_000,
        },
        index=index,
    )


class TestTickerValidation:
    def test_normalizes_common_symbols(self) -> None:
        assert normalize_ticker(" aapl ") == "AAPL"
        assert normalize_ticker("btc-usd") == "BTC-USD"
        assert normalize_ticker("BRK-B") == "BRK-B"

    @pytest.mark.parametrize("bad", ["", "   ", "AAPL; DROP TABLE", "A B", "x" * 30])
    def test_rejects_invalid_symbols(self, bad: str) -> None:
        with pytest.raises(ValueError):
            normalize_ticker(bad)

    def test_rejects_non_string(self) -> None:  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            normalize_ticker(123)  # type: ignore[arg-type]

    def test_infers_asset_type(self) -> None:
        assert infer_asset_type("BTC-USD") == "Crypto"
        assert infer_asset_type("eth-usd") == "Crypto"
        assert infer_asset_type("AAPL") == "Stock"


class TestDateValidation:
    def test_accepts_valid_range(self) -> None:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=400)
        start_ts, end_ts = validate_date_range(start, end)
        assert start_ts < end_ts
        assert end_ts.date() == end

    def test_rejects_reversed_range(self) -> None:
        with pytest.raises(ValueError, match="earlier"):
            validate_date_range(date.today(), date.today() - timedelta(days=100))

    def test_rejects_future_end(self) -> None:
        with pytest.raises(ValueError, match="future"):
            validate_date_range(
                date.today() - timedelta(days=100),
                date.today() + timedelta(days=10),
            )

    def test_rejects_too_short_range(self) -> None:
        end = date.today() - timedelta(days=5)
        with pytest.raises(ValueError, match="at least 30"):
            validate_date_range(end - timedelta(days=10), end)


class TestNormalizeOHLCV:
    def test_sorts_deduplicates_and_selects_columns(self) -> None:
        raw = make_ohlcv(35)
        raw = pd.concat([raw, raw.iloc[[0]]])  # duplicate first date
        raw = raw.iloc[::-1]  # descending order
        clean = normalize_ohlcv(raw)

        assert list(clean.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert clean.index.is_monotonic_increasing
        assert not clean.index.duplicated().any()
        assert len(clean) == 35

    def test_flattens_multiindex_columns(self) -> None:
        raw = make_ohlcv(40)
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
        clean = normalize_ohlcv(raw)
        assert "Close" in clean.columns
        assert clean["Close"].notna().all()

    def test_fills_missing_volume(self) -> None:
        raw = make_ohlcv(40).drop(columns=["Volume"])
        clean = normalize_ohlcv(raw)
        assert (clean["Volume"] == 0).all()

    def test_drops_rows_with_missing_close(self) -> None:
        raw = make_ohlcv(40)
        raw.loc[raw.index[5], "Close"] = np.nan
        clean = normalize_ohlcv(raw)
        assert len(clean) == 39

    def test_rejects_empty_or_tiny_frames(self) -> None:
        with pytest.raises(DataFetchError):
            normalize_ohlcv(pd.DataFrame())
        with pytest.raises(DataFetchError, match="at least 30"):
            normalize_ohlcv(make_ohlcv(10))

    def test_rejects_missing_close(self) -> None:
        raw = make_ohlcv(40).drop(columns=["Close", "Adj Close"])
        with pytest.raises(DataFetchError, match="Close"):
            normalize_ohlcv(raw)


class TestFetchOHLCV:
    def test_returns_normalized_data(self) -> None:
        raw = make_ohlcv(120)
        with patch("yfinance.download", return_value=raw) as download:
            result = fetch_ohlcv("AAPL", "2024-01-02", "2024-06-30")
        assert not result.empty
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert download.call_count == 1
        assert download.call_args.kwargs["tickers"] == "AAPL"

    def test_raises_for_empty_response(self) -> None:
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(DataFetchError, match="No data"):
                fetch_ohlcv("FAKE", "2024-01-02", "2024-06-30")

    def test_raises_for_download_exception(self) -> None:
        with patch("yfinance.download", side_effect=ConnectionError("offline")):
            with pytest.raises(DataFetchError, match="Could not download"):
                fetch_ohlcv("AAPL", "2024-01-02", "2024-06-30")

    def test_invalid_ticker_is_rejected_before_download(self) -> None:
        with patch("yfinance.download") as download:
            with pytest.raises(ValueError):
                fetch_ohlcv("BAD TICKER", "2024-01-02", "2024-06-30")
        download.assert_not_called()


def test_last_close_and_daily_change() -> None:
    data = make_ohlcv(30)
    data.loc[data.index[-2], "Close"] = 100.0
    data.loc[data.index[-1], "Close"] = 110.0
    assert last_close(data) == pytest.approx(110.0)
    assert daily_change_pct(data) == pytest.approx(10.0)

    with pytest.raises(DataFetchError):
        last_close(pd.DataFrame())
