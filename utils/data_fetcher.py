"""Market-data acquisition helpers built on top of ``yfinance``.

The module deliberately contains no Streamlit code.  UI-level caching is applied
in :mod:`app`, which keeps this module easy to unit-test and reuse.
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

STOCK_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "NFLX",
    "JPM",
    "V",
    "AMD",
    "SPY",
)

CRYPTO_TICKERS: tuple[str, ...] = (
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD",
    "AVAX-USD",
)

_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,19}$")


class DataFetchError(RuntimeError):
    """Raised when market data cannot be downloaded or validated."""


def normalize_ticker(ticker: str) -> str:
    """Return a cleaned, upper-case ticker symbol.

    Accepts stock symbols (``aapl``), class shares (``BRK-B``) and crypto pairs
    (``btc-usd``).  Raises :class:`ValueError` for empty or unsafe input.
    """

    if not isinstance(ticker, str):
        raise ValueError("Ticker must be a string.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol cannot be empty.")
    if not _TICKER_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid ticker symbol {ticker!r}. Use letters, numbers, '.', '-' "
            "or '=' (for example AAPL or BTC-USD)."
        )
    return cleaned


def infer_asset_type(ticker: str) -> str:
    """Infer ``"Crypto"`` or ``"Stock"`` from the ticker format."""

    symbol = normalize_ticker(ticker)
    if symbol.endswith("-USD") or symbol.endswith("USDT"):
        return "Crypto"
    return "Stock"


def validate_date_range(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize an inclusive historical date range."""

    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Start and end dates must be valid dates.")
    if start_ts >= end_ts:
        raise ValueError("Start date must be earlier than end date.")
    today = pd.Timestamp.today().normalize()
    if end_ts > today + pd.Timedelta(days=1):
        raise ValueError("End date cannot be in the future.")
    if start_ts > today:
        raise ValueError("Start date cannot be in the future.")
    if (end_ts - start_ts).days < 30:
        raise ValueError("Please select a date range of at least 30 days.")
    return start_ts, end_ts


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's optional ``(Price, Ticker)`` MultiIndex columns."""

    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance uses level 0 for the field when one ticker is requested.
        try:
            frame.columns = frame.columns.get_level_values(0)
        except IndexError:  # pragma: no cover - defensive
            frame.columns = ["_".join(map(str, col)) for col in frame.columns]
    frame.columns = [str(col).strip().title() for col in frame.columns]
    return frame


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data and guarantee a consistent schema.

    * Flattens MultiIndex columns returned by recent yfinance versions.
    * Coerces prices/volume to numeric and drops rows without a close.
    * Produces a timezone-naive, sorted, de-duplicated ``DatetimeIndex``.
    * Ensures all :data:`REQUIRED_COLUMNS` exist.

    Raises :class:`DataFetchError` when no usable observations remain.
    """

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataFetchError("No market data was returned for this ticker.")

    data = frame.copy()
    data = _flatten_columns(data)
    data.index = pd.to_datetime(data.index, utc=True, errors="coerce")
    data = data[~data.index.isna()]
    data.index = data.index.tz_convert(None).normalize()
    data = data[~data.index.duplicated(keep="last")].sort_index()

    if "Adj Close" in data.columns and "Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Close"})

    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            if column == "Volume":
                data[column] = 0.0
            else:
                raise DataFetchError(f"Market data is missing the '{column}' column.")

    data = data.loc[:, list(REQUIRED_COLUMNS)]
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data["Volume"] = data["Volume"].fillna(0.0).clip(lower=0.0)
    data = data[data["Close"] > 0]

    if data.empty:
        raise DataFetchError("Market data contained no valid closing prices.")
    if len(data) < 30:
        raise DataFetchError(
            f"Only {len(data)} trading days are available; at least 30 are required."
        )
    data.index.name = "Date"
    return data


def fetch_ohlcv(
    ticker: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    interval: str = "1d",
    timeout: int = 20,
) -> pd.DataFrame:
    """Download OHLCV history for ``ticker`` from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Stock (``AAPL``) or crypto (``BTC-USD``) symbol.
    start, end:
        Inclusive date bounds.
    interval:
        yfinance interval; the dashboard uses daily bars.
    timeout:
        Per-request network timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Indexed by date with columns ``Open, High, Low, Close, Volume``.

    Raises
    ------
    DataFetchError
        For invalid symbols, empty responses and network/API failures.
    """

    symbol = normalize_ticker(ticker)
    start_ts, end_ts = validate_date_range(start, end)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataFetchError(
            "yfinance is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    # yfinance's `end` is exclusive, so request one extra calendar day.
    request_end = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    LOGGER.info("Downloading %s from %s to %s", symbol, start_ts.date(), end_ts.date())

    try:
        with warnings.catch_warnings():
            # yfinance uses deprecated pandas in-place operations internally.
            warnings.filterwarnings(
                "ignore", message=".*inplace keyword.*", category=Warning
            )
            raw = yf.download(
                tickers=symbol,
                start=start_ts.date().isoformat(),
                end=request_end,
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=timeout,
            )
    except Exception as exc:
        LOGGER.exception("yfinance download failed for %s", symbol)
        raise DataFetchError(
            f"Could not download data for {symbol}. Check the symbol and your "
            "internet connection, then try again."
        ) from exc

    if raw is None or len(raw) == 0:
        raise DataFetchError(
            f"No data found for {symbol!r}. Verify the symbol (for example AAPL "
            "or BTC-USD) and the selected date range."
        )

    data = normalize_ohlcv(raw)
    data = data.loc[(data.index >= start_ts) & (data.index <= end_ts)]
    if data.empty:
        raise DataFetchError(
            f"No observations for {symbol!r} fall inside the selected date range."
        )
    if len(data) < 30:
        raise DataFetchError(
            f"Only {len(data)} trading days are available in this range. "
            "Choose a longer range (at least 30 days)."
        )
    return data


def fetch_asset_info(ticker: str) -> dict[str, Any]:
    """Return lightweight metadata for a ticker, with safe fallbacks.

    Network/metadata failures never prevent the dashboard from rendering; a
    best-effort dictionary is returned instead.
    """

    symbol = normalize_ticker(ticker)
    fallback = {
        "symbol": symbol,
        "name": symbol,
        "currency": "USD",
        "exchange": "",
        "quote_type": infer_asset_type(symbol),
    }
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info() or {}
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Could not load metadata for %s: %s", symbol, exc)
        return fallback

    name = info.get("longName") or info.get("shortName") or symbol
    quote_type = info.get("quoteType") or infer_asset_type(symbol)
    return {
        "symbol": symbol,
        "name": name,
        "currency": info.get("currency") or "USD",
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "quote_type": quote_type,
    }


def last_close(data: pd.DataFrame) -> float:
    """Return the most recent close from a normalized OHLCV frame."""

    if data is None or data.empty or "Close" not in data:
        raise DataFetchError("Cannot read the latest price from an empty dataset.")
    return float(data["Close"].iloc[-1])


def daily_change_pct(data: pd.DataFrame) -> float:
    """Return the latest close-to-close percentage change."""

    if data is None or len(data) < 2:
        return 0.0
    previous = float(data["Close"].iloc[-2])
    if previous == 0:
        return 0.0
    return (float(data["Close"].iloc[-1]) / previous - 1.0) * 100.0


def default_date_range(years: int = 5) -> tuple[date, date]:
    """Return a convenient historical range ending today."""

    end = date.today()
    start = end - timedelta(days=max(1, years) * 365)
    return start, end
