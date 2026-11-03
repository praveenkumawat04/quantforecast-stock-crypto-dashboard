"""Stock & Crypto Price Prediction Dashboard.

Run with:
    streamlit run app.py

The UI layer intentionally delegates data access, preprocessing, modelling and
chart construction to the modules under ``utils/`` and ``models/``.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from html import escape as html_escape
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from models.lstm_model import (
    TF_AVAILABLE,
    LSTMForecastError,
    LSTMUnavailableError,
    run_lstm_forecast,
)
from models.prophet_model import (
    PROPHET_AVAILABLE,
    ProphetForecastError,
    ProphetUnavailableError,
    run_prophet_forecast,
)
from utils.data_fetcher import (
    CRYPTO_TICKERS,
    STOCK_TICKERS,
    DataFetchError,
    daily_change_pct,
    fetch_asset_info,
    fetch_ohlcv,
    infer_asset_type,
    last_close,
    normalize_ticker,
    validate_date_range,
)
from utils.evaluation import EvaluationError
from utils.preprocessing import PreprocessingError, add_technical_indicators
from utils.visualization import (
    COLORS,
    actual_vs_predicted_chart,
    candlestick_chart,
    forecast_chart,
    price_history_chart,
    training_loss_chart,
    volume_chart,
)

# ---------------------------------------------------------------------------
# Page configuration & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantForecast · Stock & Crypto Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {
            --bg: #070d18;
            --panel: #0d1626;
            --panel-2: #111d31;
            --border: #1e2c44;
            --text: #e6edf7;
            --muted: #8b9bb4;
            --cyan: #22d3ee;
            --green: #16c784;
            --red: #ea3943;
            --amber: #f59e0b;
            --violet: #a78bfa;
        }

        html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .stApp {
            background:
                radial-gradient(1100px 600px at 82% -12%, rgba(34, 211, 238, 0.10), transparent 62%),
                radial-gradient(900px 520px at 12% 4%, rgba(167, 139, 250, 0.07), transparent 60%),
                #070d18;
            color: var(--text);
        }
        .block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 2.4rem; }
        header[data-testid="stHeader"] { background: rgba(7, 13, 24, 0.7); backdrop-filter: blur(12px); }
        #MainMenu, footer { visibility: hidden; }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1322 0%, #08101d 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] label { color: #c7d3e5 !important; font-weight: 500; }
        section[data-testid="stSidebar"] hr { border-color: var(--border); }

        .brand {
            display: flex; align-items: center; gap: 10px;
            padding: 0.2rem 0 0.9rem 0;
        }
        .brand-mark {
            width: 38px; height: 38px; border-radius: 11px;
            background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 55%, #a78bfa 100%);
            display: flex; align-items: center; justify-content: center;
            color: #041018; font-weight: 800; font-size: 18px;
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.22);
        }
        .brand-name { font-size: 16px; font-weight: 800; letter-spacing: 0.02em; color: var(--text); line-height: 1.1; }
        .brand-sub { font-size: 10.5px; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }

        .hero {
            position: relative;
            padding: 1.5rem 1.7rem 1.35rem 1.7rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background:
                linear-gradient(120deg, rgba(34, 211, 238, 0.08), rgba(13, 22, 38, 0.72) 45%, rgba(167, 139, 250, 0.07)),
                #0b1424;
            overflow: hidden;
            margin-bottom: 1.15rem;
        }
        .hero::after {
            content: ""; position: absolute; right: -90px; top: -120px;
            width: 320px; height: 320px; border-radius: 50%;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.18), transparent 68%);
            pointer-events: none;
        }
        .hero-eyebrow {
            color: var(--cyan); font-size: 10.5px; font-weight: 700;
            letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.45rem;
        }
        .hero h1 {
            font-size: 2.05rem; line-height: 1.12; font-weight: 800;
            margin: 0 0 0.4rem 0; color: #f8fafc; letter-spacing: -0.025em;
        }
        .hero h1 .grad {
            background: linear-gradient(90deg, #22d3ee, #60a5fa 55%, #a78bfa);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .hero p { color: var(--muted); margin: 0; font-size: 13.5px; max-width: 760px; line-height: 1.55; }
        .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.9rem; }
        .chip {
            font-size: 10.5px; font-weight: 600; letter-spacing: 0.04em;
            color: #c7d3e5; padding: 4px 10px; border-radius: 999px;
            border: 1px solid var(--border); background: rgba(17, 29, 49, 0.8);
        }
        .chip.live { color: #16c784; border-color: rgba(22, 199, 132, 0.35); background: rgba(22, 199, 132, 0.08); }
        .chip .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #16c784; margin-right: 6px; }

        .status-line {
            display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
            font-size: 12px; color: var(--muted); margin: 0.25rem 0 1rem 0;
        }
        .status-line strong { color: #dbe6f5; font-weight: 600; }
        .status-line .sep { color: #30425f; }
        .status-line .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 4px rgba(22,199,132,0.12); }

        .kpi-card {
            position: relative; overflow: hidden;
            border: 1px solid var(--border); border-radius: 16px;
            background: linear-gradient(150deg, rgba(17, 29, 49, 0.92), rgba(10, 18, 32, 0.92));
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            min-height: 118px;
        }
        .kpi-card::before {
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
            background: var(--accent, #22d3ee);
        }
        .kpi-label { color: var(--muted); font-size: 10.5px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; }
        .kpi-value { color: #f8fafc; font-size: 1.72rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 0.38rem; font-variant-numeric: tabular-nums; }
        .kpi-foot { font-size: 11.5px; color: var(--muted); margin-top: 0.32rem; }
        .up { color: var(--green) !important; }
        .down { color: var(--red) !important; }
        .neutral { color: var(--amber) !important; }

        .model-card {
            border: 1px solid var(--border); border-radius: 16px;
            background: linear-gradient(150deg, rgba(17, 29, 49, 0.9), rgba(10, 18, 32, 0.9));
            padding: 1rem 1.1rem 1rem 1.1rem; height: 100%;
        }
        .model-card .name { font-size: 13px; font-weight: 700; color: #f8fafc; letter-spacing: 0.02em; }
        .model-card .desc { font-size: 11px; color: var(--muted); margin-top: 2px; }
        .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; margin-top: 0.85rem; }
        .metric-box { background: rgba(7, 13, 24, 0.55); border: 1px solid #1a2940; border-radius: 10px; padding: 0.55rem 0.65rem; }
        .metric-box .m-label { font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
        .metric-box .m-value { font-size: 1.08rem; font-weight: 700; color: #e6edf7; margin-top: 2px; font-variant-numeric: tabular-nums; }

        .section-title { font-size: 15px; font-weight: 700; color: #e6edf7; margin: 0.6rem 0 0.25rem 0; }
        .section-sub { font-size: 12px; color: var(--muted); margin-bottom: 0.75rem; }

        .panel {
            border: 1px solid var(--border); border-radius: 16px;
            background: rgba(13, 22, 38, 0.72); padding: 1rem 1.1rem;
        }
        .note {
            border: 1px solid rgba(34, 211, 238, 0.18); border-left: 3px solid var(--cyan);
            background: rgba(34, 211, 238, 0.055); border-radius: 10px;
            padding: 0.7rem 0.9rem; color: #a9bdd6; font-size: 11.8px; line-height: 1.55;
        }
        .warning-note {
            border: 1px solid rgba(245, 158, 11, 0.22); border-left: 3px solid var(--amber);
            background: rgba(245, 158, 11, 0.055); border-radius: 10px;
            padding: 0.7rem 0.9rem; color: #c9b28a; font-size: 11.8px; line-height: 1.55;
        }
        .disclaimer {
            margin-top: 1.4rem; padding: 0.95rem 1.1rem;
            border: 1px solid rgba(245, 158, 11, 0.18); border-radius: 12px;
            background: rgba(245, 158, 11, 0.04); color: #9aa9be; font-size: 11.5px; line-height: 1.6;
        }
        .disclaimer strong { color: #f59e0b; }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 0; }
        .stTabs [data-baseweb="tab"] {
            background: transparent; color: var(--muted); border-radius: 9px 9px 0 0;
            padding: 9px 16px; font-size: 12.5px; font-weight: 600;
        }
        .stTabs [aria-selected="true"] { color: #f8fafc !important; background: rgba(34, 211, 238, 0.07) !important; border-bottom: 2px solid var(--cyan) !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }

        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stDownloadButton"] button {
            border-radius: 10px; font-weight: 700; font-size: 12.5px;
            border: 1px solid rgba(34, 211, 238, 0.4);
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(92deg, #22d3ee 0%, #38bdf8 48%, #60a5fa 100%);
            color: #041018; border: 0;
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.18);
        }
        div[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.06); }

        div[data-testid="stSelectbox"] > div, div[data-testid="stDateInput"] > div,
        div[data-testid="stNumberInput"] > div {
            background: #0d1626; border-color: var(--border); border-radius: 9px;
        }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; background: rgba(13, 22, 38, 0.55); }
        .stAlert { border-radius: 12px; }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #22d3ee, #a78bfa); }
        .small-muted { color: var(--muted); font-size: 11.5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------------------------------------------------------------------------
# Cached data access (Streamlit caching lives in the UI layer)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Cached OHLCV download.  TTL keeps intraday prices reasonably fresh."""

    return fetch_ohlcv(ticker, start, end)


@st.cache_data(ttl=86_400, show_spinner=False)
def load_asset_info(ticker: str) -> dict[str, Any]:
    """Cached ticker metadata (name/exchange/currency)."""

    return fetch_asset_info(ticker)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def fmt_money(value: float | None, currency: str = "USD") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if currency.upper() == "USD":
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency.upper()}"


def fmt_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.2f}%"


def delta_class(value: float | None) -> str:
    if value is None or not np.isfinite(value) or abs(value) < 1e-12:
        return "neutral"
    return "up" if value > 0 else "down"


def kpi_card(
    label: str,
    value: str,
    footer: str,
    accent: str,
    footer_class: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {footer_class}">{value}</div>
            <div class="kpi-foot {footer_class}">{footer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(name: str, metrics: dict[str, Any], description: str) -> None:
    mae = metrics.get("mae", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    mape = metrics.get("mape", float("nan"))
    direction = metrics.get("directional_accuracy", float("nan"))

    def fmt(value: float, suffix: str = "") -> str:
        return f"{value:,.2f}{suffix}" if np.isfinite(value) else "N/A"

    st.markdown(
        f"""
        <div class="model-card">
            <div class="name">{name}</div>
            <div class="desc">{description}</div>
            <div class="metric-grid">
                <div class="metric-box"><div class="m-label">MAE</div><div class="m-value">{fmt(mae)}</div></div>
                <div class="metric-box"><div class="m-label">RMSE</div><div class="m-value">{fmt(rmse)}</div></div>
                <div class="metric-box"><div class="m-label">MAPE</div><div class="m-value">{fmt(mape, "%")}</div></div>
                <div class="metric-box"><div class="m-label">Direction</div><div class="m-value">{fmt(direction, "%")}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def forecast_table(result: Any) -> pd.DataFrame:
    """Build a presentable forecast table for either model."""

    return pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(result.future_dates).strftime("%Y-%m-%d"),
            "Forecast": np.round(result.future_predictions, 2),
            "Lower Bound": np.round(result.lower_bound, 2),
            "Upper Bound": np.round(result.upper_bound, 2),
        }
    )


def combined_forecast_csv(results: dict[str, Any], ticker: str) -> bytes:
    frames = []
    for name, result in results.items():
        frame = forecast_table(result)
        frame.insert(0, "Model", name)
        frame.insert(1, "Ticker", ticker)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False).encode("utf-8")


def model_description(name: str) -> str:
    if name == "LSTM":
        return "Stacked LSTM · recursive multi-step forecast"
    return "Trend + seasonality · probabilistic intervals"


def run_forecasts(
    models: list[str],
    close: pd.Series,
    horizon: int,
    trading_days: bool,
    lstm_config: dict[str, Any],
    prophet_config: dict[str, Any],
    progress_bar: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Train each selected model, reporting progress through the UI."""

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    total = len(models)

    for index, name in enumerate(models):
        base = index / total
        span = 1.0 / total
        progress_bar.progress(min(base + 0.01, 1.0), text=f"Preparing {name}…")

        if name == "LSTM":

            def lstm_progress(
                epoch: int,
                total_epochs: int,
                logs: dict[str, float],
                _base: float = base,
                _span: float = span,
            ) -> None:
                fraction = _base + _span * 0.96 * (epoch / max(total_epochs, 1))
                loss = logs.get("loss")
                val_loss = logs.get("val_loss")
                detail = f"loss {loss:.5f}" if loss is not None else "training"
                if val_loss is not None:
                    detail += f" · val_loss {val_loss:.5f}"
                progress_bar.progress(
                    min(fraction, 0.995),
                    text=f"{name} · epoch {epoch}/{total_epochs} · {detail}",
                )

            try:
                results[name] = run_lstm_forecast(
                    close_prices=close,
                    horizon=horizon,
                    trading_days=trading_days,
                    progress_callback=lstm_progress,
                    **lstm_config,
                )
            except (LSTMUnavailableError, LSTMForecastError, PreprocessingError, EvaluationError) as exc:
                errors[name] = str(exc)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                errors[name] = f"Unexpected LSTM error: {exc}"
        else:

            def prophet_progress(
                message: str,
                fraction: float,
                _base: float = base,
                _span: float = span,
            ) -> None:
                progress_bar.progress(
                    min(_base + _span * fraction, 0.995),
                    text=f"{name} · {message}",
                )

            try:
                results[name] = run_prophet_forecast(
                    close_prices=close,
                    horizon=horizon,
                    trading_days=trading_days,
                    progress_callback=prophet_progress,
                    **prophet_config,
                )
            except (ProphetUnavailableError, ProphetForecastError, PreprocessingError, EvaluationError) as exc:
                errors[name] = str(exc)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                errors[name] = f"Unexpected Prophet error: {exc}"

    progress_bar.progress(1.0, text="Forecast complete")
    return results, errors


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div>
                <div class="brand-name">QuantForecast</div>
                <div class="brand-sub">Market ML Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("##### Configuration")

    asset_type = st.radio(
        "Asset class",
        options=["Stocks", "Crypto"],
        horizontal=True,
        help="Stock symbols use trading days; crypto pairs trade every day.",
    )

    preset_tickers = list(STOCK_TICKERS if asset_type == "Stocks" else CRYPTO_TICKERS)
    ticker_choice = st.selectbox(
        "Ticker symbol",
        options=preset_tickers + ["Custom ticker…"],
        index=0,
        key=f"ticker_choice_{asset_type}",
        help="Pick a liquid example or enter any Yahoo Finance symbol.",
    )

    if ticker_choice == "Custom ticker…":
        raw_ticker = st.text_input(
            "Yahoo Finance symbol",
            value="",
            placeholder="e.g. NVDA, BRK-B, SOL-USD",
            key=f"custom_ticker_{asset_type}",
        )
    else:
        raw_ticker = ticker_choice

    try:
        ticker = normalize_ticker(raw_ticker) if raw_ticker else None
    except ValueError as exc:
        ticker = None
        st.error(str(exc))

    today = date.today()
    preset = st.selectbox(
        "Historical range",
        options=["1Y", "2Y", "5Y", "10Y", "Year to date", "Custom"],
        index=2,
    )
    if preset == "Custom":
        default_start = today - timedelta(days=365 * 2)
        start_date = st.date_input(
            "Start date",
            value=default_start,
            max_value=today,
            key="custom_start",
        )
        end_date = st.date_input(
            "End date",
            value=today,
            max_value=today,
            key="custom_end",
        )
    else:
        if preset == "Year to date":
            start_date = date(today.year, 1, 1)
        else:
            years = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10}[preset]
            start_date = today - timedelta(days=365 * years)
        end_date = today

    horizon = st.select_slider(
        "Forecast horizon",
        options=[7, 14, 30, 90],
        value=30,
        format_func=lambda days: f"{days} days",
        help="Every horizon is generated recursively or with Prophet's future frame.",
    )

    model_options: list[str] = []
    if TF_AVAILABLE:
        model_options.append("LSTM")
    if PROPHET_AVAILABLE:
        model_options.append("Prophet")
    if TF_AVAILABLE and PROPHET_AVAILABLE:
        model_options.append("Compare both")
    if not model_options:
        model_options = ["No model available"]

    model_choice = st.selectbox(
        "Forecasting model",
        options=model_options,
        index=0,
        help="Compare both trains each model on the same chronological split.",
    )

    with st.expander("Model settings", expanded=False):
        test_size_pct = st.slider(
            "Held-out test size",
            min_value=10,
            max_value=35,
            value=20,
            step=5,
            format="%d%%",
            help="Chronological hold-out used for MAE/RMSE/MAPE. Never shuffled.",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Seeds Python, NumPy and TensorFlow/Prophet for reproducibility.",
        )

        st.markdown("**LSTM settings**")
        lstm_window = st.slider(
            "Lookback window",
            min_value=20,
            max_value=120,
            value=60,
            step=10,
            help="Number of prior trading days fed to each sequence.",
        )
        lstm_epochs = st.slider("Training epochs", 5, 80, 30, 5)
        lstm_units = st.select_slider("LSTM units", options=[32, 48, 64, 96, 128], value=64)
        lstm_batch = st.select_slider("Batch size", options=[16, 32, 64], value=32)

        st.markdown("**Prophet settings**")
        changepoint_prior = st.slider(
            "Changepoint flexibility",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.005,
            format="%.3f",
        )
        interval_width = st.slider(
            "Interval width",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            format="%.0f%%",
            help="Width of Prophet's uncertainty interval.",
        )
        seasonality_mode = st.selectbox(
            "Seasonality mode",
            options=["additive", "multiplicative"],
            index=0,
        )

    st.markdown("---")
    run_clicked = st.button("🚀  Run Forecast", type="primary", width="stretch")

    availability = []
    if TF_AVAILABLE:
        availability.append("TensorFlow ready")
    else:
        availability.append("TensorFlow unavailable")
    if PROPHET_AVAILABLE:
        availability.append("Prophet ready")
    else:
        availability.append("Prophet unavailable")
    st.caption(" · ".join(availability))
    st.caption("Data via Yahoo Finance · cached 15 min")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Market intelligence · deep learning + statistical forecasting</div>
        <h1>Stock &amp; Crypto <span class="grad">Price Prediction</span></h1>
        <p>
            Explore live market history, train leak-free LSTM networks and Prophet models,
            evaluate held-out accuracy, and inspect future scenarios with interactive charts.
        </p>
        <div class="chips">
            <span class="chip live"><span class="dot"></span>Yahoo Finance</span>
            <span class="chip">TensorFlow · LSTM</span>
            <span class="chip">Prophet · intervals</span>
            <span class="chip">MAE / RMSE / MAPE</span>
            <span class="chip">CSV export</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TF_AVAILABLE and not PROPHET_AVAILABLE:
    st.error(
        "Neither TensorFlow nor Prophet is available. Install the project requirements "
        "and restart the dashboard."
    )
    st.stop()

if ticker is None:
    st.info("Select a ticker symbol in the sidebar to load market data.")
    st.stop()

try:
    start_ts, end_ts = validate_date_range(start_date, end_date)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

try:
    with st.spinner(f"Loading {ticker} market data…"):
        market_data = load_market_data(ticker, start_date, end_date)
except DataFetchError as exc:
    st.error(f"Market data error: {exc}")
    st.info(
        "Try a different symbol or a longer date range. Examples: AAPL, MSFT, "
        "TSLA, BTC-USD, ETH-USD."
    )
    st.stop()
except Exception as exc:  # pragma: no cover - network boundary
    st.error(f"Unexpected data error: {exc}")
    st.stop()

try:
    asset_info = load_asset_info(ticker)
except Exception:  # pragma: no cover - metadata is optional
    asset_info = {
        "symbol": ticker,
        "name": ticker,
        "currency": "USD",
        "exchange": "",
        "quote_type": infer_asset_type(ticker),
    }

try:
    enriched = add_technical_indicators(market_data, ma_windows=(20, 50))
except PreprocessingError as exc:
    st.error(f"Could not prepare indicators: {exc}")
    enriched = market_data

currency = asset_info.get("currency", "USD")
trading_days = infer_asset_type(ticker) == "Stock"
close_series = market_data["Close"]
latest_price = last_close(market_data)
day_change = daily_change_pct(market_data)
last_observation = market_data.index[-1].strftime("%b %d, %Y")

asset_label = f"{asset_info.get('name', ticker)} · {ticker}"
st.markdown(
    f"""
    <div class="status-line">
        <span class="dot"></span>
        <strong>{html_escape(str(asset_label))}</strong>
        <span class="sep">|</span>
        <span>{html_escape(str(asset_info.get('exchange') or asset_info.get('quote_type', '')))}</span>
        <span class="sep">|</span>
        <span>{len(market_data):,} observations</span>
        <span class="sep">|</span>
        <span>Last bar {last_observation}</span>
        <span class="sep">|</span>
        <span>{html_escape(str(currency))}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run signature and cached result handling
# ---------------------------------------------------------------------------

model_map = {
    "LSTM": ["LSTM"],
    "Prophet": ["Prophet"],
    "Compare both": ["LSTM", "Prophet"],
}
selected_models = model_map[model_choice]

lstm_config: dict[str, Any] = {
    "window": int(lstm_window),
    "test_size": test_size_pct / 100.0,
    "epochs": int(lstm_epochs),
    "batch_size": int(lstm_batch),
    "units": int(lstm_units),
    "dropout": 0.2,
    "learning_rate": 0.001,
    "patience": 8,
    "random_state": int(random_state),
}
prophet_config: dict[str, Any] = {
    "test_size": test_size_pct / 100.0,
    "changepoint_prior_scale": float(changepoint_prior),
    "seasonality_mode": seasonality_mode,
    "interval_width": float(interval_width),
    "yearly_seasonality": "auto",
    "weekly_seasonality": "auto",
    "daily_seasonality": False,
    "random_state": int(random_state),
}

signature = (
    ticker,
    str(start_date),
    str(end_date),
    int(horizon),
    model_choice,
    json.dumps(lstm_config, sort_keys=True),
    json.dumps(prophet_config, sort_keys=True),
)

if "forecast_bundle" not in st.session_state:
    st.session_state["forecast_bundle"] = None

active_bundle = None
bundle = st.session_state.get("forecast_bundle")
if bundle is not None and bundle.get("signature") == signature:
    active_bundle = bundle

if run_clicked:
    with st.status("Running forecasting pipeline…", expanded=True) as status:
        progress_bar = st.progress(0.02, text="Preparing data…")
        try:
            results, errors = run_forecasts(
                models=selected_models,
                close=close_series,
                horizon=int(horizon),
                trading_days=trading_days,
                lstm_config=lstm_config,
                prophet_config=prophet_config,
                progress_bar=progress_bar,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            results, errors = {}, {"pipeline": f"Pipeline failed: {exc}"}

        progress_bar.empty()
        if results:
            status.update(
                label=f"Forecast complete · {', '.join(results)}",
                state="complete",
                expanded=bool(errors),
            )
        else:
            status.update(label="Forecast failed", state="error", expanded=True)

    if errors:
        for name, message in errors.items():
            st.error(f"{name}: {message}")

    if results:
        bundle = {
            "signature": signature,
            "results": results,
            "errors": errors,
            "ticker": ticker,
            "horizon": int(horizon),
            "trading_days": trading_days,
            "currency": currency,
            "latest_price": latest_price,
        }
        st.session_state["forecast_bundle"] = bundle
        active_bundle = bundle
        st.toast("Forecast ready — explore the tabs below.", icon="✅")
    else:
        st.session_state["forecast_bundle"] = None
        active_bundle = None
        bundle = None

if bundle is not None and active_bundle is None:
    st.info("Settings changed since the last run. Run the forecast again to refresh results.")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

results: dict[str, Any] = active_bundle["results"] if active_bundle else {}
primary_name = next(iter(results)) if results else None
primary_result = results[primary_name] if primary_name else None

predicted_price: float | None = None
expected_change: float | None = None
if primary_result is not None:
    predicted_price = float(primary_result.future_predictions[-1])
    if latest_price:
        expected_change = (predicted_price / latest_price - 1.0) * 100.0

kpi_cols = st.columns(4, gap="small")
with kpi_cols[0]:
    kpi_card(
        "Latest close",
        fmt_money(latest_price, currency),
        f"{last_observation} · {asset_info.get('quote_type', '')}",
        COLORS["accent"],
    )
with kpi_cols[1]:
    kpi_card(
        "Daily change",
        fmt_pct(day_change),
        "vs. previous close",
        COLORS["green"] if day_change >= 0 else COLORS["red"],
        delta_class(day_change),
    )
with kpi_cols[2]:
    if predicted_price is not None:
        kpi_card(
            f"{horizon}-day forecast",
            fmt_money(predicted_price, currency),
            f"{primary_name} · end of horizon",
            COLORS["violet"],
        )
    else:
        kpi_card(
            f"{horizon}-day forecast",
            "—",
            "Run a forecast to populate",
            COLORS["violet"],
            "neutral",
        )
with kpi_cols[3]:
    if expected_change is not None:
        kpi_card(
            "Expected change",
            fmt_pct(expected_change),
            f"from {fmt_money(latest_price, currency)} over {horizon} days",
            COLORS["amber"],
            delta_class(expected_change),
        )
    else:
        kpi_card(
            "Expected change",
            "—",
            "No forecast yet",
            COLORS["amber"],
            "neutral",
        )

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

market_tab, evaluation_tab, forecast_tab = st.tabs(
    ["📊  Market Overview", "🎯  Model Evaluation", "🔮  Forecast & Export"]
)

with market_tab:
    st.plotly_chart(
        price_history_chart(enriched, ticker, show_moving_averages=True),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        candlestick_chart(market_data, ticker, show_moving_averages=True),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        volume_chart(market_data, ticker),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    with st.expander("Historical OHLCV data", expanded=False):
        st.caption(
            f"{len(market_data):,} daily bars · {market_data.index[0]:%b %d, %Y} "
            f"to {market_data.index[-1]:%b %d, %Y}"
        )
        display_frame = enriched.tail(250).copy()
        st.dataframe(
            display_frame.style.format(
                {
                    "Open": "{:,.2f}",
                    "High": "{:,.2f}",
                    "Low": "{:,.2f}",
                    "Close": "{:,.2f}",
                    "Volume": "{:,.0f}",
                    "MA20": "{:,.2f}",
                    "MA50": "{:,.2f}",
                    "Pct Change": "{:+.2f}%",
                },
                na_rep="—",
            ),
            width="stretch",
            height=360,
        )
        st.download_button(
            "⬇  Download historical CSV",
            data=market_data.to_csv().encode("utf-8"),
            file_name=f"{ticker}_history_{end_date.isoformat()}.csv",
            mime="text/csv",
            width="content",
        )

with evaluation_tab:
    if not results:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:2.2rem;">
                <div style="font-size:1.2rem; font-weight:700; color:#e6edf7;">No evaluation yet</div>
                <div class="small-muted" style="margin-top:0.4rem;">
                    Configure the model in the sidebar and press <strong>Run Forecast</strong>.
                    Metrics are computed on a chronological hold-out set, never on training data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-title">Held-out performance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Chronological test partition · no shuffling · scaler fitted on train only</div>',
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(len(results), gap="medium")
        for col, (name, result) in zip(metric_cols, results.items()):
            with col:
                render_metric_card(name, result.metrics, model_description(name))

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        reference = next(iter(results.values()))
        predicted_map = {name: res.test_predictions for name, res in results.items()}
        st.plotly_chart(
            actual_vs_predicted_chart(
                dates=reference.test_dates,
                actual=reference.test_actual,
                predicted=predicted_map,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

        comparison_rows = []
        for name, result in results.items():
            comparison_rows.append(
                {
                    "Model": name,
                    "MAE": round(float(result.metrics["mae"]), 4),
                    "RMSE": round(float(result.metrics["rmse"]), 4),
                    "MAPE (%)": round(float(result.metrics["mape"]), 4),
                    "Direction (%)": round(float(result.metrics["directional_accuracy"]), 2),
                    "Test samples": int(result.metrics["n_samples"]),
                }
            )
        st.dataframe(
            pd.DataFrame(comparison_rows),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Lower MAE/RMSE/MAPE is better. Direction is the share of correctly predicted "
            "step-to-step moves. MAPE can be unstable for assets with prices near zero."
        )

        if "LSTM" in results and results["LSTM"].training_history:
            st.plotly_chart(
                training_loss_chart(results["LSTM"].training_history),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

        with st.expander("Model configuration & leakage safeguards", expanded=False):
            for name, result in results.items():
                st.markdown(f"**{name}**")
                st.json(result.config)
            st.markdown(
                """
                <div class="note">
                    <strong>Leakage safeguards:</strong> the split is strictly chronological;
                    the MinMaxScaler is fitted on the training partition only; test windows use
                    only prior observations; LSTM sequences are never shuffled. Future forecasts
                    are seeded with the most recent actual window after evaluation is complete.
                </div>
                """,
                unsafe_allow_html=True,
            )

with forecast_tab:
    if not results:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:2.2rem;">
                <div style="font-size:1.2rem; font-weight:700; color:#e6edf7;">Forecast awaiting execution</div>
                <div class="small-muted" style="margin-top:0.4rem;">
                    Choose LSTM, Prophet, or both, then run the pipeline to generate future prices,
                    uncertainty bands and a downloadable CSV.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, (name, result) in enumerate(results.items()):
            st.markdown(f'<div class="section-title">{name} · next {horizon} days</div>', unsafe_allow_html=True)
            if name == "Prophet":
                st.markdown(
                    f'<div class="section-sub">{int(interval_width * 100)}% uncertainty interval from Prophet’s posterior predictive samples.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="section-sub">Approximate band = ±1.96 × held-out RMSE × √step '
                    '(heuristic, not a calibrated probability interval).</div>',
                    unsafe_allow_html=True,
                )

            st.plotly_chart(
                forecast_chart(
                    history_dates=enriched.index,
                    history_prices=enriched["Close"].to_numpy(),
                    forecast_dates=result.future_dates,
                    forecast_prices=result.future_predictions,
                    lower_bound=result.lower_bound,
                    upper_bound=result.upper_bound,
                    model_name=name,
                    ticker=ticker,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

            table = forecast_table(result)
            end_forecast = float(result.future_predictions[-1])
            expected = (end_forecast / latest_price - 1.0) * 100.0 if latest_price else float("nan")
            summary_cols = st.columns([1, 1, 1, 2])
            summary_cols[0].metric("End forecast", fmt_money(end_forecast, currency))
            summary_cols[1].metric("Expected change", fmt_pct(expected))
            summary_cols[2].metric("Horizon", f"{horizon} days")
            summary_cols[3].caption(
                f"Forecast begins {pd.Timestamp(result.future_dates[0]):%b %d, %Y} and ends "
                f"{pd.Timestamp(result.future_dates[-1]):%b %d, %Y}. Bounds are model estimates."
            )
            with st.expander(f"Forecast table · {name}", expanded=(idx == 0)):
                st.dataframe(
                    table.style.format(
                        {"Forecast": "{:,.2f}", "Lower Bound": "{:,.2f}", "Upper Bound": "{:,.2f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                    height=300,
                )
                if name == "LSTM":
                    st.caption(
                        "LSTM intervals are an error-band approximation; use Prophet when a "
                        "calibrated uncertainty interval is required."
                    )
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        st.download_button(
            "⬇  Download all forecast results (CSV)",
            data=combined_forecast_csv(results, ticker),
            file_name=f"{ticker}_forecast_{horizon}d.csv",
            mime="text/csv",
            type="primary",
            width="content",
        )

st.markdown(
    """
    <div class="disclaimer">
        <strong>Financial disclaimer.</strong> This dashboard is an educational and research tool.
        Forecasts are statistical estimates generated from historical prices and are not guarantees
        of future performance. LSTM outputs are deterministic point forecasts; Prophet intervals
        reflect model uncertainty, not market risk. Nothing here is financial, investment, or
        trading advice. Markets can be volatile and you may lose capital. Do your own research and
        consult a licensed financial professional before making investment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="text-align:center; color:#52657f; font-size:10.5px; margin-top:1.1rem;">
        QuantForecast · modular Streamlit dashboard · data © Yahoo Finance ·
        generated {datetime.now():%Y-%m-%d %H:%M}
    </div>
    """,
    unsafe_allow_html=True,
)
