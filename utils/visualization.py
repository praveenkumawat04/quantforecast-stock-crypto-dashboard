"""Plotly visualization builders for the dashboard.

Every function returns a ``plotly.graph_objects.Figure`` and applies a shared
professional dark theme so the UI stays visually consistent.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS: dict[str, str] = {
    "bg": "#070d18",
    "panel": "#0d1626",
    "grid": "#1e2c44",
    "text": "#e6edf7",
    "muted": "#8b9bb4",
    "accent": "#22d3ee",
    "accent_soft": "rgba(34, 211, 238, 0.16)",
    "green": "#16c784",
    "red": "#ea3943",
    "amber": "#f59e0b",
    "purple": "#a78bfa",
    "violet": "#a78bfa",
    "blue": "#3b82f6",
    "white": "#f8fafc",
}


def apply_dark_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 440,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the dashboard's dark layout to a Plotly figure."""

    fig.update_layout(
        template="plotly_dark",
        title={
            "text": title,
            "font": {"size": 16, "color": COLORS["text"], "family": "Inter, sans-serif"},
            "x": 0.01,
            "xanchor": "left",
        },
        height=height,
        margin={"l": 16, "r": 16, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": COLORS["muted"], "size": 12},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": COLORS["panel"],
            "bordercolor": COLORS["grid"],
            "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": COLORS["muted"]},
        },
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=False,
        tickfont={"color": COLORS["muted"]},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=False,
        tickfont={"color": COLORS["muted"]},
    )
    return fig


def _close_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data.astype(float)
    else:
        series = data["Close"].astype(float)
    return series


def price_history_chart(
    data: pd.DataFrame,
    ticker: str,
    show_moving_averages: bool = True,
) -> go.Figure:
    """Line chart of close price with optional 20/50-day moving averages."""

    close = _close_series(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.to_numpy(),
            name="Close",
            mode="lines",
            line={"color": COLORS["accent"], "width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<extra></extra>",
        )
    )

    if show_moving_averages:
        for window, color in ((20, COLORS["amber"]), (50, COLORS["purple"])):
            if len(close) >= window:
                ma = close.rolling(window=window, min_periods=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=ma.index,
                        y=ma.to_numpy(),
                        name=f"MA{window}",
                        mode="lines",
                        line={"color": color, "width": 1.4},
                        hovertemplate=f"MA{window}: $%{{y:,.2f}}<extra></extra>",
                    )
                )

    apply_dark_theme(fig, f"{ticker} · Historical Close Price", height=430)
    fig.update_yaxes(tickprefix="$", tickformat=",.2f")
    return fig


def candlestick_chart(
    data: pd.DataFrame,
    ticker: str,
    show_moving_averages: bool = True,
) -> go.Figure:
    """OHLC candlestick chart with moving-average overlays."""

    fig = go.Figure(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
            increasing={
                "line": {"color": COLORS["green"], "width": 1},
                "fillcolor": COLORS["green"],
            },
            decreasing={
                "line": {"color": COLORS["red"], "width": 1},
                "fillcolor": COLORS["red"],
            },
            whiskerwidth=0.4,
        )
    )

    if show_moving_averages:
        for window, color in ((20, COLORS["amber"]), (50, COLORS["purple"])):
            if len(data) >= window:
                ma = data["Close"].rolling(window=window, min_periods=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        name=f"MA{window}",
                        mode="lines",
                        line={"color": color, "width": 1.3},
                        hovertemplate=f"MA{window}: $%{{y:,.2f}}<extra></extra>",
                    )
                )

    apply_dark_theme(fig, f"{ticker} · Candlestick", height=460)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(tickprefix="$", tickformat=",.2f")
    return fig


def volume_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Volume bars colored by up/down days."""

    changes = data["Close"].diff().fillna(0.0)
    colors = np.where(changes >= 0, COLORS["green"], COLORS["red"])
    fig = go.Figure(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker={"color": colors, "line": {"width": 0}},
            opacity=0.65,
            hovertemplate="%{x|%b %d, %Y}<br>Volume: %{y:,.0f}<extra></extra>",
        )
    )
    apply_dark_theme(fig, f"{ticker} · Trading Volume", height=260, show_legend=False)
    fig.update_yaxes(tickformat="~s", title="Shares / units")
    return fig


def actual_vs_predicted_chart(
    dates: Sequence[pd.Timestamp],
    actual: Sequence[float],
    predicted: Mapping[str, Sequence[float]],
    title: str = "Actual vs Predicted · Held-out Test Set",
) -> go.Figure:
    """Compare actual held-out prices with one or more model predictions."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=np.asarray(actual, dtype=float),
            name="Actual",
            mode="lines",
            line={"color": COLORS["white"], "width": 2.2},
            hovertemplate="Actual: $%{y:,.2f}<extra></extra>",
        )
    )

    palette = [COLORS["amber"], COLORS["purple"], COLORS["accent"]]
    for idx, (name, values) in enumerate(predicted.items()):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.asarray(values, dtype=float),
                name=name,
                mode="lines",
                line={"color": palette[idx % len(palette)], "width": 2, "dash": "dot"},
                hovertemplate=f"{name}: $%{{y:,.2f}}<extra></extra>",
            )
        )

    apply_dark_theme(fig, title, height=420)
    fig.update_yaxes(tickprefix="$", tickformat=",.2f")
    return fig


def forecast_chart(
    history_dates: Sequence[pd.Timestamp],
    history_prices: Sequence[float],
    forecast_dates: Sequence[pd.Timestamp],
    forecast_prices: Sequence[float],
    lower_bound: Sequence[float] | None,
    upper_bound: Sequence[float] | None,
    model_name: str,
    ticker: str,
    history_points: int = 180,
) -> go.Figure:
    """Future forecast line with an uncertainty band and recent history."""

    hist_dates = list(history_dates)[-history_points:]
    hist_prices = np.asarray(history_prices, dtype=float)[-history_points:]
    future_dates = pd.DatetimeIndex(forecast_dates)
    future = np.asarray(forecast_prices, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_prices,
            name="Historical",
            mode="lines",
            line={"color": COLORS["muted"], "width": 1.6},
            hovertemplate="Close: $%{y:,.2f}<extra></extra>",
        )
    )

    if lower_bound is not None and upper_bound is not None:
        lower = np.asarray(lower_bound, dtype=float)
        upper = np.asarray(upper_bound, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper,
                name="Confidence band",
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=COLORS["accent_soft"],
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future,
            name=f"{model_name} forecast",
            mode="lines+markers",
            line={"color": COLORS["accent"], "width": 2.4},
            marker={"size": 4, "color": COLORS["accent"]},
            hovertemplate=f"{model_name}: $%{{y:,.2f}}<extra></extra>",
        )
    )

    if len(hist_dates) > 0 and len(future_dates) > 0:
        bridge_x = [hist_dates[-1], future_dates[0]]
        bridge_y = [hist_prices[-1], future[0]]
        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines",
                line={"color": COLORS["accent"], "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    apply_dark_theme(fig, f"{ticker} · {model_name} Price Forecast", height=460)
    fig.update_yaxes(tickprefix="$", tickformat=",.2f")
    return fig


def training_loss_chart(history: Mapping[str, Sequence[float]]) -> go.Figure:
    """Plot training and validation loss curves for the LSTM."""

    fig = go.Figure()
    loss = list(history.get("loss", []))
    if loss:
        fig.add_trace(
            go.Scatter(
                y=loss,
                name="Training loss",
                mode="lines",
                line={"color": COLORS["accent"], "width": 2},
            )
        )
    val_loss = list(history.get("val_loss", []))
    if val_loss:
        fig.add_trace(
            go.Scatter(
                y=val_loss,
                name="Validation loss",
                mode="lines",
                line={"color": COLORS["amber"], "width": 1.8, "dash": "dot"},
            )
        )
    apply_dark_theme(fig, "Training Progress · MSE Loss", height=280)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss")
    return fig


def price_with_forecast_chart(
    data: pd.DataFrame,
    forecast_dates: Sequence[pd.Timestamp],
    forecast_prices: Sequence[float],
    lower_bound: Sequence[float] | None = None,
    upper_bound: Sequence[float] | None = None,
    model_name: str = "Model",
    ticker: str = "",
    history_points: int = 365,
) -> go.Figure:
    """Convenience wrapper combining historical close data and a forecast."""

    close = _close_series(data)
    return forecast_chart(
        history_dates=close.index,
        history_prices=close.to_numpy(),
        forecast_dates=forecast_dates,
        forecast_prices=forecast_prices,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_name=model_name,
        ticker=ticker,
        history_points=history_points,
    )
