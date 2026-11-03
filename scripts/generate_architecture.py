"""Generate assets/architecture.png.

Run with:
    python scripts/generate_architecture.py

The diagram is drawn with Matplotlib (no external Graphviz binary required) so
it can be regenerated on any machine that installs the project requirements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "architecture.png"

BG = "#070d18"
PANEL = "#0d1626"
TEXT = "#e6edf7"
MUTED = "#8b9bb4"
GRID = "#1e2c44"
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
GREEN = "#16c784"
AMBER = "#f59e0b"
BLUE = "#3b82f6"


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    accent: str,
    title_size: float = 13,
) -> None:
    """Draw a rounded panel with a colored accent edge and two text lines."""

    shadow = FancyBboxPatch(
        (x + 0.04, y - 0.05),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0,
        facecolor="#000000",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.6,
        edgecolor=accent,
        facecolor=PANEL,
        zorder=2,
    )
    ax.add_patch(box)

    # Accent strip along the top of each card.
    strip = FancyBboxPatch(
        (x + 0.01, y + height - 0.075),
        width - 0.02,
        0.055,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=0,
        facecolor=accent,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(strip)

    ax.text(
        x + width / 2,
        y + height * 0.60,
        title,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=title_size,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        subtitle,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=9.2,
        zorder=4,
    )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#38506f",
    connection: str = "arc3,rad=0.0",
    width: float = 1.8,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=1.5,
    )
    ax.add_patch(arrow)


def build() -> Path:
    fig, ax = plt.subplots(figsize=(14, 10.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.text(
        0.65,
        10.12,
        "Stock & Crypto Price Prediction Dashboard",
        color=TEXT,
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.67,
        9.68,
        "System architecture  ·  leak-free market data → model evaluation → interactive forecast",
        color=MUTED,
        fontsize=11,
        ha="left",
        va="center",
    )
    ax.plot([0.65, 13.35], [9.38, 9.38], color=GRID, linewidth=1.2)

    # Main vertical pipeline -------------------------------------------------
    draw_box(ax, 5.05, 8.55, 3.9, 0.58, "User", "asset · ticker · range · horizon · model", CYAN, 11.5)
    draw_box(
        ax,
        3.80,
        7.45,
        6.40,
        0.82,
        "Streamlit Dashboard",
        "app.py  ·  dark responsive UI  ·  st.cache_data  ·  session state",
        CYAN,
        15,
    )
    draw_box(
        ax,
        4.15,
        6.30,
        5.70,
        0.78,
        "Data Acquisition",
        "yfinance  ·  OHLCV  ·  validation + error handling   [utils/data_fetcher.py]",
        GREEN,
    )
    draw_box(
        ax,
        4.15,
        5.15,
        5.70,
        0.78,
        "Data Preprocessing",
        "Pandas + NumPy  ·  indicators  ·  MinMaxScaler  ·  sliding windows   [utils/preprocessing.py]",
        BLUE,
    )
    draw_box(
        ax,
        4.55,
        4.00,
        4.90,
        0.78,
        "Forecasting Model Selection",
        "LSTM  or  Prophet  ·  chronological train/test split",
        AMBER,
    )

    # Model branches --------------------------------------------------------
    draw_box(
        ax,
        0.85,
        2.35,
        5.45,
        1.08,
        "LSTM Model",
        "TensorFlow/Keras · stacked LSTM · recursive multi-step\n"
        "models/lstm_model.py  ·  min-max scaling fitted on train only",
        VIOLET,
        14,
    )
    draw_box(
        ax,
        7.70,
        2.35,
        5.45,
        1.08,
        "Prophet Model",
        "Trend + seasonality · uncertainty intervals\n"
        "models/prophet_model.py  ·  ds / y format  ·  refit on full history",
        VIOLET,
        14,
    )

    # Evaluation / visualization / results ----------------------------------
    draw_box(
        ax,
        3.90,
        1.28,
        6.20,
        0.80,
        "Model Evaluation",
        "MAE  ·  RMSE  ·  MAPE  ·  directional accuracy   [utils/evaluation.py]",
        AMBER,
        14,
    )
    draw_box(
        ax,
        3.90,
        0.38,
        2.95,
        0.72,
        "Plotly Visualization",
        "candles · volume · MA\nactual vs predicted · CI",
        CYAN,
        12.5,
    )
    draw_box(
        ax,
        7.15,
        0.38,
        2.95,
        0.72,
        "Dashboard Results",
        "KPIs · charts · metrics\nCSV export · disclaimer",
        GREEN,
        12.5,
    )

    # Arrows ----------------------------------------------------------------
    draw_arrow(ax, (7.00, 8.55), (7.00, 8.27), CYAN)
    draw_arrow(ax, (7.00, 7.45), (7.00, 7.08), CYAN)
    draw_arrow(ax, (7.00, 6.30), (7.00, 5.93), GREEN)
    draw_arrow(ax, (7.00, 5.15), (7.00, 4.78), BLUE)
    draw_arrow(ax, (6.55, 4.00), (3.58, 3.43), VIOLET, connection="arc3,rad=0.08")
    draw_arrow(ax, (7.45, 4.00), (10.42, 3.43), VIOLET, connection="arc3,rad=-0.08")
    draw_arrow(ax, (3.58, 2.35), (6.35, 2.08), AMBER, connection="arc3,rad=0.06")
    draw_arrow(ax, (10.42, 2.35), (7.65, 2.08), AMBER, connection="arc3,rad=-0.06")
    draw_arrow(ax, (6.20, 1.28), (5.38, 1.10), AMBER)
    draw_arrow(ax, (7.80, 1.28), (8.62, 1.10), AMBER)

    # Branch labels ---------------------------------------------------------
    ax.text(4.35, 3.72, "sequence model", color=MUTED, fontsize=8.5, ha="center", style="italic")
    ax.text(9.65, 3.72, "probabilistic model", color=MUTED, fontsize=8.5, ha="center", style="italic")

    # Footer legend ---------------------------------------------------------
    ax.text(
        0.65,
        0.10,
        "Reproducible seeds  ·  no data leakage  ·  graceful handling of invalid tickers / API failures  ·  forecasts are estimates, not financial advice",
        color=MUTED,
        fontsize=8.6,
        ha="left",
        va="center",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
