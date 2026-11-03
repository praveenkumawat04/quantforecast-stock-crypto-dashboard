"""Generate a polished architecture diagram for the project README.

Run from the project root:

    python scripts/generate_architecture.py

The diagram is drawn with Matplotlib, so no Graphviz binary is required.  It
uses a fixed, vertically aligned layout and verifies every text bounding box
against its card (and against every other label) before writing the PNG.

Output: assets/architecture.png (high-resolution, tight margins)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "architecture.png"

# ---------------------------------------------------------------------------
# Palette & typography
# ---------------------------------------------------------------------------

PAGE = "#F7F9FC"
CARD = "#FFFFFF"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
ARROW = "#94A3B8"
SHADOW = "#0F172A"

BLUE = "#1D4ED8"
TEAL = "#0F766E"
CYAN = "#0E7490"
INDIGO = "#4338CA"
SLATE = "#475569"
AMBER = "#B45309"
ORANGE = "#C2410C"
VIOLET = "#6D28D9"
ROSE = "#BE123C"
GREEN = "#15803D"

# Avenir Next / Helvetica Neue produce a cleaner result on macOS; DejaVu Sans
# is the portable fallback. The chosen fonts are embedded in the PNG.
_AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
SANS = next(
    (name for name in ("Avenir Next", "Helvetica Neue", "Inter", "Arial") if name in _AVAILABLE_FONTS),
    "DejaVu Sans",
)
MONO = next(
    (name for name in ("Menlo", "SF Mono", "JetBrains Mono", "DejaVu Sans Mono") if name in _AVAILABLE_FONTS),
    "DejaVu Sans Mono",
)

plt.rcParams.update(
    {
        "font.family": SANS,
        "font.size": 9,
        "axes.unicode_minus": False,
        "savefig.facecolor": PAGE,
    }
)

# Canvas (data units mirror inches for easy reasoning)
FIG_W, FIG_H = 8.4, 10.6
DPI = 200


def tint(accent: str, strength: float = 0.035) -> tuple[float, float, float]:
    """Mix an accent into white for a very subtle card tint."""

    rgb = np.array(to_rgb(accent))
    white = np.ones(3)
    return tuple(white + (rgb - white) * strength)


@dataclass
class BoxRecord:
    name: str
    x: float
    y: float
    width: float
    height: float
    accent: str
    texts: list[Text] = field(default_factory=list)

    @property
    def bbox(self) -> Bbox:
        return Bbox.from_bounds(self.x, self.y, self.width, self.height)


class Diagram:
    """Small drawing helper that records geometry for verification."""

    def __init__(self, ax: plt.Axes) -> None:
        self.ax = ax
        self.boxes: list[BoxRecord] = []
        self.outside_texts: list[tuple[str, Text]] = []

    def card(
        self,
        name: str,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        secondary: str,
        accent: str,
        title_size: float = 12.0,
        secondary_size: float = 8.0,
    ) -> BoxRecord:
        # Soft shadow
        self.ax.add_patch(
            FancyBboxPatch(
                (x + 0.025, y - 0.045),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.115",
                linewidth=0,
                facecolor=SHADOW,
                alpha=0.055,
                zorder=1,
            )
        )

        # Card
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.115",
                linewidth=1.1,
                edgecolor=BORDER,
                facecolor=tint(accent, 0.035),
                zorder=2,
            )
        )

        # Colored spine
        self.ax.add_patch(
            FancyBboxPatch(
                (x + 0.075, y + 0.095),
                0.055,
                height - 0.19,
                boxstyle="round,pad=0,rounding_size=0.0275",
                linewidth=0,
                facecolor=accent,
                zorder=3,
            )
        )

        title_text = self.ax.text(
            x + 0.30,
            y + height * 0.625,
            title,
            ha="left",
            va="center",
            color=INK,
            fontsize=title_size,
            fontweight="bold",
            zorder=4,
        )
        secondary_text = self.ax.text(
            x + 0.30,
            y + height * 0.265,
            secondary,
            ha="left",
            va="center",
            color=MUTED,
            fontsize=secondary_size,
            fontfamily=MONO,
            zorder=4,
        )

        record = BoxRecord(name, x, y, width, height, accent, [title_text, secondary_text])
        self.boxes.append(record)
        return record

    def annotation(self, box: BoxRecord, side: str, text: str) -> None:
        """Attach a short external label to a card with a small accent tick."""

        center_y = box.y + box.height / 2
        if side == "right":
            tick_x = (box.x + box.width + 0.065, box.x + box.width + 0.155)
            text_x = box.x + box.width + 0.225
            ha = "left"
        else:
            tick_x = (box.x - 0.155, box.x - 0.065)
            text_x = box.x - 0.225
            ha = "right"

        self.ax.plot(
            tick_x,
            (center_y, center_y),
            color=box.accent,
            linewidth=1.3,
            solid_capstyle="round",
            zorder=3,
        )
        label = self.ax.text(
            text_x,
            center_y,
            text,
            ha=ha,
            va="center",
            color="#475569",
            fontsize=8.0,
            linespacing=1.38,
            zorder=4,
        )
        self.outside_texts.append((f"{box.name} annotation", label))

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = ARROW,
        width: float = 1.25,
    ) -> None:
        self.ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10.5,
                linewidth=width,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )

    def connector(self, points: list[tuple[float, float]], color: str = ARROW) -> None:
        xs, ys = zip(*points)
        self.ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.25,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _text_bbox(fig: plt.Figure, ax: plt.Axes, text: Text) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    display_bbox = text.get_window_extent(renderer=renderer)
    points = ax.transData.inverted().transform(display_bbox.get_points())
    return Bbox.from_extents(points[0][0], points[0][1], points[1][0], points[1][1])


def _intersects(a: Bbox, b: Bbox, tolerance: float = 0.0) -> bool:
    return not (
        a.x1 <= b.x0 + tolerance
        or b.x1 <= a.x0 + tolerance
        or a.y1 <= b.y0 + tolerance
        or b.y1 <= a.y0 + tolerance
    )


def verify_layout(
    fig: plt.Figure,
    ax: plt.Axes,
    diagram: Diagram,
    title_texts: list[Text],
) -> dict[str, float | int]:
    """Fail loudly on clipped text, card overflow, or label collisions."""

    fig.canvas.draw()
    errors: list[str] = []

    # 1. Every in-card text must sit fully inside its card with padding.
    for box in diagram.boxes:
        for text in box.texts:
            tb = _text_bbox(fig, ax, text)
            padding = 0.085
            if not (
                tb.x0 >= box.x + padding
                and tb.x1 <= box.x + box.width - padding
                and tb.y0 >= box.y + padding
                and tb.y1 <= box.y + box.height - padding
            ):
                errors.append(
                    f"Text {text.get_text()!r} overflows card {box.name!r}: "
                    f"text={tuple(round(v, 3) for v in tb.extents)}, "
                    f"card={tuple(round(v, 3) for v in box.bbox.extents)}"
                )

    # 2. Cards must not overlap each other.
    for i, first in enumerate(diagram.boxes):
        for second in diagram.boxes[i + 1 :]:
            if _intersects(first.bbox, second.bbox):
                errors.append(f"Cards overlap: {first.name!r} and {second.name!r}")

    # 3. External annotations must not touch a card or each other.
    outside_bboxes: list[tuple[str, Bbox]] = []
    for label, text in diagram.outside_texts:
        tb = _text_bbox(fig, ax, text)
        outside_bboxes.append((label, tb))
        for box in diagram.boxes:
            if _intersects(tb, box.bbox):
                errors.append(f"Annotation {label!r} collides with card {box.name!r}")
        if not (0.20 <= tb.x0 and tb.x1 <= FIG_W - 0.20 and 0.05 <= tb.y0 and tb.y1 <= FIG_H - 0.05):
            errors.append(f"Annotation {label!r} is outside the safe canvas: {tb.extents}")

    for i, (name_a, bbox_a) in enumerate(outside_bboxes):
        for name_b, bbox_b in outside_bboxes[i + 1 :]:
            if _intersects(bbox_a, bbox_b):
                errors.append(f"Annotations overlap: {name_a!r} and {name_b!r}")

    # 4. Header/footer copy must fit the safe canvas and stay clear of cards.
    title_bboxes: list[tuple[str, Bbox]] = []
    for text in title_texts:
        tb = _text_bbox(fig, ax, text)
        name = f"title:{text.get_text()[:24]}"
        title_bboxes.append((name, tb))
        if not (0.20 <= tb.x0 and tb.x1 <= FIG_W - 0.20 and 0.05 <= tb.y0 and tb.y1 <= FIG_H - 0.05):
            errors.append(f"{name!r} is outside the safe canvas: {tb.extents}")
        for box in diagram.boxes:
            if _intersects(tb, box.bbox):
                errors.append(f"{name!r} collides with card {box.name!r}")

    # 5. All text objects (titles, card labels, annotations) must be mutually clear.
    all_texts: list[tuple[str, Bbox]] = []
    all_texts.extend(title_bboxes)
    for box in diagram.boxes:
        for text in box.texts:
            all_texts.append((f"{box.name}:{text.get_text()[:24]}", _text_bbox(fig, ax, text)))
    all_texts.extend(outside_bboxes)

    for i, (name_a, bbox_a) in enumerate(all_texts):
        for name_b, bbox_b in all_texts[i + 1 :]:
            if _intersects(bbox_a, bbox_b, tolerance=0.005):
                errors.append(f"Text overlap: {name_a!r} and {name_b!r}")

    if errors:
        raise RuntimeError("Architecture layout verification failed:\n- " + "\n- ".join(errors))

    return {
        "cards": len(diagram.boxes),
        "labels": len(all_texts),
        "annotations": len(diagram.outside_texts),
    }


def verify_png(path: Path, expected_size: tuple[int, int] | None = None) -> dict[str, int]:
    """Check the saved PNG for clipping, transparency and unsafe margins."""

    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3]
        if int(alpha.min()) != 255:
            raise RuntimeError("Saved PNG contains transparent pixels.")

        rgb = np.asarray(rgba.convert("RGB")).astype(int)
        background = np.array([int(PAGE[1:3], 16), int(PAGE[3:5], 16), int(PAGE[5:7], 16)])
        distance = np.abs(rgb - background).sum(axis=2)
        content = distance > 18
        ys, xs = np.where(content)
        if xs.size == 0:
            raise RuntimeError("Saved PNG appears to be blank.")

        margins = {
            "left": int(xs.min()),
            "right": int(width - 1 - xs.max()),
            "top": int(ys.min()),
            "bottom": int(height - 1 - ys.max()),
        }
        if min(margins.values()) < 8:
            raise RuntimeError(f"Content is clipped too close to the PNG edge: {margins}")

    if expected_size is not None and (width, height) != expected_size:
        raise RuntimeError(f"Unexpected PNG size {width}x{height}; expected {expected_size}.")

    return {"width": width, "height": height, **margins}


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------


def build() -> Path:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    diagram = Diagram(ax)

    # Header -----------------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (0.75, 10.500),
            6.90,
            0.038,
            boxstyle="round,pad=0,rounding_size=0.019",
            linewidth=0,
            facecolor=BLUE,
            zorder=3,
        )
    )

    title_texts: list[Text] = []
    title_texts.append(
        ax.text(
            0.75,
            10.330,
            "SYSTEM ARCHITECTURE",
            ha="left",
            va="center",
            color=BLUE,
            fontsize=8.4,
            fontweight="bold",
            zorder=4,
        )
    )
    title_texts.append(
        ax.text(
            7.65,
            10.330,
            "PYTHON 3.11+  ·  STREAMLIT  ·  TENSORFLOW  ·  PROPHET",
            ha="right",
            va="center",
            color=MUTED,
            fontsize=7.4,
            fontfamily=MONO,
            zorder=4,
        )
    )
    title_texts.append(
        ax.text(
            0.75,
            9.980,
            "Stock & Crypto Price Prediction Dashboard",
            ha="left",
            va="center",
            color=INK,
            fontsize=18.5,
            fontweight="bold",
            zorder=4,
        )
    )
    title_texts.append(
        ax.text(
            0.75,
            9.600,
            "Leak-free market data  ·  ML forecasting  ·  held-out evaluation  ·  interactive results",
            ha="left",
            va="center",
            color="#475569",
            fontsize=9.4,
            zorder=4,
        )
    )
    ax.plot([0.75, 7.65], [9.430, 9.430], color=BORDER, linewidth=1.0, zorder=3)

    # Vertical pipeline ------------------------------------------------------
    cx = 4.20
    main_x, main_w, main_h = cx - 1.60, 3.20, 0.66

    user = diagram.card(
        "User Input",
        main_x,
        8.68,
        main_w,
        main_h,
        "User Input",
        "sidebar controls",
        SLATE,
        title_size=11.8,
    )
    dashboard = diagram.card(
        "Streamlit Dashboard",
        main_x,
        7.84,
        main_w,
        main_h,
        "Streamlit Dashboard",
        "app.py",
        CYAN,
    )
    acquisition = diagram.card(
        "Data Acquisition",
        main_x,
        6.88,
        main_w,
        main_h,
        "Data Acquisition",
        "utils/data_fetcher.py",
        BLUE,
    )
    preprocessing = diagram.card(
        "Data Preprocessing",
        main_x,
        5.92,
        main_w,
        main_h,
        "Data Preprocessing",
        "utils/preprocessing.py",
        INDIGO,
    )
    selection = diagram.card(
        "Forecasting Model Selection",
        main_x,
        4.96,
        main_w,
        main_h,
        "Forecasting Model Selection",
        "models/  ·  shared split",
        SLATE,
        title_size=11.2,
    )

    # Parallel branches ------------------------------------------------------
    branch_w = 2.60
    branch_y, branch_h = 3.56, 0.66
    lstm_x = 1.45
    prophet_x = 4.35

    lstm = diagram.card(
        "LSTM Model",
        lstm_x,
        branch_y,
        branch_w,
        branch_h,
        "LSTM Model",
        "models/lstm_model.py",
        VIOLET,
        title_size=11.8,
    )
    prophet = diagram.card(
        "Prophet Model",
        prophet_x,
        branch_y,
        branch_w,
        branch_h,
        "Prophet Model",
        "models/prophet_model.py",
        ROSE,
        title_size=11.8,
    )

    evaluation = diagram.card(
        "Model Evaluation",
        main_x,
        2.46,
        main_w,
        main_h,
        "Model Evaluation",
        "utils/evaluation.py",
        ORANGE,
    )
    visualization = diagram.card(
        "Plotly Visualization",
        main_x,
        1.48,
        main_w,
        main_h,
        "Plotly Visualization",
        "utils/visualization.py",
        TEAL,
    )
    results = diagram.card(
        "Dashboard Results",
        main_x,
        0.50,
        main_w,
        main_h,
        "Dashboard Results",
        "KPIs  ·  tables  ·  CSV",
        GREEN,
    )

    # External labels: descriptions stay outside the cards -------------------
    diagram.annotation(user, "right", "ticker · range · horizon")
    diagram.annotation(dashboard, "right", "dark responsive UI\n15-min data cache")
    diagram.annotation(acquisition, "left", "OHLCV · yfinance\nvalidation + errors")
    diagram.annotation(preprocessing, "right", "chronological split\ntrain-only scaling")
    diagram.annotation(selection, "left", "parallel branches\nshared test partition")
    diagram.annotation(lstm, "left", "recursive\nmulti-step")
    diagram.annotation(prophet, "right", "probabilistic\nintervals")
    diagram.annotation(evaluation, "right", "MAE · RMSE · MAPE\nheld-out data only")
    diagram.annotation(visualization, "left", "candles · volume · MAs\nforecast bands")
    diagram.annotation(results, "right", "KPIs · tables · CSV\nfinancial disclaimer")

    # Arrows: straight main spine -------------------------------------------
    diagram.arrow((cx, 8.68), (cx, 8.50))
    diagram.arrow((cx, 7.84), (cx, 7.54))
    diagram.arrow((cx, 6.88), (cx, 6.58))
    diagram.arrow((cx, 5.92), (cx, 5.62))

    # Fork into the two model branches --------------------------------------
    fork_y = 4.59
    diagram.connector([(cx, 4.96), (cx, fork_y)])
    diagram.connector([(lstm.x + lstm.width / 2, fork_y), (prophet.x + prophet.width / 2, fork_y)])
    diagram.arrow((lstm.x + lstm.width / 2, fork_y), (lstm.x + lstm.width / 2, branch_y + branch_h))
    diagram.arrow((prophet.x + prophet.width / 2, fork_y), (prophet.x + prophet.width / 2, branch_y + branch_h))

    # Merge into evaluation --------------------------------------------------
    merge_y = 3.34
    diagram.connector(
        [
            (lstm.x + lstm.width / 2, branch_y),
            (lstm.x + lstm.width / 2, merge_y),
            (prophet.x + prophet.width / 2, merge_y),
            (prophet.x + prophet.width / 2, branch_y),
        ]
    )
    diagram.arrow((cx, merge_y), (cx, 3.12))

    # Remaining vertical spine ----------------------------------------------
    diagram.arrow((cx, 2.46), (cx, 2.14))
    diagram.arrow((cx, 1.48), (cx, 1.16))

    # Footer -----------------------------------------------------------------
    title_texts.append(
        ax.text(
            0.75,
            0.160,
            "Reproducible seeds  ·  chronological evaluation  ·  no data leakage  ·  "
            "forecasts are estimates, not financial advice",
            ha="left",
            va="center",
            color=MUTED,
            fontsize=7.6,
            zorder=4,
        )
    )

    stats = verify_layout(fig, ax, diagram, title_texts)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=PAGE,
        edgecolor=PAGE,
        pad_inches=0.12,
    )
    plt.close(fig)

    png_stats = verify_png(OUTPUT)
    print(
        "Layout verified: "
        f"{stats['cards']} cards, {stats['labels']} text labels, "
        f"{stats['annotations']} external annotations."
    )
    print(
        "PNG verified: "
        f"{png_stats['width']}x{png_stats['height']} px, "
        f"margins L{png_stats['left']} R{png_stats['right']} "
        f"T{png_stats['top']} B{png_stats['bottom']} px."
    )
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
