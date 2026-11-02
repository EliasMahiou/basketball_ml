"""Reliability chart for win probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .model import calibration, calibration_error  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
COLORS = {"logistic": "#2a78d6", "elo": "#eb6834", "boosting": "#1baf7a"}
NAMES = {"logistic": "Logistic regression", "elo": "Elo alone", "boosting": "Gradient boosting"}


def calibration_plot(results: pd.DataFrame, out_path: str | Path,
                     methods: Sequence[str] = ("logistic", "elo"), bins: int = 10,
                     min_games: int = 30) -> Path:
    """Predicted probability against observed home win rate, with games per bin below.

    Bins with fewer than ``min_games`` games are left out (their win rate is noise).
    """
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bars):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.62, 0.66, "perfect calibration", color=MUTED, fontsize=8.5, ha="center", rotation=45,
            rotation_mode="anchor", transform_rotates_text=True)
    width = 0.8 / bins / len(methods)
    for i, m in enumerate(methods):
        table = calibration(results, m, bins)
        table = table[table["games"] >= min_games]
        ece = calibration_error(results, m, bins)
        ax.plot(table["predicted"], table["observed"], color=COLORS.get(m, INK), linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{NAMES.get(m, m)} (average gap {ece:.1%})")
        offset = (i - (len(methods) - 1) / 2) * width
        bars.bar(table["predicted"] + offset, table["games"], width=width * 0.9,
                 color=COLORS.get(m, INK), edgecolor=SURFACE, linewidth=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Home teams that won", color=INK, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(f"Predicted vs actual, {len(results):,} out-of-sample games", color=INK,
                 fontsize=11, loc="left")
    bars.set_ylabel("Games", color=INK, fontsize=10)
    bars.set_xlabel("Predicted home win probability", color=INK, fontsize=10)
    bars.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
