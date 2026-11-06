"""Chart: how much more often open shots go in than their location says."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..predictions.plot import GRID, INK, MUTED, SURFACE  # noqa: E402

COLORS = {"two": "#2a78d6", "three": "#eb6834"}
NAMES = {"two": "Twos", "three": "Threes"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def defender_chart(shots: pd.DataFrame, expected: np.ndarray, out_path: str | Path, max_feet: int = 10,
                   min_shots: int = 300) -> Path:
    """FG% minus xFG without tracking, by the closest defender's distance (1-foot bins), twos and threes."""
    feet = np.minimum(np.floor(shots["def_dist"].to_numpy()), max_feet).astype(int)
    table = pd.DataFrame({"feet": feet, "above": shots["made"].to_numpy() - expected,
                          "shot": np.where(shots["value"].to_numpy() == 3, "three", "two")})
    stats = table.groupby(["shot", "feet"])["above"].agg(["mean", "size"]).reset_index()

    fig, (ax, bars) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bars):
        _style(a)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    width = 0.4
    for i, shot in enumerate(("two", "three")):
        s = stats[(stats["shot"] == shot) & (stats["size"] >= min_shots)]
        x = s["feet"] + 0.5
        ax.plot(x, 100 * s["mean"], color=COLORS[shot], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[shot])
        bars.bar(x + (i - 0.5) * width, s["size"] / 1000, width=width * 0.9, color=COLORS[shot], linewidth=0)
    ax.set_ylabel("FG% minus expected from location (points)", color=INK, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(f"Open shots go in more than their location says: {len(shots):,} shots of 2014-15",
                 color=INK, fontsize=11, loc="left")
    bars.set_ylabel("Shots (000s)", color=INK, fontsize=10)
    bars.set_xlabel("Closest defender's distance at the release (feet)", color=INK, fontsize=10)
    ticks = np.arange(0, max_feet + 1, 2)
    bars.set_xticks(ticks)
    bars.set_xticklabels([f"{t}+" if t == max_feet else str(t) for t in ticks])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
