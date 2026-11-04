"""Half-court map of expected FG% by shot location."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
LINE = "#3d3c39"
# One-hue sequential ramp (light = low, dark = high).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def zone_of(shots: pd.DataFrame) -> pd.Series:
    """The NBA's basic shot zones, from location and shot value (hoop at 0, 0; tenths of feet)."""
    x, y = shots["loc_x"].to_numpy(float), shots["loc_y"].to_numpy(float)
    three = shots["value"].to_numpy() == 3
    distance = np.hypot(x, y) / 10
    zone = np.select(
        [distance > 47, three & (y <= 92.5), three, distance <= 4, (np.abs(x) <= 80) & (y <= 142.5)],
        ["Backcourt", "Corner 3", "Above the break 3", "Restricted area", "Paint (non-RA)"], "Mid-range")
    return pd.Series(zone, index=shots.index)


def draw_court(ax: plt.Axes, color: str = LINE, lw: float = 1.2) -> None:
    """Half-court lines in shot chart units: rim, backboard, paint, restricted area, 3-point line."""
    style = dict(color=color, lw=lw, fill=False, zorder=3)
    ax.add_patch(Circle((0, 0), 7.5, **style))
    ax.plot([-30, 30], [-7.5, -7.5], color=color, lw=lw, zorder=3)
    ax.add_patch(Rectangle((-80, -47.5), 160, 190, **style))
    ax.add_patch(Rectangle((-60, -47.5), 120, 190, **style))
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, **style))
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=180, theta2=360, linestyle=(0, (4, 4)), **style))
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, **style))
    corner = np.degrees(np.arcsin(92.5 / 237.5))
    ax.plot([-220, -220], [-47.5, 92.5], color=color, lw=lw, zorder=3)
    ax.plot([220, 220], [-47.5, 92.5], color=color, lw=lw, zorder=3)
    ax.add_patch(Arc((0, 0), 475, 475, theta1=corner, theta2=180 - corner, **style))
    ax.add_patch(Rectangle((-250, -47.5), 500, 470, **style))
    ax.add_patch(Arc((0, 422.5), 120, 120, theta1=180, theta2=360, **style))


def court_map(shots: pd.DataFrame, out_path: str | Path, column: str = "xfg", title: str | None = None,
              gridsize: int = 34, min_shots: int = 20, vmin: float = 0.25, vmax: float = 0.70) -> Path:
    """Average predicted make probability of the shots taken from each spot of the half court.

    ``shots`` needs ``loc_x``, ``loc_y``, ``value``, ``made`` and the prediction
    ``column``. Areas with fewer than ``min_shots`` shots are left blank. The
    side table gives each basic zone's expected FG% and expected points per shot.
    """
    half = shots[(shots["loc_y"] <= 417.5) & (shots["loc_x"].abs() <= 250)]
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("xfg", RAMP)
    hexes = ax.hexbin(half["loc_x"], half["loc_y"], C=half[column], reduce_C_function=np.mean, gridsize=gridsize,
                      extent=(-250, 250, -47.5, 422.5), mincnt=min_shots, cmap=cmap, vmin=vmin, vmax=vmax,
                      linewidths=0.4, edgecolors=SURFACE, zorder=2)
    hexes.set_clip_path(Rectangle((-250, -47.5), 500, 470, transform=ax.transData))
    draw_court(ax)
    ax.set_xlim(-252, 252)
    ax.set_ylim(-50, 425)
    ax.set_aspect("equal")
    ax.axis("off")

    zones = half.assign(zone=zone_of(half), xpts=half[column] * half["value"]).groupby("zone")
    table = pd.DataFrame({"xfg": zones[column].mean(), "xpts": zones["xpts"].mean(), "share": zones.size() / len(half)})
    order = ["Restricted area", "Paint (non-RA)", "Mid-range", "Corner 3", "Above the break 3"]
    table = table.reindex([z for z in order if z in table.index])
    lines = [f"{'Zone':<19}{'xFG%':>6}{'Pts/shot':>10}{'Shots':>7}"]
    lines += [f"{z:<19}{r.xfg:>6.0%}{r.xpts:>10.2f}{r.share:>7.0%}" for z, r in table.iterrows()]
    ax.text(0, -62, "\n".join(lines), ha="center", va="top", family="monospace", fontsize=8.6, color=INK,
            linespacing=1.45)

    bar = fig.colorbar(hexes, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40,
                       location="top")
    bar.set_label("Expected FG% (average prediction of the shots taken there)", color=INK, fontsize=9.5)
    bar.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    bar.ax.tick_params(colors=MUTED, labelsize=8.5)
    bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, color=INK, fontsize=11.5, x=0.5, y=0.985)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
