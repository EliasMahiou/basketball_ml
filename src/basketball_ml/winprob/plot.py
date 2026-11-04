"""Charts: one game's live win probability, and the models' calibration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import pbp  # noqa: E402
from ..predictions.plot import GRID, INK, MUTED, SURFACE  # noqa: E402
from .model import NAMES, calibration, calibration_error, endgame  # noqa: E402

COLORS = {"stern": "#eb6834", "logistic": "#2a78d6", "boosting": "#1baf7a"}
HOME_WASH, AWAY_WASH = "#2a78d6", "#eb6834"
PERIODS = ("Q1", "Q2", "Q3", "Q4")
LABEL_BOX = dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none", alpha=0.92)
PLAYS = (pbp.MADE_SHOT, pbp.MISSED_SHOT, pbp.FREE_THROW, pbp.REBOUND, pbp.TURNOVER, pbp.FOUL)
OVERTIMES = {1: "overtime", 2: "double overtime", 3: "triple overtime"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _pct(p: float) -> str:
    """A probability as a percentage, with enough digits to show how small it is."""
    if p >= 0.1:
        return f"{p:.0%}"
    if p >= 0.01:
        return f"{p:.1%}"
    return f"{p:.2%}" if p >= 0.0001 else "<0.01%"


def _clock_text(period: int, seconds_left: int) -> str:
    clock = seconds_left - (4 - period) * 720 if period <= 4 else seconds_left
    where = f"Q{period}" if period <= 4 else ("OT" if period == 5 else f"{period - 4}OT")
    return f"{clock // 60}:{clock % 60:02d} left in {where}"


def _hits(box, x: np.ndarray, y: np.ndarray) -> int:
    """How many pieces of the step curve through (x, y) cross ``box`` (in data coordinates)."""
    nxt = np.append(x[1:], x[-1])
    flat = (nxt >= box.x0) & (x <= box.x1) & (y >= box.y0) & (y <= box.y1)
    lo, hi = np.minimum(y[:-1], y[1:]), np.maximum(y[:-1], y[1:])
    jump = (x[1:] >= box.x0) & (x[1:] <= box.x1) & (hi >= box.y0) & (lo <= box.y1)
    return int(flat.sum() + jump.sum())


def _crosses(box, start: tuple[float, float], end: tuple[float, float]) -> bool:
    """Whether the straight line from ``start`` to ``end`` passes through ``box``."""
    t = np.linspace(0.02, 0.98, 50)
    px, py = start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])
    return bool(((px >= box.x0) & (px <= box.x1) & (py >= box.y0) & (py <= box.y1)).any())


def _label(ax: plt.Axes, point: tuple[float, float], text: str, levels: Sequence[float], left: bool,
           curve: tuple[np.ndarray, np.ndarray], taken: list) -> None:
    """Annotate ``point`` at the first height in ``levels`` whose label clears the curve and earlier labels.

    ``taken`` holds the boxes and leader lines of earlier labels; this label's are added to it.
    """
    x0, x1 = ax.get_xlim()
    dx = (x1 - x0) * 0.02
    best = None
    for level in levels:
        anchor = (point[0] - dx if left else point[0] + dx, level)
        note = ax.annotate(text, point, xytext=anchor, textcoords="data", ha="right" if left else "left",
                           va="center", color=INK, fontsize=9, bbox=LABEL_BOX,
                           arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=5,
                                           relpos=(1.0, 0.5) if left else (0.0, 0.5)))
        ax.figure.canvas.draw()
        box = note.get_bbox_patch().get_window_extent().transformed(ax.transData.inverted())
        hits = _hits(box, *curve) + 500 * (box.x0 < x0 or box.x1 > x1)
        for other, start, end in taken:
            hits += 1000 * (box.overlaps(other) or _crosses(other, anchor, point) or _crosses(box, start, end))
        if best is not None and hits >= best[0]:
            note.remove()
            continue
        if best is not None:
            best[1].remove()
        best = (hits, note, (box, anchor, point))
        if hits == 0:
            break
    taken.append(best[2])


def game_plot(curve: pd.DataFrame, game: pd.Series, out_path: str | Path) -> Path:
    """Home win probability (top) and home lead (bottom) over one game, from ``wpa.game_curve``."""
    home, away = game["home"], game["away"]
    end = float(curve["minute"].max())
    fig, (ax, low_ax) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, low_ax):
        _style(a)
        a.grid(axis="x", visible=False)

    x, wp = curve["minute"].to_numpy(), curve["wp"].to_numpy()
    ax.fill_between(x, wp, 0.5, where=wp >= 0.5, step="post", color=HOME_WASH, alpha=0.12, linewidth=0)
    ax.fill_between(x, wp, 0.5, where=wp <= 0.5, step="post", color=AWAY_WASH, alpha=0.12, linewidth=0)
    ax.step(x, wp, where="post", color=INK, linewidth=1.6)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, end)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(f"{home} (home) win probability", color=INK, fontsize=10)
    for level, team in ((0.75, home), (0.25, away)):
        ax.text(1.01, level, f"{team}\nfavored", color=MUTED, fontsize=9, ha="left", va="center",
                transform=ax.get_yaxis_transform())

    # Quarter (and overtime) markers.
    bounds = [0, 12, 24, 36, 48] + [48 + 5 * k for k in range(1, round((end - 48) / 5) + 1)]
    labels = list(PERIODS) + ["OT" if k == 1 else f"{k}OT" for k in range(1, len(bounds) - 4)]
    for a in (ax, low_ax):
        for b in bounds[1:-1]:
            a.axvline(b, color=GRID, linewidth=1.0, zorder=0)
    for left, right, label in zip(bounds[:-1], bounds[1:], labels):
        ax.text((left + right) / 2, 1.015, label, color=MUTED, fontsize=9, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())

    # The eventual winner's lowest point.
    home_won = bool(game["home_win"])
    winner, loser = (home, away) if home_won else (away, home)
    winner_wp = wp if home_won else 1 - wp
    i = int(np.argmin(winner_wp[:-1]))
    row = curve.iloc[i]
    trailing = int(-row["margin"] if home_won else row["margin"])
    ax.plot([x[i]], [wp[i]], marker="o", markersize=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=5)
    note = f"{winner} at {_pct(winner_wp[i])}\n{_clock_text(int(row['period']), int(row['seconds_left']))}"
    if trailing > 0:
        note += f", down {trailing}"
    steps = (0.12, 0.22, 0.32, 0.45)
    levels = [min(wp[i] + d, 0.9) for d in steps] if wp[i] < 0.5 else [max(wp[i] - d, 0.1) for d in steps]
    taken: list = []
    _label(ax, (x[i], wp[i]), note, levels, x[i] > end * 0.6, (x, wp), taken)

    # The play that helped the winner most (a player's action, not the clock or a period start).
    if "description" in curve:
        gain = (curve["delta"] * (1 if home_won else -1)).to_numpy()
        plays = curve["event"].isin(PLAYS) & curve["description"].fillna("").astype(str).str.strip().ne("")
        gain = np.where(plays.to_numpy(), gain, -np.inf)[:-1]
        j = int(np.argmax(gain))
        play = curve.iloc[j]
        text = re.sub(r"\s*\(\d+ PTS\)", "", str(play["description"]))
        text = re.sub(r"\s+", " ", re.sub(r" \d+ AST\)", " assist)", text)).strip()
        text = text if len(text) <= 42 else text[:41] + "..."
        ax.plot([x[j]], [wp[j]], marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5)
        # Start on the side of 50% that the curve stayed away from in the minutes before.
        before = wp[(x >= x[j] - 8) & (x <= x[j])]
        levels = (0.8, 0.9, 0.7, 0.6) if before.mean() < 0.5 else (0.2, 0.1, 0.3, 0.4)
        _label(ax, (x[j], wp[j]), f"{text}\n{_clock_text(int(play['period']), int(play['seconds_left']))}: "
               f"{winner} +{gain[j]:.0%}", levels, x[j] > end * 0.6, (x, wp), taken)

    margin = curve["margin"].to_numpy()
    low_ax.fill_between(x, margin, 0, where=margin >= 0, step="post", color=HOME_WASH, alpha=0.12, linewidth=0)
    low_ax.fill_between(x, margin, 0, where=margin <= 0, step="post", color=AWAY_WASH, alpha=0.12, linewidth=0)
    low_ax.step(x, margin, where="post", color=INK, linewidth=1.2)
    low_ax.axhline(0, color=MUTED, linewidth=0.8)
    span = max(5, int(np.abs(margin).max()))
    low_ax.set_ylim(-span * 1.15, span * 1.15)
    low_ax.set_ylabel(f"{home} lead", color=INK, fontsize=10)
    low_ax.set_xlim(0, end)
    low_ax.set_xticks(bounds)
    low_ax.set_xlabel("Minutes played", color=INK, fontsize=10)

    date = pd.Timestamp(game["date"])
    title = f"{away} at {home}, {date:%B} {date.day}, {date.year}"
    overtimes = int(curve["period"].max()) - 4
    extra = f" in {OVERTIMES.get(overtimes, f'{overtimes} overtimes')}" if overtimes > 0 else ""
    sub = (f"{winner} beat {loser} {max(game['home_pts'], game['away_pts'])}-"
           f"{min(game['home_pts'], game['away_pts'])}{extra} after falling to a {_pct(winner_wp[i])} chance")
    fig.text(0.125, 0.975, title, color=INK, fontsize=12.5, ha="left", va="bottom")
    fig.text(0.125, 0.948, sub, color=MUTED, fontsize=9.5, ha="left", va="bottom")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def calibration_plot(results: pd.DataFrame, out_path: str | Path,
                     methods: Sequence[str] = ("stern", "logistic", "boosting"), bins: int = 10,
                     min_states: int = 200) -> Path:
    """Predicted vs actual home win rate, for all states and for the last 2 minutes, with states per bin."""
    late = endgame(results)
    panels = (("All game states", None), ("Last 2 minutes of the 4th quarter and overtime", late))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.18})
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / bins / len(methods)
    for col, (title, mask) in enumerate(panels):
        ax, bars = axes[0, col], axes[1, col]
        for a in (ax, bars):
            _style(a)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        n = len(results) if mask is None else int(mask.sum())
        for k, m in enumerate(methods):
            table = calibration(results, m, bins, mask)
            table = table[table["states"] >= min_states]
            gap = calibration_error(results, m, bins, mask)
            ax.plot(table["predicted"], table["observed"], color=COLORS[m], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{NAMES[m]} (average gap {gap:.1%})")
            offset = (k - (len(methods) - 1) / 2) * width
            centers = (np.clip(np.floor(table["predicted"] * bins), 0, bins - 1) + 0.5) / bins
            bars.bar(centers + offset, table["states"] / 1000, width=width * 0.9, color=COLORS[m], linewidth=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{title}: {n:,} states", color=INK, fontsize=10.5, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
        bars.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        bars.set_xlabel("Predicted home win probability", color=INK, fontsize=10)
        if col == 0:
            ax.set_ylabel("Home teams that won", color=INK, fontsize=10)
            bars.set_ylabel("States (thousands)", color=INK, fontsize=10)
    games = results["game_id"].nunique()
    fig.text(0.125, 0.965, f"Live win probability, predicted vs actual: {games:,} out-of-sample games",
             color=INK, fontsize=12, ha="left", va="bottom")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
