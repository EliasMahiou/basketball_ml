"""Group players into style archetypes and draw a 2D style map."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

LABELS = {
    "share_rim": "rim attempts", "share_paint": "paint attempts", "share_mid": "mid-range",
    "share_corner3": "corner 3s", "share_above3": "above-break 3s",
    "fg_rim": "rim finishing", "fg_paint": "paint touch", "fg_mid": "mid-range making",
    "fg_corner3": "corner 3 making", "fg_above3": "3-point making",
    "type_dunk": "dunks", "type_layup": "layups", "type_putback": "putbacks", "type_floater": "floaters",
    "type_hook": "hooks", "type_post_fade": "post fadeaways", "type_pullup": "pull-ups",
    "type_driving": "drives", "type_cutting": "cuts", "type_alley_oop": "lobs",
    "type_catch_shoot": "catch and shoot", "rate_fga": "shooting volume", "rate_fta": "free throws",
    "rate_ast": "assists", "rate_oreb": "offensive boards", "rate_dreb": "defensive boards",
    "rate_stl": "steals", "rate_blk": "blocks", "rate_tov": "turnovers", "rate_pf": "fouls",
    "rate_fouls_drawn": "fouls drawn", "assisted_share": "assisted makes",
    "bad_pass_share": "risky passing", "shot_distance": "shot distance",
}


def cluster(table: pd.DataFrame, columns: Sequence[str], k: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means archetypes on style z-scores.

    Returns the table with an ``archetype`` column and one row per archetype
    naming its three most distinctive traits (highest average z-scores).
    """
    cols = list(columns)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(table[cols].to_numpy(float))
    out = table.assign(archetype=km.labels_)
    centers = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = []
    for i, row in centers.iterrows():
        top = row.sort_values(ascending=False).index[:3]
        names.append({"archetype": i, "traits": ", ".join(LABELS.get(c, c) for c in top),
                      "players": int((km.labels_ == i).sum())})
    return out, pd.DataFrame(names)


def style_axes(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Two main style axes (PCA), oriented so that x grows toward the rim and y toward spot-up play."""
    cols = list(columns)
    pca = PCA(n_components=2, random_state=0).fit(table[cols].to_numpy(float))
    xy = pca.transform(table[cols].to_numpy(float))
    loadings = pd.DataFrame(pca.components_.T, index=cols)
    if "share_rim" in loadings.index and loadings.loc["share_rim", 0] < 0:
        xy[:, 0] *= -1
    if "assisted_share" in loadings.index and loadings.loc["assisted_share", 1] < 0:
        xy[:, 1] *= -1
    return xy


def style_map(table: pd.DataFrame, columns: Sequence[str], out_path: str | Path,
              names: pd.DataFrame | None = None, highlight: Sequence[str] = (), title: str = "") -> Path:
    """Scatter plot of players on the two main style axes, colored by archetype."""
    xy = style_axes(table, columns)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    groups = table["archetype"].to_numpy() if "archetype" in table else np.zeros(len(table), int)
    labels = dict(zip(names["archetype"], names["traits"])) if names is not None else {}
    cmap = plt.get_cmap("tab10")
    for g in np.unique(groups):
        m = groups == g
        ax.scatter(xy[m, 0], xy[m, 1], s=18, alpha=0.65, color=cmap(int(g) % 10),
                   label=labels.get(g, f"Archetype {g}"))
    ax.set_xlabel("Perimeter  \u2190  style axis 1  \u2192  Interior")
    ax.set_ylabel("Self-created  \u2190  style axis 2  \u2192  Spot-up")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _label_points(fig, ax, xy, table["player"].to_numpy(), highlight)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _label_points(fig, ax, xy: np.ndarray, player_names: np.ndarray, highlight: Sequence[str]) -> None:
    """Mark and name the ``highlight`` players without labels covering each other or their dots.

    Each label tries positions around its dot, nearest first, and keeps the
    first free one; a thin leader line ties it to the dot.
    """
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    points = [np.where(player_names == name)[0] for name in highlight]
    placed = [Bbox.from_bounds(*(ax.transData.transform(xy[i[0]]) - 7), 14, 14) for i in points if len(i)]
    offsets = [(dx * r, dy * r) for r in (8, 16, 26)
               for dx, dy in ((1, 0.6), (-1, 0.6), (1, -0.9), (-1, -0.9), (0, 1.2), (0, -1.5))]
    for name, idx in zip(highlight, points):
        if not len(idx):
            continue
        i = idx[0]
        ax.scatter(xy[i, 0], xy[i, 1], s=60, color="black", zorder=3)
        for k, (dx, dy) in enumerate(offsets):
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top"
            label = ax.annotate(name, xy[i], xytext=(dx, dy), textcoords="offset points", fontsize=9,
                                fontweight="bold", ha=ha, va=va, zorder=4,
                                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6, shrinkA=1, shrinkB=4))
            label.update_positions(renderer)
            box = Text.get_window_extent(label, renderer).expanded(1.04, 1.1)
            if k == len(offsets) - 1 or not any(box.overlaps(other) for other in placed):
                break
            label.remove()
        placed.append(box)
