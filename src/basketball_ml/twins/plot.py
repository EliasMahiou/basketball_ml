"""Player vs twin radar chart, in percentiles (0 = lowest, 100 = highest)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..data import FEATURES  # noqa: E402


def add_percentiles(table: pd.DataFrame, features: Sequence[str] = tuple(FEATURES)) -> pd.DataFrame:
    """Add ``<stat>_pct``: percentile within the season, or among all careers."""
    out = table.copy()
    feats = list(features)
    ranked = out.groupby("season")[feats] if "season" in out else out[feats]
    pct = ranked.rank(pct=True) * 100
    out[[f"{f}_pct" for f in feats]] = pct.to_numpy()
    return out


def _label(row: pd.Series) -> str:
    if "season" in row and not pd.isna(row.get("season")):
        s = int(row["season"])
        return f"{row['player']} ({s - 1}-{str(s)[-2:]})"
    return f"{row['player']} ({int(row['first_season']) - 1}-{str(int(row['last_season']))[-2:]})"


def radar(rows: Sequence[pd.Series], out_path: str | Path, title: str = "",
          features: Sequence[str] = tuple(FEATURES)) -> Path:
    """Draw a percentile radar for each row and save it as a PNG."""
    feats = list(features)
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#552583", "#FDB927", "#1D428A", "#CE1141"]
    for row, color in zip(rows, colors):
        values = [float(row[f"{f}_pct"]) for f in feats]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=_label(row))
        ax.fill(angles, values, color=color, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FEATURES.get(f, f) for f in feats], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], fontsize=8, color="grey")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False)
    if title:
        ax.set_title(title, pad=24, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
