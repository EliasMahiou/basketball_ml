"""Project a player's next seasons from his statistical twins at the same age.

How it works:
1. Find the ``k`` player-seasons most similar to the target season (per-season
   z-scores, see :mod:`basketball_ml.twins`), among players of the same age
   whose season ``h`` years later is already known.
2. For each twin, measure how each stat changed between that season and the
   season ``h`` years later (raw units, e.g. +1.2 BPM).
3. The projection is the target's current value plus the similarity-weighted
   average of those changes.

Only seasons that ended before the target season can serve as comparables,
so backtests do not peek into the future.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..data import FEATURES
from ..twins.similarity import similarity_score

# Stats projected by default: the 12 twin features plus overall impact.
TARGETS: tuple[str, ...] = (*FEATURES, "bpm")


def build_pairs(seasons: pd.DataFrame, z: pd.DataFrame, horizon: int = 1,
                targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    """Pair each player-season with the same player's season ``horizon`` years later.

    Returns the z-scores of the starting season (for matching) and the raw
    change of each target stat (``<stat>_delta``).
    """
    feats = list(FEATURES)
    targets = list(targets)
    base = z[["player_id", "player", "season", "age", *feats]].copy()
    raw = seasons[["player_id", "season", *targets]]
    now = raw.rename(columns={t: f"{t}_now" for t in targets})
    later = raw.rename(columns={t: f"{t}_later" for t in targets})
    later = later.assign(season=later["season"] - horizon)
    pairs = base.merge(now, on=["player_id", "season"]).merge(later, on=["player_id", "season"])
    for t in targets:
        pairs[f"{t}_delta"] = pairs[f"{t}_later"] - pairs[f"{t}_now"]
    pairs["age"] = pairs["age"].astype(int)
    return pairs


def project_season(
    target_z: pd.Series,
    target_raw: pd.Series,
    pairs: pd.DataFrame,
    horizon: int = 1,
    k: int = 25,
    age_window: int = 0,
    targets: Sequence[str] = TARGETS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Project ``target`` ``horizon`` seasons ahead.

    ``target_z`` holds the target season's z-scores, ``target_raw`` its raw
    stats. Returns the projected raw stats and the comparables used.
    """
    feats = list(FEATURES)
    age = int(target_z["age"])
    season = int(target_z["season"])
    pool = pairs[
        (pairs["player_id"] != target_z["player_id"])
        & ((pairs["age"] - age).abs() <= age_window)
        & (pairs["season"] + horizon < season + 1)  # comparable outcome known before the target season ends
    ]
    if len(pool) == 0:
        raise ValueError(f"No comparables at age {age} before {season}")

    diff = pool[feats].to_numpy(dtype=float) - target_z[feats].to_numpy(dtype=float)
    dist = np.sqrt((diff**2).sum(axis=1))
    order = np.argsort(dist)[:k]
    comps = pool.iloc[order].copy()
    comps["similarity"] = similarity_score(dist[order], len(feats))

    w = comps["similarity"].to_numpy()
    w = w / w.sum()
    deltas = comps[[f"{t}_delta" for t in targets]].to_numpy(dtype=float)
    projected = target_raw[list(targets)].to_numpy(dtype=float) + w @ deltas
    return pd.Series(projected, index=list(targets)), comps.reset_index(drop=True)


def age_curve(pairs: pd.DataFrame, targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    """Average change of each stat by age: the classic aging-curve baseline."""
    return pairs.groupby("age")[[f"{t}_delta" for t in targets]].mean()
