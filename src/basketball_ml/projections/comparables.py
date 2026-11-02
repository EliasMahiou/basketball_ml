"""Project a player's next seasons from his statistical twins at the same age.

How it works:
1. Find the ``k`` player-seasons most similar to the target season (per-season
   z-scores, see :mod:`basketball_ml.twins`), among players of the same age
   whose season ``h`` years later is already known.
2. For each twin, measure how each stat changed between that season and the
   season ``h`` years later (raw units, e.g. +1.2 BPM).
3. The projection is the target's starting point plus the similarity-weighted
   average of those changes. The starting point blends the current season with
   a 3-season weighted average (5/4/3 recency weights times minutes), which
   smooths out one-season noise. The 10th-90th weighted percentiles of the same
   changes give an 80% range.

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


HISTORY_WEIGHTS = (5, 4, 3)


def add_history(seasons: pd.DataFrame, targets: Sequence[str] = TARGETS,
                weights: Sequence[float] = HISTORY_WEIGHTS) -> pd.DataFrame:
    """Add ``<stat>_hist``: recency- and minutes-weighted average over the last 3 seasons.

    The current season gets weight 5, the one before 4, then 3, each times
    minutes played; missing seasons are skipped.
    """
    targets = list(targets)
    out = seasons.copy()
    num = {t: np.zeros(len(out)) for t in targets}
    den = np.zeros(len(out))
    key = out[["player_id", "season"]]
    for lag, w in enumerate(weights):
        past = seasons[["player_id", "season", "mp", *targets]].assign(season=seasons["season"] + lag)
        past = key.merge(past, on=["player_id", "season"], how="left")
        mp = past["mp"].fillna(0).to_numpy(dtype=float) * w
        den += mp
        for t in targets:
            num[t] += mp * past[t].fillna(0).to_numpy(dtype=float)
    for t in targets:
        out[f"{t}_hist"] = np.where(den > 0, num[t] / np.where(den > 0, den, 1), out[t])
    return out


def starting_point(raw: pd.Series, targets: Sequence[str], history_weight: float) -> np.ndarray:
    """Blend of the current season and its ``<stat>_hist`` average (when present)."""
    now = np.array([float(raw[t]) for t in targets])
    if history_weight == 0 or f"{targets[0]}_hist" not in raw:
        return now
    hist = np.array([float(raw[f"{t}_hist"]) for t in targets])
    return (1 - history_weight) * now + history_weight * hist


def build_pairs(seasons: pd.DataFrame, z: pd.DataFrame, horizon: int = 1,
                targets: Sequence[str] = TARGETS, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Pair each player-season with the same player's season ``horizon`` years later.

    Returns the z-scores of the starting season (for matching), any ``extra``
    matching columns of ``z``, and the raw change of each target stat
    (``<stat>_delta``).
    """
    feats = list(FEATURES) + [c for c in extra if c not in FEATURES]
    targets = list(targets)
    base = z[["player_id", "player", "season", "age", *feats]].copy()
    hist = [f"{t}_hist" for t in targets if f"{t}_hist" in seasons]
    raw = seasons[["player_id", "season", *targets]]
    now = seasons[["player_id", "season", *targets, *hist]].rename(columns={t: f"{t}_now" for t in targets})
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
    match: Sequence[str] = tuple(FEATURES),
    history_weight: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Project ``target`` ``horizon`` seasons ahead.

    ``target_z`` holds the target season's z-scores, ``target_raw`` its raw
    stats (and ``<stat>_hist`` averages, see :func:`add_history`). ``match``
    lists the z-score columns used to find comparables, and ``history_weight``
    how much the starting point leans on the 3-season average.
    Returns the projected raw stats and the comparables used.
    """
    feats = list(match)
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
    projected = starting_point(target_raw, list(targets), history_weight) + w @ deltas
    return pd.Series(projected, index=list(targets)), comps.reset_index(drop=True)


def age_curve(pairs: pd.DataFrame, targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    """Average change of each stat by age: the classic aging-curve baseline."""
    return pairs.groupby("age")[[f"{t}_delta" for t in targets]].mean()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile ``q`` of ``values`` where each value counts with its weight."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def projection_range(
    comps: pd.DataFrame,
    target_raw: pd.Series,
    targets: Sequence[str] = TARGETS,
    low: float = 0.1,
    high: float = 0.9,
    history_weight: float = 0.0,
) -> pd.DataFrame:
    """Range of plausible outcomes: weighted percentiles of the comparables' changes.

    Returns one row per stat with ``low`` and ``high`` columns (raw units).
    """
    w = comps["similarity"].to_numpy(dtype=float)
    targets = list(targets)
    bases = dict(zip(targets, starting_point(target_raw, targets, history_weight)))
    rows = {}
    for t in targets:
        d = comps[f"{t}_delta"].to_numpy(dtype=float)
        base = bases[t]
        rows[t] = {"low": base + weighted_quantile(d, w, low), "high": base + weighted_quantile(d, w, high)}
    return pd.DataFrame(rows).T
