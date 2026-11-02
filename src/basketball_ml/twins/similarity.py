"""Find statistical twins with nearest neighbors.

How it works:
1. Each stat is standardized *within its season* (z-score), which removes
   era effects (pace, 3-point volume...).
2. A player is a vector of z-scores, either for one season or for a whole
   career (mean of season z-scores weighted by minutes played).
3. Twins are the closest vectors by Euclidean distance, converted into a
   similarity score out of 100.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import FEATURES


def standardize_by_season(df: pd.DataFrame, features: Sequence[str] = tuple(FEATURES)) -> pd.DataFrame:
    """Replace each stat with its z-score within the season."""
    out = df.copy()
    grouped = out.groupby("season")[list(features)]
    out[list(features)] = (out[list(features)] - grouped.transform("mean")) / grouped.transform("std")
    return out


def career_profiles(z: pd.DataFrame, features: Sequence[str] = tuple(FEATURES)) -> pd.DataFrame:
    """Career profile: mean of season z-scores, weighted by minutes played."""
    feats = list(features)
    weighted = z[feats].mul(z["mp"], axis=0)
    weighted[["player_id", "mp"]] = z[["player_id", "mp"]]
    sums = weighted.groupby("player_id").sum()
    prof = sums[feats].div(sums["mp"], axis=0)
    info = z.groupby("player_id").agg(
        player=("player", "last"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        mp=("mp", "sum"),
    )
    return info.join(prof).reset_index()


def similarity_score(distance: np.ndarray, n_features: int) -> np.ndarray:
    """Convert a distance into a score out of 100 (100 = identical profile).

    The distance is scaled to the average gap per stat, then passed through a
    decaying exponential: an average gap of half a standard deviation per stat
    gives about 61/100.
    """
    return 100.0 * np.exp(-distance / np.sqrt(n_features))


def _normalize(name: str) -> str:
    """Lowercase and strip accents: "Jokić" -> "jokic"."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def resolve_player(table: pd.DataFrame, name: str) -> str:
    """Find the ``player_id`` for a name (case- and accent-insensitive).

    "nikola jokic" matches "Nikola Jokić". When several players share the
    name, pick the one with the most minutes.
    """
    names = table["player"].map(_normalize)
    query = _normalize(name)
    hits = table[names == query]
    if hits.empty:
        hits = table[names.str.contains(query, regex=False)]
    if hits.empty:
        raise KeyError(f"Player not found: {name!r}")
    return hits.groupby("player_id")["mp"].sum().idxmax()


def find_twins(
    table: pd.DataFrame,
    player_id: str,
    season: int | None = None,
    k: int = 10,
    features: Sequence[str] = tuple(FEATURES),
    weights: Mapping[str, float] | None = None,
    one_row_per_player: bool = True,
) -> pd.DataFrame:
    """Return the ``k`` rows of ``table`` closest to the target player.

    ``table`` is either the output of :func:`standardize_by_season`
    (one row per season, ``season`` required) or that of
    :func:`career_profiles` (one row per player, ``season=None``).
    The target player is excluded from the results.
    """
    feats = list(features)
    w = np.array([1.0 if weights is None else weights.get(f, 1.0) for f in feats])

    mask = table["player_id"] == player_id
    if season is not None:
        mask &= table["season"] == season
    if not mask.any():
        raise KeyError(f"No row for {player_id} (season {season})")
    target = table.loc[mask, feats].to_numpy(dtype=float)[0]

    cands = table[table["player_id"] != player_id].copy()
    diff = (cands[feats].to_numpy(dtype=float) - target) * np.sqrt(w)
    cands["distance"] = np.sqrt((diff**2).sum(axis=1))
    cands["similarity"] = similarity_score(cands["distance"].to_numpy(), len(feats))
    cands = cands.sort_values("distance")
    if one_row_per_player:
        cands = cands.drop_duplicates("player_id")
    return cands.head(k).reset_index(drop=True)
