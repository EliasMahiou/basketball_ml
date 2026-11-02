"""Backtest comparables projections against simple baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .comparables import TARGETS, age_curve, project_season, projection_range


def backtest(
    pairs: pd.DataFrame,
    seasons: Sequence[int],
    horizon: int = 1,
    k: int = 25,
    targets: Sequence[str] = TARGETS,
    match: Sequence[str] | None = None,
    history_weight: float = 0.0,
) -> pd.DataFrame:
    """Project every player-season in ``seasons`` and compare with what happened.

    Each projection only uses comparables whose outcome was known by the end of
    the target season. Returns one row per projected player-season with the
    actual value, the comparables projection and two baselines for each stat:
    ``<stat>_same`` (no change) and ``<stat>_age`` (average aging curve), plus
    the comparables' 80% range (``<stat>_low``, ``<stat>_high``).
    """
    targets = list(targets)
    rows = []
    for season in seasons:
        history = pairs[pairs["season"] + horizon <= season]
        curve = age_curve(history, targets)
        for _, target in pairs[pairs["season"] == season].iterrows():
            raw_now = pd.Series({t: target[f"{t}_now"] for t in targets})
            for t in targets:
                if f"{t}_hist" in target:
                    raw_now[f"{t}_hist"] = target[f"{t}_hist"]
            try:
                proj, comps = project_season(target, raw_now, history, horizon=horizon, k=k, targets=targets,
                                             history_weight=history_weight,
                                             **({"match": match} if match is not None else {}))
            except ValueError:
                continue
            row = {"player": target["player"], "season": season, "age": target["age"]}
            rng = projection_range(comps, raw_now, targets, history_weight=history_weight)
            age_delta = curve.loc[target["age"]] if target["age"] in curve.index else None
            for t in targets:
                row[f"{t}_actual"] = target[f"{t}_later"]
                row[f"{t}_comps"] = proj[t]
                row[f"{t}_low"] = rng.loc[t, "low"]
                row[f"{t}_high"] = rng.loc[t, "high"]
                row[f"{t}_same"] = target[f"{t}_now"]
                row[f"{t}_age"] = target[f"{t}_now"] + (age_delta[f"{t}_delta"] if age_delta is not None else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def score(results: pd.DataFrame, targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    """Mean absolute error of each method per stat, and how often the 80% range held."""
    out = {}
    for t in targets:
        actual = results[f"{t}_actual"].to_numpy()
        out[t] = {m: float(np.mean(np.abs(results[f"{t}_{m}"].to_numpy() - actual)))
                  for m in ("comps", "age", "same")}
        if f"{t}_low" in results:
            inside = (actual >= results[f"{t}_low"].to_numpy()) & (actual <= results[f"{t}_high"].to_numpy())
            out[t]["coverage"] = float(inside.mean())
    return pd.DataFrame(out).T.rename(columns={"comps": "comparables", "age": "age_curve", "same": "no_change",
                                               "coverage": "range_80_coverage"})
