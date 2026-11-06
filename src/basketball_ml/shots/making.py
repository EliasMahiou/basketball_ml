"""Shot making and shot quality: points scored above what an average shooter would score on the same shots.

The expected points of a shot are its xFG times its value (2 or 3). xFG comes
from the walk-forward model (out of sample), shifted so that each season the
league's makes equal its expected makes, separately for twos and threes: the
model is trained on earlier seasons and would otherwise carry their league
averages (2-point accuracy has risen almost every season). Twos are also
corrected for each arena's scorekeeper bias (see :func:`arena_bias`).

A season still being played is passed as ``in_progress``: its arena biases
lean on the previous season's until enough games are in.

- Shot quality: average expected points per shot (xPPS).
- Shot making: actual points per shot minus xPPS.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

MIN_SHOTS = 300   # attempts for leaderboards and year-to-year comparisons
PRIOR_MIN_SHOTS = 200   # player-seasons used to estimate the spread of true shot making
CARRY_MIN_SHOTS = 100   # attempts last season for its shot making to be this season's starting point
# Weight of last season's arena biases in a season in progress, in games: n games in
# weigh n / (n + 600). An arena's bias carries over (year-to-year r 0.77 from 2015-16
# to 2025-26), and this blend predicted the rest of the season best from 60 to 800 games in.
ARENA_PRIOR_GAMES = 600


def _league_level(shots: pd.DataFrame, p: pd.Series) -> pd.Series:
    gap = (shots["made"] - p).groupby([shots["season"], shots["value"]]).transform("mean")
    return p + gap


def arena_bias(shots: pd.DataFrame, p: pd.Series, in_progress: Collection[int] = ()) -> pd.Series:
    """Scorekeeper bias of each arena and season on 2-point shots, in make-probability units (0 for threes).

    Some arenas log shots farther from the rim than others (fewer
    restricted-area shots), so twos there go in more often than their recorded
    location implies, for both teams. A team's own skill is the same at home
    and on the road, and the home edge cancels between its offense and its
    defense, so an arena's bias is half the sum of two home-minus-road gaps
    (makes minus ``p``): the home team's shots and its opponents' shots.
    Needs ``home`` (1 when the shooter is at home), ``team`` and ``opponent``.

    A few weeks in, a season's own games are too few to measure its biases,
    so for seasons in ``in_progress`` the estimate after n games is blended
    with the previous season's: weight n / (n + :data:`ARENA_PRIOR_GAMES`)
    on the season's own (an arena missing last season counts as 0).
    """
    two = (shots["value"] == 2).to_numpy()
    rows = pd.DataFrame({"season": shots["season"], "gap": shots["made"] - p, "home": shots["home"] == 1,
                         "offense": shots["team"].astype(str), "defense": shots["opponent"].astype(str)})[two]
    off = rows.groupby(["season", "offense", "home"])["gap"].mean().unstack()
    dfn = rows.groupby(["season", "defense", "home"])["gap"].mean().unstack()
    # The defense is at home when the shooter is not.
    bias = 0.5 * ((off[True] - off[False]) + (dfn[False] - dfn[True]).reindex(off.index))
    bias = bias - bias.groupby(level=0).transform("mean")
    seasons = bias.index.get_level_values(0)
    for season in set(in_progress) & set(seasons):
        now = seasons == season
        games = shots.loc[shots["season"] == season, "game_id"].nunique()
        weight = games / (games + ARENA_PRIOR_GAMES)
        last = bias[seasons == season - 1].droplevel(0)
        prior = pd.Series(bias.index.get_level_values(1)[now]).map(last).fillna(0.0).to_numpy()
        blend = weight * bias[now].to_numpy() + (1 - weight) * prior
        bias[now] = blend - blend.mean()
    arena = np.where(shots["home"] == 1, shots["team"].astype(str), shots["opponent"].astype(str))
    out = bias.reindex(pd.MultiIndex.from_arrays([shots["season"], arena])).to_numpy()
    return pd.Series(np.where(two, np.nan_to_num(out), 0.0), index=shots.index)


def recenter(shots: pd.DataFrame, xfg: str = "xfg", arenas: bool = True,
             in_progress: Collection[int] = ()) -> pd.Series:
    """xFG of an average shooter in that season and arena.

    Shifted so that league makes equal expected makes in every season, for
    twos and threes separately, and, when ``arenas`` and a ``home`` column are
    there, corrected for each arena's scorekeeper bias on twos (:func:`arena_bias`).
    """
    p = _league_level(shots, shots[xfg].astype(float))
    if arenas and "home" in shots:
        p = _league_level(shots, p + arena_bias(shots, p, in_progress))
    return p.clip(0.001, 0.999)


def _shot_values(shots: pd.DataFrame, xfg: str, in_progress: Collection[int] = ()) -> pd.DataFrame:
    p = recenter(shots, xfg, in_progress=in_progress).to_numpy(float)
    value = shots["value"].to_numpy(float)
    made = shots["made"].to_numpy(float)
    return pd.DataFrame({"made": made, "three": (value == 3).astype(float), "three_made": made * (value == 3),
                         "pts": made * value, "xpts": p * value, "xfg": p, "var": value ** 2 * p * (1 - p)},
                        index=shots.index)


def player_seasons(shots: pd.DataFrame, xfg: str = "xfg", prior_min_shots: int = PRIOR_MIN_SHOTS,
                   in_progress: Collection[int] = ()) -> pd.DataFrame:
    """One row per player and season: attempts, FG%, eFG%, xFG%, points added and shrunk shot making.

    ``points_added`` is actual minus expected points; ``per_100`` is the same
    per 100 shots, ``per_100_shrunk`` its empirical-Bayes estimate (see
    :func:`shrink`) and ``per_100_skill`` the same starting from last
    season's (see :func:`carry_over`). ``team`` is where he took the most
    shots. Seasons in ``in_progress`` are still being played (see
    :func:`arena_bias`).
    """
    v = _shot_values(shots, xfg, in_progress)
    keys = [shots["season"], shots["player_id"]]
    g = v.groupby(keys)
    table = pd.DataFrame({"fga": g.size(), "fgm": g["made"].sum(), "fg3a": g["three"].sum(),
                          "fg3m": g["three_made"].sum(), "pts": g["pts"].sum(), "xpts": g["xpts"].sum(),
                          "xfg_pct": g["xfg"].mean(), "var": g["var"].sum()})
    table.index.names = ["season", "player_id"]
    who = shots[["season", "player_id", "player", "team"]].astype({"player": str, "team": str})
    table["player"] = who.groupby(["season", "player_id"])["player"].last()
    main_team = who.groupby(["season", "player_id", "team"]).size().reset_index(name="n").sort_values("n")
    table["team"] = main_team.drop_duplicates(["season", "player_id"], keep="last").set_index(
        ["season", "player_id"])["team"]
    table["fg_pct"] = table["fgm"] / table["fga"]
    table["efg_pct"] = (table["fgm"] + 0.5 * table["fg3m"]) / table["fga"]
    table["xefg_pct"] = table["xpts"] / (2 * table["fga"])
    table["fg_above"] = table["fg_pct"] - table["xfg_pct"]
    table["pps"] = table["pts"] / table["fga"]
    table["xpps"] = table["xpts"] / table["fga"]
    table["points_added"] = table["pts"] - table["xpts"]
    table["per_100"] = 100 * table["points_added"] / table["fga"]
    return carry_over(shrink(table.reset_index(), prior_min_shots), prior_min_shots, in_progress)


def shrink(table: pd.DataFrame, prior_min_shots: int = PRIOR_MIN_SHOTS) -> pd.DataFrame:
    """Add ``per_100_shrunk``: each player's shot making pulled toward 0 (average) by its noise.

    A season's shot making is true skill plus luck. The luck's variance is
    known from the shots themselves (each shot's value squared times
    xFG (1 - xFG), summed, over attempts squared); the spread of true skill,
    tau, is the spread left after removing it, among player-seasons with at
    least ``prior_min_shots`` attempts. Each estimate keeps the share
    tau^2 / (tau^2 + noise) of the observed value. ``attrs`` hold tau (points
    per 100 shots) and ``half_shots``, the attempts at which half of the
    observed spread is skill.
    """
    t = table.copy()
    per_shot = t["points_added"] / t["fga"]
    noise = t["var"] / t["fga"] ** 2
    base = t["fga"] >= prior_min_shots
    tau2 = max(float(per_shot[base].var() - noise[base].mean()), 1e-8)
    t["per_100_shrunk"] = 100 * per_shot * tau2 / (tau2 + noise)
    t.attrs.update(tau=100 * np.sqrt(tau2), half_shots=float((t["var"] / t["fga"]).mean() / tau2))
    return t


def carry_over(table: pd.DataFrame, prior_min_shots: int = PRIOR_MIN_SHOTS,
               in_progress: Collection[int] = ()) -> pd.DataFrame:
    """Add ``per_100_skill``: shot making shrunk toward what last season's predicts instead of toward 0.

    Shot making carries over: a player's per-shot making is on average
    ``rho`` times last season's shrunk estimate, and true skill spreads less
    around that (``s``) than around 0 (tau). Both are fitted on players with
    ``prior_min_shots`` attempts in a finished season and
    :data:`CARRY_MIN_SHOTS` the season before. The estimate keeps
    s^2 / (s^2 + noise) of the season's own making and the rest of the
    prediction, so it starts from last season and moves as shots come in.
    Players without :data:`CARRY_MIN_SHOTS` attempts last season keep
    ``per_100_shrunk``. ``attrs`` hold ``rho`` and ``s`` (points per 100 shots).
    """
    t = table.copy()
    last = t[["season", "player_id", "fga"]].assign(season=t["season"] + 1, m=t["per_100_shrunk"] / 100)
    prev = last.set_index(["season", "player_id"]).reindex(pd.MultiIndex.from_arrays([t["season"], t["player_id"]]))
    has = (prev["fga"] >= CARRY_MIN_SHOTS).to_numpy()
    before = prev["m"].fillna(0.0).to_numpy()
    per_shot = (t["points_added"] / t["fga"]).to_numpy()
    noise = (t["var"] / t["fga"] ** 2).to_numpy()
    fit = has & (t["fga"] >= prior_min_shots).to_numpy() & ~t["season"].isin(in_progress).to_numpy()
    if not fit.any():
        t["per_100_skill"] = t["per_100_shrunk"]
        return t
    rho = float((per_shot[fit] * before[fit]).sum() / (before[fit] ** 2).sum())
    s2 = max(float(np.var(per_shot[fit] - rho * before[fit], ddof=1) - noise[fit].mean()), 1e-8)
    skill = (rho * before * noise + per_shot * s2) / (s2 + noise)
    t["per_100_skill"] = np.where(has, 100 * skill, t["per_100_shrunk"])
    t.attrs.update(rho=rho, s=100 * np.sqrt(s2))
    return t


def team_quality(shots: pd.DataFrame, xfg: str = "xfg", in_progress: Collection[int] = ()) -> pd.DataFrame:
    """Per team and season: shot quality and shot making on offense, and what it allowed on defense.

    ``off_xpps`` is the average expected points of the team's shots, ``off_pps``
    its actual points per shot and ``off_making`` the difference; the
    ``def_`` columns are the same for its opponents' shots. Seasons in
    ``in_progress`` are still being played (see :func:`arena_bias`).
    """
    v = _shot_values(shots, xfg, in_progress)
    parts = []
    for side, team in (("off", "team"), ("def", "opponent")):
        g = v.groupby([shots["season"], shots[team].astype(str)])
        part = pd.DataFrame({f"{side}_fga": g.size(), f"{side}_pps": g["pts"].mean(), f"{side}_xpps": g["xpts"].mean()})
        part[f"{side}_making"] = part[f"{side}_pps"] - part[f"{side}_xpps"]
        parts.append(part)
    out = pd.concat(parts, axis=1)
    out.index.names = ["season", "team"]
    out["net_pps"] = out["off_pps"] - out["def_pps"]
    out["net_xpps"] = out["off_xpps"] - out["def_xpps"]
    return out.reset_index()


def efficiency_shares(teams: pd.DataFrame) -> pd.DataFrame:
    """How much of the spread in teams' points per shot comes from shot quality vs shot making.

    Points per shot = xPPS + shot making, so within each season the variance of
    PPS splits exactly into cov(PPS, xPPS) + cov(PPS, making). Returns one row
    per side (offense, defense, net) with the two shares and the standard
    deviations of each part, from deviations to the season average.
    """
    rows = {}
    for side, label in (("off", "offense"), ("def", "defense"), ("net", "net")):
        if side == "net":
            pps, xpps = teams["net_pps"], teams["net_xpps"]
        else:
            pps, xpps = teams[f"{side}_pps"], teams[f"{side}_xpps"]
        dev = pd.DataFrame({"pps": pps, "xpps": xpps}).groupby(teams["season"]).transform(lambda s: s - s.mean())
        dev["making"] = dev["pps"] - dev["xpps"]
        var = dev["pps"].var()
        rows[label] = {"quality_share": dev["pps"].cov(dev["xpps"]) / var,
                       "making_share": dev["pps"].cov(dev["making"]) / var,
                       "sd_pps": dev["pps"].std(), "sd_quality": dev["xpps"].std(), "sd_making": dev["making"].std()}
    return pd.DataFrame(rows).T


def consecutive(table: pd.DataFrame, key: str = "player_id", min_shots: int = MIN_SHOTS) -> pd.DataFrame:
    """Pairs of consecutive seasons of the same player (or team), both with ``min_shots`` attempts.

    Columns of the first season keep their names; the next season's get a ``next_`` prefix.
    """
    fga = "fga" if "fga" in table else "off_fga"
    t = table[table[fga] >= min_shots]
    nxt = t.assign(season=t["season"] - 1).add_prefix("next_").rename(
        columns={"next_season": "season", f"next_{key}": key})
    return t.merge(nxt, on=["season", key])


def year_to_year(pairs: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Correlation of each metric with itself the next season, pooled and per pair of seasons.

    The pooled ``r`` uses deviations from each pair of seasons' averages, so
    league-wide changes (more threes every year) don't count as persistence.
    """
    rows = {}
    for m in metrics:
        d = pairs[[m, f"next_{m}"]].groupby(pairs["season"]).transform(lambda s: s - s.mean())
        per_season = pairs.groupby("season")[[m, f"next_{m}"]].apply(lambda g: g[m].corr(g[f"next_{m}"]))
        rows[m] = {"r": d[m].corr(d[f"next_{m}"]), "r_lowest": per_season.min(),
                   "r_highest": per_season.max(), "pairs": len(pairs)}
    return pd.DataFrame(rows).T.astype({"pairs": int})


def predict_next(pairs: pd.DataFrame, target: str, predictors: dict[str, Sequence[str]]) -> pd.DataFrame:
    """How well linear fits on last season's numbers predict ``target`` the next season.

    Each pair of seasons is predicted from a fit on all the other pairs
    (leave one season out). Returns the out-of-sample correlation and RMSE
    for every predictor set (empty with a single pair of seasons).
    """
    y = pairs[f"next_{target}"].to_numpy(float)
    seasons = pairs["season"].to_numpy()
    rows = {}
    if len(np.unique(seasons)) < 2:
        return pd.DataFrame(columns=["r", "rmse"], dtype=float)
    for name, cols in predictors.items():
        X = np.c_[np.ones(len(pairs)), pairs[list(cols)].to_numpy(float)]
        pred = np.empty(len(pairs))
        for s in np.unique(seasons):
            test = seasons == s
            beta = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0]
            pred[test] = X[test] @ beta
        rows[name] = {"r": float(np.corrcoef(pred, y)[0, 1]), "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    return pd.DataFrame(rows).T
