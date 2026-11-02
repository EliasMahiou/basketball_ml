"""Per player-season style features from play-by-play events and shot charts.

Every feature is a share or a rate per "touch", so it describes *how* a
player plays rather than how much he plays:

- Touches ``T = FGA + 0.44 FTA + TOV + AST + OREB + DREB + STL + BLK + PF``;
  each of those counts (and fouls drawn) is divided by ``T``.
- Shot diet: share of attempts by zone and by shot type (dunk, layup, pull-up...).
- Shot making: field goal percentage by zone, shrunk toward the league average.
- Creation: share of made shots that were assisted, share of turnovers that were bad passes.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp

ZONES = {
    "rim": ["Restricted Area"],
    "paint": ["In The Paint (Non-RA)"],
    "mid": ["Mid-Range"],
    "corner3": ["Left Corner 3", "Right Corner 3"],
    "above3": ["Above the Break 3"],
}

# Shot types matched on the ACTION_TYPE text (a shot can match several).
SHOT_TYPES = {
    "dunk": "Dunk",
    "layup": "Layup|Finger Roll",
    "putback": "Tip|Putback",
    "floater": "Float",
    "hook": "Hook",
    "post_fade": "Turnaround|Fadeaway",
    "pullup": "Pull|Step Back",
    "driving": "Driving",
    "cutting": "Cutting",
    "alley_oop": "Alley Oop",
}

FG_PRIOR = 20  # attempts of league-average shooting added to each zone


def _is_player(ids: pd.Series) -> pd.Series:
    """True for player ids (team ids start at 1610612737)."""
    ids = pd.to_numeric(ids, errors="coerce")
    return ids.notna() & (ids > 0) & (ids < 1_610_000_000)


def _count(df: pd.DataFrame, id_col: str, name: str) -> pd.Series:
    rows = df[_is_player(df[id_col])]
    return rows.groupby(rows[id_col].astype(int)).size().rename(name)


def shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Shot diet and zone shooting per player, from shot chart rows."""
    shots = shots[shots["SHOT_ZONE_BASIC"] != "Backcourt"]
    g = shots.groupby("PLAYER_ID")
    out = pd.DataFrame({"fga": g.size(), "fgm": g["SHOT_MADE_FLAG"].sum(),
                        "shot_distance": g["SHOT_DISTANCE"].mean()})
    for zone, labels in ZONES.items():
        in_zone = shots["SHOT_ZONE_BASIC"].isin(labels)
        att = in_zone.groupby(shots["PLAYER_ID"]).sum()
        made = (in_zone & (shots["SHOT_MADE_FLAG"] == 1)).groupby(shots["PLAYER_ID"]).sum()
        league = made.sum() / max(att.sum(), 1)
        out[f"share_{zone}"] = att / out["fga"]
        out[f"fg_{zone}"] = (made + FG_PRIOR * league) / (att + FG_PRIOR)
    action = shots["ACTION_TYPE"].fillna("")
    for name, pattern in SHOT_TYPES.items():
        out[f"type_{name}"] = action.str.contains(pattern, regex=True).groupby(shots["PLAYER_ID"]).mean()
    out["type_catch_shoot"] = (action == "Jump Shot").groupby(shots["PLAYER_ID"]).mean()
    out.index = out.index.astype(int)
    return out


def event_features(events: pd.DataFrame) -> pd.DataFrame:
    """Counts of assists, rebounds, turnovers, steals, blocks and fouls per player."""
    e = events
    made = e[e["EVENTMSGTYPE"] == pbp.MADE_SHOT]
    fouls = e[e["EVENTMSGTYPE"] == pbp.FOUL]
    tov = e[e["EVENTMSGTYPE"] == pbp.TURNOVER]
    counts = [
        _count(e[e["EVENTMSGTYPE"] == pbp.FREE_THROW], "PLAYER1_ID", "fta"),
        _count(made, "PLAYER2_ID", "ast"),
        _count(made[_is_player(made["PLAYER2_ID"])], "PLAYER1_ID", "fgm_assisted"),
        _count(made, "PLAYER1_ID", "fgm_pbp"),
        _count(tov, "PLAYER1_ID", "tov"),
        _count(tov[tov["EVENTMSGACTIONTYPE"].isin([1, 45])], "PLAYER1_ID", "tov_bad_pass"),
        _count(tov, "PLAYER2_ID", "stl"),
        _count(e[e["EVENTMSGTYPE"] == pbp.MISSED_SHOT], "PLAYER3_ID", "blk"),
        _count(fouls, "PLAYER1_ID", "pf"),
        _count(fouls, "PLAYER2_ID", "fouls_drawn"),
    ]
    # Rebounds: descriptions carry running per-game totals, e.g. "REBOUND (Off:1 Def:2)".
    reb = e[(e["EVENTMSGTYPE"] == pbp.REBOUND) & _is_player(e["PLAYER1_ID"])]
    text = reb["HOMEDESCRIPTION"].fillna("") + " " + reb["VISITORDESCRIPTION"].fillna("")
    parsed = text.str.extract(r"Off:(\d+) Def:(\d+)").astype(float)
    reb = reb.assign(off=parsed[0], dfn=parsed[1]).dropna(subset=["off"])
    per_game = reb.groupby([reb["PLAYER1_ID"].astype(int), "GAME_ID"])[["off", "dfn"]].max()
    totals = per_game.groupby(level=0).sum().rename(columns={"off": "oreb", "dfn": "dreb"})
    out = pd.concat([*counts, totals], axis=1).fillna(0)
    names = e[_is_player(e["PLAYER1_ID"])].groupby(e["PLAYER1_ID"].astype(int))["PLAYER1_NAME"].last()
    out["player"] = names
    return out


RATE_COUNTS = ["fga", "fta", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf", "fouls_drawn"]


def season_features(season: int, data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """All style features for one season, one row per player."""
    shots = shot_features(pbp.load_shots(season, data_dir))
    events = event_features(pbp.load_pbp(season, data_dir))
    df = shots.join(events, how="inner")
    touches = (df["fga"] + 0.44 * df["fta"] + df["tov"] + df["ast"] + df["oreb"] + df["dreb"]
               + df["stl"] + df["blk"] + df["pf"])
    df["touches"] = touches
    for c in RATE_COUNTS:
        df[f"rate_{c}"] = df[c] / touches
    df["assisted_share"] = df["fgm_assisted"] / df["fgm_pbp"].clip(lower=1)
    df["bad_pass_share"] = df["tov_bad_pass"] / df["tov"].clip(lower=1)
    df["season"] = season
    df.index.name = "player_id"
    return df.reset_index()


def style_columns(df: pd.DataFrame) -> list[str]:
    """The feature columns used for embeddings (shares, rates and percentages)."""
    prefixes = ("share_", "fg_", "type_", "rate_")
    return [c for c in df.columns if c.startswith(prefixes)] + ["assisted_share", "bad_pass_share", "shot_distance"]


def build_style_table(seasons: Iterable[int], data_dir: str | Path = "data/pbp",
                      min_fga: int = 300) -> pd.DataFrame:
    """Style features for several seasons, standardized within each season.

    Shares and rates get a square-root transform first (it evens out their
    variance), then every column becomes a z-score within its season.
    """
    df = pd.concat([season_features(s, data_dir) for s in seasons], ignore_index=True)
    df = df[df["fga"] >= min_fga].reset_index(drop=True)
    cols = style_columns(df)
    z = df.copy()
    sqrt_cols = [c for c in cols if c != "shot_distance" and not c.startswith("fg_")]
    z[sqrt_cols] = np.sqrt(z[sqrt_cols].clip(lower=0))
    grouped = z.groupby("season")[cols]
    z[cols] = (z[cols] - grouped.transform("mean")) / grouped.transform("std")
    # A shot label missing from a season (e.g. "Cutting" before 2015-16) has no
    # spread that season: treat everyone as average on it.
    z[cols] = z[cols].fillna(0.0)
    return z
