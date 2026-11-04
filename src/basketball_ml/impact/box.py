"""Per-game box scores from the play-by-play, the input of the RAPM prior.

Rebounds are split into offensive and defensive by the team of the last miss
before them. Technical fouls are left out of personal fouls (they can be
called on the bench and say little about defense).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from ..predictions.games import TEAM_ALIASES
from .stints import TECHNICALS

BOX_STATS = ("pts", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "ast", "oreb", "dreb",
             "stl", "blk", "tov", "pf", "pfd")
# Action types of fouls that are not personal fouls in the cdn.nba.com data (technical, defensive 3 seconds).
_NOT_PERSONAL = TECHNICALS | {17}


def _player(ids: pd.Series) -> pd.Series:
    ids = pd.to_numeric(ids, errors="coerce")
    return ids.where((ids > 0) & (ids < 1_610_000_000))


def season_box(season: int, data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """One row per player and game of ``season`` with the counts in :data:`BOX_STATS`."""
    out = box_scores(pbp.load_pbp(season, data_dir))
    out.insert(0, "season", season)
    return out


def box_scores(events: pd.DataFrame) -> pd.DataFrame:
    """Box scores of every player in ``events`` (play-by-play rows in file order), one row per game."""
    e = events.sort_values(["GAME_ID", "PERIOD"], kind="stable").reset_index(drop=True)
    kind = e["EVENTMSGTYPE"].to_numpy()
    text = (e["HOMEDESCRIPTION"].fillna("") + " " + e["VISITORDESCRIPTION"].fillna("")).str.upper()
    three = text.str.contains("3PT").to_numpy()
    ft_miss = (kind == pbp.FREE_THROW) & text.str.contains("MISS").to_numpy()
    miss = (kind == pbp.MISSED_SHOT) | ft_miss
    # Team of the last miss at or before each event, within its game.
    last_miss = e["PLAYER1_TEAM_ABBREVIATION"].where(miss).groupby(e["GAME_ID"]).ffill()
    rebound = kind == pbp.REBOUND
    offensive = rebound & (e["PLAYER1_TEAM_ABBREVIATION"] == last_miss).to_numpy()
    foul = (kind == pbp.FOUL) & ~e["EVENTMSGACTIONTYPE"].isin(_NOT_PERSONAL).to_numpy()

    parts = []

    def credit(slot: int, mask: np.ndarray, **stats) -> None:
        ids = _player(e[f"PLAYER{slot}_ID"])
        keep = mask & ids.notna().to_numpy()
        frame = pd.DataFrame({"game_id": e.loc[keep, "GAME_ID"].to_numpy(),
                              "player_id": ids[keep].astype(np.int64).to_numpy(),
                              "team": e.loc[keep, f"PLAYER{slot}_TEAM_ABBREVIATION"].to_numpy()})
        for name, value in stats.items():
            frame[name] = value[keep] if isinstance(value, np.ndarray) else value
        parts.append(frame)

    made = kind == pbp.MADE_SHOT
    shot = made | (kind == pbp.MISSED_SHOT)
    credit(1, shot, fg2a=(~three).astype(float), fg3a=three.astype(float),
           fg2m=(made & ~three).astype(float), fg3m=(made & three).astype(float),
           pts=np.where(made, np.where(three, 3.0, 2.0), 0.0))
    credit(2, made, ast=1.0)
    ft = kind == pbp.FREE_THROW
    credit(1, ft, fta=1.0, ftm=(~ft_miss).astype(float), pts=(~ft_miss).astype(float))
    credit(3, kind == pbp.MISSED_SHOT, blk=1.0)
    credit(1, rebound, oreb=offensive.astype(float), dreb=(~offensive).astype(float))
    credit(1, kind == pbp.TURNOVER, tov=1.0)
    credit(2, kind == pbp.TURNOVER, stl=1.0)
    credit(1, foul, pf=1.0)
    credit(2, foul, pfd=1.0)

    box = pd.concat(parts, ignore_index=True)
    box = box.dropna(subset=["team"])
    box["team"] = box["team"].replace(TEAM_ALIASES)
    stats = list(BOX_STATS)
    box[stats] = box.reindex(columns=stats).fillna(0.0)
    return box.groupby(["game_id", "team", "player_id"], as_index=False)[stats].sum()


def build_box(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    return pd.concat([season_box(s, data_dir) for s in seasons], ignore_index=True)
