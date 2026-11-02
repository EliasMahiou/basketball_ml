"""Who played in each game, and how much of a team's usual production was missing.

A player "played" in a game when he appears in its play-by-play (shot, free
throw, rebound, assist, turnover, steal, block or foul). In practice who plays
is known shortly before tip-off from injury reports, so the model treats the
game's actual lineup as known.

Production per game is a simple box-score sum from the play-by-play:
points + assists + rebounds + steals + blocks - turnovers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from .games import TEAM_ALIASES

RECENT_GAMES = 10   # a player counts as part of the team if he played in one of its last 10 games
MIN_GAMES = 3       # ...and in at least 3 games this season


def player_games(season: int, data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """One row per player per game: team and production from the play-by-play."""
    e = pbp.load_pbp(season, data_dir)
    text = e["HOMEDESCRIPTION"].fillna("") + " " + e["VISITORDESCRIPTION"].fillna("")
    rows = []

    def credit(mask, id_col, team_col, value):
        name_col = id_col.replace("_ID", "_NAME")
        sub = e.loc[mask, ["GAME_ID", id_col, team_col, name_col]].dropna()
        sub = sub[(pd.to_numeric(sub[id_col], errors="coerce") > 0)
                  & (pd.to_numeric(sub[id_col], errors="coerce") < 1_610_000_000)]
        vals = value if np.isscalar(value) else value[sub.index]
        rows.append(pd.DataFrame({"game_id": sub["GAME_ID"].to_numpy(), "player_id": sub[id_col].astype(int).to_numpy(),
                                  "player": sub[name_col].to_numpy(), "team": sub[team_col].to_numpy(),
                                  "production": vals}))

    made = e["EVENTMSGTYPE"] == pbp.MADE_SHOT
    pts = pd.Series(np.where(text.str.contains("3PT"), 3, 2), index=e.index)
    credit(made, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", pts)
    credit(made, "PLAYER2_ID", "PLAYER2_TEAM_ABBREVIATION", 1.0)                       # assist
    ft = e["EVENTMSGTYPE"] == pbp.FREE_THROW
    credit(ft, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION",
           pd.Series(np.where(text.str.contains("MISS"), 0.0, 1.0), index=e.index))
    credit(e["EVENTMSGTYPE"] == pbp.MISSED_SHOT, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", 0.0)
    credit(e["EVENTMSGTYPE"] == pbp.MISSED_SHOT, "PLAYER3_ID", "PLAYER3_TEAM_ABBREVIATION", 1.0)  # block
    credit(e["EVENTMSGTYPE"] == pbp.REBOUND, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", 1.0)
    tov = e["EVENTMSGTYPE"] == pbp.TURNOVER
    credit(tov, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", -1.0)
    credit(tov, "PLAYER2_ID", "PLAYER2_TEAM_ABBREVIATION", 1.0)                         # steal
    credit(e["EVENTMSGTYPE"] == pbp.FOUL, "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", 0.0)
    credit(e["EVENTMSGTYPE"] == pbp.FOUL, "PLAYER2_ID", "PLAYER2_TEAM_ABBREVIATION", 0.0)

    out = pd.concat(rows, ignore_index=True)
    out["team"] = out["team"].replace(TEAM_ALIASES)
    out["season"] = season
    return out.groupby(["season", "game_id", "team", "player_id"], as_index=False).agg(
        player=("player", "last"), production=("production", "sum"))


def build_player_games(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    return pd.concat([player_games(s, data_dir) for s in seasons], ignore_index=True)


def add_availability(games: pd.DataFrame, appearances: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Add lineup features from earlier games only.

    - ``strength_home`` / ``strength_away``: summed per-game production of the
      ``top`` best players who play in this game, using each player's average
      so far this season (last season's average until he has 3 games)
    - ``missing_home`` / ``missing_away``: summed per-game production of the
      team's regular players (3+ games this season, seen in its last 10 games)
      who did not play in this game
    - ``strength_diff`` and ``missing_diff``: home minus away
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    lineups = appearances.groupby(["game_id", "team"])["player_id"].apply(set).to_dict()
    prod = appearances.set_index(["game_id", "player_id"])["production"].to_dict()

    team_totals: dict = defaultdict(float)   # (team, player) -> production this season
    team_counts: dict = defaultdict(int)
    last_seen: dict = {}                     # (team, player) -> team game number
    team_games: dict = defaultdict(int)
    totals: dict = defaultdict(float)        # player -> production this season, any team
    counts: dict = defaultdict(int)
    previous: dict = {}                      # player -> last season's average
    season = None
    out = {k: [] for k in ("strength_home", "strength_away", "missing_home", "missing_away")}

    def average(pid: int) -> float:
        if counts[pid] >= MIN_GAMES:
            return totals[pid] / counts[pid]
        return previous.get(pid, totals[pid] / counts[pid] if counts[pid] else 0.0)

    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:
                previous = {pid: totals[pid] / counts[pid] for pid in counts if counts[pid] >= MIN_GAMES}
            for d in (team_totals, team_counts, last_seen, team_games, totals, counts):
                d.clear()
            season = g.season
        for side, team in (("home", g.home), ("away", g.away)):
            playing = lineups.get((g.game_id, team), set())
            values = sorted((average(pid) for pid in playing), reverse=True)
            out[f"strength_{side}"].append(float(sum(values[:top])))
            n = team_games[team]
            missing = 0.0
            for (t, pid), seen in last_seen.items():
                if (t == team and pid not in playing and n - seen < RECENT_GAMES
                        and team_counts[(t, pid)] >= MIN_GAMES):
                    missing += team_totals[(t, pid)] / team_counts[(t, pid)]
            out[f"missing_{side}"].append(missing)
        for team in (g.home, g.away):
            team_games[team] += 1
            for pid in lineups.get((g.game_id, team), set()):
                value = prod.get((g.game_id, pid), 0.0)
                team_totals[(team, pid)] += value
                team_counts[(team, pid)] += 1
                last_seen[(team, pid)] = team_games[team]
                totals[pid] += value
                counts[pid] += 1

    games = games.assign(**out)
    games["strength_diff"] = games["strength_home"] - games["strength_away"]
    games["missing_diff"] = games["missing_home"] - games["missing_away"]
    return games


def expected_lineup(appearances: pd.DataFrame, games: pd.DataFrame, team: str, out: Iterable[str] = (),
                    top: int = 10) -> tuple[float, pd.DataFrame]:
    """Strength of ``team``'s regulars after the last game loaded, minus players listed in ``out``.

    Regulars are players whose most recent game was for ``team``, with 3+ games
    that season and an appearance in its last 10 games. Returns the strength
    and the players counted.
    """
    from ..twins.similarity import _normalize

    season = int(games["season"].max())
    apps = appearances[appearances["season"] == season]
    order = {gid: i for i, gid in enumerate(games.sort_values(["date", "game_id"])["game_id"])}
    apps = apps.assign(order=apps["game_id"].map(order)).dropna(subset=["order"])
    last_team = apps.sort_values("order").groupby("player_id")["team"].last()
    team_game_ids = apps.loc[apps["team"] == team, "order"].drop_duplicates().sort_values()
    recent = set(team_game_ids.tail(RECENT_GAMES))
    stats = apps.groupby("player_id").agg(player=("player", "last"), games=("game_id", "nunique"),
                                          production=("production", "mean"))
    seen_recent = set(apps.loc[(apps["team"] == team) & apps["order"].isin(recent), "player_id"])
    regulars = stats[(last_team.reindex(stats.index) == team) & (stats["games"] >= MIN_GAMES)
                     & stats.index.isin(seen_recent)]
    skip = {_normalize(n) for n in out}
    regulars = regulars[~regulars["player"].map(_normalize).isin(skip)]
    regulars = regulars.sort_values("production", ascending=False).head(top)
    return float(regulars["production"].sum()), regulars.reset_index()
