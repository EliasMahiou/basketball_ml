"""Who played in each game, and how much of a team's usual production was missing.

A player "played" in a game when he appears in its play-by-play (shot, free
throw, rebound, assist, turnover, steal, block or foul). Who plays is only
partly known before tip-off, from injury reports, so the lineup features can
use one of three lineups (see :func:`expected_lineups`): the players who
actually played (known only once the game is over, an upper bound), the
players of the team's previous game, or its regular rotation.

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
DEFAULT_SECONDS = 720.0  # expected playing time of a player with no history (12 minutes)


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


LINEUP_RULES = ("played", "previous", "regulars")


def expected_lineups(games: pd.DataFrame, appearances: pd.DataFrame, rule: str = "previous") -> dict:
    """The players each team is expected to field in each game: ``{(game_id, team): {player_id, ...}}``.

    - ``played``: the players who appeared in the game. Known only once it is
      over, it is an upper bound on what injury reports can tell.
    - ``previous``: the players who appeared in the team's previous game of the
      season, from earlier games only. A player still injured keeps missing
      games; one back from injury is missed once.
    - ``regulars``: players whose last game was for the team, seen in one of its
      last 10 games: the rotation with no injury news.

    A team's first game of a season has no earlier game: its players stand in
    for the roster announced before it, under every rule.
    """
    if rule not in LINEUP_RULES:
        raise ValueError(f"Unknown lineup rule {rule!r}, expected one of {LINEUP_RULES}")
    played = appearances.groupby(["game_id", "team"])["player_id"].apply(frozenset).to_dict()
    if rule == "played":
        return played
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    out: dict = {}
    history: dict = defaultdict(list)   # team -> lineups of its games this season, oldest first
    last_team: dict = {}                # player -> team of his latest game this season
    season = None
    for g in games.itertuples(index=False):
        if g.season != season:
            history.clear()
            last_team.clear()
            season = g.season
        for team in (g.home, g.away):
            actual = played.get((g.game_id, team), frozenset())
            past = history[team]
            if not past:
                out[(g.game_id, team)] = actual
            elif rule == "previous":
                out[(g.game_id, team)] = past[-1]
            else:
                recent = frozenset().union(*past[-RECENT_GAMES:])
                out[(g.game_id, team)] = frozenset(pid for pid in recent if last_team.get(pid) == team)
        for team in (g.home, g.away):
            actual = played.get((g.game_id, team), frozenset())
            history[team].append(actual)
            for pid in actual:
                last_team[pid] = team
    return out


def add_availability(games: pd.DataFrame, appearances: pd.DataFrame, top: int = 10,
                     lineups: dict | None = None) -> pd.DataFrame:
    """Add lineup features from earlier games only.

    - ``strength_home`` / ``strength_away``: summed per-game production of the
      ``top`` best players expected to play in this game, using each player's
      average so far this season (last season's average until he has 3 games)
    - ``missing_home`` / ``missing_away``: summed per-game production of the
      team's regular players (3+ games this season, seen in its last 10 games)
      not expected to play in this game
    - ``strength_diff`` and ``missing_diff``: home minus away

    ``lineups`` (from :func:`expected_lineups`) says who is expected to play;
    by default, the players who actually played.
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    played = appearances.groupby(["game_id", "team"])["player_id"].apply(set).to_dict()
    if lineups is None:
        lineups = played
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
            playing = _weights(lineups.get((g.game_id, team), ()))
            out[f"strength_{side}"].append(top_sum(((average(pid), w) for pid, w in playing.items()), top))
            n = team_games[team]
            missing = 0.0
            for (t, pid), seen in last_seen.items():
                if t == team and n - seen < RECENT_GAMES and team_counts[(t, pid)] >= MIN_GAMES:
                    missing += (1 - playing.get(pid, 0.0)) * team_totals[(t, pid)] / team_counts[(t, pid)]
            out[f"missing_{side}"].append(missing)
        for team in (g.home, g.away):
            team_games[team] += 1
            for pid in played.get((g.game_id, team), set()):
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


def _weights(lineup) -> dict:
    """``{player_id: chance he plays}`` from a lineup given as ids (each certain to play) or as that dict."""
    return dict(lineup) if isinstance(lineup, dict) else dict.fromkeys(lineup, 1.0)


def top_sum(values: Iterable[tuple[float, float]], top: int) -> float:
    """Sum of the ``top`` largest values, each counted by its weight (the chance its player plays).

    With weights of 1 it is the plain sum of the ``top`` largest values; a
    player with half a chance to play fills half a place.
    """
    total, room = 0.0, float(top)
    for value, weight in sorted(values, reverse=True):
        if room <= 0:
            break
        take = min(weight, room)
        total += value * take
        room -= take
    return float(total)


def add_impact(games: pd.DataFrame, seconds: pd.DataFrame, ratings: pd.DataFrame,
               lineups: dict | None = None) -> pd.DataFrame:
    """Add ``rapm_home``, ``rapm_away`` and ``rapm_diff``: the impact of who plays.

    ``seconds`` is time on the court per player and game (see
    :func:`basketball_ml.impact.player_seconds`). ``ratings`` holds RAPM
    ratings with the ``season`` they serve and the date ``as_of`` which they
    can be used (see :func:`basketball_ml.impact.rating_schedule`); each game
    uses the latest ratings of its season dated on or before it. A side's value
    is the minutes-weighted average rating of the players who play, times 5:
    the expected margin per 100 possessions of its lineups against average
    players. Expected minutes are each player's average so far this season
    (last season's until he has 3 games). Games without ratings get 0.
    ``lineups`` (from :func:`expected_lineups`, keyed by team) says who is
    expected to play; by default, the players who were on the court.
    """
    from ..impact.rapm import NEWCOMER

    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    played = {key: dict(zip(g["player_id"], g["seconds"])) for key, g in seconds.groupby(["game_id", "side"])}
    ratings = ratings.assign(as_of=pd.to_datetime(ratings["as_of"]))
    schedule: dict[int, tuple[np.ndarray, list[dict]]] = {}
    for season, block in ratings.groupby("season"):
        stamps = sorted(block["as_of"].unique())
        maps = [dict(zip(r["player_id"], r["rapm"])) for _, r in block.groupby("as_of", sort=True)]
        schedule[int(season)] = (np.array(stamps, dtype="datetime64[ns]"), maps)

    total: dict = defaultdict(float)
    count: dict = defaultdict(int)
    previous: dict = {}
    season = None
    out = {"rapm_home": [], "rapm_away": []}
    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:
                previous = {pid: total[pid] / count[pid] for pid in count if count[pid] >= MIN_GAMES}
            total.clear()
            count.clear()
            season = g.season
        rating = None
        if g.season in schedule:
            stamps, maps = schedule[g.season]
            i = int(np.searchsorted(stamps, np.datetime64(g.date, "ns"), side="right")) - 1
            rating = maps[i] if i >= 0 else None
        for side in ("home", "away"):
            if lineups is None:
                players = dict.fromkeys(played.get((g.game_id, side), {}), 1.0)
            else:
                players = _weights(lineups.get((g.game_id, getattr(g, side)), ()))
            if rating is None or not players:
                out[f"rapm_{side}"].append(0.0)
                continue
            expected = {pid: w * (total[pid] / count[pid] if count[pid] >= MIN_GAMES
                                  else previous.get(pid, DEFAULT_SECONDS))
                        for pid, w in players.items() if w > 0}
            out[f"rapm_{side}"].append(lineup_rapm(expected, rating, NEWCOMER))
        for side in ("home", "away"):
            for pid, sec in played.get((g.game_id, side), {}).items():
                total[pid] += sec
                count[pid] += 1
    games = games.assign(**out)
    games["rapm_diff"] = games["rapm_home"] - games["rapm_away"]
    return games


def lineup_rapm(expected_seconds: dict, rating: dict, default: float) -> float:
    """5 x the minutes-weighted average rating of the players in ``expected_seconds``."""
    weight = sum(expected_seconds.values())
    if weight <= 0:
        return 0.0
    return 5 * sum(rating.get(pid, default) * sec for pid, sec in expected_seconds.items()) / weight


def expected_lineup(appearances: pd.DataFrame, games: pd.DataFrame, team: str, out: Iterable[str] = (),
                    top: int = 10, rule: str = "previous") -> tuple[float, pd.DataFrame]:
    """Strength of the players ``team`` is expected to field after the last game loaded, minus those in ``out``.

    With the ``previous`` rule (as in the backtest), they are the players of
    its latest game. With ``regulars``, players whose most recent game was for
    ``team``, with 3+ games that season and an appearance in its last 10 games
    (better for the first game of a new season: the last game of a season is
    often played without the stars). Returns the strength and the players,
    best first (the strength counts the first ``top``).
    """
    from ..twins.similarity import _normalize

    if rule not in ("previous", "regulars"):
        raise ValueError(f"Unknown lineup rule {rule!r}")
    season = int(games["season"].max())
    apps = appearances[appearances["season"] == season]
    order = {gid: i for i, gid in enumerate(games.sort_values(["date", "game_id"])["game_id"])}
    apps = apps.assign(order=apps["game_id"].map(order)).dropna(subset=["order"])
    last_team = apps.sort_values("order").groupby("player_id")["team"].last()
    team_game_ids = apps.loc[apps["team"] == team, "order"].drop_duplicates().sort_values()
    stats = apps.groupby("player_id").agg(player=("player", "last"), games=("game_id", "nunique"),
                                          production=("production", "mean"))
    if rule == "previous":
        latest = set(team_game_ids.tail(1))
        players = stats[stats.index.isin(apps.loc[(apps["team"] == team) & apps["order"].isin(latest), "player_id"])]
    else:
        recent = set(team_game_ids.tail(RECENT_GAMES))
        seen_recent = set(apps.loc[(apps["team"] == team) & apps["order"].isin(recent), "player_id"])
        players = stats[(last_team.reindex(stats.index) == team) & (stats["games"] >= MIN_GAMES)
                        & stats.index.isin(seen_recent)]
    skip = {_normalize(n) for n in out}
    players = players[~players["player"].map(_normalize).isin(skip)]
    players = players.sort_values("production", ascending=False)
    return float(players["production"].head(top).sum()), players.reset_index()
