"""The season in progress, from the NBA's public S3 feed.

The play-by-play archives the models learn from are published after each
season. The season being played comes from the JSON files the NBA's own
sites read, on a public S3 bucket:

- the league schedule, with the final score of every game played;
- each finished game's box score: who played, for how long, and his stats;
- today's odds from several sportsbooks.

:func:`season_games` and :func:`season_appearances` turn them into the
tables the game model is built on, so predictions can follow the season.
Box scores of finished games are cached in ``data/live/``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

S3_URL = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA"
SCHEDULE_URL = f"{S3_URL}/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = S3_URL + "/liveData/boxscore/boxscore_{game_id}.json"
ODDS_URL = f"{S3_URL}/liveData/odds/odds_todaysGames.json"
FINAL = 3


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def season_games(schedule: dict | None = None) -> pd.DataFrame:
    """Every regular-season game of the current season, played or not.

    Columns: ``game_id``, ``date`` (US Eastern), ``home``, ``away``,
    ``status`` (3 when final), ``home_pts``, ``away_pts``, ``season`` (end
    year), plus ``margin`` and ``home_win`` for finished games.
    """
    schedule = _fetch(SCHEDULE_URL) if schedule is None else schedule
    league = schedule["leagueSchedule"]
    season = int(league["seasonYear"][:4]) + 1
    rows = [{"game_id": int(g["gameId"]), "date": g["gameDateEst"][:10], "home": g["homeTeam"]["teamTricode"],
             "away": g["awayTeam"]["teamTricode"], "status": int(g["gameStatus"]),
             "home_pts": g["homeTeam"]["score"], "away_pts": g["awayTeam"]["score"]}
            for day in league["gameDates"] for g in day["games"] if g["gameId"].startswith("002")]
    games = pd.DataFrame(rows).assign(season=season)
    games["date"] = pd.to_datetime(games["date"])
    games["margin"] = (games["home_pts"] - games["away_pts"]).where(games["status"] == FINAL)
    games["home_win"] = (games["margin"] > 0).astype(int).where(games["status"] == FINAL)
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def boxscore(game_id: int, cache_dir: str | Path = "data/live") -> dict | None:
    """The box score of a game (cached once the game is final); None before it starts."""
    path = Path(cache_dir) / "boxscores" / f"{game_id:010d}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        data = _fetch(BOXSCORE_URL.format(game_id=f"{game_id:010d}"))
    except urllib.error.HTTPError as err:
        if err.code in (403, 404):
            return None
        raise
    if data["game"]["gameStatus"] == FINAL:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def _seconds(clock: str) -> float:
    """"PT33M15.00S" -> 1995.0."""
    minutes, _, rest = clock.removeprefix("PT").partition("M")
    return 60 * float(minutes or 0) + float(rest.removesuffix("S") or 0)


def appearances_from_boxscore(data: dict, season: int) -> pd.DataFrame:
    """Players who played in a box score, with the columns of the play-by-play appearances.

    ``production`` is the same box-score sum (points + assists + rebounds +
    steals + blocks - turnovers); ``seconds`` and ``side`` are the playing
    time and "home"/"away".
    """
    game = data["game"]
    rows = []
    for side in ("home", "away"):
        team = game[f"{side}Team"]
        for p in team["players"]:
            s = p["statistics"]
            if str(p.get("played")) != "1":
                continue
            rows.append({"season": season, "game_id": int(game["gameId"]), "team": team["teamTricode"],
                         "player_id": int(p["personId"]), "player": p["name"], "side": side,
                         "seconds": _seconds(s["minutes"]),
                         "production": s["points"] + s["assists"] + s["reboundsTotal"] + s["steals"] + s["blocks"]
                         - s["turnovers"]})
    return pd.DataFrame(rows)


def season_appearances(games: pd.DataFrame, cache_dir: str | Path = "data/live") -> pd.DataFrame:
    """Appearances in every finished game of ``games`` (see :func:`appearances_from_boxscore`)."""
    parts = []
    for g in games[games["status"] == FINAL].itertuples(index=False):
        data = boxscore(g.game_id, cache_dir)
        if data is not None and data["game"]["gameStatus"] == FINAL:
            parts.append(appearances_from_boxscore(data, g.season))
    return pd.concat(parts, ignore_index=True)


def decimal_no_vig(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    """Home win probability from decimal odds, with the bookmaker's margin removed."""
    home, away = 1 / np.asarray(home, dtype=float), 1 / np.asarray(away, dtype=float)
    return home / (home + away)


def odds_today(odds: dict | None = None) -> pd.Series:
    """Median no-vig home win probability over the sportsbooks in today's odds feed, by ``game_id``."""
    odds = _fetch(ODDS_URL) if odds is None else odds
    out = {}
    for game in odds.get("games", []):
        probs = {}
        for market in game.get("markets", []):
            if market.get("name") != "2way":
                continue
            for book in market.get("books", []):
                price = {o["type"]: float(o["odds"]) for o in book.get("outcomes", []) if o.get("odds")}
                if price.get("home", 0) > 1 and price.get("away", 0) > 1:
                    probs[book["id"]] = float(decimal_no_vig(price["home"], price["away"]))
        if probs:
            out[int(game["gameId"])] = float(np.median(list(probs.values())))
    return pd.Series(out, name="p_market", dtype=float)
