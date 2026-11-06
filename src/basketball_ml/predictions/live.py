"""The season in progress, from the NBA's public S3 feed.

The play-by-play archives the models learn from are published after each
season. The season being played comes from the JSON files the NBA's own
sites read, on a public S3 bucket:

- the league schedule, with the final score of every game played;
- each finished game's box score: who played, for how long, and his stats;
- each finished game's play-by-play, in the cdn.nba.com format the
  play-by-play archive of the last season uses;
- today's odds from several sportsbooks; the ``odds archive`` workflow saves
  them several times a day on the repository's ``odds-archive`` branch, so
  each game's last odds before tip-off can be read back (:func:`closing_odds`).

:func:`season_games`, :func:`season_appearances` and :func:`season_events`
turn them into the tables the game model, the player ratings and the shot
models are built on, so they can follow the season. Files of finished games
are cached in ``data/live/``.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

S3_URL = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA"
SCHEDULE_URL = f"{S3_URL}/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = S3_URL + "/liveData/boxscore/boxscore_{game_id}.json"
PLAYBYPLAY_URL = S3_URL + "/liveData/playbyplay/playbyplay_{game_id}.json"
ODDS_URL = f"{S3_URL}/liveData/odds/odds_todaysGames.json"
ARCHIVE_URL = "https://raw.githubusercontent.com/EliasMahiou/basketball_ml/odds-archive/odds"
CLOSING_WINDOW = pd.Timedelta(hours=12)   # how early a snapshot can be and still count as a game's last odds
FINAL = 3


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def season_games(schedule: dict | None = None) -> pd.DataFrame:
    """Every regular-season game of the current season, played or not.

    Columns: ``game_id``, ``date`` (US Eastern), ``tip`` (tip-off, UTC),
    ``home``, ``away``, ``status`` (3 when final), ``home_pts``,
    ``away_pts``, ``season`` (end year), plus ``margin`` and ``home_win``
    for finished games.
    """
    schedule = _fetch(SCHEDULE_URL) if schedule is None else schedule
    league = schedule["leagueSchedule"]
    season = int(league["seasonYear"][:4]) + 1
    rows = [{"game_id": int(g["gameId"]), "date": g["gameDateEst"][:10], "tip": g.get("gameDateTimeUTC"),
             "home": g["homeTeam"]["teamTricode"], "away": g["awayTeam"]["teamTricode"],
             "status": int(g["gameStatus"]), "home_pts": g["homeTeam"]["score"], "away_pts": g["awayTeam"]["score"]}
            for day in league["gameDates"] for g in day["games"] if g["gameId"].startswith("002")]
    games = pd.DataFrame(rows).assign(season=season)
    games["date"] = pd.to_datetime(games["date"])
    games["tip"] = pd.to_datetime(games["tip"], utc=True)
    games["margin"] = (games["home_pts"] - games["away_pts"]).where(games["status"] == FINAL)
    games["home_win"] = (games["margin"] > 0).astype(int).where(games["status"] == FINAL)
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def _cached(url: str, path: Path, final: bool | Callable[[dict], bool]) -> dict | None:
    """A feed file, read from ``path`` if it is there; saved there when ``final``. None when it doesn't exist yet."""
    if path.exists():
        return json.loads(path.read_text())
    try:
        data = _fetch(url)
    except urllib.error.HTTPError as err:
        if err.code in (403, 404):
            return None
        raise
    if final(data) if callable(final) else final:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def boxscore(game_id: int, cache_dir: str | Path = "data/live") -> dict | None:
    """The box score of a game (cached once the game is final); None before it starts."""
    return _cached(BOXSCORE_URL.format(game_id=f"{game_id:010d}"),
                   Path(cache_dir) / "boxscores" / f"{game_id:010d}.json",
                   final=lambda data: data["game"]["gameStatus"] == FINAL)


def playbyplay(game_id: int, cache_dir: str | Path = "data/live", final: bool = True) -> pd.DataFrame | None:
    """The play-by-play actions of a game, with its ``gameId``; None if there are none yet.

    Cached when ``final``; a game in progress is fetched again each time.
    """
    data = _cached(PLAYBYPLAY_URL.format(game_id=f"{game_id:010d}"),
                   Path(cache_dir) / "playbyplay" / f"{game_id:010d}.json", final=final)
    if data is None or not data["game"].get("actions"):
        return None
    return pd.DataFrame(data["game"]["actions"]).assign(gameId=int(data["game"]["gameId"]))


def season_actions(games: pd.DataFrame, cache_dir: str | Path = "data/live") -> pd.DataFrame:
    """The play-by-play actions of every finished game of ``games``, in the feed's own columns."""
    parts = [playbyplay(g.game_id, cache_dir) for g in games[games["status"] == FINAL].itertuples(index=False)]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise SystemExit("No finished game of the season has play-by-play yet")
    return pd.concat(parts, ignore_index=True)


def season_events(games: pd.DataFrame, cache_dir: str | Path = "data/live") -> pd.DataFrame:
    """Play-by-play of every finished game of ``games``, in the stats.nba.com columns (see :func:`pbp.from_cdn`)."""
    from .. import pbp

    events = pbp.from_cdn(season_actions(games, cache_dir))
    events["season"] = int(games["season"].iloc[0])
    return events


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


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def odds_table(odds: dict) -> pd.DataFrame:
    """One row per game of an odds feed snapshot, indexed by ``game_id``.

    ``p_market`` is the median no-vig home win probability over the
    sportsbooks' moneylines, ``spread`` their median home spread (negative
    when the home team is favored), ``books`` the number of moneylines.
    """
    rows = []
    for game in odds.get("games", []):
        probs, spreads = {}, {}
        for market in game.get("markets", []):
            for book in market.get("books", []):
                home = next((o for o in book.get("outcomes", []) if o.get("type") == "home"), {})
                away = next((o for o in book.get("outcomes", []) if o.get("type") == "away"), {})
                if market.get("name") == "2way":
                    h, a = _number(home.get("odds")), _number(away.get("odds"))
                    if h > 1 and a > 1:
                        probs[book.get("id")] = float(decimal_no_vig(h, a))
                elif market.get("name") == "spread" and np.isfinite(_number(home.get("spread"))):
                    spreads[book.get("id")] = _number(home.get("spread"))
        rows.append({"game_id": int(game["gameId"]),
                     "p_market": float(np.median(list(probs.values()))) if probs else np.nan,
                     "spread": float(np.median(list(spreads.values()))) if spreads else np.nan,
                     "books": len(probs)})
    return pd.DataFrame(rows, columns=["game_id", "p_market", "spread", "books"]).set_index("game_id")


def odds_today(odds: dict | None = None) -> pd.Series:
    """Median no-vig home win probability over the sportsbooks in today's odds feed, by ``game_id``."""
    return odds_table(_fetch(ODDS_URL) if odds is None else odds)["p_market"].dropna()


def odds_snapshots(cache_dir: str | Path = "data/live", url: str = ARCHIVE_URL) -> list[tuple[pd.Timestamp, dict]]:
    """Every saved snapshot of the odds feed, oldest first, as (UTC time, feed).

    The ``odds archive`` workflow saves the feed several times a day on the
    ``odds-archive`` branch; snapshots are cached in ``cache_dir/odds``.
    """
    folder = Path(cache_dir) / "odds"
    with urllib.request.urlopen(f"{url}/index.txt", timeout=60) as resp:
        names = resp.read().decode().split()
    out = []
    for name in names:
        path = folder / name
        if not path.exists():
            with urllib.request.urlopen(f"{url}/{name}", timeout=60) as resp:
                data = resp.read()
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        taken = pd.to_datetime(name[:16], format="%Y-%m-%dT%H%MZ", utc=True)
        out.append((taken, json.loads(gzip.decompress(path.read_bytes()))))
    return out


def closing_odds(games: pd.DataFrame, snapshots: Iterable[tuple[pd.Timestamp, dict]],
                 within: pd.Timedelta = CLOSING_WINDOW) -> pd.DataFrame:
    """Each game's last odds before tip-off: the :func:`odds_table` columns plus ``minutes`` before the tip.

    ``games`` needs ``game_id`` and ``tip`` (see :func:`season_games`).
    Snapshots taken after the tip, or more than ``within`` before it, are
    ignored; games without one are left out.
    """
    parts = [odds_table(data).reset_index().assign(taken=taken) for taken, data in snapshots]
    columns = ["p_market", "spread", "books", "minutes"]
    if not parts:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="game_id"))
    table = pd.concat(parts, ignore_index=True).merge(games[["game_id", "tip"]], on="game_id")
    table = table[(table["taken"] <= table["tip"]) & (table["taken"] >= table["tip"] - within)
                  & table["p_market"].notna()]
    last = table.sort_values("taken", kind="stable").groupby("game_id").tail(1).copy()
    last["minutes"] = (last["tip"] - last["taken"]).dt.total_seconds() / 60
    return last.set_index("game_id")[columns].sort_index()
