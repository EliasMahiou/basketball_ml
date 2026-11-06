"""NCAA seasons of NBA players, for the forecast's values of newcomers.

Draft slot and age say little about which rookies will help. A player's last
college season says more: ESPN's men's college basketball box scores, in the
public files of the sportsdataverse project (2009-10 on, one file per
season), give each player's season totals, and the game scores give each
team's schedule-adjusted margin (:func:`team_ratings`).

NBA players are matched to their college seasons by name, then by school when
Basketball-Reference lists one (:func:`match_players`): 94% of the NBA
players with a US college since 2012 are matched name and school. The
features are rates per 40 minutes, shooting, minutes per game and the
team's margin, standardized over the college seasons of NBA players
(:func:`college_features`).
"""

from __future__ import annotations

import re
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

BOX_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           "espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet")
FIRST_SEASON = 2010
MIN_MINUTES = 200.0      # a college season with fewer minutes is ignored
MAX_GAP = 6              # seasons between the last college season and the NBA season it informs
TOTALS = ("minutes", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
          "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted", "offensive_rebounds",
          "defensive_rebounds", "assists", "steals", "blocks", "turnovers", "fouls", "points")
STATS = ("c_pts", "c_ast", "c_oreb", "c_dreb", "c_stl", "c_blk", "c_tov", "c_pf", "c_ts", "c_3par", "c_ftp",
         "c_team", "c_mpg")
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def download_box(seasons: Iterable[int], data_dir: str | Path = "data/college", overwrite: bool = False) -> Path:
    """Download the player box scores of ``seasons`` (end years) to ``data_dir``; seasons not published are skipped."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        out = data_dir / f"player_box_{season}.parquet"
        if overwrite or not out.exists():
            try:
                with urllib.request.urlopen(BOX_URL.format(season=season), timeout=300) as resp:
                    out.write_bytes(resp.read())
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
    return data_dir


def team_ratings(box: pd.DataFrame, damping: float = 0.5) -> pd.Series:
    """Each team's schedule-adjusted margin per game (points better than an average team), by ``team_id``.

    A least-squares fit of every game's home margin on the two teams and a
    home edge, from one season's box score rows.
    """
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr

    games = box.drop_duplicates(["game_id", "team_id"])
    games = games[games["home_away"] == "home"]
    teams = pd.Index(sorted(set(games["team_id"]) | set(games["opponent_team_id"])))
    n = len(games)
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[teams.get_indexer(games["team_id"]), teams.get_indexer(games["opponent_team_id"])]
    design = hstack([csr_matrix((np.r_[np.ones(n), -np.ones(n)], (rows, cols)), shape=(n, len(teams))),
                     csr_matrix(np.ones((n, 1)))]).tocsr()
    margin = (games["team_score"] - games["opponent_team_score"]).to_numpy(float)
    solution = lsqr(design, margin, damp=damping)[0][:-1]
    played = pd.concat([games["team_id"], games["opponent_team_id"]]).value_counts().reindex(teams).to_numpy()
    return pd.Series(solution - np.average(solution, weights=played), index=teams, name="team_rating")


def season_totals(box: pd.DataFrame) -> pd.DataFrame:
    """One row per player of a season's box scores: name, main team and school, games, the totals of :data:`TOTALS`.

    ``team_rating`` is the main team's schedule-adjusted margin (see :func:`team_ratings`).
    """
    box = box[box["athlete_id"].notna() & ~box["did_not_play"].fillna(False).astype(bool)]
    totals = box.groupby("athlete_id").agg(name=("athlete_display_name", "last"), games=("game_id", "nunique"),
                                           **{c: (c, "sum") for c in TOTALS})
    main = (box.groupby(["athlete_id", "team_id"])["minutes"].sum().reset_index()
               .sort_values("minutes", kind="stable").drop_duplicates("athlete_id", keep="last").set_index("athlete_id"))
    totals["team_id"] = main["team_id"].reindex(totals.index)
    totals["school"] = totals["team_id"].map(box.drop_duplicates("team_id").set_index("team_id")["team_location"])
    totals["team_rating"] = totals["team_id"].map(team_ratings(box))
    return totals.reset_index()


def college_seasons(seasons: Iterable[int], data_dir: str | Path = "data/college") -> pd.DataFrame:
    """:func:`season_totals` of every season in ``seasons`` that has a box score file, with ``season``.

    Files are downloaded on first use; the table is cached in ``data_dir``.
    """
    from ..cache import cached_csv

    seasons = list(seasons)
    data_dir = Path(data_dir)

    def build() -> pd.DataFrame:
        download_box(seasons, data_dir)
        parts = []
        for season in seasons:
            path = data_dir / f"player_box_{season}.parquet"
            if path.exists():
                columns = ["game_id", "athlete_id", "athlete_display_name", "team_id", "team_location", "home_away",
                           "team_score", "opponent_team_id", "opponent_team_score", "did_not_play", *TOTALS]
                parts.append(season_totals(pd.read_parquet(path, columns=columns)).assign(season=season))
        return pd.concat(parts, ignore_index=True)

    return cached_csv(data_dir / f"college_seasons_{seasons[0]}_{seasons[-1]}.csv", build)


def normalize_name(name: object) -> str:
    """Lowercase ASCII without punctuation or suffixes: "Gary Trent Jr." -> "gary trent"."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`,]", "", s).replace("-", " ")
    return re.sub(r"\s+", " ", _SUFFIXES.sub("", s)).strip()


def school_key(name: object) -> str:
    """A school name reduced for matching: "University of Kentucky" -> "kentucky", "Ohio State" -> "ohio st"."""
    s = re.sub(r"\(.*?\)", "", normalize_name(name))
    for word, short in (("university of ", ""), ("university", ""), ("college", ""), ("state", "st")):
        s = s.replace(word, short)
    return re.sub(r"\s+", " ", s).strip()


def match_players(players: pd.DataFrame, seasons: pd.DataFrame, window: int = MAX_GAP + 1) -> pd.DataFrame:
    """The college player (``athlete_id``) of each NBA player, matched by name and school.

    ``players`` has ``bref_id``, ``name``, ``colleges`` (comma-separated,
    may be missing) and ``first_nba`` (first NBA season, end year);
    ``seasons`` is :func:`college_seasons`. Candidates have the same
    normalized name and a college season in the ``window`` seasons before the
    first NBA one; when the player has a listed college, candidates at a
    matching school are kept if there are any. The candidate with the most
    minutes wins. Returns ``bref_id``, ``athlete_id`` and ``by`` ("name and
    school" or "name").
    """
    by_name = {key: group for key, group in seasons.assign(key=seasons["name"].map(normalize_name)).groupby("key")}
    rows = []
    for p in players.itertuples(index=False):
        found = by_name.get(normalize_name(p.name))
        if found is None or pd.isna(p.first_nba):
            continue
        found = found[found["season"].between(p.first_nba - window, p.first_nba - 1)]
        if found.empty:
            continue
        schools = [school_key(s) for s in str(p.colleges).split(",")] if isinstance(p.colleges, str) else []
        schools = [s for s in schools if s]
        keys = found["school"].map(school_key)
        at_school = found[np.array([bool(key) and any(key in k or k in key for k in schools) for key in keys], dtype=bool)]
        how = "name and school" if not at_school.empty else "name"
        found = at_school if not at_school.empty else found
        rows.append({"bref_id": p.bref_id, "athlete_id": found.groupby("athlete_id")["minutes"].sum().idxmax(),
                     "by": how})
    return pd.DataFrame(rows, columns=["bref_id", "athlete_id", "by"])


def college_features(seasons: pd.DataFrame, min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """The raw :data:`STATS` of each college season with at least ``min_minutes``.

    Rates are per 40 minutes (points, assists, offensive and defensive
    rebounds, steals, blocks, turnovers, fouls); ``c_ts`` is true shooting,
    ``c_3par`` the share of shots from three, ``c_ftp`` free-throw
    percentage shrunk toward 70% by 10 attempts, ``c_team`` the team's
    margin over an average team in tens of points and ``c_mpg`` minutes per
    game over 40.
    """
    s = seasons[seasons["minutes"] >= min_minutes]
    per40 = 40 / s["minutes"]
    out = pd.DataFrame({
        "athlete_id": s["athlete_id"], "season": s["season"],
        "c_pts": s["points"] * per40, "c_ast": s["assists"] * per40, "c_oreb": s["offensive_rebounds"] * per40,
        "c_dreb": s["defensive_rebounds"] * per40, "c_stl": s["steals"] * per40, "c_blk": s["blocks"] * per40,
        "c_tov": s["turnovers"] * per40, "c_pf": s["fouls"] * per40,
        "c_ts": s["points"] / (2 * (s["field_goals_attempted"] + 0.44 * s["free_throws_attempted"])).clip(lower=1),
        "c_3par": s["three_point_field_goals_attempted"] / s["field_goals_attempted"].clip(lower=1),
        "c_ftp": (s["free_throws_made"] + 7) / (s["free_throws_attempted"] + 10),
        "c_team": s["team_rating"].fillna(0.0) / 10, "c_mpg": s["minutes"] / s["games"] / 40})
    return out.reset_index(drop=True)


def last_college_season(players: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Each matched NBA player's last college season of :func:`college_features`, standardized.

    ``players`` as in :func:`match_players`. Each stat is centered and scaled
    over the college seasons of all matched NBA players. Returns
    ``bref_id``, ``college_season`` and :data:`STATS`.
    """
    matches = match_players(players, seasons)
    feats = college_features(seasons).merge(matches[["bref_id", "athlete_id"]], on="athlete_id")
    cols = list(STATS)
    feats[cols] = (feats[cols] - feats[cols].mean()) / feats[cols].std()
    last = feats.sort_values("season", kind="stable").drop_duplicates("bref_id", keep="last")
    return last.rename(columns={"season": "college_season"})[["bref_id", "college_season", *cols]].reset_index(drop=True)
