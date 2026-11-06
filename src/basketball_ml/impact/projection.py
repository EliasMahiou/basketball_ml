"""From past ratings to next season: rookies, playing time and age.

RAPM from the seasons before describes how good a player was. To forecast
the next season, three things it ignores matter:

- Players without a rating (mostly rookies) are not all alike: high draft
  picks, older newcomers (often veterans of other leagues), draft picks who
  stayed abroad first and good college players are better, and players
  coming back to the NBA after two seasons away are worse.
- Players rated from few possessions are worse than their shrunk ratings say.
- Young players improve and old ones decline.

:func:`fit_projection` learns these effects from earlier seasons: the margin
of every stint that the ratings fitted before its season leave unexplained
is regressed on the players' features (see :func:`features`). Draft slots,
birth dates, NBA seasons, colleges and the link from NBA.com to
Basketball-Reference player ids come from public GitHub mirrors of
Basketball-Reference; college seasons from :mod:`basketball_ml.impact.college`.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from . import college
from .rapm import NEWCOMER, predict_margin
from .stints import player_columns

BIO_URLS = {
    "Draft_Pick_History.csv": "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/"
                              "Data/Draft%20Pick%20History.csv",
    "Player_Career_Info.csv": "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/"
                              "Data/Player%20Career%20Info.csv",
    "Player_Season_Info.csv": "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/"
                              "Data/Player%20Season%20Info.csv",
    "index_master.csv": "https://raw.githubusercontent.com/gabriel1200/site_Data/master/index_master.csv",
}
COLLEGE_FEATURES = ("new_college", *(f"new_{stat}" for stat in college.STATS))
FEATURES = ("new_const", "new_log_pick", "new_undrafted", "new_age", "rated_const", "rated_share", "rated_age",
            "new_returning", "new_stash", *COLLEGE_FEATURES)
UNDRAFTED_PICK = 75     # undrafted players count as pick 75 in log(pick), plus their own term
SHARE_POSSESSIONS = 1000.0
RIDGE = 1.0
# Stint margins per 100 possessions have a noise variance of about 2e4 per possession; 1e5 is a prior standard
# deviation of about 0.45 points per standard deviation of a college stat, so few seasons of rookies don't overfit.
COLLEGE_RIDGE = 1e5


def download_bio(data_dir: str | Path = "data/bref", overwrite: bool = False) -> Path:
    """Download the draft, birth date and player id tables to ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in BIO_URLS.items():
        out = data_dir / name
        if overwrite or not out.exists():
            with urllib.request.urlopen(url, timeout=300) as resp:
                out.write_bytes(resp.read())
    return data_dir


def last_college_year(today: pd.Timestamp | None = None) -> int:
    """End year of the last finished college season: 2026 from May 2026 to April 2027."""
    today = pd.Timestamp.today() if today is None else today
    return today.year if today.month >= 5 else today.year - 1


def load_bio(data_dir: str | Path = "data/bref", college_dir: str | Path | None = "data/college") -> pd.DataFrame:
    """Birth date, height, draft, NBA debut and last college season of every player, indexed by NBA.com player id.

    ``pick`` is the overall pick of the player's last NBA draft and
    ``draft_year`` the year of that draft (both missing for undrafted
    players); ``first_nba`` is the player's first NBA season (end year).
    ``college_season`` and the stats of :data:`college.STATS` describe his
    last college season (see :func:`college.last_college_season`), missing
    when he has none; ``college_dir=None`` leaves them all missing.
    """
    data_dir = download_bio(data_dir)
    ids = pd.read_csv(data_dir / "index_master.csv", usecols=["bref_id", "nba_id"]).dropna()
    ids = ids.drop_duplicates("nba_id").astype({"nba_id": np.int64})
    career = pd.read_csv(data_dir / "Player_Career_Info.csv",
                         usecols=["player_id", "player", "colleges", "birth_date", "ht_in_in"]).drop_duplicates("player_id")
    draft = pd.read_csv(data_dir / "Draft_Pick_History.csv", usecols=["season", "lg", "overall_pick", "player_id"])
    draft = draft[draft["lg"] == "NBA"].sort_values("season").drop_duplicates("player_id", keep="last")
    draft = draft.rename(columns={"season": "draft_year", "overall_pick": "pick"})
    seasons = pd.read_csv(data_dir / "Player_Season_Info.csv", usecols=["season", "lg", "player_id"])
    career["first_nba"] = career["player_id"].map(seasons[seasons["lg"] == "NBA"].groupby("player_id")["season"].min())
    bio = (ids.merge(career, left_on="bref_id", right_on="player_id", how="left").drop(columns="player_id")
              .merge(draft[["player_id", "draft_year", "pick"]], left_on="bref_id", right_on="player_id", how="left")
              .drop(columns="player_id"))
    bio = bio.rename(columns={"nba_id": "player_id", "ht_in_in": "height"})
    bio["birth"] = pd.to_datetime(bio["birth_date"], errors="coerce")
    stats = ["college_season", *college.STATS]
    if college_dir is None:
        bio[stats] = np.nan
    else:
        table = college.college_seasons(range(college.FIRST_SEASON, last_college_year() + 1), college_dir)
        players = career.rename(columns={"player_id": "bref_id", "player": "name"})
        last = college.last_college_season(players, table).set_index("bref_id")
        bio = bio.join(last, on="bref_id")
    return bio.set_index("player_id")[["bref_id", "birth", "height", "draft_year", "pick", "first_nba", *stats]]


def ages(players: Iterable[int], season: int, bio: pd.DataFrame) -> pd.Series:
    """Age on February 1 of ``season`` (Basketball-Reference's convention), NaN when unknown."""
    birth = bio["birth"].reindex(list(players))
    return (pd.Timestamp(f"{season}-02-01") - birth).dt.days / 365.25


def features(players: Iterable[int], rated: pd.DataFrame, season: int, bio: pd.DataFrame) -> pd.DataFrame:
    """The projection features of :data:`FEATURES` for ``players`` in ``season``.

    ``rated`` holds the ratings fitted before ``season`` (with ``possessions``).
    Unrated players get a constant, the log of their draft pick, an undrafted
    flag, their age minus 22 (clipped to 19-30), a flag if they played in the
    NBA before (``new_returning``), a flag for draft picks who start in the
    NBA a season or more after their draft (``new_stash``), and their last
    college season if it ended in the 6 seasons before: a flag
    (``new_college``) and the standardized stats of :data:`college.STATS`.
    Rated players get a constant, their share ``poss / (poss + 1000)`` and
    their age minus 27 (clipped to 19-38). Unknown ages count as 0.
    """
    players = list(players)
    poss = rated.set_index("player_id")["possessions"].reindex(players)
    new = poss.isna().to_numpy().astype(float)
    old = 1.0 - new
    age = ages(players, season, bio).to_numpy()
    known = ~np.isnan(age)
    b = bio.reindex(players)

    def column(name: str) -> np.ndarray:
        """A bio column as floats, all missing when ``bio`` lacks it."""
        return b[name].to_numpy(float) if name in b else np.full(len(players), np.nan)

    pick = column("pick")
    out = pd.DataFrame(index=pd.Index(players, name="player_id"))
    out["new_const"] = new
    out["new_log_pick"] = new * np.log(np.where(np.isnan(pick), UNDRAFTED_PICK, np.minimum(pick, UNDRAFTED_PICK)))
    out["new_undrafted"] = new * np.isnan(pick)
    out["new_age"] = new * np.where(known, np.clip(np.nan_to_num(age), 19, 30) - 22, 0.0)
    possessions = poss.fillna(0.0).to_numpy()
    out["rated_const"] = old
    out["rated_share"] = old * possessions / (possessions + SHARE_POSSESSIONS)
    out["rated_age"] = old * np.where(known, np.clip(np.nan_to_num(age), 19, 38) - 27, 0.0)
    returning = column("first_nba") < season          # missing compares False
    out["new_returning"] = new * returning
    out["new_stash"] = new * ((column("draft_year") < season - 1) & ~returning)
    last = column("college_season")
    has = (last <= season - 1) & (last >= season - college.MAX_GAP)
    out["new_college"] = new * has
    for stat in college.STATS:
        out[f"new_{stat}"] = new * np.where(has, np.nan_to_num(column(stat)), 0.0)
    return out


def _stint_features(stints: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Home-minus-away sums of the players' features for each stint."""
    M = table.to_numpy()
    row = {pid: i for i, pid in enumerate(table.index)}
    X = np.zeros((len(stints), M.shape[1]))
    for side, sign in (("home", 1.0), ("away", -1.0)):
        for col in player_columns(side):
            X += sign * M[[row[pid] for pid in stints[col].to_numpy()]]
    return X


def _players(stints: pd.DataFrame) -> np.ndarray:
    return np.unique(stints[player_columns("home") + player_columns("away")].to_numpy())


def penalties(names: Iterable[str] = FEATURES, ridge: float = RIDGE, college_ridge: float = COLLEGE_RIDGE) -> np.ndarray:
    """Ridge penalty of each feature: ``college_ridge`` for :data:`COLLEGE_FEATURES`, ``ridge`` for the others."""
    return np.array([college_ridge if name in COLLEGE_FEATURES else ridge for name in names])


def weighted_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """Ridge coefficients with an intercept, one penalty per column, rows weighted by ``weights``."""
    total = weights.sum()
    Xc = X - weights @ X / total
    yc = y - weights @ y / total
    gram = (Xc * weights[:, None]).T @ Xc + np.diag(penalty)
    return np.linalg.solve(gram, (Xc * weights[:, None]).T @ yc)


def fit_projection(stints: pd.DataFrame, ratings: Mapping[int, pd.DataFrame], bio: pd.DataFrame,
                   seasons: Iterable[int], ridge: float = RIDGE) -> pd.Series:
    """Coefficients of :data:`FEATURES`, learned from ``seasons``.

    ``ratings`` maps each season to the ratings fitted before it. For each
    stint of those seasons, the margin per 100 possessions left after those
    ratings (unrated players count as :data:`NEWCOMER`) is regressed on the
    home-minus-away sums of the players' features, weighted by possessions,
    with a ridge penalty of ``ridge`` (:data:`COLLEGE_RIDGE` for the college
    stats, which few rookies inform in the first seasons).
    """
    Xs, ys, ws = [], [], []
    for season in seasons:
        part = stints[stints["season"] == season]
        if part.empty or season not in ratings:
            continue
        poss = ((part["home_poss"] + part["away_poss"]) / 2).to_numpy()
        part, poss = part[poss > 0], poss[poss > 0]
        margin = 100 * (part["home_pts"] - part["away_pts"]).to_numpy() / poss
        ys.append(margin - predict_margin(part, ratings[season]))
        Xs.append(_stint_features(part, features(_players(part), ratings[season], season, bio)))
        ws.append(poss)
    if not Xs:
        raise ValueError("No season to learn the projection from")
    coef = weighted_ridge(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), penalties(FEATURES, ridge))
    return pd.Series(coef, index=list(FEATURES))


def project(ratings: pd.DataFrame, season: int, bio: pd.DataFrame, coef: pd.Series,
            players: Iterable[int] = ()) -> pd.DataFrame:
    """Ratings fitted before ``season`` turned into a forecast for it.

    Rated players get the projection added to ``rapm`` (and half of it to
    ``orapm`` and ``drapm``); unrated ``players`` get rows of their own,
    :data:`NEWCOMER` plus their projection. ``shift`` holds the change.

    Stint margins only compare the ten players on the court, so they leave
    the forecast's zero free: everyone can move by the same amount. The
    forecast is centered so that the rated players' average, weighted by
    their possessions, is 0: values are points per 100 possessions above an
    average player.
    """
    rated = set(ratings["player_id"])
    new = [pid for pid in dict.fromkeys(players) if pid not in rated]
    table = features([*ratings["player_id"], *new], ratings, season, bio)
    shift = table.to_numpy() @ coef.reindex(table.columns).fillna(0.0).to_numpy()
    out = pd.concat([ratings, pd.DataFrame({"player_id": new, "orapm": NEWCOMER / 2, "drapm": NEWCOMER / 2,
                                            "rapm": NEWCOMER, "possessions": 0.0})], ignore_index=True)
    weight = out["possessions"].fillna(0.0).to_numpy(float)
    if (weight > 0).any():
        shift = shift - np.average((out["rapm"].to_numpy(float) + shift)[weight > 0], weights=weight[weight > 0])
    out["shift"] = shift
    out["rapm"] = out["rapm"] + shift
    out["orapm"] = out["orapm"] + shift / 2
    out["drapm"] = out["drapm"] + shift / 2
    return out


def projection_coefficients(stints: pd.DataFrame, before: Mapping[int, pd.DataFrame], bio: pd.DataFrame,
                            seasons: Iterable[int], first: int | None = None) -> pd.DataFrame:
    """The projection's coefficients for each of ``seasons`` (one row per season, columns :data:`FEATURES`).

    The row of season t is learned from the seasons from ``first`` (default:
    the earliest in ``before``) to t - 1 that have stints; seasons with none
    are left out.
    """
    first = min(before) if first is None else first
    played = set(stints["season"])
    rows = {}
    for season in seasons:
        earlier = [s for s in range(first, season) if s in before and s in played]
        if earlier:
            rows[season] = fit_projection(stints, before, bio, earlier)
    return pd.DataFrame(rows, index=list(FEATURES)).T.rename_axis("season")


def preseason_ratings(stints: pd.DataFrame, before: Mapping[int, pd.DataFrame], bio: pd.DataFrame,
                      seasons: Iterable[int], first: int | None = None,
                      coefficients: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast ratings for each of ``seasons`` from data before it only.

    ``before`` maps seasons to the ratings fitted before them (see
    :func:`basketball_ml.impact.ratings_before`); the projection for season t
    is learned from the seasons from ``first`` (default: the earliest in
    ``before``) to t - 1, unless ``coefficients`` (from
    :func:`projection_coefficients`) are given. Every player of season t's
    stints gets a row; for a season not played yet, the rookies of the draft
    before it do.
    """
    seasons = [s for s in seasons if s in before]
    if coefficients is None:
        coefficients = projection_coefficients(stints, before, bio, seasons, first)
    parts = []
    for season in seasons:
        if season not in coefficients.index:
            continue
        played = stints[stints["season"] == season]
        players = _players(played) if len(played) else bio.index[bio["draft_year"] == season - 1]
        parts.append(project(before[season], season, bio, coefficients.loc[season], players).assign(season=season))
    return pd.concat(parts, ignore_index=True)


def project_schedule(schedule: pd.DataFrame, stints: pd.DataFrame, bio: pd.DataFrame,
                     coefficients: pd.DataFrame) -> pd.DataFrame:
    """In-season ratings (see :func:`basketball_ml.impact.rating_schedule`) with the projection added.

    Each refresh of season t gets season t's coefficients, learned from
    earlier seasons only, and rows for the season's players it doesn't rate
    yet. Seasons without coefficients are left as they are.
    """
    parts = []
    for season, group in schedule.groupby("season"):
        if season not in coefficients.index:
            parts.append(group)
            continue
        players = _players(stints[stints["season"] == season])
        for as_of, ratings in group.groupby("as_of"):
            projected = project(ratings.drop(columns=["season", "as_of"]), season, bio, coefficients.loc[season], players)
            parts.append(projected.assign(season=season, as_of=as_of))
    return pd.concat(parts, ignore_index=True)
