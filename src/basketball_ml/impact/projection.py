"""From past ratings to next season: rookies, playing time and age.

RAPM from the seasons before describes how good a player was. To forecast
the next season, three things it ignores matter:

- Players without a rating (mostly rookies) are not all alike: high draft
  picks and older newcomers (often veterans of other leagues) are better.
- Players rated from few possessions are worse than their shrunk ratings say.
- Young players improve and old ones decline.

:func:`fit_projection` learns these effects from earlier seasons: the margin
of every stint that the ratings fitted before its season leave unexplained
is regressed on the players' features (see :func:`features`). Draft slots,
birth dates and the link from NBA.com to Basketball-Reference player ids come
from public GitHub mirrors of Basketball-Reference.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .rapm import NEWCOMER, predict_margin
from .stints import player_columns

BIO_URLS = {
    "Draft_Pick_History.csv": "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/"
                              "Data/Draft%20Pick%20History.csv",
    "Player_Career_Info.csv": "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/"
                              "Data/Player%20Career%20Info.csv",
    "index_master.csv": "https://raw.githubusercontent.com/gabriel1200/site_Data/master/index_master.csv",
}
FEATURES = ("new_const", "new_log_pick", "new_undrafted", "new_age", "rated_const", "rated_share", "rated_age")
UNDRAFTED_PICK = 75     # undrafted players count as pick 75 in log(pick), plus their own term
SHARE_POSSESSIONS = 1000.0


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


def load_bio(data_dir: str | Path = "data/bref") -> pd.DataFrame:
    """Birth date, height and draft of every player, indexed by NBA.com player id.

    ``pick`` is the overall pick of the player's last NBA draft and
    ``draft_year`` the year of that draft (both missing for undrafted players).
    """
    data_dir = download_bio(data_dir)
    ids = pd.read_csv(data_dir / "index_master.csv", usecols=["bref_id", "nba_id"]).dropna()
    ids = ids.drop_duplicates("nba_id").astype({"nba_id": np.int64})
    career = pd.read_csv(data_dir / "Player_Career_Info.csv", usecols=["player_id", "birth_date", "ht_in_in"])
    draft = pd.read_csv(data_dir / "Draft_Pick_History.csv", usecols=["season", "lg", "overall_pick", "player_id"])
    draft = draft[draft["lg"] == "NBA"].sort_values("season").drop_duplicates("player_id", keep="last")
    draft = draft.rename(columns={"season": "draft_year", "overall_pick": "pick"})
    bio = (ids.merge(career, left_on="bref_id", right_on="player_id", how="left").drop(columns="player_id")
              .merge(draft[["player_id", "draft_year", "pick"]], left_on="bref_id", right_on="player_id", how="left")
              .drop(columns="player_id"))
    bio = bio.rename(columns={"nba_id": "player_id", "ht_in_in": "height"})
    bio["birth"] = pd.to_datetime(bio["birth_date"], errors="coerce")
    return bio.set_index("player_id")[["bref_id", "birth", "height", "draft_year", "pick"]]


def ages(players: Iterable[int], season: int, bio: pd.DataFrame) -> pd.Series:
    """Age on February 1 of ``season`` (Basketball-Reference's convention), NaN when unknown."""
    birth = bio["birth"].reindex(list(players))
    return (pd.Timestamp(f"{season}-02-01") - birth).dt.days / 365.25


def features(players: Iterable[int], rated: pd.DataFrame, season: int, bio: pd.DataFrame) -> pd.DataFrame:
    """The projection features of :data:`FEATURES` for ``players`` in ``season``.

    ``rated`` holds the ratings fitted before ``season`` (with ``possessions``).
    Unrated players get a constant, the log of their draft pick, an undrafted
    flag and their age minus 22 (clipped to 19-30); rated players a constant,
    their share ``poss / (poss + 1000)`` and their age minus 27 (clipped to
    19-38). Unknown ages count as 0.
    """
    players = list(players)
    poss = rated.set_index("player_id")["possessions"].reindex(players)
    new = poss.isna().to_numpy().astype(float)
    old = 1.0 - new
    age = ages(players, season, bio).to_numpy()
    known = ~np.isnan(age)
    pick = bio["pick"].reindex(players).to_numpy()
    out = pd.DataFrame(index=pd.Index(players, name="player_id"))
    out["new_const"] = new
    out["new_log_pick"] = new * np.log(np.where(np.isnan(pick), UNDRAFTED_PICK, np.minimum(pick, UNDRAFTED_PICK)))
    out["new_undrafted"] = new * np.isnan(pick)
    out["new_age"] = new * np.where(known, np.clip(np.nan_to_num(age), 19, 30) - 22, 0.0)
    possessions = poss.fillna(0.0).to_numpy()
    out["rated_const"] = old
    out["rated_share"] = old * possessions / (possessions + SHARE_POSSESSIONS)
    out["rated_age"] = old * np.where(known, np.clip(np.nan_to_num(age), 19, 38) - 27, 0.0)
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


def fit_projection(stints: pd.DataFrame, ratings: Mapping[int, pd.DataFrame], bio: pd.DataFrame,
                   seasons: Iterable[int], ridge: float = 1.0) -> pd.Series:
    """Coefficients of :data:`FEATURES`, learned from ``seasons``.

    ``ratings`` maps each season to the ratings fitted before it. For each
    stint of those seasons, the margin per 100 possessions left after those
    ratings (unrated players count as :data:`NEWCOMER`) is regressed on the
    home-minus-away sums of the players' features, weighted by possessions.
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
    model = Ridge(alpha=ridge).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return pd.Series(model.coef_, index=list(FEATURES))


def project(ratings: pd.DataFrame, season: int, bio: pd.DataFrame, coef: pd.Series,
            players: Iterable[int] = ()) -> pd.DataFrame:
    """Ratings fitted before ``season`` turned into a forecast for it.

    Rated players get the projection added to ``rapm`` (and half of it to
    ``orapm`` and ``drapm``); unrated ``players`` get rows of their own,
    :data:`NEWCOMER` plus their projection. ``shift`` holds the change.
    """
    rated = set(ratings["player_id"])
    new = [pid for pid in dict.fromkeys(players) if pid not in rated]
    table = features([*ratings["player_id"], *new], ratings, season, bio)
    shift = table.to_numpy() @ coef.reindex(table.columns).fillna(0.0).to_numpy()
    out = pd.concat([ratings, pd.DataFrame({"player_id": new, "orapm": NEWCOMER / 2, "drapm": NEWCOMER / 2,
                                            "rapm": NEWCOMER, "possessions": 0.0})], ignore_index=True)
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
