"""Regularized adjusted plus-minus (RAPM): each player's effect on scoring.

Every stint gives two observations, one per team on offense:

    points per 100 possessions = league average + home edge
                                 + sum of the 5 offensive players' O ratings
                                 + sum of the 5 defensive players' D ratings

weighted by the stint's possessions. Ridge regression shrinks every rating
toward zero (average), which keeps players with few possessions, or who
always share the court with the same teammates, from getting wild values.

``orapm`` is points added per 100 possessions on offense, ``drapm`` points
prevented on defense, and ``rapm`` their sum (a +2 player makes his team
about 2 points per 100 possessions better than an average player would).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from .stints import player_columns

# Chosen on the 2015-16 to 2018-19 seasons by how well ratings from the two
# previous seasons explain each game's margin (see :func:`backtest`).
DEFAULT_ALPHA = 4000.0
DEFAULT_HALF_LIFE = 2.0
DEFAULT_WINDOW = 2
# Rating for players without one (mostly rookies), who are below average on the whole.
NEWCOMER = -1.5


def _players(stints: pd.DataFrame) -> np.ndarray:
    return np.unique(stints[player_columns("home") + player_columns("away")].to_numpy())


def _rows(stints: pd.DataFrame):
    """Offense rows (home then away) as (offense ids, defense ids, home flag, points, possessions)."""
    home = stints[player_columns("home")].to_numpy()
    away = stints[player_columns("away")].to_numpy()
    offense = np.vstack([home, away])
    defense = np.vstack([away, home])
    is_home = np.r_[np.ones(len(stints)), np.zeros(len(stints))]
    pts = np.r_[stints["home_pts"].to_numpy(float), stints["away_pts"].to_numpy(float)]
    poss = np.r_[stints["home_poss"].to_numpy(float), stints["away_poss"].to_numpy(float)]
    return offense, defense, is_home, pts, poss


def design(stints: pd.DataFrame, players: np.ndarray, season_weights: dict[int, float] | None = None):
    """Sparse design matrix, target (points per 100) and weights (possessions)."""
    offense, defense, is_home, pts, poss = _rows(stints)
    keep = poss > 0
    offense, defense, is_home, pts, poss = offense[keep], defense[keep], is_home[keep], pts[keep], poss[keep]
    n, p = len(pts), len(players)
    index = {pid: i for i, pid in enumerate(players)}
    o_cols = np.vectorize(index.get)(offense)
    d_cols = np.vectorize(index.get)(defense) + p
    rows = np.repeat(np.arange(n), 5)
    X = sparse.csr_matrix(
        (np.ones(10 * n), (np.r_[rows, rows], np.r_[o_cols.ravel(), d_cols.ravel()])), shape=(n, 2 * p))
    X = sparse.hstack([X, sparse.csr_matrix(is_home.reshape(-1, 1))]).tocsr()
    y = 100 * pts / poss
    w = poss
    if season_weights is not None:
        seasons = np.r_[stints["season"].to_numpy(), stints["season"].to_numpy()][keep]
        w = w * np.vectorize(lambda s: season_weights.get(s, 0.0))(seasons)
    return X, y, w


def fit_rapm(stints: pd.DataFrame, alpha: float = DEFAULT_ALPHA,
             season_weights: dict[int, float] | None = None) -> pd.DataFrame:
    """Fit RAPM on ``stints``. Returns one row per player with orapm, drapm, rapm and possessions."""
    players = _players(stints)
    X, y, w = design(stints, players, season_weights)
    model = Ridge(alpha=alpha, solver="sparse_cg", max_iter=5000, tol=1e-6).fit(X, y, sample_weight=w)
    p = len(players)
    table = pd.DataFrame({"player_id": players, "orapm": model.coef_[:p], "drapm": -model.coef_[p:2 * p]})
    table["rapm"] = table["orapm"] + table["drapm"]
    table = table.merge(possessions(stints), on="player_id", how="left")
    table.attrs.update(intercept=float(model.intercept_), home_edge=float(model.coef_[-1]))
    return table


def possessions(stints: pd.DataFrame) -> pd.DataFrame:
    """Offensive possessions and seconds on court per player."""
    parts = []
    for side in ("home", "away"):
        ids = stints[player_columns(side)].to_numpy()
        parts.append(pd.DataFrame({"player_id": ids.ravel(),
                                   "possessions": np.repeat(stints[f"{side}_poss"].to_numpy(), 5),
                                   "seconds": np.repeat(stints["seconds"].to_numpy(), 5)}))
    return pd.concat(parts).groupby("player_id", as_index=False).sum()


def predict_margin(stints: pd.DataFrame, ratings: pd.DataFrame, default: float = NEWCOMER,
                   column: str = "rapm") -> np.ndarray:
    """Predicted home-minus-away points per 100 possessions for each stint (no home edge).

    Players without a rating get ``default``.
    """
    r = dict(zip(ratings["player_id"], ratings[column]))
    out = np.zeros(len(stints))
    for side, sign in (("home", 1.0), ("away", -1.0)):
        ids = stints[player_columns(side)].to_numpy()
        out += sign * np.vectorize(lambda pid: r.get(pid, default))(ids).sum(axis=1)
    return out


def recency_weights(seasons: Sequence[int], half_life: float = 1.0) -> dict[int, float]:
    """Weight 1 for the latest season, halving every ``half_life`` seasons before it."""
    last = max(seasons)
    return {s: 0.5 ** ((last - s) / half_life) for s in seasons}


def ratings_before(stints: pd.DataFrame, season: int, window: int = DEFAULT_WINDOW,
                   half_life: float = DEFAULT_HALF_LIFE, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """RAPM from the ``window`` seasons before ``season`` (recent seasons weigh more)."""
    seasons = [s for s in range(season - window, season) if s in set(stints["season"])]
    if not seasons:
        raise ValueError(f"No stints before {season}")
    return fit_rapm(stints[stints["season"].isin(seasons)], alpha, recency_weights(seasons, half_life))


def ratings_as_of(stints: pd.DataFrame, dates: pd.Series, season: int, as_of: pd.Timestamp,
                  window: int = DEFAULT_WINDOW, half_life: float = DEFAULT_HALF_LIFE,
                  alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """RAPM during ``season``: the ``window`` seasons before plus its games played before ``as_of``.

    ``dates`` maps game ids to game dates.
    """
    game_dates = stints["game_id"].map(dates)
    earlier = stints["season"].between(season - window, season - 1)
    so_far = (stints["season"] == season) & (game_dates < as_of)
    seasons = list(range(season - window, season + 1))
    return fit_rapm(stints[earlier | so_far], alpha, recency_weights(seasons, half_life))


def rating_schedule(stints: pd.DataFrame, dates: pd.Series, seasons: Sequence[int],
                    every_days: int = 30) -> pd.DataFrame:
    """Ratings refreshed every ``every_days`` days through each season, each from earlier games only.

    Returns a long table with ``season`` and ``as_of`` (first date the ratings
    can be used) next to each player's ratings.
    """
    stint_seasons = set(stints["season"])
    season_of = stints.drop_duplicates("game_id").set_index("game_id")["season"]
    game_dates = pd.DataFrame({"season": season_of, "date": season_of.index.map(dates)}).dropna()
    parts = []
    for season in seasons:
        if not any(s in stint_seasons for s in range(season - DEFAULT_WINDOW, season)):
            continue
        played = game_dates.loc[game_dates["season"] == season, "date"]
        if played.empty:
            continue
        for as_of in pd.date_range(played.min(), played.max(), freq=f"{every_days}D"):
            parts.append(ratings_as_of(stints, dates, season, as_of).assign(season=season, as_of=as_of))
    return pd.concat(parts, ignore_index=True)


def on_court_net(stints: pd.DataFrame, shrink: float = 1000.0) -> pd.DataFrame:
    """Unadjusted plus-minus: team margin per 100 possessions with the player on the court.

    ``shrink`` possessions of a zero margin are added to every player.
    """
    poss = (stints["home_poss"] + stints["away_poss"]).to_numpy() / 2
    margin = (stints["home_pts"] - stints["away_pts"]).to_numpy()
    parts = []
    for side, sign in (("home", 1), ("away", -1)):
        ids = stints[player_columns(side)].to_numpy()
        parts.append(pd.DataFrame({"player_id": ids.ravel(), "margin": np.repeat(sign * margin, 5),
                                   "possessions": np.repeat(poss, 5)}))
    t = pd.concat(parts).groupby("player_id", as_index=False).sum()
    t["rapm"] = 100 * t["margin"] / (t["possessions"] + shrink)
    return t


def game_r2(stints: pd.DataFrame, predicted_per_100: np.ndarray) -> float:
    """Share of the variance of game margins explained by the stint predictions."""
    poss = (stints["home_poss"] + stints["away_poss"]).to_numpy() / 2
    g = pd.DataFrame({"game_id": stints["game_id"].to_numpy(), "pred": predicted_per_100 * poss / 100,
                      "margin": (stints["home_pts"] - stints["away_pts"]).to_numpy()}).groupby("game_id").sum()
    return float(np.corrcoef(g["pred"], g["margin"])[0, 1] ** 2)


def backtest(stints: pd.DataFrame, test_seasons: Sequence[int], raters: dict | None = None,
             window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """How well each rating, fitted on the seasons before, explains next season's game margins.

    Every rating gets the actual lineups of each stint, so the test isolates how
    good the player values are. ``raters`` maps a name to a function
    ``(stints of earlier seasons, test season) -> table with player_id and rapm``;
    the default compares RAPM with unregularized APM and raw on-court net rating.
    Unrated players get the rating's 30th percentile. Returns one row per season.
    """
    if raters is None:
        raters = {
            "RAPM": lambda train, t: ratings_before(train, t, window),
            "APM (no shrinkage)": lambda train, t: ratings_before(train, t, window, alpha=1.0),
            "On-court net rating": lambda train, t: on_court_net(train),
        }
    rows = {}
    for t in test_seasons:
        train = stints[stints["season"].between(t - window, t - 1)]
        test = stints[stints["season"] == t]
        if train.empty or test.empty:
            continue
        rows[t] = {}
        for name, rate in raters.items():
            ratings = rate(train, t)
            default = NEWCOMER if name == "RAPM" else float(ratings["rapm"].quantile(0.3))
            rows[t][name] = game_r2(test, predict_margin(test, ratings, default=default))
    return pd.DataFrame(rows).T
