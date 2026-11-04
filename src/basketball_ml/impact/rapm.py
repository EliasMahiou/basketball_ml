"""Regularized adjusted plus-minus (RAPM): each player's effect on scoring.

Every stint gives two observations, one per team on offense:

    points per 100 possessions = league average + home edge
                                 + sum of the 5 offensive players' O ratings
                                 + sum of the 5 defensive players' D ratings

weighted by the stint's possessions. Ridge regression shrinks every rating
toward a prior, which keeps players with few possessions, or who always
share the court with the same teammates, from getting wild values. The plain
version shrinks toward zero (average); with box-score data, each player is
shrunk toward a box-score estimate of his rating instead (see :func:`box_prior`).

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
# Possessions of league-average production added to each player's box-score rate.
PRIOR_SHRINK = 300.0
# Box-score counts of the offense and the defense used by the separate priors (see :func:`box_prior_od`).
OFFENSE_STATS = ("fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "ast", "oreb", "tov", "pfd")
DEFENSE_STATS = ("dreb", "stl", "blk", "pf")
# With the offense/defense prior the ratings can lean on it more: chosen on 2015-16 to 2018-19.
PRIOR_ALPHA = 8000.0
# Share of the opponents' 3-point luck removed from the defensive ratings (see :func:`fit_rapm`).
# All of it did slightly better than 0.5 or 0.75 on 2015-16 to 2018-19.
LUCK_THREES = 1.0


def _players(stints: pd.DataFrame) -> np.ndarray:
    return np.unique(stints[player_columns("home") + player_columns("away")].to_numpy())


def _points(stints: pd.DataFrame, side: str, luck: float | None) -> np.ndarray:
    """A side's points; with ``luck``, free throws count as expected and that share of 3-point luck is removed."""
    pts = stints[f"{side}_pts"].to_numpy(float)
    if luck is None:
        return pts
    pts = pts - stints[f"{side}_ftm"].to_numpy(float) + stints[f"{side}_xftm"].to_numpy(float)
    return pts - luck * 3 * (stints[f"{side}_fg3m"].to_numpy(float) - stints[f"{side}_xfg3m"].to_numpy(float))


def _rows(stints: pd.DataFrame, luck: float | None = None):
    """Offense rows (home then away) as (offense ids, defense ids, home flag, points, possessions)."""
    home = stints[player_columns("home")].to_numpy()
    away = stints[player_columns("away")].to_numpy()
    offense = np.vstack([home, away])
    defense = np.vstack([away, home])
    is_home = np.r_[np.ones(len(stints)), np.zeros(len(stints))]
    pts = np.r_[_points(stints, "home", luck), _points(stints, "away", luck)]
    poss = np.r_[stints["home_poss"].to_numpy(float), stints["away_poss"].to_numpy(float)]
    return offense, defense, is_home, pts, poss


def design(stints: pd.DataFrame, players: np.ndarray, season_weights: dict[int, float] | None = None,
           luck: float | None = None):
    """Sparse design matrix, target (points per 100) and weights (possessions).

    ``luck`` adjusts the points as in :func:`_points`.
    """
    offense, defense, is_home, pts, poss = _rows(stints, luck)
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
             season_weights: dict[int, float] | None = None,
             prior: pd.Series | pd.DataFrame | None = None, luck: bool = False) -> pd.DataFrame:
    """Fit RAPM on ``stints``. Returns one row per player with orapm, drapm, rapm and possessions.

    ``prior`` gives the ratings each player is shrunk toward: a Series of net
    ratings (split evenly between offense and defense) or a DataFrame with
    ``orapm`` and ``drapm`` columns, indexed by player id. Players missing from
    it, and every player when it is None, are shrunk toward zero.

    With ``luck``, free throws count as the shooters' expected makes, and the
    defensive ratings come from a second fit that also removes
    :data:`LUCK_THREES` of the opponents' 3-point luck (defenses have little
    control over 3-point percentage; shooters do, so offense keeps it).
    """
    players = _players(stints)
    X, y, w = design(stints, players, season_weights, luck=0.0 if luck else None)
    p = len(players)
    if prior is None:
        base_o = base_d = np.zeros(p)
    elif isinstance(prior, pd.DataFrame):
        base_o = prior["orapm"].reindex(players).fillna(0.0).to_numpy(float)
        base_d = prior["drapm"].reindex(players).fillna(0.0).to_numpy(float)
    else:
        base_o = base_d = prior.reindex(players).fillna(0.0).to_numpy(float) / 2
    offset = X[:, :p] @ base_o - X[:, p:2 * p] @ base_d
    model = Ridge(alpha=alpha, solver="sparse_cg", max_iter=5000, tol=1e-6).fit(X, y - offset, sample_weight=w)
    defense = model
    if luck:
        y_d = design(stints, players, season_weights, luck=LUCK_THREES)[1]
        defense = Ridge(alpha=alpha, solver="sparse_cg", max_iter=5000, tol=1e-6).fit(
            X, y_d - offset, sample_weight=w)
    table = pd.DataFrame({"player_id": players, "orapm": base_o + model.coef_[:p],
                          "drapm": base_d - defense.coef_[p:2 * p]})
    table["rapm"] = table["orapm"] + table["drapm"]
    table = table.merge(possessions(stints), on="player_id", how="left")
    table.attrs.update(intercept=float(model.intercept_), home_edge=float(model.coef_[-1]))
    return table


def box_prior(stints: pd.DataFrame, production: pd.DataFrame, season_weights: dict[int, float] | None = None,
              plain: pd.DataFrame | None = None, shrink: float = PRIOR_SHRINK, alpha: float = DEFAULT_ALPHA) -> pd.Series:
    """Box-score estimate of each player's rating, for RAPM to shrink toward.

    Each player's box-score production (one row per player and game, with
    ``season``, ``game_id``, ``player_id`` and ``production``, for example
    :func:`basketball_ml.predictions.build_player_games`) per 100 possessions
    over the games in ``stints``, pulled toward the league rate by ``shrink``
    possessions, is mapped to the rating scale by a possession-weighted
    regression of plain RAPM (``plain``, fitted here when not given) on it.
    """
    weights = season_weights or {s: 1.0 for s in stints["season"].unique()}
    if plain is None:
        plain = fit_rapm(stints, alpha, season_weights)
    games = production[production["game_id"].isin(set(stints["game_id"]))]
    made = (games["production"] * games["season"].map(weights)).groupby(games["player_id"]).sum()
    w = stints["season"].map(weights)
    poss = possessions(stints.assign(home_poss=stints["home_poss"] * w, away_poss=stints["away_poss"] * w))
    table = poss.set_index("player_id")[["possessions"]].join(made.rename("made")).fillna(0.0)
    league = table["made"].sum() / table["possessions"].sum()
    rate = 100 * (table["made"] + shrink * league) / (table["possessions"] + shrink)
    target = plain.set_index("player_id")["rapm"].reindex(table.index).fillna(0.0)
    root = np.sqrt(table["possessions"].to_numpy())
    X = np.c_[np.ones(len(rate)), rate.to_numpy()]
    beta = np.linalg.lstsq(X * root[:, None], target.to_numpy() * root, rcond=None)[0]
    return pd.Series(X @ beta, index=table.index, name="prior")


def on_court(stints: pd.DataFrame, season_weights: dict[int, float] | None = None) -> pd.DataFrame:
    """Offensive and defensive possessions and seconds on the court per player (index: player id)."""
    w = np.ones(len(stints)) if season_weights is None else stints["season"].map(season_weights).fillna(0.0).to_numpy()
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        ids = stints[player_columns(side)].to_numpy()
        parts.append(pd.DataFrame({"player_id": ids.ravel(),
                                   "o_poss": np.repeat(stints[f"{side}_poss"].to_numpy() * w, 5),
                                   "d_poss": np.repeat(stints[f"{other}_poss"].to_numpy() * w, 5),
                                   "seconds": np.repeat(stints["seconds"].to_numpy() * w, 5)}))
    return pd.concat(parts).groupby("player_id").sum()


def box_rates(stints: pd.DataFrame, box: pd.DataFrame, season_weights: dict[int, float] | None = None,
              shrink: float = PRIOR_SHRINK) -> pd.DataFrame:
    """Box-score rates per 100 possessions over the games in ``stints``, one row per player.

    Offensive counts are per 100 of the team's possessions with the player on
    the court, defensive counts (:data:`DEFENSE_STATS`) per 100 of the
    opponent's; each rate is pulled toward the league rate by ``shrink``
    possessions. ``mpg`` is minutes per game, pulled toward 12 by 3 games.
    ``possessions`` is the average of offensive and defensive possessions.
    """
    weights = season_weights or {s: 1.0 for s in stints["season"].unique()}
    games = box[box["game_id"].isin(set(stints["game_id"]))]
    w = games["season"].map(weights).fillna(0.0)
    stats = [*OFFENSE_STATS, *DEFENSE_STATS]
    counts = games[stats].mul(w, axis=0).groupby(games["player_id"]).sum()
    played = w.groupby(games["player_id"]).sum().rename("games")
    table = on_court(stints, weights).join(counts).join(played).fillna(0.0)
    out = pd.DataFrame(index=table.index)
    for stat in stats:
        poss = table["d_poss"] if stat in DEFENSE_STATS else table["o_poss"]
        league = table[stat].sum() / poss.sum()
        out[stat] = 100 * (table[stat] + shrink * league) / (poss + shrink)
    out["mpg"] = (table["seconds"] / 60 + 3 * 12.0) / (table["games"] + 3)
    out["possessions"] = (table["o_poss"] + table["d_poss"]) / 2
    return out


def box_prior_od(stints: pd.DataFrame, box: pd.DataFrame, season_weights: dict[int, float] | None = None,
                 plain: pd.DataFrame | None = None, shrink: float = PRIOR_SHRINK, alpha: float = DEFAULT_ALPHA,
                 ridge: float = 1.0, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Separate box-score estimates of each player's offensive and defensive rating.

    ``box`` holds per-game box scores (see :func:`basketball_ml.impact.build_box`).
    Plain RAPM (``plain``, fitted here when not given) is regressed on the
    standardized rates of :func:`box_rates`, offense and defense separately,
    with every rate available to both (steals say something about offense too),
    weighted by possessions and lightly ridge-penalized. ``features`` picks
    the rates used (default: all of them and minutes per game). Returns a
    DataFrame with ``orapm`` and ``drapm`` indexed by player id, for :func:`fit_rapm`.
    """
    if plain is None:
        plain = fit_rapm(stints, alpha, season_weights)
    rates = box_rates(stints, box, season_weights, shrink)
    features = list(features) if features is not None else [c for c in rates.columns if c != "possessions"]
    weight = rates["possessions"].to_numpy()
    weight = weight / weight.mean()
    X = rates[features].to_numpy(float)
    mean = np.average(X, axis=0, weights=weight)
    std = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weight))
    Z = (X - mean) / np.where(std > 0, std, 1.0)
    target = plain.set_index("player_id").reindex(rates.index).fillna(0.0)
    out = pd.DataFrame(index=rates.index)
    for column in ("orapm", "drapm"):
        model = Ridge(alpha=ridge).fit(Z, target[column].to_numpy(), sample_weight=weight)
        out[column] = model.predict(Z)
    return out


def _fit(stints: pd.DataFrame, alpha: float, weights: dict[int, float],
         production: pd.DataFrame | None, box: pd.DataFrame | None = None,
         luck: bool = False, prior_alpha: float = PRIOR_ALPHA) -> pd.DataFrame:
    """Plain RAPM, or RAPM shrunk toward a box-score prior.

    With ``box`` (per-game box scores) the prior has separate offense and
    defense parts (:func:`box_prior_od`) and the final fit uses ``prior_alpha``;
    with only ``production`` it is a net rating (:func:`box_prior`). ``luck``
    is passed to :func:`fit_rapm`.
    """
    plain = fit_rapm(stints, alpha, weights, luck=luck)
    if box is not None:
        return fit_rapm(stints, prior_alpha, weights, prior=box_prior_od(stints, box, weights, plain), luck=luck)
    if production is None:
        return plain
    return fit_rapm(stints, alpha, weights, prior=box_prior(stints, production, weights, plain), luck=luck)


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
                   half_life: float = DEFAULT_HALF_LIFE, alpha: float = DEFAULT_ALPHA,
                   production: pd.DataFrame | None = None, box: pd.DataFrame | None = None,
                   luck: bool = False) -> pd.DataFrame:
    """RAPM from the ``window`` seasons before ``season`` (recent seasons weigh more).

    With ``production`` (box-score production per player and game) or ``box``
    (per-game box scores), ratings are shrunk toward a box-score prior instead
    of zero; ``luck`` takes free-throw and opponent 3-point luck out (see
    :func:`_fit` and :func:`fit_rapm`).
    """
    seasons = [s for s in range(season - window, season) if s in set(stints["season"])]
    if not seasons:
        raise ValueError(f"No stints before {season}")
    return _fit(stints[stints["season"].isin(seasons)], alpha, recency_weights(seasons, half_life), production, box,
                luck)


def ratings_as_of(stints: pd.DataFrame, dates: pd.Series, season: int, as_of: pd.Timestamp,
                  window: int = DEFAULT_WINDOW, half_life: float = DEFAULT_HALF_LIFE,
                  alpha: float = DEFAULT_ALPHA, production: pd.DataFrame | None = None,
                  box: pd.DataFrame | None = None, luck: bool = False) -> pd.DataFrame:
    """RAPM during ``season``: the ``window`` seasons before plus its games played before ``as_of``.

    ``dates`` maps game ids to game dates; ``production``, ``box`` and ``luck`` are as in :func:`ratings_before`.
    """
    game_dates = stints["game_id"].map(dates)
    earlier = stints["season"].between(season - window, season - 1)
    so_far = (stints["season"] == season) & (game_dates < as_of)
    seasons = list(range(season - window, season + 1))
    return _fit(stints[earlier | so_far], alpha, recency_weights(seasons, half_life), production, box, luck)


def rating_schedule(stints: pd.DataFrame, dates: pd.Series, seasons: Sequence[int],
                    every_days: int = 30, production: pd.DataFrame | None = None,
                    box: pd.DataFrame | None = None, luck: bool = False) -> pd.DataFrame:
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
            ratings = ratings_as_of(stints, dates, season, as_of, production=production, box=box, luck=luck)
            parts.append(ratings.assign(season=season, as_of=as_of))
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
            default = NEWCOMER if name.startswith("RAPM") else float(ratings["rapm"].quantile(0.3))
            rows[t][name] = game_r2(test, predict_margin(test, ratings, default=default))
    return pd.DataFrame(rows).T
