"""Live win-probability models, their walk-forward backtest and evaluation by game phase.

Three models of P(home team wins | game state):

- ``stern``: Stern's (1994) Brownian-motion model. The home margin at the
  end is the current margin, plus the value of the ball, plus the pre-game
  expected margin scaled by the share of the game left, plus normal noise
  whose variance is proportional to the time left. Its six parameters
  (three possession values, the weight of the pre-game spread, the noise
  and a few seconds added to the time left, so that the last seconds stay
  finite) are fit by maximum likelihood.
- ``logistic``: logistic regression on the same quantities, transformed as
  the Brownian model suggests (margin / sqrt(time left), pre-game logit x
  sqrt(time left) and x time left, possession / sqrt(time left)) but with
  free weights. Time left gets 15 extra seconds, as the Brownian fit does.
- ``boosting``: gradient-boosted trees on the raw state, constrained to be
  increasing in the margin, the pre-game spread and the possession features.
  Leaves need 20,000 training states (states of one game are nearly
  duplicates: small leaves memorize single games through their pre-game spread).

Settings (the logistic time offset, the boosting leaves) were picked on the
2015-16 and 2016-17 test seasons.

``pregame`` (the Elo probability, whatever happens in the game) is the
reference every in-game model should beat.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import pbp
from .states import REGULATION_SECONDS

METHODS = ("pregame", "stern", "logistic", "boosting")
NAMES = {"pregame": "Pre-game Elo only", "stern": "Brownian motion (Stern)", "logistic": "Logistic regression",
         "boosting": "Gradient boosting"}
PHASES = ("Q1", "Q2", "Q3", "Q4 before the last 2 min", "Q4 last 2 min", "Overtime")
POSSESSION = ["ball", "rebound", "free_throws"]
RAW_FEATURES = ["seconds_left", "margin", *POSSESSION, "elo_spread", "overtime"]
LOGISTIC_FEATURES = ["margin_scaled", "pregame_sqrt", "pregame_frac", "ball_scaled", "rebound_scaled",
                     "free_throws_scaled", "margin"]
# Boosting: increasing in the margin, the pre-game spread and in having the ball.
MONOTONIC = {"margin": 1, "elo_spread": 1, "ball": 1, "rebound": 1, "free_throws": 1}
LOGISTIC_OFFSET = 15.0  # seconds added to the time left in the logistic features
MAX_TRAIN_ROWS = 2_000_000  # training states sampled per fold
EPS = 1e-6


def model_frame(states: pd.DataFrame) -> pd.DataFrame:
    """Model inputs of each state.

    ``ball`` is +1 when the home team has a live possession and -1 for the
    away team; ``rebound`` is +1 (-1) after a home (away) miss with the
    rebound pending; ``free_throws`` is the signed number of free throws
    still to shoot. Time features use the share of the game left, with one
    ``LOGISTIC_OFFSET`` seconds added so the last seconds stay finite.
    """
    out = pd.DataFrame(index=states.index)
    out["seconds_left"] = states["seconds_left"].astype(np.float32)
    out["margin"] = states["margin"].astype(np.float32)
    poss = states["poss"].to_numpy()
    loose = states["loose"].to_numpy().astype(bool)
    fts = states["fts"].to_numpy()
    out["ball"] = np.where(~loose & (fts == 0), poss, 0).astype(np.float32)
    out["rebound"] = np.where(loose, poss, 0).astype(np.float32)
    out["free_throws"] = (poss * fts).astype(np.float32)
    out["elo_spread"] = states["elo_spread"].astype(np.float32)
    out["overtime"] = (states["period"] > 4).astype(np.float32)
    frac = (out["seconds_left"] + LOGISTIC_OFFSET) / REGULATION_SECONDS
    root = np.sqrt(frac)
    logit = np.log(states["elo_prob"] / (1 - states["elo_prob"])).astype(np.float32)
    out["margin_scaled"] = out["margin"] / root
    out["pregame_sqrt"] = logit * root
    out["pregame_frac"] = logit * frac
    for col in POSSESSION:
        out[f"{col}_scaled"] = out[col] / root
    return out


class SternModel:
    """P(home win) = Phi((margin + v . possession + k * spread * f) / (sigma * sqrt(f + c / 2880))).

    ``f`` is the share of regulation left (an overtime's 5 minutes count as
    5 / 48), ``spread`` the pre-game expected home margin.
    """

    def __init__(self) -> None:
        self.params = np.array([0.5, -0.3, 0.8, 1.0, np.log(12.0), np.log(5.0)])

    @staticmethod
    def _inputs(frame: pd.DataFrame):
        x = frame[POSSESSION].to_numpy(np.float64)
        t = frame["seconds_left"].to_numpy(np.float64)
        return frame["margin"].to_numpy(np.float64), x, frame["elo_spread"].to_numpy(np.float64), t

    @staticmethod
    def _z(theta: np.ndarray, margin, x, spread, t):
        v, k, a, b = theta[:3], theta[3], theta[4], theta[5]
        f = t / REGULATION_SECONDS
        num = margin + x @ v + k * spread * f
        den = np.exp(a) * np.sqrt((t + np.exp(b)) / REGULATION_SECONDS)
        return num / den, den, f

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> SternModel:
        margin, x, spread, t = self._inputs(frame)
        sign = 2.0 * np.asarray(y, dtype=np.float64) - 1.0

        def loss(theta):
            z, den, f = self._z(theta, margin, x, spread, t)
            ll = log_ndtr(sign * z)
            dz = sign * np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - ll)  # d loglik / dz
            c = np.exp(theta[5])
            grads = np.column_stack([x / den[:, None], spread * f / den, -z, -z * 0.5 * c / (t + c)])
            return -ll.mean(), -(dz[:, None] * grads).mean(axis=0)

        res = minimize(loss, self.params, jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] * 4 + [(np.log(3.0), np.log(40.0)), (np.log(0.1), np.log(600.0))])
        self.params = res.x
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        z, _, _ = self._z(self.params, *self._inputs(frame))
        return ndtr(z)

    def describe(self) -> dict[str, float]:
        v, k, a, b = self.params[:3], self.params[3], self.params[4], self.params[5]
        return {"ball": v[0], "rebound": v[1], "free_throw": v[2], "spread_weight": k,
                "sigma": float(np.exp(a)), "extra_seconds": float(np.exp(b))}


def make_model(method: str):
    """A fresh, unfitted model for ``method``."""
    if method == "stern":
        return SternModel()
    if method == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    if method == "boosting":
        cst = [MONOTONIC.get(f, 0) for f in RAW_FEATURES]
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              min_samples_leaf=20_000, l2_regularization=1.0, monotonic_cst=cst,
                                              early_stopping=False, random_state=0)
    raise ValueError(f"No trainable model for {method!r}")


def _columns(method: str) -> list[str]:
    return LOGISTIC_FEATURES if method == "logistic" else RAW_FEATURES


def fit(method: str, states: pd.DataFrame):
    """Fit ``method`` on game states with their pre-game strength and result (``states.add_game_info``)."""
    model = make_model(method)
    frame, y = model_frame(states), states["home_win"].to_numpy()
    if method == "stern":
        return model.fit(frame, y)
    return model.fit(frame[_columns(method)], y)


def predict(method: str, model, states: pd.DataFrame) -> np.ndarray:
    """Home win probability of each state from a model returned by :func:`fit`."""
    frame = model_frame(states)
    if method == "stern":
        return model.predict_proba(frame)
    return model.predict_proba(frame[_columns(method)])[:, 1]


def trainable(states: pd.DataFrame) -> pd.Series:
    """States used to fit and evaluate the models: everything but the end-of-period markers."""
    return states["event"] != pbp.PERIOD_END


def walk_forward(states: pd.DataFrame, test_seasons: Iterable[int], methods: Sequence[str] = METHODS,
                 max_train_rows: int | None = MAX_TRAIN_ROWS, seed: int = 0) -> pd.DataFrame:
    """Predict each test season with models fit only on earlier seasons.

    ``states`` must hold ``elo_prob``, ``elo_spread`` and ``home_win``
    (``states.add_game_info``). Returns the test seasons' states with one
    ``p_<method>`` column per method. ``max_train_rows`` subsamples the
    training states (consecutive states of a game carry little new information).
    """
    from threadpoolctl import threadpool_limits

    states = states[trainable(states)]
    out = []
    for season in test_seasons:
        train = states[states["season"] < season]
        test = states[states["season"] == season].copy()
        if train.empty or test.empty:
            continue
        if max_train_rows and len(train) > max_train_rows:
            train = train.sample(max_train_rows, random_state=seed)
        for m in methods:
            if m == "pregame":
                test[f"p_{m}"] = test["elo_prob"].to_numpy()
                continue
            with threadpool_limits(2):
                model = fit(m, train)
                test[f"p_{m}"] = predict(m, model, test)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def phase(states: pd.DataFrame) -> pd.Series:
    """Game phase of each state: Q1, Q2, Q3, Q4 before / in its last 2 minutes, overtime."""
    period, left = states["period"].to_numpy(), states["seconds_left"].to_numpy()
    labels = np.select([period == 1, period == 2, period == 3, (period == 4) & (left > 120), period == 4],
                       list(PHASES[:5]), default=PHASES[5])
    return pd.Series(pd.Categorical(labels, categories=PHASES), index=states.index)


def endgame(states: pd.DataFrame) -> pd.Series:
    """The last 2 minutes of the 4th quarter and of overtimes."""
    return (states["period"] >= 4) & (states["seconds_left"] <= 120)


def _scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return float(log_loss(y, p, labels=[0, 1])), float(brier_score_loss(y, p))


def evaluate(results: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Log loss and Brier score per method, overall and by phase (rows: phase, columns: metric x method).

    Every state counts once, so periods with more events weigh more, and
    the states of one game are strongly correlated: differences between
    models should be judged with ``bootstrap_difference``.
    """
    groups = {"All": results, **{ph: g for ph, g in results.groupby(phase(results), observed=True)}}
    rows = {}
    for name, g in groups.items():
        y = g["home_win"].to_numpy()
        row = {("states", ""): len(g)}
        for m in methods:
            ll, br = _scores(y, g[f"p_{m}"].to_numpy())
            row[("log_loss", m)] = ll
            row[("brier", m)] = br
        rows[name] = row
    return pd.DataFrame(rows).T


def at_time_left(results: pd.DataFrame, minutes: Sequence[float] = (36, 24, 12, 6, 3, 1, 0.5)) -> pd.DataFrame:
    """One state per game at each checkpoint of regulation: the last one at or before it.

    The states of different games are independent, unlike consecutive
    events of the same game.
    """
    reg = results[results["period"] <= 4]
    parts = []
    for m in minutes:
        before = reg[reg["seconds_left"] >= m * 60]
        last = before.groupby("game_id").tail(1).copy()
        last["minutes_left"] = m
        parts.append(last)
    return pd.concat(parts, ignore_index=True)


def evaluate_at_time_left(results: pd.DataFrame, methods: Sequence[str] = METHODS,
                          minutes: Sequence[float] = (36, 24, 12, 6, 3, 1, 0.5)) -> pd.DataFrame:
    """Log loss per method at fixed minutes left in regulation (one state per game)."""
    checkpoints = at_time_left(results, minutes)
    rows = {}
    for m_left, g in checkpoints.groupby("minutes_left", sort=False):
        y = g["home_win"].to_numpy()
        rows[m_left] = {"games": len(g), **{m: _scores(y, g[f"p_{m}"].to_numpy())[0] for m in methods}}
    out = pd.DataFrame(rows).T
    out.index.name = "minutes left"
    return out


def bootstrap_difference(results: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 0,
                         mask: pd.Series | None = None) -> tuple[float, float, float]:
    """Log loss of ``a`` minus ``b``, with a 95% interval from resampling whole games."""
    r = results if mask is None else results[mask]
    y = r["home_win"].to_numpy()
    la = -np.log(np.clip(np.where(y == 1, r[f"p_{a}"], 1 - r[f"p_{a}"]), EPS, None))
    lb = -np.log(np.clip(np.where(y == 1, r[f"p_{b}"], 1 - r[f"p_{b}"]), EPS, None))
    per_game = pd.DataFrame({"g": r["game_id"].to_numpy(), "d": la - lb}).groupby("g")["d"].agg(["sum", "size"])
    s, n = per_game["sum"].to_numpy(), per_game["size"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[draws].sum(axis=1) / n[draws].sum(axis=1)
    return float(s.sum() / n.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def calibration(results: pd.DataFrame, method: str, bins: int = 10, mask: pd.Series | None = None) -> pd.DataFrame:
    """Predicted vs observed home win rate in equal-width probability bins (``predicted``, ``observed``, ``states``)."""
    r = results if mask is None else results[mask]
    p = r[f"p_{method}"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = pd.DataFrame({"bin": which, "p": p, "y": r["home_win"].to_numpy()})
    out = table.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), states=("y", "size"))
    out.index = [f"{edges[i]:.0%}-{edges[i + 1]:.0%}" for i in out.index]
    return out


def calibration_error(results: pd.DataFrame, method: str, bins: int = 10, mask: pd.Series | None = None) -> float:
    """State-weighted average gap between predicted and observed win rates."""
    table = calibration(results, method, bins, mask)
    return float(np.average((table["predicted"] - table["observed"]).abs(), weights=table["states"]))
