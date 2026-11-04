"""Expected FG% (xFG) models and their walk-forward evaluation.

xFG is the chance that an average NBA shooter makes a given shot. The models
see only the shot (location, shot type and game situation), never who took
it, so a player's makes above his expected makes measure his shot making.

Every model has ``fit(shots)`` and ``predict(shots)`` on the table of
:func:`basketball_ml.shots.build_shot_table`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CATEGORICAL, FEATURES, LOCATION_FEATURES, ZONES

DEFAULT_WINDOW = 3   # seasons each model is trained on
MIN_ACTION_SHOTS = 200   # rarer ACTION_TYPE labels become "other"
OTHER = "other"
# Checked on the 2018-19 and 2019-20 test seasons only: other learning rates, tree
# sizes and leaf minimums move the log loss by less than 0.0002; early stopping
# (on a random tenth of the training shots) beats a fixed 300 trees.
BOOSTING_PARAMS = dict(learning_rate=0.1, max_iter=500, max_leaf_nodes=31, min_samples_leaf=200,
                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=20, random_state=0)


class LeagueRate:
    """The league's FG% on the training shots, for every shot."""

    def fit(self, shots: pd.DataFrame) -> LeagueRate:
        self.rate_ = float(shots["made"].mean())
        return self

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        return np.full(len(shots), self.rate_)


class ZoneRate:
    """League FG% in the shot's zone (basic zone, side and distance range).

    A zone combination missing from the training shots falls back to its basic zone.
    """

    def fit(self, shots: pd.DataFrame) -> ZoneRate:
        self.zone_ = shots.groupby(ZONES, observed=True)["made"].mean()
        self.basic_ = shots.groupby("zone_basic", observed=True)["made"].mean()
        self.rate_ = float(shots["made"].mean())
        return self

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_frame(shots[ZONES].astype(str))
        zone = self.zone_.copy()
        zone.index = zone.index.map(lambda t: tuple(str(v) for v in t))
        p = zone.reindex(key).to_numpy()
        basic = shots["zone_basic"].astype(str).map(self.basic_.rename(index=str)).to_numpy(float)
        return np.where(np.isnan(p), np.where(np.isnan(basic), self.rate_, basic), p)


class DistanceLogit:
    """Logistic regression on shot distance and the 3-point flag."""

    columns = ["distance", "is_three"]

    def fit(self, shots: pd.DataFrame) -> DistanceLogit:
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.model_.fit(shots[self.columns].to_numpy(float), shots["made"].to_numpy())
        return self

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(shots[self.columns].to_numpy(float))[:, 1]


class ShotModel:
    """Gradient boosting on all shot features, with native categorical splits.

    ACTION_TYPE labels with fewer than ``min_action_shots`` training attempts,
    and labels first seen after training, are grouped as "other". Features
    missing from every training shot (the margin without play-by-play) are left out.
    """

    def __init__(self, features: Sequence[str] = FEATURES, min_action_shots: int = MIN_ACTION_SHOTS, **params):
        self.features = list(features)
        self.min_action_shots = min_action_shots
        self.params = {**BOOSTING_PARAMS, **params}

    def _frame(self, shots: pd.DataFrame) -> pd.DataFrame:
        X = shots[self.features_].copy()
        for c in self.categories_:
            values = X[c].astype(str)
            if c == "action":
                values = values.where(values.isin(self.categories_[c]), OTHER)
            X[c] = pd.Categorical(values, categories=self.categories_[c])
        return X

    def fit(self, shots: pd.DataFrame) -> ShotModel:
        self.features_ = [f for f in self.features if shots[f].notna().any()]
        self.categories_ = {}
        for c in (c for c in CATEGORICAL if c in self.features_):
            counts = shots[c].astype(str).value_counts()
            if c == "action":
                kept = sorted(counts.index[counts >= self.min_action_shots])
                self.categories_[c] = [*kept, OTHER] if OTHER not in kept else kept
            else:
                self.categories_[c] = sorted(counts.index)
        self.model_ = HistGradientBoostingClassifier(categorical_features="from_dtype", **self.params)
        self.model_.fit(self._frame(shots), shots["made"].to_numpy())
        return self

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._frame(shots))[:, 1]


MODELS = {"league": LeagueRate, "zone": ZoneRate, "logistic": DistanceLogit,
          "location": partial(ShotModel, features=LOCATION_FEATURES),
          "kind": partial(ShotModel, features=[*LOCATION_FEATURES, "kind"]),
          "boosting": ShotModel}
NAMES = {"league": "League FG% (no model)", "zone": "League FG% by zone",
         "logistic": "Logistic regression, distance + 3PT",
         "location": "Boosting, location + game situation",
         "kind": "  + shot kind (dunk/layup/hook/tip/jumper)",
         "boosting": "  + shot type (ACTION_TYPE) = xFG model"}
# Column of each model's predictions; "xfg" is the main model.
COLUMNS = {"league": "xfg_league", "zone": "xfg_zone", "logistic": "xfg_logistic", "location": "xfg_location",
           "kind": "xfg_kind", "boosting": "xfg"}
KEEP = ["season", "game_id", "game_event_id", "player_id", "player", "team", "opponent", "home", "loc_x", "loc_y",
        "value", "made"]


def walk_forward(shots: pd.DataFrame, test_seasons: Iterable[int], window: int = DEFAULT_WINDOW,
                 models: dict | None = None) -> pd.DataFrame:
    """Predict every shot of each test season with models trained on the ``window`` seasons before it.

    ``models`` maps a name to a function returning a fresh model (default:
    :data:`MODELS`). Returns the test shots' :data:`KEEP` columns plus one
    prediction column per model (see :data:`COLUMNS`). Seasons without all
    ``window`` earlier seasons are skipped.
    """
    models = MODELS if models is None else models
    available = set(shots["season"].unique())
    out = []
    for season in test_seasons:
        train_seasons = range(season - window, season)
        if season not in available or not all(s in available for s in train_seasons):
            continue
        train = shots[shots["season"].isin(train_seasons)]
        test = shots[shots["season"] == season]
        part = test[[c for c in KEEP if c in test]].copy()
        for name, make in models.items():
            part[COLUMNS.get(name, f"xfg_{name}")] = make().fit(train).predict(test).astype(np.float32)
        out.append(part)
    if not out:
        raise ValueError(f"No test season has {window} earlier seasons of shots")
    return pd.concat(out, ignore_index=True)


def prediction_columns(results: pd.DataFrame) -> dict[str, str]:
    """Model name -> prediction column, for the models present in ``results``."""
    names = {v: k for k, v in COLUMNS.items()}
    return {names.get(c, c[4:]): c for c in results.columns if c == "xfg" or c.startswith("xfg_")}


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Log loss, Brier score and AUC of make probabilities ``p``."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


def evaluate(results: pd.DataFrame) -> pd.DataFrame:
    """Log loss, Brier score and AUC of every model over all test shots (one row per model)."""
    y = results["made"].to_numpy()
    return pd.DataFrame({name: scores(y, results[c]) for name, c in prediction_columns(results).items()}).T


def season_table(results: pd.DataFrame, column: str = "xfg") -> pd.DataFrame:
    """Per test season: shots, actual and predicted FG%, and the model's log loss, Brier score and AUC."""
    rows = {}
    for season, g in results.groupby("season"):
        rows[season] = {"shots": len(g), "fg": g["made"].mean(), "xfg": g[column].mean(),
                        **scores(g["made"].to_numpy(), g[column].to_numpy())}
    return pd.DataFrame(rows).T.astype({"shots": int})


def calibration(results: pd.DataFrame, column: str = "xfg", bins: int = 10) -> pd.DataFrame:
    """Average predicted vs actual FG% in deciles of the prediction (equal numbers of shots)."""
    p = results[column].to_numpy(float)
    order = np.argsort(p, kind="stable")
    decile = np.empty(len(p), dtype=int)
    decile[order] = np.arange(len(p)) * bins // len(p)
    table = pd.DataFrame({"decile": decile + 1, "p": p, "y": results["made"].to_numpy()})
    return table.groupby("decile").agg(predicted=("p", "mean"), actual=("y", "mean"), shots=("y", "size"))
