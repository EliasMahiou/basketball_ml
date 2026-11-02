"""Win-probability models and their walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .games import FEATURES, LINEUP_FEATURES

METHODS = ("home_court", "elo", "logistic", "boosting")


def make_model(method: str):
    """A fresh, unfitted scikit-learn classifier for ``method``."""
    if method == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    if method == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=100, random_state=0)
    raise ValueError(f"No trainable model for {method!r}")


def default_features(games: pd.DataFrame) -> list[str]:
    """Lineup-aware features when ``strength_diff`` is available, else the base set."""
    return list(LINEUP_FEATURES) if "strength_diff" in games else list(FEATURES)


def walk_forward(games: pd.DataFrame, test_seasons: Iterable[int],
                 methods: Sequence[str] = METHODS, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Predict each test season with models trained only on earlier seasons.

    Returns the test games with one ``p_<method>`` column (home win probability) per method.
    """
    feats = list(features) if features is not None else default_features(games)
    out = []
    for season in test_seasons:
        train = games[games["season"] < season]
        test = games[games["season"] == season].copy()
        if train.empty or test.empty:
            continue
        for m in methods:
            if m == "home_court":
                test[f"p_{m}"] = train["home_win"].mean()
            elif m == "elo":
                test[f"p_{m}"] = test["elo_prob"]
            else:
                model = make_model(m).fit(train[feats], train["home_win"])
                test[f"p_{m}"] = model.predict_proba(test[feats])[:, 1]
        out.append(test)
    return pd.concat(out, ignore_index=True)


def evaluate(results: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Accuracy, log loss and Brier score per method (lower loss is better)."""
    y = results["home_win"].to_numpy()
    rows = {}
    for m in methods:
        p = np.clip(results[f"p_{m}"].to_numpy(), 1e-6, 1 - 1e-6)
        rows[m] = {"accuracy": float(((p > 0.5) == y).mean()),
                   "log_loss": float(log_loss(y, p, labels=[0, 1])),
                   "brier": float(brier_score_loss(y, p))}
    return pd.DataFrame(rows).T


def calibration(results: pd.DataFrame, method: str = "logistic", bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed home win rate, in equal-width probability bins.

    A calibrated model's games predicted at 70% are won about 70% of the time.
    Returns one row per non-empty bin: ``predicted`` (mean probability),
    ``observed`` (share of home wins) and ``games``.
    """
    p = results[f"p_{method}"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = pd.DataFrame({"bin": which, "p": p, "y": results["home_win"].to_numpy()})
    out = table.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), games=("y", "size"))
    out.index = [f"{edges[i]:.0%}-{edges[i + 1]:.0%}" for i in out.index]
    return out


def calibration_error(results: pd.DataFrame, method: str = "logistic", bins: int = 10) -> float:
    """Expected calibration error: game-weighted average gap between predicted and observed."""
    table = calibration(results, method, bins)
    return float(np.average((table["predicted"] - table["observed"]).abs(), weights=table["games"]))
