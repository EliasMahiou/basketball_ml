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

from .games import FEATURES

METHODS = ("home_court", "elo", "logistic", "boosting")


def make_model(method: str):
    """A fresh, unfitted scikit-learn classifier for ``method``."""
    if method == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    if method == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=100, random_state=0)
    raise ValueError(f"No trainable model for {method!r}")


def walk_forward(games: pd.DataFrame, test_seasons: Iterable[int],
                 methods: Sequence[str] = METHODS, features: Sequence[str] = FEATURES) -> pd.DataFrame:
    """Predict each test season with models trained only on earlier seasons.

    Returns the test games with one ``p_<method>`` column (home win probability) per method.
    """
    feats = list(features)
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
