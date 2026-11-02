"""Learn compact style embeddings and measure how well they identify players."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class StyleAutoencoder:
    """A small autoencoder: style features -> ``dim`` numbers -> style features.

    The bottleneck layer is the embedding: ``dim`` numbers that must hold
    enough of a player's style to rebuild all his features.
    """

    def __init__(self, dim: int = 8, hidden: int = 24, seed: int = 0, max_iter: int = 2000):
        self.dim = dim
        self.net = MLPRegressor(hidden_layer_sizes=(hidden, dim, hidden), activation="tanh",
                                alpha=1e-3, learning_rate_init=1e-3, max_iter=max_iter,
                                early_stopping=True, n_iter_no_change=50, random_state=seed)

    def fit(self, X: np.ndarray) -> "StyleAutoencoder":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net.fit(X, X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=float)
        for W, b in zip(self.net.coefs_[:2], self.net.intercepts_[:2]):
            h = np.tanh(h @ W + b)
        return h

    def reconstruction_r2(self, X: np.ndarray) -> float:
        """Share of the features' variance the embedding can rebuild."""
        X = np.asarray(X, dtype=float)
        err = ((self.net.predict(X) - X) ** 2).sum()
        return float(1 - err / ((X - X.mean(axis=0)) ** 2).sum())


def embed(table: pd.DataFrame, columns: Sequence[str], method: str = "autoencoder",
          dim: int = 8, train_mask: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Embed every row of ``table``; the model is fit on ``train_mask`` rows only."""
    X = table[list(columns)].to_numpy(dtype=float)
    fit_X = X if train_mask is None else X[train_mask]
    if method == "autoencoder":
        model = StyleAutoencoder(dim=dim, seed=seed).fit(fit_X)
    elif method == "pca":
        model = PCA(n_components=dim, random_state=seed).fit(fit_X)
    elif method == "raw":
        return X
    else:
        raise ValueError(method)
    return model.transform(X)


def identification(table: pd.DataFrame, vectors: np.ndarray, seasons: Sequence[int]) -> pd.DataFrame:
    """Can a player's season-t vector find his own season t+1 among everyone's t+1?

    For each player who qualifies in both seasons, ranks all season t+1
    vectors by distance to his season t vector. Returns one row per player
    with the rank of his own next season (1 = found first).
    """
    rows = []
    ids = table["player_id"].to_numpy()
    season_col = table["season"].to_numpy()
    for s in seasons:
        now = np.where(season_col == s)[0]
        nxt = np.where(season_col == s + 1)[0]
        nxt_ids = ids[nxt]
        for i in now:
            if ids[i] not in set(nxt_ids):
                continue
            d = np.linalg.norm(vectors[nxt] - vectors[i], axis=1)
            own = d[nxt_ids == ids[i]][0]
            rows.append({"player_id": ids[i], "season": s, "rank": int((d < own).sum()) + 1,
                         "candidates": len(nxt)})
    return pd.DataFrame(rows)


def summarize(ranks: pd.DataFrame) -> dict[str, float]:
    return {"top1": float((ranks["rank"] == 1).mean()), "top5": float((ranks["rank"] <= 5).mean()),
            "median_rank": float(ranks["rank"].median()), "players": len(ranks)}
