"""What tracking cameras add to shot quality, from the 2014-15 shot logs.

Cameras in every arena turn game video into player tracking: each player's
and the ball's position 25 times a second (SportVU from 2013-14, then Second
Spectrum and Hawk-Eye). The positions themselves aren't public, but for
2014-15 the NBA's stats site published shot logs with what tracking adds to
each shot: the closest defender and his distance when the shot left the
hand, the shot clock, and how many dribbles and seconds the shooter had
the ball. The file (128,069 shots, October 2014 to early March 2015) is
mirrored in many public GitHub repositories.

The logs are matched to the play-by-play shot charts by game, shooter and
time: the tracked release comes 0 to 8 seconds before the time the scorer
wrote down. The xFG model is then fit with and without the tracking
features, by 5-fold cross-validation over whole games of that season.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from ..shots.features import LOCATION_FEATURES
from ..shots.model import ShotModel

SEASON = 2015   # 2014-15, the only season with public shot logs
SHOT_LOG_URLS = (
    "https://raw.githubusercontent.com/SergioLlana/nba-shot-analysis/master/data/shot_logs.csv",
    "https://raw.githubusercontent.com/BugraAKKUS/NBA-Shot-Logs/master/shot_logs.csv",
    "https://raw.githubusercontent.com/alberthtan/NBA-Shot-Logs/master/shot_logs.csv",
)
SHOT_LOG_MD5 = "3a709ad0568bc66b32e9b1556a95b891"
SHOT_LOG_FILE = "shot_logs_2014_15.csv"
MATCH_SECONDS = 8   # the tracked release is at most this many seconds before the scorer's clock
TRACKING_FEATURES = ["def_dist", "shot_clock", "dribbles", "touch_time"]
# Shot type (ACTION_TYPE) gives away part of the result before 2017-18 (see
# basketball_ml.shots.features), so the main comparison builds on the coarse shot kind.
FEATURE_SETS = {
    "location": LOCATION_FEATURES,
    "kind": [*LOCATION_FEATURES, "kind"],
    "kind + defender": [*LOCATION_FEATURES, "kind", "def_dist"],
    "kind + tracking": [*LOCATION_FEATURES, "kind", *TRACKING_FEATURES],
    "type": [*LOCATION_FEATURES, "action"],
    "type + tracking": [*LOCATION_FEATURES, "action", *TRACKING_FEATURES],
}
NAMES = {
    "location": "Location + game situation",
    "kind": "  + shot kind (dunk, layup, hook, tip, jumper)",
    "kind + defender": "  + shot kind + closest defender's distance",
    "kind + tracking": "  + shot kind + all tracking (defender, shot clock, dribbles, touch time)",
    "type": "  + shot type (the xFG model)",
    "type + tracking": "  + shot type + all tracking",
}
DISTANCE_BINS = (0, 2, 4, 6, np.inf)
DISTANCE_LABELS = ("0-2 ft", "2-4 ft", "4-6 ft", "6+ ft")


def download_shot_logs(data_dir: str | Path = "data/tracking", overwrite: bool = False) -> Path:
    """Download the 2014-15 shot logs to ``data_dir`` (from the first mirror with the expected file)."""
    out = Path(data_dir) / SHOT_LOG_FILE
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in SHOT_LOG_URLS:
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                body = resp.read()
        except OSError as e:
            errors.append(f"{url}: {e}")
            continue
        if hashlib.md5(body).hexdigest() != SHOT_LOG_MD5:
            errors.append(f"{url}: unexpected content")
            continue
        out.write_bytes(body)
        return out
    raise RuntimeError("Could not download the shot logs:\n" + "\n".join(errors))


def load_shot_logs(path: str | Path) -> pd.DataFrame:
    """One row per logged shot: ``game_id``, ``player_id``, ``period``, ``seconds_left``, the result and tracking.

    Tracking columns: ``def_dist`` (feet between the shooter and the closest
    defender at the release), ``defender_id``, ``shot_clock`` (empty when it
    was off), ``dribbles``, ``touch_time`` (seconds he had the ball; a few
    negative values become 0) and ``tracked_distance`` (feet from the basket).
    """
    raw = pd.read_csv(path)
    clock = raw["GAME_CLOCK"].str.split(":", expand=True).astype(int)
    return pd.DataFrame({
        "game_id": raw["GAME_ID"].astype(np.int64),
        "player_id": raw["player_id"].astype(np.int64),
        "period": raw["PERIOD"].astype(np.int64),
        "seconds_left": (60 * clock[0] + clock[1]).astype(np.int64),
        "made": raw["FGM"].astype(np.int8),
        "value": raw["PTS_TYPE"].astype(np.int8),
        "def_dist": raw["CLOSE_DEF_DIST"].astype(float),
        "defender_id": raw["CLOSEST_DEFENDER_PLAYER_ID"].astype(np.int64),
        "defender": raw["CLOSEST_DEFENDER"].astype(str),
        "shot_clock": raw["SHOT_CLOCK"].astype(float),
        "dribbles": raw["DRIBBLES"].astype(float),
        "touch_time": raw["TOUCH_TIME"].astype(float).clip(lower=0),
        "tracked_distance": raw["SHOT_DIST"].astype(float),
    })


def elapsed(period, seconds_left) -> np.ndarray:
    """Seconds since tip-off (12-minute quarters, 5-minute overtimes)."""
    period, left = np.asarray(period), np.asarray(seconds_left)
    start = np.where(period <= 4, (period - 1) * 720, 2880 + (period - 5) * 300)
    return start + np.where(period <= 4, 720, 300) - left


def match_shots(shots: pd.DataFrame, logs: pd.DataFrame, tolerance: int = MATCH_SECONDS) -> pd.DataFrame:
    """``shots`` (with the unclipped ``raw_period``) that have a logged shot, with its tracking columns.

    Each logged shot goes to the shooter's first charted shot at most
    ``tolerance`` seconds later; a charted shot claimed twice keeps the closest
    one. Pairs whose result or shot value disagree are dropped.
    """
    left = logs.assign(t=elapsed(logs["period"], logs["seconds_left"]).astype(np.int64)).sort_values("t")
    right = shots.assign(t=elapsed(shots["raw_period"], shots["seconds_left"]).astype(np.int64),
                         row=np.arange(len(shots))).sort_values("t")
    pairs = pd.merge_asof(left[["t", "game_id", "player_id", "made", "value", *TRACKING_FEATURES, "defender_id",
                                "defender", "tracked_distance"]],
                          right[["t", "game_id", "player_id", "row"]].rename(columns={"t": "t_chart"}),
                          left_on="t", right_on="t_chart", by=["game_id", "player_id"], direction="forward",
                          tolerance=tolerance)
    pairs = pairs.dropna(subset=["row"]).astype({"row": np.int64})
    pairs["gap"] = pairs["t_chart"] - pairs["t"]
    pairs = pairs.sort_values(["gap", "t"]).drop_duplicates("row")
    matched = shots.iloc[pairs["row"].to_numpy()].reset_index(drop=True)
    same = (matched["made"].to_numpy() == pairs["made"].to_numpy()) & \
           (matched["value"].to_numpy() == pairs["value"].to_numpy())
    extra = pairs[[*TRACKING_FEATURES, "defender_id", "defender", "tracked_distance", "gap"]].reset_index(drop=True)
    out = pd.concat([matched, extra], axis=1)[same]
    return out.sort_values(["game_id", "game_event_id"]).reset_index(drop=True)


def cross_validate(shots: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]] = FEATURE_SETS,
                   folds: int = 5) -> pd.DataFrame:
    """Out-of-fold xFG of every shot for each feature set, folds made of whole games."""
    shots = shots.reset_index(drop=True)
    splits = list(GroupKFold(folds).split(shots, groups=shots["game_id"]))
    out = pd.DataFrame(index=shots.index)
    for name, features in feature_sets.items():
        p = np.zeros(len(shots))
        for train, test in splits:
            p[test] = ShotModel(features=features).fit(shots.iloc[train]).predict(shots.iloc[test])
        out[name] = p
    return out


def _losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(shots: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Log loss, Brier score and AUC of each feature set's out-of-fold predictions."""
    y = shots["made"].to_numpy()
    return pd.DataFrame({name: {"log_loss": log_loss(y, oof[name]), "brier": brier_score_loss(y, oof[name]),
                                "auc": roc_auc_score(y, oof[name])} for name in oof}).T


def paired_difference(shots: pd.DataFrame, oof: pd.DataFrame, a: str, b: str, n_boot: int = 1000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Log loss of ``a`` minus ``b``, with a 95% interval from resampling whole games."""
    y = shots["made"].to_numpy()
    diff = pd.DataFrame({"g": shots["game_id"].to_numpy(),
                         "d": _losses(y, oof[a].to_numpy()) - _losses(y, oof[b].to_numpy())})
    per_game = diff.groupby("g")["d"].agg(["sum", "size"])
    s, n = per_game["sum"].to_numpy(), per_game["size"].to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boot = s[draws].sum(axis=1) / n[draws].sum(axis=1)
    return float(s.sum() / n.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def by_defender_distance(shots: pd.DataFrame, expected: np.ndarray) -> pd.DataFrame:
    """FG% by the closest defender's distance, for twos and threes, next to ``expected`` (xFG without tracking)."""
    bins = pd.cut(shots["def_dist"], DISTANCE_BINS, right=False, labels=DISTANCE_LABELS)
    table = shots.assign(bin=bins, xfg=expected, shot=np.where(shots["value"] == 3, "three", "two"))
    out = table.groupby(["shot", "bin"], observed=True).agg(shots=("made", "size"), fg=("made", "mean"),
                                                            xfg=("xfg", "mean"))
    out["above"] = out["fg"] - out["xfg"]
    return out


def making_change(shots: pd.DataFrame, before: np.ndarray, after: np.ndarray, min_shots: int = 250) -> pd.DataFrame:
    """Each player's FG% above expected without (``before``) and with (``after``) tracking, and how open he was."""
    table = shots.assign(before=shots["made"] - before, after=shots["made"] - after)
    out = table.groupby("player_id").agg(player=("player", "first"), shots=("made", "size"), fg=("made", "mean"),
                                         before=("before", "mean"), after=("after", "mean"),
                                         defender=("def_dist", "mean"))
    out["change"] = out["after"] - out["before"]
    return out[out["shots"] >= min_shots].sort_values("after", ascending=False)


def split_half(shots: pd.DataFrame, values: np.ndarray, key: str = "player_id", min_shots: int = 125) -> float:
    """Correlation between odd and even games of the mean of ``values`` per ``key``: how stable it is."""
    table = shots.assign(v=values, half=shots["game_id"] % 2)
    halves = table.groupby([key, "half"])["v"].agg(["mean", "size"]).unstack()
    halves = halves[(halves["size"] >= min_shots).all(axis=1)]
    return float(np.corrcoef(halves["mean"][0], halves["mean"][1])[0, 1])


def contests(shots: pd.DataFrame, expected: np.ndarray, within: float = 6.0, min_shots: int = 300) -> pd.DataFrame:
    """Defenders by how far below expected the shots they contest go in.

    A contest is a shot whose closest defender is within ``within`` feet;
    ``expected`` is xFG without the tracking features, so a defender gets
    credit both for getting close and for what he does once there. The
    closest defender isn't always the one responsible for the shot.
    """
    near = shots["def_dist"].to_numpy() < within
    table = shots[near].assign(above=shots["made"].to_numpy()[near] - expected[near])
    out = table.groupby("defender_id").agg(defender=("defender", "first"), shots=("made", "size"),
                                           fg=("made", "mean"), above=("above", "mean"),
                                           distance=("def_dist", "mean"))
    out["defender"] = out["defender"].map(_first_last)
    return out[out["shots"] >= min_shots].sort_values("above")


def _first_last(name: str) -> str:
    """"Gobert, Rudy" -> "Rudy Gobert"."""
    last, _, first = str(name).partition(", ")
    return f"{first} {last}".strip() if first else last
