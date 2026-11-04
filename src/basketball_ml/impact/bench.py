"""Score preseason ratings on the Hoop Impact Bench game-margin test.

Hoop Impact Bench (https://github.com/lumahoops/hoop-impact-bench, data under
CC BY 4.0) scores public impact metrics the same way: each player's rating
from before the season, weighted by the minutes he actually played, gives
each team's strength in every game, and the score is the share of game
margins explained (squared correlation). This re-implements that test on the
benchmark's own game and minutes files, which are downloaded on first use.

Two rules for players a rating file doesn't cover: under the "established"
rule every player with fewer than 250 minutes the season before (rookies
included) counts as -2.0; under the "full forecast" rule only players
missing from the file do.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_URL = "https://raw.githubusercontent.com/lumahoops/hoop-impact-bench/main/data"
SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
REPLACEMENT = -2.0
ESTABLISHED_MINUTES = 250.0


def download_bench(data_dir: str | Path = "data/bench", overwrite: bool = False) -> Path:
    """Download the benchmark's games and minutes tables to ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("games.csv", "minutes.csv"):
        out = data_dir / name
        if overwrite or not out.exists():
            with urllib.request.urlopen(f"{BENCH_URL}/{name}", timeout=300) as resp:
                out.write_bytes(resp.read())
    return data_dir


def _previous(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def score_ratings(ratings: pd.DataFrame, data_dir: str | Path = "data/bench",
                  seasons: tuple[str, ...] = SEASONS) -> pd.DataFrame:
    """Game-margin R2 of ``ratings`` (``season`` like "2024-25", ``nba_id``, ``rating``), overall and by season.

    Returns one row per rule ("established", "full forecast").
    """
    data_dir = download_bench(data_dir)
    games = pd.read_csv(data_dir / "games.csv", dtype={"game_id": str})
    minutes = pd.read_csv(data_dir / "minutes.csv", dtype={"game_id": str})
    games = games[games["season"].isin(seasons)].reset_index(drop=True)
    margin = (games["home_pts"] - games["away_pts"]).to_numpy(float)
    played = minutes[minutes["game_id"].isin(games["game_id"])].reset_index(drop=True)
    game = played["game_id"].map(pd.Series(games.index, index=games["game_id"])).to_numpy()
    side = np.where(played["team"].to_numpy() == games["home"].to_numpy()[game], 1.0, -1.0)
    weight = side * 5 * played["min"] / played.groupby(["game_id", "team"])["min"].transform("sum")
    by_season = minutes.groupby(["season", "nba_id"])["min"].sum()
    last = by_season.reindex(pd.MultiIndex.from_arrays([played["season"].map(_previous), played["nba_id"]]))
    last = last.fillna(0.0).to_numpy()
    raw = ratings.set_index(["season", "nba_id"])["rating"].reindex(
        pd.MultiIndex.from_arrays([played["season"], played["nba_id"]])).to_numpy()
    rows = {}
    for rule, minimum in (("established", ESTABLISHED_MINUTES), ("full forecast", 0.0)):
        value = np.where((last >= minimum) & ~np.isnan(raw), raw, REPLACEMENT)
        predicted = np.bincount(game, weights=weight * value, minlength=len(games))
        row = {"all": np.corrcoef(predicted, margin)[0, 1] ** 2}
        for season in seasons:
            m = (games["season"] == season).to_numpy()
            row[season] = np.corrcoef(predicted[m], margin[m])[0, 1] ** 2
        rows[rule] = row
    return pd.DataFrame(rows).T
