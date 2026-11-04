"""Closing betting lines, to benchmark the game model against the market.

The lines come from the public SQLite file of
``kyleskom/NBA-Machine-Learning-Sports-Betting`` (Sportsbook Review closing
moneylines, 2007-08 to January 2026), downloaded into ``data/odds/`` on
first use. A moneyline pair becomes the home team's win probability with the
bookmaker's margin removed proportionally. Games are matched on date (one day
either way, for time zones) and teams.

The market is the bar to clear: closing lines know everything public before
tip-off, injuries included. :func:`market_backtest` scores the market, the
model, and the model with the market's opinion as one more feature, which
shows how much the model adds to the market.
"""

from __future__ import annotations

import sqlite3
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .model import default_features, walk_forward

ODDS_URL = ("https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/"
            "OddsData.sqlite")
ODDS_FILE = "OddsData.sqlite"
TEAM_NAMES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "New Jersey Nets": "BKN",
    "Charlotte Bobcats": "CHA", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def download_odds(data_dir: str | Path = "data/odds", overwrite: bool = False) -> Path:
    """Download the betting-line file to ``data_dir`` and return its path."""
    out = Path(data_dir) / ODDS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not out.exists():
        with urllib.request.urlopen(ODDS_URL, timeout=300) as resp:
            out.write_bytes(resp.read())
    return out


def no_vig(home_ml, away_ml) -> np.ndarray:
    """Home win probability from American moneylines, with the bookmaker's margin removed.

    Pairs that don't look like real moneylines (a margin outside -1% to 12%) give NaN.
    """
    home_ml, away_ml = np.asarray(home_ml, dtype=float), np.asarray(away_ml, dtype=float)

    def implied(ml):
        return np.where(ml < 0, -ml, 100.0) / (np.abs(ml) + 100.0)

    home, away = implied(home_ml), implied(away_ml)
    margin = home + away - 1
    ok = (np.abs(home_ml) >= 100) & (np.abs(away_ml) >= 100) & (margin > -0.01) & (margin < 0.12)
    return np.where(ok, home / (home + away), np.nan)


def load_lines(path: str | Path) -> pd.DataFrame:
    """Every closing moneyline in the SQLite file: ``date``, ``home``, ``away``, ``p_market``.

    Some seasons have two tables; tables whose dates are not ISO dates (older
    copies) are skipped and duplicate games dropped.
    """
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type = 'table'")]
        parts = [pd.read_sql(f'select Date, Home, Away, ML_Home, ML_Away from "{name}"', con) for name in names]
    finally:
        con.close()
    rows = pd.concat([p for p in parts if p["Date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()],
                     ignore_index=True)
    lines = pd.DataFrame({
        "date": pd.to_datetime(rows["Date"]), "home": rows["Home"].map(TEAM_NAMES), "away": rows["Away"].map(TEAM_NAMES),
        "p_market": no_vig(pd.to_numeric(rows["ML_Home"], errors="coerce"),
                           pd.to_numeric(rows["ML_Away"], errors="coerce")),
    })
    return lines.dropna().drop_duplicates(["date", "home", "away"]).reset_index(drop=True)


def match_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    """Each game's ``p_market`` (indexed by ``game_id``), matched on date (one day either way) and teams."""
    found = pd.Series(np.nan, index=games["game_id"].to_numpy())
    for shift in (0, -1, 1):
        moved = lines.assign(date=lines["date"] + pd.Timedelta(days=shift))
        hit = games[["game_id", "date", "home", "away"]].merge(moved, on=["date", "home", "away"])
        hit = hit.drop_duplicates("game_id").set_index("game_id")["p_market"]
        found = found.fillna(hit.reindex(found.index))
    return found


def add_market(games: pd.DataFrame, data_dir: str | Path = "data/odds") -> pd.DataFrame:
    """``games`` with the closing line's home win probability ``p_market`` (NaN without a line) and its log-odds."""
    p = match_lines(games, load_lines(download_odds(data_dir)))
    out = games.copy()
    out["p_market"] = p.reindex(out["game_id"]).to_numpy()
    q = out["p_market"].clip(1e-4, 1 - 1e-4)
    out["market_logit"] = np.log(q / (1 - q))
    return out


def market_backtest(games: pd.DataFrame, test_seasons: Iterable[int],
                    features: Sequence[str] | None = None) -> pd.DataFrame:
    """Walk-forward predictions on the games with a closing line.

    Returns those games with ``p_market`` (the line), ``p_logistic`` (the
    model on ``features``) and ``p_blend`` (the same model given the line's
    log-odds as one more feature), all fitted on earlier seasons only.
    """
    feats = list(features) if features is not None else default_features(games)
    lined = games.dropna(subset=["p_market"])
    model = walk_forward(lined, test_seasons, methods=("logistic",), features=feats)
    blend = walk_forward(lined, test_seasons, methods=("logistic",), features=[*feats, "market_logit"])
    return model.assign(p_blend=blend["p_logistic"].to_numpy())
