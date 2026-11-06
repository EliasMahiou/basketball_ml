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

The file's point spreads are rougher: before 2022-23 their sign often
disagrees with the moneyline and some hold the over/under instead. Each
spread is signed by the moneyline's favorite and kept only when it is within
:data:`SPREAD_TOLERANCE` points of the margin the moneyline implies (94% of
the test games). :func:`spread_backtest` predicts point margins with the
model, with the market alone (spread and moneyline) and with both, and
:func:`against_the_spread` scores them against the spread.
"""

from __future__ import annotations

import sqlite3
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .model import default_features, walk_forward

ODDS_URL = ("https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/"
            "OddsData.sqlite")
ODDS_FILE = "OddsData.sqlite"
MARGIN_SD = 12.5   # spread of final margins around the expected one, to turn a win probability into a margin
SPREAD_TOLERANCE = 3.0   # points a spread may differ from the moneyline's implied margin and still be used
BREAK_EVEN = 110 / 210   # share of spread bets to win at -110 to break even
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


def implied_margin(p: np.ndarray) -> np.ndarray:
    """Home margin expected from a home win probability, if margins spread normally (sd :data:`MARGIN_SD`)."""
    return MARGIN_SD * norm.ppf(np.clip(np.asarray(p, dtype=float), 0.005, 0.995))


def load_lines(path: str | Path) -> pd.DataFrame:
    """Every closing line in the SQLite file: ``date``, ``home``, ``away``, ``p_market`` and ``line_margin``.

    ``line_margin`` is the home margin the closing spread expects: the
    spread's size, signed by the moneyline's favorite, or NaN when it is more
    than :data:`SPREAD_TOLERANCE` points from the moneyline's implied margin.
    Some seasons have two tables; tables whose dates are not ISO dates (older
    copies) are skipped and duplicate games dropped.
    """
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type = 'table'")]
        parts = [pd.read_sql(f'select * from "{name}"', con) for name in names]
    finally:
        con.close()
    rows = pd.concat([p for p in parts if p["Date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()],
                     ignore_index=True)
    home_ml = pd.to_numeric(rows["ML_Home"], errors="coerce").to_numpy()
    away_ml = pd.to_numeric(rows["ML_Away"], errors="coerce").to_numpy()
    p = no_vig(home_ml, away_ml)
    size = pd.to_numeric(rows["Spread"], errors="coerce").abs() if "Spread" in rows else pd.Series(np.nan, rows.index)
    spread = size.to_numpy() * np.sign(away_ml - home_ml)   # the favorite has the lower moneyline
    lines = pd.DataFrame({
        "date": pd.to_datetime(rows["Date"]), "home": rows["Home"].map(TEAM_NAMES), "away": rows["Away"].map(TEAM_NAMES),
        "p_market": p, "line_margin": np.where(np.abs(spread - implied_margin(p)) <= SPREAD_TOLERANCE, spread, np.nan),
    })
    return lines.dropna(subset=["date", "home", "away", "p_market"]).drop_duplicates(
        ["date", "home", "away"]).reset_index(drop=True)


def match_lines(games: pd.DataFrame, lines: pd.DataFrame, column: str = "p_market") -> pd.Series:
    """Each game's ``column`` of ``lines`` (indexed by ``game_id``), matched on date (one day either way) and teams.

    A line on the game's own date wins over one a day off, even when its
    ``column`` is missing: the same teams can meet on consecutive days.
    """
    found = pd.Series(np.nan, index=games["game_id"].to_numpy())
    matched = pd.Series(False, index=found.index)
    for shift in (0, -1, 1):
        moved = lines.assign(date=lines["date"] + pd.Timedelta(days=shift))
        hit = games[["game_id", "date", "home", "away"]].merge(moved, on=["date", "home", "away"])
        hit = hit.drop_duplicates("game_id").set_index("game_id")[column]
        new = hit.index[~matched[hit.index].to_numpy()]
        found[new], matched[new] = hit[new], True
    return found


def add_market(games: pd.DataFrame, data_dir: str | Path = "data/odds") -> pd.DataFrame:
    """``games`` with the closing line: home win probability ``p_market`` and its log-odds, and ``line_margin``.

    Games without a line get NaN; ``line_margin`` is also NaN where the spread looks wrong (see :func:`load_lines`).
    """
    lines = load_lines(download_odds(data_dir))
    out = games.copy()
    for column in ("p_market", "line_margin"):
        out[column] = match_lines(games, lines, column).reindex(out["game_id"]).to_numpy()
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


def spread_backtest(games: pd.DataFrame, test_seasons: Iterable[int],
                    features: Sequence[str] | None = None) -> pd.DataFrame:
    """Walk-forward point-margin predictions on the games with a usable closing spread.

    Linear regressions of the home margin, each fitted on earlier seasons
    only: ``margin_model`` on the game model's features (all earlier games),
    ``margin_market`` on the spread and the moneyline's implied margin, and
    ``margin_blend`` on both (earlier games with a spread). With the
    player-impact feature ``rapm_diff``, ``margin_impact`` is the market plus
    that feature alone. Returns the test games with a ``line_margin``.
    """
    feats = list(features) if features is not None else default_features(games)
    games = games.assign(moneyline_margin=implied_margin(games["p_market"]))
    market = ["line_margin", "moneyline_margin"]
    sets = {"margin_market": market, "margin_blend": [*feats, *market]}
    if "rapm_diff" in games:
        sets["margin_impact"] = ["rapm_diff", *market]
    out = []
    for season in test_seasons:
        train, test = games[games["season"] < season], games[games["season"] == season]
        lined_train = train.dropna(subset=["line_margin"])
        test = test.dropna(subset=["line_margin"]).copy()
        if test.empty or lined_train.empty:
            continue
        test["margin_model"] = LinearRegression().fit(train[feats], train["margin"]).predict(test[feats])
        for column, cols in sets.items():
            test[column] = LinearRegression().fit(lined_train[cols], lined_train["margin"]).predict(test[cols])
        out.append(test)
    return pd.concat(out, ignore_index=True)


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int = 1000) -> tuple[float, float]:
    """95% interval of the mean of ``values``, resampling them."""
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def against_the_spread(results: pd.DataFrame, column: str = "margin_model", baseline: str = "line_margin",
                       seed: int = 0) -> dict:
    """How ``column``'s predicted margins do against the closing spread.

    ``rmse``; ``gap``, its squared error per game minus ``baseline``'s (the
    spread unless named), with a 95% interval from resampling games; and
    ``ats``, the share of games where the side it prefers against the spread
    won, pushes and exact agreements left out, with its 95% interval
    (:data:`BREAK_EVEN` is what a bet at -110 needs).
    """
    rng = np.random.default_rng(seed)
    y = results["margin"].to_numpy(float)
    line, pred = results["line_margin"].to_numpy(float), results[column].to_numpy(float)
    gap = (y - pred) ** 2 - (y - results[baseline].to_numpy(float)) ** 2
    side, cover = np.sign(pred - line), np.sign(y - line)
    won = (side == cover)[(side != 0) & (cover != 0)].astype(float)
    return {"games": len(results), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))), "gap": float(gap.mean()),
            "gap_interval": _interval(gap, rng), "bets": len(won),
            "ats": float(won.mean()) if len(won) else float("nan"), "ats_interval": _interval(won, rng)}
