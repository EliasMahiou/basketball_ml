"""Game results and pre-game features built from play-by-play data."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp

# Elo settings in the style of FiveThirtyEight's NBA model. Its 100-point home
# edge is too large since home teams stopped winning ~59% of games (~55% since
# 2019-20); 60 points was best on the 2017-18 to 2020-21 backtest seasons.
ELO_START = 1505.0
ELO_K = 20.0
ELO_HOME = 60.0
ELO_CARRYOVER = 0.75
ROLLING_GAMES = 10
MAX_REST = 4

FEATURES = ["elo_diff", "form_diff", "season_diff", "rest_diff", "b2b_home", "b2b_away"]
LINEUP_FEATURES = [*FEATURES, "strength_diff"]

TEAM_ALIASES = {"NOH": "NOP", "NJN": "BKN", "CHH": "CHA", "SEA": "OKC", "VAN": "MEM"}


def build_games(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """One row per regular-season game: date, teams and final score.

    The final score is the highest score reached in the play-by-play
    (``SCORE`` reads "visitor - home"; event numbers aren't always in game
    order, so the last-numbered event can be an earlier score). Date and
    teams come from the shot charts.
    """
    frames = []
    for season in seasons:
        events = pbp.load_pbp(season, data_dir)
        scored = events.dropna(subset=["SCORE"])
        pts = scored["SCORE"].str.split(" - ", expand=True).astype(int)
        final = (pd.DataFrame({"GAME_ID": scored["GAME_ID"], "away_pts": pts[0], "home_pts": pts[1]})
                 .groupby("GAME_ID", as_index=False).max())

        shots = pbp.load_shots(season, data_dir)
        info = shots.groupby("GAME_ID").agg(date=("GAME_DATE", "first"), home=("HTM", "first"),
                                            away=("VTM", "first")).reset_index()
        games = info.merge(final, on="GAME_ID")
        games[["home", "away"]] = games[["home", "away"]].replace(TEAM_ALIASES)
        games["date"] = pd.to_datetime(games["date"].astype(str), format="%Y%m%d")
        games["season"] = season
        frames.append(games)
    games = pd.concat(frames, ignore_index=True).rename(columns={"GAME_ID": "game_id"})
    games["margin"] = games["home_pts"] - games["away_pts"]
    games["home_win"] = (games["margin"] > 0).astype(int)
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def elo_win_prob(elo_diff: np.ndarray | float) -> np.ndarray | float:
    """Home win probability from (home Elo + home advantage - away Elo)."""
    return 1.0 / (1.0 + 10 ** (-np.asarray(elo_diff, dtype=float) / 400.0))


class LeagueState:
    """Running team ratings, updated game by game in date order."""

    def __init__(self) -> None:
        self.elo: dict[str, float] = defaultdict(lambda: ELO_START)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_GAMES))
        self.season_sum: dict[str, float] = defaultdict(float)
        self.season_n: dict[str, int] = defaultdict(int)
        self.last_date: dict[str, pd.Timestamp] = {}
        self.season: int | None = None

    def start_season(self, season: int) -> None:
        """Regress Elo toward the mean and reset the season's running stats."""
        for team in list(self.elo):
            self.elo[team] = ELO_CARRYOVER * self.elo[team] + (1 - ELO_CARRYOVER) * ELO_START
        self.recent.clear()
        self.season_sum.clear()
        self.season_n.clear()
        self.last_date.clear()
        self.season = season

    def features(self, home: str, away: str, date: pd.Timestamp | None = None) -> dict[str, float]:
        """Pre-game features for ``home`` vs ``away`` from the current state."""
        diff = self.elo[home] + ELO_HOME - self.elo[away]
        form = [float(np.mean(self.recent[t])) if self.recent[t] else 0.0 for t in (home, away)]
        avg = [self.season_sum[t] / self.season_n[t] if self.season_n[t] else 0.0 for t in (home, away)]
        out = {"elo_diff": diff, "elo_prob": float(elo_win_prob(diff)),
               "form_diff": form[0] - form[1], "season_diff": avg[0] - avg[1],
               "games_played": min(self.season_n[home], self.season_n[away])}
        for side, team in (("home", home), ("away", away)):
            if date is not None and team in self.last_date:
                rest = (date - self.last_date[team]).days
            else:
                rest = MAX_REST
            out[f"rest_{side}"] = min(rest, MAX_REST)
            out[f"b2b_{side}"] = int(rest == 1)
        out["rest_diff"] = out["rest_home"] - out["rest_away"]
        return out

    def update(self, home: str, away: str, home_pts: int, away_pts: int, date: pd.Timestamp) -> None:
        """Apply a final score (Elo with margin-of-victory multiplier)."""
        diff = self.elo[home] + ELO_HOME - self.elo[away]
        margin = home_pts - away_pts
        mov_mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * (diff if margin > 0 else -diff))
        shift = ELO_K * mov_mult * ((1.0 if margin > 0 else 0.0) - float(elo_win_prob(diff)))
        self.elo[home] += shift
        self.elo[away] -= shift
        for team, m in ((home, margin), (away, -margin)):
            self.recent[team].append(m)
            self.season_sum[team] += m
            self.season_n[team] += 1
            self.last_date[team] = date


def add_features(games: pd.DataFrame) -> tuple[pd.DataFrame, LeagueState]:
    """Add pre-game features computed only from earlier games.

    - ``elo_diff``: home Elo + home advantage - away Elo (``elo_prob`` is its win probability)
    - ``form_diff``: difference of average margin over each team's last 10 games
    - ``season_diff``: difference of season-to-date average margin
    - ``rest_diff``: home minus away days of rest (each capped at 4)
    - ``b2b_home`` / ``b2b_away``: second night of a back-to-back

    Also returns the league state after the last game, for predicting new games.
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    state = LeagueState()
    rows = []
    for g in games.itertuples(index=False):
        if g.season != state.season:
            state.start_season(g.season)
        rows.append(state.features(g.home, g.away, g.date))
        state.update(g.home, g.away, g.home_pts, g.away_pts, g.date)
    return pd.concat([games, pd.DataFrame(rows)], axis=1), state
