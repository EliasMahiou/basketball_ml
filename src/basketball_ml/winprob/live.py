"""Win probability for the season in progress, from the NBA's live feed.

The season being played isn't in the play-by-play archives yet. Its games
come from the NBA's public S3 feed (:mod:`basketball_ml.predictions.live`):
the schedule with each game's status, and the play-by-play of every game
that has started, final or in progress. Pre-game Elo is brought up to the
last finished game, and the in-game model is fit on every archived season.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from ..cache import cached_csv
from ..predictions import live as feed
from .model import MAX_TRAIN_ROWS, fit, predict, trainable
from .states import elo_spread, prepare_events, states_from_events

GAME_COLUMNS = ["game_id", "date", "home", "away", "home_pts", "away_pts", "season", "margin", "home_win"]
IN_PROGRESS = 2


def fitted_model(method: str, data_dir: str | Path, first: int, last: int, max_rows: int = MAX_TRAIN_ROWS):
    """The in-game model ``method`` fit on every state from ``first`` to ``last``, cached in ``data_dir``."""
    path = Path(data_dir) / f"winprob_model_{method}_{first}_{last}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from .cli import load_games, load_states

    states = load_states(data_dir, range(first, last + 1), load_games(data_dir, first, last))
    states = states[trainable(states)]
    if max_rows and len(states) > max_rows:
        states = states.sample(max_rows, random_state=0)
    model = fit(method, states)
    path.write_bytes(pickle.dumps(model))
    return model


def season_elo(schedule: pd.DataFrame, data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """``schedule`` (from :func:`feed.season_games`) with each game's pre-game ``elo_prob`` and ``elo_diff``.

    Finished games get the Elo before them; games not finished yet get it
    from the league after the last finished game.
    """
    from ..predictions import add_features, build_games

    data_dir = Path(data_dir)
    raw = cached_csv(data_dir / f"games_{first}_{last}.csv", lambda: build_games(range(first, last + 1), data_dir),
                     parse_dates=["date"])
    played = schedule[schedule["status"] == feed.FINAL]
    games, state = add_features(pd.concat([raw[GAME_COLUMNS], played[GAME_COLUMNS]], ignore_index=True))
    elo = games[games["game_id"].isin(played["game_id"])][["game_id", "elo_prob", "elo_diff"]]
    later = [{"game_id": g.game_id, "elo_prob": f["elo_prob"], "elo_diff": f["elo_diff"]}
             for g in schedule[schedule["status"] != feed.FINAL].itertuples(index=False)
             for f in [state.features(g.home, g.away)]]
    return schedule.merge(pd.concat([elo, pd.DataFrame(later, columns=elo.columns)], ignore_index=True),
                          on="game_id", how="left")


def game_states(game: pd.Series, cache_dir: str | Path = "data/live") -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """States and events (stats.nba.com columns) of one game of the schedule; None before tip-off.

    ``game`` is a row of :func:`season_elo`. The play-by-play of a final game
    is cached in ``cache_dir``; a game in progress is read again each time.
    """
    actions = feed.playbyplay(int(game["game_id"]), cache_dir, final=int(game["status"]) == feed.FINAL)
    if actions is None:
        return None
    events = pbp.from_cdn(actions).sort_values(["GAME_ID", "PERIOD"], kind="stable").reset_index(drop=True)
    states = states_from_events(prepare_events(events, pd.Series({int(game["game_id"]): game["home"]})))
    states.insert(1, "season", int(game["season"]))
    states["elo_prob"] = float(game["elo_prob"])
    states["elo_spread"] = float(elo_spread(game["elo_diff"]))
    states["home_win"] = game["home_win"]
    return states, events


def game_probabilities(states: pd.DataFrame, model, method: str) -> pd.DataFrame:
    """``states`` with ``p_<method>``: the home win probability after each event."""
    out = states[trainable(states)].reset_index(drop=True)
    out[f"p_{method}"] = predict(method, model, out)
    return out


def default_date(schedule: pd.DataFrame) -> pd.Timestamp:
    """The last day with a game in progress or finished."""
    started = schedule[schedule["status"].isin([IN_PROGRESS, feed.FINAL])]
    if started.empty:
        raise SystemExit("No game of the season has started yet")
    return started["date"].max()


def status_text(game: pd.Series, states: pd.DataFrame, p_now: float) -> str:
    """"Final, DET won 137-135 in OT" or "Q3 4:12, DET leads 87-80, DET 78%"."""
    home, away = game["home"], game["away"]
    if int(game["status"]) == feed.FINAL:
        winner = home if game["home_pts"] > game["away_pts"] else away
        extra = int(states["period"].max()) - 4
        when = "" if extra <= 0 else (" in OT" if extra == 1 else f" in {extra}OT")
        return f"Final, {winner} won {max(game['home_pts'], game['away_pts'])}-" \
               f"{min(game['home_pts'], game['away_pts'])}{when}"
    last = states.iloc[-1]
    period, left = int(last["period"]), int(last["seconds_left"])
    clock = left - (4 - period) * 720 if period <= 4 else left
    where = f"Q{period}" if period <= 4 else ("OT" if period == 5 else f"{period - 4}OT")
    margin = int(last["margin"])
    score = f"{home} leads by {margin}" if margin > 0 else (f"{away} leads by {-margin}" if margin < 0 else "tied")
    return f"{where} {clock // 60}:{clock % 60:02d}, {score}, {home} {p_now:.0%}"


def lowest_for_winner(probs: pd.DataFrame, method: str, home_won: bool) -> float:
    p = probs[f"p_{method}"].to_numpy()
    return float(np.min(p if home_won else 1 - p))
