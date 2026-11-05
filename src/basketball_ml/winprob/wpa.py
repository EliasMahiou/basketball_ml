"""Win probability added (WPA): how much each player's plays moved his team's chances.

Each event's change in the home team's win probability (from the state
before it to the state after it, so the time that ran off in between is
included) goes to the players credited on the event, from the point of view
of each player's team:

- made shot: the shooter, or 70% to the shooter and 30% to the passer when assisted;
- missed shot: the shooter, or half to the shooter and half to the blocker;
- free throw: the shooter; rebound: the rebounder;
- turnover: the ball handler, or half to him and half to the stealer;
- foul: the fouler (the free throws it gives away are counted at the foul).

Team rebounds and turnovers, violations, jump balls and the clock running
out after the last event are not credited to anyone. "Clutch" is the NBA's
definition: the last 5 minutes of the 4th quarter or overtime with the
margin (before the event) within 5 points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from ..impact.stints import player_columns
from .states import elapsed_seconds

ASSIST_SHARE = 0.3
STEAL_SHARE = 0.5
BLOCK_SHARE = 0.5
CLUTCH_SECONDS = 300
CLUTCH_MARGIN = 5
SOURCES = {pbp.MADE_SHOT: "shots", pbp.MISSED_SHOT: "shots", pbp.FREE_THROW: "free_throws",
           pbp.REBOUND: "rebounds", pbp.TURNOVER: "turnovers", pbp.FOUL: "fouls"}


def win_prob_changes(results: pd.DataFrame, method: str) -> pd.DataFrame:
    """States with the home win probability ``wp``, its change ``delta`` and the margin before each event.

    ``results`` is in game order (as ``walk_forward`` returns it); the first
    state of a game (the opening tip) has no change.
    """
    r = results.copy()
    r["wp"] = r[f"p_{method}"].to_numpy()
    g = r.groupby("game_id", sort=False)
    r["delta"] = (r["wp"] - g["wp"].shift()).fillna(0.0)
    r["margin_before"] = g["margin"].shift().fillna(0).astype(int)
    r["clutch"] = (((r["period"] == 4) & (r["seconds_left"] <= CLUTCH_SECONDS)) | (r["period"] > 4)) & (
        r["margin_before"].abs() <= CLUTCH_MARGIN)
    return r


def credits(changes: pd.DataFrame) -> pd.DataFrame:
    """One row per credited player and event: ``player_id``, ``side`` (+1 home, -1 away), ``source`` and ``wpa``."""
    c = changes
    parts = []

    def add(mask, id_col, sign, share, source):
        sub = c[mask & (c[id_col] > 0)]
        side = sign * sub["side"].to_numpy()
        wpa = sub["delta"].to_numpy() * side * (share if np.isscalar(share) else share[sub.index])
        parts.append(pd.DataFrame({"season": sub["season"].to_numpy(), "game_id": sub["game_id"].to_numpy(),
                                   "player_id": sub[id_col].to_numpy(), "side": side,
                                   "source": source, "wpa": wpa, "clutch": sub["clutch"].to_numpy()}))

    made, missed = c["event"] == pbp.MADE_SHOT, c["event"] == pbp.MISSED_SHOT
    tov = c["event"] == pbp.TURNOVER
    acted = c["side"] != 0
    add(made & acted, "p1", 1, pd.Series(np.where(c["p2"] > 0, 1 - ASSIST_SHARE, 1.0), index=c.index), "shots")
    add(made & acted, "p2", 1, ASSIST_SHARE, "assists")
    add(missed & acted, "p1", 1, pd.Series(np.where(c["p3"] > 0, 1 - BLOCK_SHARE, 1.0), index=c.index), "shots")
    add(missed & acted, "p3", -1, BLOCK_SHARE, "defense")
    add(tov & acted, "p1", 1, pd.Series(np.where(c["p2"] > 0, 1 - STEAL_SHARE, 1.0), index=c.index), "turnovers")
    add(tov & acted, "p2", -1, STEAL_SHARE, "defense")
    for kind in (pbp.FREE_THROW, pbp.REBOUND, pbp.FOUL):
        add((c["event"] == kind) & acted, "p1", 1, 1.0, SOURCES[kind])
    return pd.concat(parts, ignore_index=True)


def _team(table: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """Abbreviation of the team each credited row played for."""
    teams = games.set_index("game_id")[["home", "away"]]
    home = table["game_id"].map(teams["home"])
    away = table["game_id"].map(teams["away"])
    return pd.Series(np.where(table["side"] > 0, home, away), index=table.index)


def player_wpa(results: pd.DataFrame, games: pd.DataFrame, method: str) -> pd.DataFrame:
    """Season WPA per player (in wins): total, clutch, and the total split by kind of play."""
    c = credits(win_prob_changes(results, method))
    c["team"] = _team(c, games)
    keys = ["season", "player_id"]
    out = c.groupby(keys).agg(team=("team", lambda s: s.value_counts().index[0]), games=("game_id", "nunique"),
                              wpa=("wpa", "sum"))
    out["clutch_wpa"] = c[c["clutch"]].groupby(keys)["wpa"].sum().reindex(out.index).fillna(0.0)
    split = c.pivot_table(index=keys, columns="source", values="wpa", aggfunc="sum").fillna(0.0)
    for source in ("shots", "free_throws", "assists", "rebounds", "turnovers", "defense", "fouls"):
        out[f"from_{source}"] = split[source].reindex(out.index).fillna(0.0) if source in split else 0.0
    return out.reset_index()


def on_court_wpa(results: pd.DataFrame, games: pd.DataFrame, stints: pd.DataFrame, method: str) -> pd.DataFrame:
    """Team WPA while each player is on the court (a plus-minus in wins), per season.

    Events are matched to the stints of ``impact.stints`` by the number of
    substitutions earlier in the period; periods whose stints don't match
    (or that the stint builder skipped) are left out.
    """
    r = win_prob_changes(results, method)
    s = stints[["game_id", "period", *player_columns("home"), *player_columns("away")]].copy()
    s["subs"] = s.groupby(["game_id", "period"]).cumcount()
    m = r.merge(s, on=["game_id", "period", "subs"], how="inner")
    parts = []
    for side, sign in (("home", 1), ("away", -1)):
        ids = m[player_columns(side)].to_numpy()
        parts.append(pd.DataFrame({"season": np.repeat(m["season"].to_numpy(), 5),
                                   "game_id": np.repeat(m["game_id"].to_numpy(), 5),
                                   "player_id": ids.ravel(), "side": sign,
                                   "on_court_wpa": np.repeat(sign * m["delta"].to_numpy(), 5)}))
    long = pd.concat(parts, ignore_index=True)
    long["team"] = _team(long, games)
    out = long.groupby(["season", "player_id"]).agg(team=("team", lambda t: t.value_counts().index[0]),
                                                    games=("game_id", "nunique"), on_court_wpa=("on_court_wpa", "sum"))
    return out.reset_index()


def biggest_comebacks(results: pd.DataFrame, games: pd.DataFrame, method: str) -> pd.DataFrame:
    """Games sorted by the eventual winner's lowest win probability, with where it happened."""
    r = results[["game_id", "period", "seconds_left", "margin", "home_win", f"p_{method}"]].copy()
    r["winner_wp"] = np.where(r["home_win"] == 1, r[f"p_{method}"], 1 - r[f"p_{method}"])
    low = r.loc[r.groupby("game_id")["winner_wp"].idxmin()]
    info = games.set_index("game_id")
    out = pd.DataFrame({"game_id": low["game_id"].to_numpy()})
    out["date"] = out["game_id"].map(info["date"]).dt.date
    out["away"] = out["game_id"].map(info["away"])
    out["home"] = out["game_id"].map(info["home"])
    out["final"] = (out["game_id"].map(info["away_pts"]).astype(str) + "-"
                    + out["game_id"].map(info["home_pts"]).astype(str))
    home_won = low["home_win"].to_numpy() == 1
    out["winner"] = np.where(home_won, out["home"], out["away"])
    out["lowest_wp"] = low["winner_wp"].round(4).to_numpy()
    out["period"] = low["period"].to_numpy()
    period, left = low["period"].to_numpy(np.int64), low["seconds_left"].to_numpy(np.int64)
    clock = np.where(period <= 4, left - (4 - np.minimum(period, 4)) * 720, left)
    out["clock"] = [f"{c // 60}:{c % 60:02d}" for c in clock]
    out["winner_trailed_by"] = np.where(home_won, -low["margin"], low["margin"])
    return out.sort_values("lowest_wp").reset_index(drop=True)


def game_curve(results: pd.DataFrame, games: pd.DataFrame, game_id: int, method: str,
               data_dir: str | Path | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """One game's states with ``minute`` (played), ``wp``, ``margin`` and ``delta``, ending on the result.

    Adds each event's ``description`` from the game's play-by-play: ``events``
    (in the file order its states were built from), or the archive in
    ``data_dir``. A game still in progress (``home_win`` missing) ends on its
    last event.
    """
    r = win_prob_changes(results[results["game_id"] == game_id], method)
    if r.empty:
        raise SystemExit(f"No out-of-sample probabilities for game {game_id}")
    r["minute"] = elapsed_seconds(r["period"].to_numpy(), r["seconds_left"].to_numpy()) / 60
    if events is None and data_dir is not None:
        events = pbp.load_pbp(int(r["season"].iloc[0]), data_dir)
        events = events[events["GAME_ID"] == game_id].sort_values("PERIOD", kind="stable").reset_index(drop=True)
    if events is not None:
        text = events["HOMEDESCRIPTION"].fillna(events["VISITORDESCRIPTION"]).fillna("")
        both = events["HOMEDESCRIPTION"].notna() & events["VISITORDESCRIPTION"].notna()
        # Steals and blocks share a row with the turnover or miss: keep the play's own text.
        own = np.where(r["side"].to_numpy() > 0, events["HOMEDESCRIPTION"].reindex(r["row"]).to_numpy(),
                       events["VISITORDESCRIPTION"].reindex(r["row"]).to_numpy())
        r["description"] = np.where(both.reindex(r["row"]).to_numpy(), own, text.reindex(r["row"]).to_numpy())
    game = games[games["game_id"] == game_id].iloc[0]
    if pd.isna(game["home_win"]):
        return r.reset_index(drop=True)
    end = r.iloc[[-1]].copy()
    end["wp"] = float(game["home_win"])
    end["delta"] = end["wp"] - r["wp"].iloc[-1]
    end["minute"] = 48 + 5 * max(int(r["period"].max()) - 4, 0)
    end["margin"] = int(game["home_pts"] - game["away_pts"])
    if "description" in end:
        end["description"] = "Final"
    return pd.concat([r, end], ignore_index=True)
