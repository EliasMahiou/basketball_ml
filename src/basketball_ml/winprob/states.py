"""Game states from the play-by-play: time left, score margin and who has the ball, event by event.

The play-by-play doesn't say who has the ball, so it is inferred from the
event sequence, following possession accounting:

- a made shot, a defensive rebound, a turnover or the last made free throw
  of a trip gives the ball to the other team;
- a missed shot or a missed last free throw leaves the ball *loose* until
  the rebound; an offensive rebound keeps the possession;
- a foul that sends a team to the line puts it in a *free-throw* state,
  with the number of free throws awarded (known when the foul is called);
- technical free throws don't change possession, and flagrant, clear-path,
  away-from-play and transition-take trips give the ball back to the
  shooting team;
- a loose-ball foul gives the ball to the other team; after a jump ball,
  and at the start of the 2nd to 4th quarters, the ball belongs to the team
  that next shoots, goes to the line or turns the ball over (the recorded
  "tip to" player is often wrong for jump balls during play).

A rebound recorded just after the putback it led to (same team, same clock)
is moved back before the shot.

Each row is the state right after an event. ``poss`` is +1 when the home
team has the possession, -1 for the away team and 0 before the opening tip;
``loose`` flags a missed shot waiting for its rebound; ``fts`` counts the free
throws the team in possession still has to shoot.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from ..impact.stints import TECHNICALS, _home_teams

PERIOD_SECONDS = 720
OVERTIME_SECONDS = 300
REGULATION_SECONDS = 4 * PERIOD_SECONDS
# FiveThirtyEight's conversion of an Elo gap into an expected point margin.
ELO_PER_POINT = 28.0
TEAM_ID_MIN = 1_610_612_737

# Event types kept as states. Substitutions, timeouts, ejections and replay reviews
# don't change the state; they are still walked through (the substitution count
# maps events to the stints of ``impact.stints``).
STATE_EVENTS = (pbp.MADE_SHOT, pbp.MISSED_SHOT, pbp.FREE_THROW, pbp.REBOUND, pbp.TURNOVER, pbp.FOUL,
                pbp.VIOLATION, pbp.JUMP_BALL, pbp.PERIOD_START, pbp.PERIOD_END)
# Events that end a dead-ball stretch: a free-throw trip's state is not carried back past them.
PLAY_EVENTS = {pbp.MADE_SHOT, pbp.MISSED_SHOT, pbp.REBOUND, pbp.TURNOVER, pbp.JUMP_BALL, pbp.PERIOD_START}
# stats.nba.com foul action types after which the fouled team keeps the ball once its free throws are
# shot: clear path (9), flagrant 1 and 2 (14, 15), transition take (31); away from play (6) only in the
# last two minutes of the 4th quarter and of overtimes.
KEEP_BALL_FOULS = {9, 14, 15, 31}
AWAY_FROM_PLAY = 6
TECHNICAL_FOULS = set(TECHNICALS) | {17, 32, 33}

STATE_COLUMNS = ["game_id", "row", "event", "period", "seconds_left", "margin", "poss", "loose", "fts",
                 "side", "p1", "p2", "p3", "subs"]


def seconds_left(period: np.ndarray | int, clock: np.ndarray | int) -> np.ndarray:
    """Seconds left in regulation, or in the current overtime."""
    period = np.asarray(period, dtype=np.int64)
    clock = np.asarray(clock, dtype=np.int64)
    return np.where(period <= 4, (4 - np.minimum(period, 4)) * PERIOD_SECONDS + clock, clock)


def elapsed_seconds(period: np.ndarray | int, left: np.ndarray | int) -> np.ndarray:
    """Seconds played since tip-off, from ``seconds_left`` (overtimes add 5 minutes each)."""
    period = np.asarray(period, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    overtime = REGULATION_SECONDS + (period - 5) * OVERTIME_SECONDS + (OVERTIME_SECONDS - left)
    return np.where(period <= 4, REGULATION_SECONDS - left, overtime)


def parse_clock(text: pd.Series) -> pd.Series:
    """"M:SS" -> seconds left in the period."""
    parts = text.astype(str).str.split(":", n=1, expand=True)
    return parts[0].astype(int) * 60 + parts[1].astype(float).astype(int)


def free_throw_details(text: pd.Series, late: pd.Series | bool = False) -> pd.DataFrame:
    """Number in the trip (``n`` of ``m``), technical, keeps-the-ball and missed flags of free-throw texts.

    Handles both layouts: "Free Throw Flagrant 1 of 2" (stats.nba.com) and
    "flagrant Free Throw 1 of 2" (cdn.nba.com). ``late`` marks the last two
    minutes of the 4th quarter and of overtimes, where an away-from-play
    foul also gives the ball back.
    """
    text = text.fillna("")
    nm = text.str.extract(r"(\d) of (\d)").fillna(1).astype(int)
    keeps = text.str.contains(r"flagrant|clear.path|transition take", case=False)
    keeps |= text.str.contains(r"away.from.play", case=False) & late
    return pd.DataFrame({
        "n": nm[0].to_numpy(), "m": nm[1].to_numpy(),
        "technical": text.str.contains("technical", case=False).to_numpy(),
        "keeps": keeps.to_numpy(),
        "missed": text.str.contains("MISS").to_numpy(),
    }, index=text.index)


def _sides(events: pd.DataFrame, homes: pd.Series) -> pd.DataFrame:
    """Side (+1 home, -1 away, 0 unknown) of each event's acting team, of its 3 players and team ids."""
    home = events["GAME_ID"].map(homes)
    hdesc, vdesc = events["HOMEDESCRIPTION"].notna(), events["VISITORDESCRIPTION"].notna()
    out = pd.DataFrame(index=events.index)
    ids = {}
    for i in (1, 2, 3):
        pid = pd.to_numeric(events[f"PLAYER{i}_ID"], errors="coerce").fillna(0).astype(np.int64)
        team = events[f"PLAYER{i}_TEAM_ABBREVIATION"]
        player = (pid > 0) & (pid < TEAM_ID_MIN) & team.notna()
        out[f"side{i}"] = np.where(player, np.where(team == home, 1, -1), 0)
        out[f"p{i}"] = np.where(player, pid, 0)
        ids[i] = pid
    # Team ids -> side, from team events whose text sits in a single description column.
    team_event = ids[1] >= TEAM_ID_MIN
    clear = team_event & (hdesc ^ vdesc)
    known = pd.DataFrame({"g": events["GAME_ID"][clear], "t": ids[1][clear], "s": np.where(hdesc[clear], 1, -1)})
    team_side = known.groupby(["g", "t"])["s"].agg(lambda s: int(np.sign(s.sum())))

    def side_of_team(pid: pd.Series) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([events["GAME_ID"], pid])
        return team_side.reindex(key).fillna(0).astype(int).to_numpy()

    by_desc = np.where(hdesc & ~vdesc, 1, np.where(vdesc & ~hdesc, -1, 0))
    team_side_1 = np.where(side_of_team(ids[1]) != 0, side_of_team(ids[1]), by_desc)
    out["side"] = np.where(out["side1"] != 0, out["side1"], np.where(team_event, team_side_1, 0))
    out["team_event"] = team_event.to_numpy()
    # Jump balls: the side of the player (or team) who recovers the tip.
    out["jump_side"] = np.where(out["side3"] != 0, out["side3"],
                                np.where(ids[3] >= TEAM_ID_MIN, side_of_team(ids[3]), 0))
    return out


def prepare_events(events: pd.DataFrame, homes: pd.Series) -> pd.DataFrame:
    """Column-wise inputs of the possession walk: clock, sides, score and free-throw details.

    ``events`` must be in file order within each game (EVENTNUM is not chronological).
    """
    e = events.reset_index(drop=True)
    kind = e["EVENTMSGTYPE"].astype(int)
    action = pd.to_numeric(e["EVENTMSGACTIONTYPE"], errors="coerce").fillna(0).astype(int)
    text = e["HOMEDESCRIPTION"].fillna("") + " " + e["VISITORDESCRIPTION"].fillna("")
    out = pd.DataFrame({"game_id": e["GAME_ID"].to_numpy(), "kind": kind.to_numpy(),
                        "period": e["PERIOD"].astype(int).to_numpy(),
                        "clock": parse_clock(e["PCTIMESTRING"]).to_numpy()})
    # A clock that runs backwards in file order is a recording slip: never let time go back up.
    out["clock"] = out.groupby(["game_id", "period"])["clock"].cummin()
    out = pd.concat([out, _sides(e, homes).reset_index(drop=True)], axis=1)

    score = e["SCORE"].where(e["SCORE"].astype(str).str.contains(" - ", regex=False))
    pts = score.str.split(" - ", expand=True)
    away = pd.to_numeric(pts[0], errors="coerce") if 0 in pts else pd.Series(np.nan, index=e.index)
    home = pd.to_numeric(pts[1], errors="coerce") if 1 in pts else pd.Series(np.nan, index=e.index)
    out["score_margin"] = (home - away).to_numpy()  # set on scoring events only, forward-filled later

    late = pd.Series((out["period"] >= 4).to_numpy() & (out["clock"] <= 120).to_numpy(), index=e.index)
    ft = kind == pbp.FREE_THROW
    details = free_throw_details(text[ft], late[ft])
    for col, default in (("n", 1), ("m", 1), ("technical", False), ("keeps", False), ("missed", False)):
        values = np.full(len(e), default, dtype=type(default))
        values[ft.to_numpy()] = details[col].to_numpy()
        out[f"ft_{col}"] = values
    foul = (kind == pbp.FOUL).to_numpy()
    tech_text = text.str.contains(r"tech", case=False).to_numpy()
    away_from_play = ((action == AWAY_FROM_PLAY) | text.str.contains(r"away.from.play", case=False)) & late
    keeps = action.isin(KEEP_BALL_FOULS) | text.str.contains(r"transition take", case=False) | away_from_play
    out["foul_technical"] = foul & (action.isin(TECHNICAL_FOULS).to_numpy() | tech_text)
    out["foul_keeps"] = foul & keeps.to_numpy()
    out["subs"] = (kind == pbp.SUBSTITUTION).astype(int).groupby([e["GAME_ID"], e["PERIOD"]]).cumsum().to_numpy()
    return out


def _next_offense(kind: list, period: list, side: list, trip_start: list) -> list:
    """For each event, the side of the next shot, turnover or free-throw trip at or after it in the same period."""
    n = len(kind)
    out = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and period[i] != period[i + 1]:
            cur = 0
        if side[i] and (kind[i] in (pbp.MADE_SHOT, pbp.MISSED_SHOT, pbp.TURNOVER) or trip_start[i]):
            cur = side[i]
        out[i] = cur
    return out


def putback_order(kind: list, period: list, clock: list, side: list) -> list:
    """Event order with each rebound recorded after its own putback moved back before the shot."""
    order: list[int] = []
    last_play = -1  # position in ``order`` of the last play event
    for i in range(len(kind)):
        if kind[i] == pbp.REBOUND and last_play >= 0:
            q = order[last_play]
            if (kind[q] == pbp.MADE_SHOT and side[q] == side[i] and side[i]
                    and period[q] == period[i] and clock[q] == clock[i]):
                order.insert(last_play, i)
                last_play += 1
                continue
        order.append(i)
        if kind[i] in PLAY_EVENTS:
            last_play = len(order) - 1
    return order


def walk_possession(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Possession (+1/-1/0), loose-ball flag and pending free throws after each event of one game."""
    kind, period, clock, side = (g[c].tolist() for c in ("kind", "period", "clock", "side"))
    team_event, jump_side = g["team_event"].tolist(), g["jump_side"].tolist()
    ft_n, ft_m, ft_tech, ft_keeps, ft_missed = (g[f"ft_{c}"].tolist()
                                                for c in ("n", "m", "technical", "keeps", "missed"))
    foul_tech, foul_keeps = g["foul_technical"].tolist(), g["foul_keeps"].tolist()
    trip_start = [k == pbp.FREE_THROW and a == 1 and not t and not kp
                  for k, a, t, kp in zip(kind, ft_n, ft_tech, ft_keeps)]
    nxt = _next_offense(kind, period, side, trip_start)
    n = len(kind)
    poss, loose, fts = [0] * n, [0] * n, [0] * n
    p = lo = ft = 0
    keep_next = False
    for i in range(n):
        k, s = kind[i], side[i]
        if k == pbp.PERIOD_START:
            # Quarters 1 and overtimes open with a jump ball; the others with an inbound.
            p = 0 if period[i] == 1 or period[i] > 4 else nxt[i]
            lo = ft = 0
            keep_next = False
        elif k == pbp.MADE_SHOT and s:
            p, lo, ft, keep_next = -s, 0, 0, False
        elif k == pbp.MISSED_SHOT and s:
            p, lo, ft, keep_next = s, 1, 0, False
        elif k == pbp.FREE_THROW and s and not ft_tech[i]:
            p = s
            if ft_n[i] < ft_m[i]:
                ft, lo = ft_m[i] - ft_n[i], 0
            else:
                ft = 0
                if ft_keeps[i] or keep_next:
                    lo = 0
                elif ft_missed[i]:
                    lo = 1
                else:
                    p, lo = -s, 0
                keep_next = False
        elif k == pbp.REBOUND and s and (lo or not team_event[i]):
            # Team rebounds outside a loose ball are dead-ball rebounds (missed non-final free throws).
            p, lo, ft, keep_next = s, 0, 0, False
        elif k == pbp.TURNOVER and s:
            p, lo, ft, keep_next = -s, 0, 0, False
        elif k == pbp.FOUL:
            if foul_keeps[i]:
                keep_next = True
            if lo and s and not foul_tech[i]:
                p, lo = -s, 0  # loose-ball foul: the other team gets the ball (or free throws)
        elif k == pbp.JUMP_BALL:
            p = (nxt[i + 1] if i + 1 < n else 0) or jump_side[i]
            lo = ft = 0
            keep_next = False
        poss[i], loose[i], fts[i] = p, lo, ft

    # A trip's free throws are known when the foul is called: carry the state back to it.
    for i in range(n):
        if kind[i] == pbp.FREE_THROW and side[i] and not ft_tech[i] and ft_n[i] == 1:
            j = i - 1
            while (j >= 0 and period[j] == period[i] and clock[j] == clock[i] and kind[j] not in PLAY_EVENTS
                   and not (kind[j] == pbp.FREE_THROW and not ft_tech[j])):
                poss[j], loose[j], fts[j] = side[i], 0, ft_m[i]
                j -= 1
    return np.array(poss), np.array(loose), np.array(fts)


def game_states(game: pd.DataFrame, home: str) -> pd.DataFrame:
    """States after each event of one game (events in file order), for ``home`` the home team's abbreviation."""
    gid = game["GAME_ID"].iloc[0]
    prepared = prepare_events(game, pd.Series({gid: home}))
    return states_from_events(prepared)


def states_from_events(prepared: pd.DataFrame) -> pd.DataFrame:
    """Run the possession walk game by game (on :func:`prepare_events` output) and keep the state-changing events.

    ``row`` is each event's position in the game's file order; the table is in walk order.
    """
    parts = []
    for _, g in prepared.groupby("game_id", sort=False):
        order = putback_order(g["kind"].tolist(), g["period"].tolist(), g["clock"].tolist(), g["side"].tolist())
        g = g.iloc[order]
        poss, loose, fts = walk_possession(g)
        part = g[["game_id", "kind", "period", "side", "p1", "p2", "p3", "subs"]].copy()
        part.insert(1, "row", np.asarray(order))
        part["margin"] = g["score_margin"].ffill().fillna(0).astype(int).to_numpy()
        part["seconds_left"] = seconds_left(g["period"].to_numpy(), g["clock"].to_numpy())
        part["poss"], part["loose"], part["fts"] = poss, loose, fts
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).rename(columns={"kind": "event"})
    out = out[out["event"].isin(STATE_EVENTS)]
    return out[STATE_COLUMNS].reset_index(drop=True)


def season_states(season: int, data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """States of every regular-season game of ``season``."""
    events = pbp.load_pbp(season, data_dir).sort_values(["GAME_ID", "PERIOD"], kind="stable")
    events = events.drop(columns=[c for c in ("EVENTNUM", "SCOREMARGIN", "PLAYER1_NAME", "PLAYER2_NAME",
                                              "PLAYER3_NAME") if c in events])
    homes = _home_teams(events)
    events = events[events["GAME_ID"].isin(homes.index)]
    out = states_from_events(prepare_events(events, homes))
    out.insert(1, "season", season)
    return out


def build_states(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    return pd.concat([season_states(s, data_dir) for s in seasons], ignore_index=True)


def elo_spread(elo_diff: np.ndarray | pd.Series) -> np.ndarray:
    """Expected home margin (points) of a pre-game Elo gap (home advantage included)."""
    return np.asarray(elo_diff, dtype=float) / ELO_PER_POINT


def add_game_info(states: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join pre-game strength (``elo_prob``, ``elo_spread``) and the final result (``home_win``) to each state.

    ``games`` is the output of ``predictions.add_features``. States of games
    missing from it are dropped.
    """
    info = games[["game_id", "elo_prob", "elo_diff", "home_win"]].copy()
    info["elo_spread"] = elo_spread(info.pop("elo_diff"))
    return states.merge(info, on="game_id", how="inner")


def possession_check(states: pd.DataFrame) -> pd.Series:
    """How often the inferred possession agrees with the team that acts next.

    For every shot and turnover (except the first after a period start or a
    jump ball, which is used to set who has the ball), the possession
    inferred just before it should belong to the shooter's or ball handler's
    team, and every player rebound should follow a loose ball. Returns the
    share of agreement for each check.
    """
    s = states
    prev = s.groupby("game_id", sort=False)[["poss", "loose", "fts", "period"]].shift()
    acts = s["event"].isin([pbp.MADE_SHOT, pbp.MISSED_SHOT, pbp.TURNOVER]) & (s["side"] != 0)
    segment = s["event"].isin([pbp.PERIOD_START, pbp.JUMP_BALL]).astype(int).groupby(s["game_id"]).cumsum()
    nth = acts.astype(int).groupby([s["game_id"], segment]).cumsum()
    action = acts & (nth > 1) & (prev["period"] == s["period"])
    shots = action & s["event"].isin([pbp.MADE_SHOT, pbp.MISSED_SHOT])
    tov = action & (s["event"] == pbp.TURNOVER)
    reb = (s["event"] == pbp.REBOUND) & (s["p1"] > 0)
    return pd.Series({
        "shots": float((prev["poss"][shots] == s["side"][shots]).mean()),
        "turnovers": float((prev["poss"][tov] == s["side"][tov]).mean()),
        "rebounds after a loose ball": float((prev["loose"][reb] == 1).mean()),
    })
