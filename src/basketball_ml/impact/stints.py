"""Who is on the court for every stretch of play, from the play-by-play.

The play-by-play lists substitutions but not who starts each period, so the
starters of a period are the players whose first appearance in it is not
being subbed in. A player who plays a whole period without recording a
single event can't be seen; such periods (fewer than 1 in 100) are skipped.

A stint is a stretch of play with the same ten players on the court. For
each stint the builder records the score change and each side's estimated
possessions, ``FGA + 0.44 FTA - OREB + TOV``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp

SUBSTITUTION = 8
JUMP_BALL = 10
# Technical-foul action types: they can be called on players on the bench.
TECHNICALS = {11, 12, 13, 16, 18, 19, 25, 30}
# For each event type, the player slots that prove a player was on the court.
ON_COURT_SLOTS = {
    pbp.MADE_SHOT: (1, 2),     # shooter, assist
    pbp.MISSED_SHOT: (1, 3),   # shooter, blocker
    pbp.FREE_THROW: (1,),
    pbp.REBOUND: (1,),
    pbp.TURNOVER: (1, 2),      # ball handler, stealer
    pbp.FOUL: (1, 2),          # fouler, fouled player
    JUMP_BALL: (1, 2, 3),
}
STAT_KEYS = ("pts", "fga", "fta", "oreb", "tov")


def _clock(text: str) -> int:
    minutes, seconds = str(text).split(":")[:2]
    return int(minutes) * 60 + int(float(seconds))


def _home_teams(events: pd.DataFrame) -> pd.Series:
    """Home team of each game: the team of players named in the home description."""
    shots = events[events["EVENTMSGTYPE"].isin([pbp.MADE_SHOT, pbp.MISSED_SHOT])
                   & events["HOMEDESCRIPTION"].notna()]
    return shots.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(lambda s: s.mode().iat[0])


def _is_player(value) -> bool:
    return isinstance(value, (int, np.integer)) and 0 < value < 1_610_000_000


def game_stints(game: pd.DataFrame, home: str) -> tuple[list[dict], int, int]:
    """Stints of one game. Returns the stints, periods used and periods skipped."""
    rows = game[["PERIOD", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PCTIMESTRING", "SCORE",
                 "HOMEDESCRIPTION", "VISITORDESCRIPTION",
                 "PLAYER1_ID", "PLAYER1_TEAM_ABBREVIATION", "PLAYER2_ID", "PLAYER2_TEAM_ABBREVIATION",
                 "PLAYER3_ID", "PLAYER3_TEAM_ABBREVIATION"]].itertuples(index=False, name=None)
    by_period: dict[int, list[tuple]] = {}
    for r in rows:
        by_period.setdefault(r[0], []).append(r)

    stints, used, skipped = [], 0, 0
    score = (0, 0)  # (home, away)
    for period, events in sorted(by_period.items()):
        # 1. Starters: players whose first appearance in the period isn't being subbed in.
        first: dict[int, bool] = {}
        team_of: dict[int, str] = {}
        for _, kind, action, _, _, _, _, p1, t1, p2, t2, p3, t3 in events:
            players = ((p1, t1), (p2, t2), (p3, t3))
            if kind == SUBSTITUTION:
                if _is_player(p1):
                    first.setdefault(p1, True)
                    team_of[p1] = t1
                if _is_player(p2):
                    first.setdefault(p2, False)
                    team_of[p2] = t2
            elif kind in ON_COURT_SLOTS and not (kind == pbp.FOUL and action in TECHNICALS):
                for slot in ON_COURT_SLOTS[kind]:
                    pid, team = players[slot - 1]
                    if _is_player(pid) and isinstance(team, str):
                        first.setdefault(pid, True)
                        team_of[pid] = team
        on = {"home": set(), "away": set()}
        for pid, starter in first.items():
            if starter:
                on["home" if team_of[pid] == home else "away"].add(pid)
        if len(on["home"]) != 5 or len(on["away"]) != 5:
            skipped += 1
            # Keep the running score right for the next periods.
            for r in events:
                if isinstance(r[4], str):
                    v, h = r[4].split(" - ")
                    score = (int(h), int(v))
            continue
        used += 1

        # 2. Walk the period, closing a stint at every lineup change.
        def new_stint(clock: int) -> dict:
            s = {f"{side}_{k}": 0.0 for side in ("home", "away") for k in STAT_KEYS}
            s.update(period=period, start=clock, home_players=tuple(sorted(on["home"])),
                     away_players=tuple(sorted(on["away"])))
            return s

        clock = _clock(events[0][3])
        cur = new_stint(clock)
        last_miss = None
        broken = False
        for _, kind, action, time, sc, hdesc, vdesc, p1, t1, p2, t2, p3, t3 in events:
            clock = _clock(time)
            if kind == SUBSTITUTION:
                side = "home" if t1 == home else "away"
                if p1 not in on[side] or not _is_player(p2):
                    broken = True
                    break
                cur["end"] = clock
                stints.append(cur)
                on[side].discard(p1)
                on[side].add(p2)
                cur = new_stint(clock)
                continue
            if _is_player(p1) and isinstance(t1, str):
                side = "home" if t1 == home else "away"
            else:  # team event: the side whose description holds it
                side = "home" if isinstance(hdesc, str) else "away"
            if kind in (pbp.MADE_SHOT, pbp.MISSED_SHOT):
                cur[f"{side}_fga"] += 1
            if kind == pbp.MISSED_SHOT:
                last_miss = side
            elif kind == pbp.FREE_THROW:
                cur[f"{side}_fta"] += 1
                if "MISS" in f"{hdesc} {vdesc}":
                    last_miss = side
            elif kind == pbp.REBOUND:
                if _is_player(p1) and side == last_miss:
                    cur[f"{side}_oreb"] += 1
                last_miss = None
            elif kind == pbp.TURNOVER:
                cur[f"{side}_tov"] += 1
            if isinstance(sc, str):
                v, h = sc.split(" - ")
                new = (int(h), int(v))
                cur["home_pts"] += new[0] - score[0]
                cur["away_pts"] += new[1] - score[1]
                score = new
        if broken:
            # A substitution named a player we didn't have on the court: drop the period.
            stints = [s for s in stints if s["period"] != period]
            used -= 1
            skipped += 1
            for r in events:
                if isinstance(r[4], str):
                    v, h = r[4].split(" - ")
                    score = (int(h), int(v))
            continue
        cur["end"] = clock
        stints.append(cur)
    return stints, used, skipped


def season_stints(season: int, data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """All stints of a regular season, one row each."""
    # File order is chronological; EVENTNUM is not (substitutions are often renumbered).
    events = pbp.load_pbp(season, data_dir).sort_values(["GAME_ID", "PERIOD"], kind="stable")
    for col in ("PLAYER1_ID", "PLAYER2_ID", "PLAYER3_ID"):
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(np.int64)
    homes = _home_teams(events)
    rows, used, skipped = [], 0, 0
    for game_id, game in events.groupby("GAME_ID", sort=False):
        if game_id not in homes:
            continue
        stints, u, s = game_stints(game, homes[game_id])
        used, skipped = used + u, skipped + s
        for st in stints:
            st["game_id"] = game_id
        rows.extend(stints)
    out = pd.DataFrame(rows)
    for side in ("home", "away"):
        out[player_columns(side)] = pd.DataFrame(out.pop(f"{side}_players").tolist(), index=out.index)
    out["season"] = season
    out["seconds"] = (out["start"] - out["end"]).clip(lower=0)
    for side in ("home", "away"):
        out[f"{side}_poss"] = (out[f"{side}_fga"] + 0.44 * out[f"{side}_fta"]
                               - out[f"{side}_oreb"] + out[f"{side}_tov"])
    out.attrs["periods_used"], out.attrs["periods_skipped"] = used, skipped
    return out


def build_stints(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    return pd.concat([season_stints(s, data_dir) for s in seasons], ignore_index=True)


def player_columns(side: str) -> list[str]:
    """The five player-id columns of a side: ``home_1`` ... ``home_5``."""
    return [f"{side}_{i}" for i in range(1, 6)]


def player_seconds(stints: pd.DataFrame) -> pd.DataFrame:
    """Seconds on the court per player per game, with the side he played for."""
    parts = []
    for side in ("home", "away"):
        ids = stints[player_columns(side)].to_numpy()
        parts.append(pd.DataFrame({"season": np.repeat(stints["season"].to_numpy(), 5),
                                   "game_id": np.repeat(stints["game_id"].to_numpy(), 5), "side": side,
                                   "player_id": ids.ravel(), "seconds": np.repeat(stints["seconds"].to_numpy(), 5)}))
    return pd.concat(parts).groupby(["season", "game_id", "side", "player_id"], as_index=False)["seconds"].sum()
