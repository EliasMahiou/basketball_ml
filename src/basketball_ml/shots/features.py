"""Shot features for expected field goal percentage (xFG), from shot charts and play-by-play.

Every feature describes the shot, never the shooter: where it was taken, what
kind of shot it was and the game situation (period, clock, score). One row per
field goal attempt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp

ZONES = ["zone_basic", "zone_area", "zone_range"]
CATEGORICAL = ["action", "kind", *ZONES]
# Where the shot was taken and the game situation, without the shot type.
LOCATION_FEATURES = ["distance", "loc_x", "loc_y", "angle", "is_three", *ZONES,
                     "period", "seconds_left", "heave", "margin"]
# The xFG model: plus the shot type (normalized ACTION_TYPE). Until 2016-17,
# scorers gave made shots the detailed labels (pull-up, driving, step back...)
# more often than misses, so in those seasons the label partly gives away the
# result; the coarse ``kind`` (dunk, layup, hook, tip, jumper) does not.
FEATURES = [*LOCATION_FEATURES, "action"]

# An end-of-period heave: at least this far out with at most this many seconds left.
HEAVE_FEET = 30.0
HEAVE_SECONDS = 3

_SHOT_KINDS = ("jump", "hook", "layup", "dunk", "tip")


def shot_kind(action: pd.Series) -> pd.Series:
    """Coarse shot kind of a normalized label: tip, dunk, layup, hook or jump (jump shots and floaters)."""
    def kind(label: str) -> str:
        return next((k for k in ("tip", "dunk", "layup", "hook") if k in label), "jump")

    labels = action.astype(str)
    return labels.map({label: kind(label) for label in labels.unique()})


def normalize_action(action: pd.Series) -> pd.Series:
    """Shot-type labels (ACTION_TYPE) with outcome-dependent words removed and names made consistent.

    - "Bank" is dropped: scorers mostly call a shot a bank shot when it goes in
      off the glass, so bank shots from 3 to 25 feet are made 11 to 19 points
      more often than other shots from the same distance. Keeping the word
      would leak the result into the features.
    - Labels renamed in 2015-16 are merged: "Slam Dunk" becomes "Dunk", "Tip
      Shot" becomes "Tip Layup Shot", "Jump Hook" becomes "Hook".
    - Case is ignored, and "Turnaround shot" and "Turnaround Jump Shot" are the same label.
    """
    def clean(label: str) -> str:
        a = re.sub(r"\bbank\b", " ", str(label).lower())
        a = " ".join(a.split())
        a = a.replace("slam dunk", "dunk").replace("jump hook", "hook")
        a = re.sub(r"\btip shot$", "tip layup shot", a)
        if a.endswith(" shot") and a != "no shot" and not any(k in a for k in _SHOT_KINDS):
            a = a[:-len("shot")] + "jump shot"
        return a

    labels = action.fillna("unknown").astype(str)
    mapping = {label: clean(label) for label in labels.unique()}
    return labels.map(mapping)


def team_abbreviations(raw: pd.DataFrame) -> pd.DataFrame:
    """Each team's abbreviation per season: the one of HTM and VTM it has in (almost) every game."""
    games = raw.drop_duplicates(["season", "GAME_ID", "TEAM_ID"])
    both = pd.concat([games[["season", "TEAM_ID", side]].rename(columns={side: "team"}) for side in ("HTM", "VTM")])
    counts = both.groupby(["season", "TEAM_ID", "team"]).size().reset_index(name="games")
    return counts.sort_values("games").drop_duplicates(["season", "TEAM_ID"], keep="last")[["season", "TEAM_ID", "team"]]


def score_before(events: pd.DataFrame) -> pd.DataFrame:
    """Home score minus away score just before every made or missed shot in the play-by-play.

    Scores are only written when they change ("away - home"); they are carried
    forward in file order and read from the previous event, so a made shot's
    own points are never part of its margin.
    """
    score = events["SCORE"].astype("string").str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    game = events["GAME_ID"]
    margin = (score[1] - score[0]).groupby(game).ffill().fillna(0.0)
    before = margin.groupby(game).shift(1).fillna(0.0)
    is_shot = events["EVENTMSGTYPE"].isin([pbp.MADE_SHOT, pbp.MISSED_SHOT])
    out = pd.DataFrame({"GAME_ID": game[is_shot].astype(np.int64),
                        "GAME_EVENT_ID": pd.to_numeric(events.loc[is_shot, "EVENTNUM"]).astype(np.int64),
                        "home_margin": before[is_shot].astype(np.float32)})
    return out.drop_duplicates(["GAME_ID", "GAME_EVENT_ID"])


def build_margins(seasons: Iterable[int], data_dir: str | Path = "data/pbp") -> pd.DataFrame:
    """Score margin before every shot of ``seasons`` (GAME_ID, GAME_EVENT_ID, home_margin)."""
    return pd.concat([score_before(pbp.load_pbp(s, data_dir)) for s in seasons], ignore_index=True)


def prepare_shots(raw: pd.DataFrame, margins: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model-ready shots from shot chart rows (with a ``season`` column).

    The shooter's team is ``TEAM_ABBREVIATION`` when there is one, otherwise
    the abbreviation his team has in its games (see :func:`team_abbreviations`).

    ``margins`` (from :func:`build_margins`) adds the shooting team's lead before
    the shot; without it, or for shots missing from the play-by-play, the
    margin is left empty.
    """
    if "TEAM_ABBREVIATION" in raw:
        team = raw["TEAM_ABBREVIATION"].to_numpy()
    else:
        team = raw[["season", "TEAM_ID"]].merge(team_abbreviations(raw), on=["season", "TEAM_ID"],
                                                how="left")["team"].to_numpy()
    home = team == raw["HTM"].to_numpy()
    x = raw["LOC_X"].to_numpy(np.float32)
    y = raw["LOC_Y"].to_numpy(np.float32)
    value = np.where(raw["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2).astype(np.int8)
    seconds = (60 * raw["MINUTES_REMAINING"] + raw["SECONDS_REMAINING"]).to_numpy(np.int16)
    distance = np.hypot(x, y) / 10
    out = pd.DataFrame({
        "season": raw["season"].to_numpy(np.int16),
        "game_id": raw["GAME_ID"].to_numpy(np.int64),
        "game_event_id": raw["GAME_EVENT_ID"].to_numpy(np.int64),
        "player_id": raw["PLAYER_ID"].to_numpy(np.int64),
        "player": raw["PLAYER_NAME"].to_numpy(),
        "team": team,
        "opponent": np.where(home, raw["VTM"].to_numpy(), raw["HTM"].to_numpy()),
        "home": home.astype(np.int8),
        "made": raw["SHOT_MADE_FLAG"].to_numpy(np.int8),
        "value": value,
        "distance": distance.astype(np.float32),
        "loc_x": x,
        "loc_y": y,
        "angle": np.degrees(np.arctan2(x, y)).astype(np.float32),
        "is_three": (value == 3).astype(np.int8),
        "period": raw["PERIOD"].clip(upper=5).to_numpy(np.int8),
        "seconds_left": seconds,
        "heave": ((seconds <= HEAVE_SECONDS) & (distance >= HEAVE_FEET)).astype(np.int8),
        "action": normalize_action(raw["ACTION_TYPE"]).to_numpy(),
        "zone_basic": raw["SHOT_ZONE_BASIC"].to_numpy(),
        "zone_area": raw["SHOT_ZONE_AREA"].to_numpy(),
        "zone_range": raw["SHOT_ZONE_RANGE"].to_numpy(),
    })
    out["kind"] = shot_kind(out["action"]).to_numpy()
    out["margin"] = np.float32(np.nan)
    if margins is not None:
        key = pd.MultiIndex.from_arrays([out["game_id"], out["game_event_id"]])
        lookup = margins.set_index(["GAME_ID", "GAME_EVENT_ID"])["home_margin"]
        home_margin = lookup.reindex(key).to_numpy(np.float32)
        out["margin"] = np.where(home, home_margin, -home_margin).astype(np.float32)
    return out


def fill_zones(shots: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """``shots`` with missing zones filled in from where they were taken, as ``reference`` shots have them.

    The live feed leaves out the shot chart's three zone fields. They are
    fixed areas of the court, so each spot (``loc_x``, ``loc_y``, 2 or 3)
    takes the zones most reference shots from that spot have, and a spot no
    reference shot came from takes those of the nearest one that did. From
    the 2023-24 and 2024-25 shot charts, 99.9% of 2025-26 shots get all three
    zones right this way.
    """
    missing = shots[ZONES].isna().any(axis=1).to_numpy()
    if not missing.any():
        return shots
    key = ["loc_x", "loc_y", "is_three"]
    ref = reference[[*key, *ZONES]].astype({z: str for z in ZONES})
    spots = (ref.groupby([*key, *ZONES]).size().reset_index(name="n")
             .sort_values("n", ascending=False, kind="stable").drop_duplicates(key).reset_index(drop=True))
    zones = spots.set_index(key)[ZONES].reindex(pd.MultiIndex.from_frame(shots.loc[missing, key])).to_numpy()
    unseen = pd.isna(zones[:, 0])
    if unseen.any():
        from sklearn.neighbors import KDTree

        where = shots.loc[missing, key].to_numpy(float)
        for three in (0, 1):
            todo = unseen & (where[:, 2] == three)
            pool = spots[spots["is_three"] == three]
            if todo.any() and len(pool):
                nearest = KDTree(pool[["loc_x", "loc_y"]].to_numpy(float)).query(where[todo, :2], k=1)[1][:, 0]
                zones[todo] = pool[ZONES].to_numpy()[nearest]
    out = shots.copy()
    for i, z in enumerate(ZONES):
        out[z] = out[z].astype(object)
        out.loc[missing, z] = zones[:, i]
    return out


def build_shot_table(seasons: Iterable[int], data_dir: str | Path = "data/pbp",
                     margins: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prepared shots of several seasons, with text columns stored as categories."""
    parts = [prepare_shots(pbp.load_shots(s, data_dir), margins) for s in seasons]
    shots = pd.concat(parts, ignore_index=True)
    for c in ("player", "team", "opponent", *CATEGORICAL):
        shots[c] = shots[c].astype("category")
    return shots
