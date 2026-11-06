"""Download and load NBA play-by-play and shot data.

Source: the public GitHub repository ``shufinskiy/nba_data``, which mirrors
stats.nba.com play-by-play (since 1996-97) and shot charts as one
``.tar.xz`` archive per season. Seasons missing from its stats.nba.com
play-by-play (2025-26 so far) are read from its copy of the cdn.nba.com feed
and converted to the same columns; the shots of that feed can be turned into
shot chart rows too (:func:`shots_from_cdn`), for the season in progress.

Seasons are named by the year they end, like everywhere else in this
package: ``2025`` is the 2024-25 season.
"""

from __future__ import annotations

import io
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets"
KINDS = ("nbastats", "shotdetail", "cdnnba")

PBP_COLUMNS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
    "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE", "SCOREMARGIN",
    "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION",
    "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ABBREVIATION",
]

# stats.nba.com EVENTMSGTYPE codes
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER, FOUL = 1, 2, 3, 4, 5, 6
VIOLATION, SUBSTITUTION, TIMEOUT, JUMP_BALL, EJECTION, PERIOD_START, PERIOD_END = 7, 8, 9, 10, 11, 12, 13


def _archive_name(kind: str, season: int, playoffs: bool) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    # The source names archives by the year a season starts.
    return f"{kind}{'_po' if playoffs else ''}_{season - 1}"


def local_path(kind: str, season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> Path:
    return Path(data_dir) / f"{_archive_name(kind, season, playoffs)}.csv"


def download(kind: str, season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False,
             overwrite: bool = False) -> Path:
    """Download one season of ``kind`` ("nbastats" or "shotdetail") as CSV."""
    out = local_path(kind, season, data_dir, playoffs)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{_archive_name(kind, season, playoffs)}.tar.xz"
    with urllib.request.urlopen(url, timeout=300) as resp:
        payload = resp.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.isfile() and m.name.endswith(".csv"))
        with tar.extractfile(member) as src:
            out.write_bytes(src.read())
    return out


def load_pbp(season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> pd.DataFrame:
    """Play-by-play events for one season (downloaded on first use)."""
    converted = Path(data_dir) / f"{_archive_name('nbastats', season, playoffs)}_from_cdn.csv"
    if converted.exists():
        path = converted
    else:
        try:
            path = download("nbastats", season, data_dir, playoffs)
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            cdn = pd.read_csv(download("cdnnba", season, data_dir, playoffs), low_memory=False)
            events = from_cdn(cdn)
            # The cdn feed only has short names ("S. Gilgeous-Alexander"): take full ones from the shot charts.
            shots = load_shots(season, data_dir, playoffs)
            full = shots.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"]
            for i in (1, 2, 3):
                ids = events[f"PLAYER{i}_ID"]
                events[f"PLAYER{i}_NAME"] = ids.map(full).fillna(events[f"PLAYER{i}_NAME"])
            events.to_csv(converted, index=False)
            path = converted
    df = pd.read_csv(path, usecols=lambda c: c in PBP_COLUMNS, low_memory=False)
    df["season"] = season
    return df


CDN_TYPES = {"freethrow": FREE_THROW, "rebound": REBOUND, "turnover": TURNOVER, "foul": FOUL,
             "violation": VIOLATION, "substitution": SUBSTITUTION, "timeout": TIMEOUT, "jumpball": JUMP_BALL,
             "ejection": EJECTION}


def _cdn_clock(clock: pd.Series) -> pd.Series:
    """"PT11M36.00S" -> "11:36"."""
    parts = clock.str.extract(r"PT(\d+)M(\d+)")
    return parts[0].astype(int).astype(str) + ":" + parts[1].str.zfill(2)


def _cdn_action_type(d: pd.DataFrame) -> pd.Series:
    """The stats.nba.com action types the package relies on (bad passes, technical fouls)."""
    sub = d["subType"].fillna("").str.lower()
    desc = d["descriptor"].fillna("").str.lower() if "descriptor" in d else pd.Series("", index=d.index)
    text = d["description"].fillna("").str.lower()
    out = np.zeros(len(d), dtype=int)
    turnover, foul = d["actionType"] == "turnover", d["actionType"] == "foul"
    out[(turnover & (sub == "bad pass")).to_numpy()] = 1
    out[(turnover & (sub == "out-of-bounds") & (desc.str.contains("bad pass") | text.str.contains("bad pass"))).to_numpy()] = 45
    out[(foul & (sub == "personal")).to_numpy()] = 1
    out[(foul & (desc == "shooting")).to_numpy()] = 2
    out[(foul & (sub == "technical")).to_numpy()] = 11
    out[(foul & (sub == "technical") & (desc == "defensive-3-second")).to_numpy()] = 17
    return pd.Series(out, index=d.index)


def from_cdn(cdn: pd.DataFrame) -> pd.DataFrame:
    """Convert cdn.nba.com play-by-play to the stats.nba.com columns used in this package.

    Steals, blocks and assists become the second and third players of the
    turnover, missed shot or made shot they belong to; each substitution's
    "out" and "in" rows become one event; scores are written only where they
    change; descriptions go to the home or visitor column by team.
    """
    d = cdn.sort_values(["gameId", "orderNumber"], kind="stable").reset_index(drop=True)
    # The live feed leaves out the fields no action of the game has used yet (no block, no jump ball...).
    for col in ("personId", "assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId",
                "jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId", "teamId"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(np.int64) if col in d else 0
    for col in ("teamTricode", "playerNameI", "subType", "shotResult", "description"):
        if col not in d:
            d[col] = None
    team_of = d[d["personId"] > 0].drop_duplicates(["gameId", "personId"]).set_index(
        ["gameId", "personId"])["teamTricode"]
    name_of = d[d["personId"] > 0].drop_duplicates(["gameId", "personId"]).set_index(
        ["gameId", "personId"])["playerNameI"]

    # Home team: the one whose made baskets raise scoreHome.
    score_home = pd.to_numeric(d["scoreHome"], errors="coerce")
    raised = score_home.groupby(d["gameId"]).diff().fillna(score_home) > 0
    home = d[raised & d["teamTricode"].notna()].groupby("gameId")["teamTricode"].agg(lambda s: s.mode().iat[0])

    # Substitutions: pair the i-th "out" with the i-th "in" of the same team at the same moment.
    subs = d[d["actionType"] == "substitution"].copy()
    subs["slot"] = subs.groupby(["gameId", "period", "clock", "teamTricode", "subType"]).cumcount()
    key = ["gameId", "period", "clock", "teamTricode", "slot"]
    pairs = subs[subs["subType"] == "out"].reset_index().merge(
        subs[subs["subType"] == "in"][[*key, "personId"]].rename(columns={"personId": "inId"}), on=key, how="left")
    sub_in = pairs.set_index("index")["inId"].fillna(0).astype(np.int64)

    keep = ~d["actionType"].isin(["steal", "block", "heave", "game"]) & ~(
        (d["actionType"] == "substitution") & (d["subType"] != "out"))
    d = d[keep].copy()

    kind = d["actionType"].map(CDN_TYPES)
    shot = d["actionType"].isin(["2pt", "3pt"])
    kind[shot] = np.where(d.loc[shot, "shotResult"] == "Made", MADE_SHOT, MISSED_SHOT)
    kind[(d["actionType"] == "period") & (d["subType"] == "start")] = PERIOD_START
    kind[(d["actionType"] == "period") & (d["subType"] == "end")] = PERIOD_END
    d = d[kind.notna()].copy()
    kind = kind[kind.notna()].astype(int)

    p1 = d["personId"].where(d["personId"] > 0, d["teamId"])
    p2 = pd.Series(0, index=d.index, dtype=np.int64)
    p3 = pd.Series(0, index=d.index, dtype=np.int64)
    p2[kind == MADE_SHOT] = d.loc[kind == MADE_SHOT, "assistPersonId"]
    p3[kind == MISSED_SHOT] = d.loc[kind == MISSED_SHOT, "blockPersonId"]
    p2[kind == TURNOVER] = d.loc[kind == TURNOVER, "stealPersonId"]
    p2[kind == FOUL] = d.loc[kind == FOUL, "foulDrawnPersonId"]
    is_sub = kind == SUBSTITUTION
    p2[is_sub] = sub_in.reindex(d.index[is_sub]).fillna(0).astype(np.int64).to_numpy()
    jump = kind == JUMP_BALL
    if "jumpBallWonPersonId" in d:
        p1[jump] = d.loc[jump, "jumpBallWonPersonId"]
        p2[jump] = d.loc[jump, "jumpBallLostPersonId"]
        p3[jump] = d.loc[jump, "jumpBallRecoverdPersonId"]

    def team(ids: pd.Series) -> pd.Series:
        return pd.Series(team_of.reindex(pd.MultiIndex.from_arrays([d["gameId"], ids])).to_numpy(), index=d.index)

    def name(ids: pd.Series) -> pd.Series:
        return pd.Series(name_of.reindex(pd.MultiIndex.from_arrays([d["gameId"], ids])).to_numpy(), index=d.index)

    home_team = d["gameId"].map(home)
    description = d["description"]
    score = pd.to_numeric(d["scoreAway"], errors="coerce").astype("Int64").astype(str) + " - " + \
        pd.to_numeric(d["scoreHome"], errors="coerce").astype("Int64").astype(str)
    changed = score != score.groupby(d["gameId"]).shift().fillna("0 - 0")
    margin = pd.to_numeric(d["scoreHome"], errors="coerce") - pd.to_numeric(d["scoreAway"], errors="coerce")
    out = pd.DataFrame({
        "GAME_ID": d["gameId"], "EVENTNUM": d["actionNumber"], "EVENTMSGTYPE": kind,
        "EVENTMSGACTIONTYPE": _cdn_action_type(d), "PERIOD": d["period"], "PCTIMESTRING": _cdn_clock(d["clock"]),
        "HOMEDESCRIPTION": description.where(d["teamTricode"] == home_team),
        "VISITORDESCRIPTION": description.where(d["teamTricode"].notna() & (d["teamTricode"] != home_team)),
        "SCORE": score.where(changed), "SCOREMARGIN": margin.where(changed),
        "PLAYER1_ID": p1, "PLAYER1_NAME": name(p1), "PLAYER1_TEAM_ABBREVIATION": team(p1),
        "PLAYER2_ID": p2, "PLAYER2_NAME": name(p2), "PLAYER2_TEAM_ABBREVIATION": team(p2),
        "PLAYER3_ID": p3, "PLAYER3_NAME": name(p3), "PLAYER3_TEAM_ABBREVIATION": team(p3),
    })
    return out.reset_index(drop=True)


# Shot types of the cdn.nba.com feed (subType, descriptor) as stats.nba.com's shot charts name them
# (ACTION_TYPE): on the 2025-26 shots that are in both, each pair has one name 99.8% of the time or more.
CDN_SHOT_TYPES = {
    ("DUNK", ""): "Dunk Shot", ("DUNK", "alley-oop"): "Alley Oop Dunk Shot",
    ("DUNK", "cutting"): "Cutting Dunk Shot", ("DUNK", "driving"): "Driving Dunk Shot",
    ("DUNK", "driving reverse"): "Driving Reverse Dunk Shot", ("DUNK", "putback"): "Putback Dunk Shot",
    ("DUNK", "reverse"): "Reverse Dunk Shot", ("DUNK", "running"): "Running Dunk Shot",
    ("DUNK", "running alley-oop"): "Running Alley Oop Dunk Shot", ("DUNK", "running reverse"): "Running Reverse Dunk Shot",
    ("DUNK", "tip"): "Tip Dunk Shot",
    ("Hook", ""): "Hook Shot", ("Hook", "bank"): "Hook Bank Shot", ("Hook", "driving"): "Driving Hook Shot",
    ("Hook", "driving bank"): "Driving Bank Hook Shot", ("Hook", "turnaround"): "Turnaround Hook Shot",
    ("Hook", "turnaround bank"): "Turnaround Bank Hook Shot",
    ("Jump Shot", ""): "Jump Shot", ("Jump Shot", "bank"): "Jump Bank Shot",
    ("Jump Shot", "driving bank"): "Driving Floating Bank Jump Shot",
    ("Jump Shot", "driving floating"): "Driving Floating Jump Shot",
    ("Jump Shot", "driving floating bank"): "Driving Floating Bank Jump Shot",
    ("Jump Shot", "fadeaway"): "Fadeaway Jump Shot", ("Jump Shot", "fadeaway bank"): "Fadeaway Bank shot",
    ("Jump Shot", "floating"): "Floating Jump shot", ("Jump Shot", "pullup"): "Pullup Jump shot",
    ("Jump Shot", "pullup bank"): "Pullup Jump shot", ("Jump Shot", "running"): "Running Jump Shot",
    ("Jump Shot", "running pullup"): "Running Pull-Up Jump Shot", ("Jump Shot", "step back"): "Step Back Jump shot",
    ("Jump Shot", "step back bank"): "Step Back Bank Jump Shot", ("Jump Shot", "turnaround"): "Turnaround Jump Shot",
    ("Jump Shot", "turnaround bank"): "Turnaround Bank shot",
    ("Jump Shot", "turnaround fadeaway"): "Turnaround Fadeaway shot",
    ("Jump Shot", "turnaround fadeaway bank"): "Turnaround Fadeaway Bank Jump Shot",
    ("Layup", ""): "Layup Shot", ("Layup", "alley-oop"): "Alley Oop Layup shot",
    ("Layup", "cutting"): "Cutting Layup Shot", ("Layup", "cutting finger roll"): "Cutting Finger Roll Layup Shot",
    ("Layup", "driving"): "Driving Layup Shot", ("Layup", "driving finger roll"): "Driving Finger Roll Layup Shot",
    ("Layup", "driving reverse"): "Driving Reverse Layup Shot", ("Layup", "finger roll"): "Finger Roll Layup Shot",
    ("Layup", "putback"): "Putback Layup Shot", ("Layup", "reverse"): "Reverse Layup Shot",
    ("Layup", "running"): "Running Layup Shot", ("Layup", "running alley-oop"): "Running Alley Oop Layup Shot",
    ("Layup", "running finger roll"): "Running Finger Roll Layup Shot",
    ("Layup", "running reverse"): "Running Reverse Layup Shot", ("Layup", "tip"): "Tip Layup Shot",
}
CDN_ZONE_AREAS = {"Left": "Left Side(L)", "Left Center": "Left Side Center(LC)", "Center": "Center(C)",
                  "Right Center": "Right Side Center(RC)", "Right": "Right Side(R)"}
CDN_ZONE_RANGES = {"0-8": "Less Than 8 ft.", "8-16": "8-16 ft.", "16-24": "16-24 ft.", "24+": "24+ ft."}


def _cdn_shot_type(sub: str, descriptor: str) -> str:
    """ACTION_TYPE of a cdn shot; pairs the 2025-26 shots don't have get "<Descriptor> <subType> Shot"."""
    known = CDN_SHOT_TYPES.get((sub, descriptor))
    if known:
        return known
    base = {"DUNK": "Dunk Shot", "Layup": "Layup Shot", "Hook": "Hook Shot"}.get(sub, "Jump Shot")
    return f"{descriptor.title()} {base}".strip()


def shots_from_cdn(cdn: pd.DataFrame, games: pd.DataFrame, names: pd.Series | None = None) -> pd.DataFrame:
    """Shot chart rows (the columns of :func:`load_shots`) from cdn.nba.com play-by-play.

    ``games`` gives each ``game_id`` its ``date``, ``home`` and ``away``
    tricodes (shots of other games are left out); ``names`` maps player ids
    to full names (default: the feed's "A. Name"). The shooter's team is
    also given as ``TEAM_ABBREVIATION``, which the shot charts don't have.

    On 2025-26, where both exist, the converted shots match the shot charts'
    event ids, shooters, results and locations (99.9%) and shot types
    (99.97%). The zones come from the feed's ``area`` and ``areaDetail``
    (97% to 99% right) when it has them; the live feed doesn't, and leaves
    them empty for :func:`basketball_ml.shots.fill_zones`.
    """
    info = games.set_index("game_id")
    d = cdn[cdn["actionType"].isin(["2pt", "3pt"]) & cdn["gameId"].astype(np.int64).isin(info.index)].copy()
    for col in ("descriptor", "area", "areaDetail", "playerNameI"):
        if col not in d:
            d[col] = None
    clock = d["clock"].str.extract(r"PT(\d+)M(\d+)").astype(int)
    detail = d["areaDetail"].fillna("").str.extract(r"^(\S+)\s*(.*)$")
    game_id = d["gameId"].astype(np.int64)
    person = pd.to_numeric(d["personId"], errors="coerce").fillna(0).astype(np.int64)
    name = d["playerNameI"] if names is None else person.map(names).fillna(d["playerNameI"])
    made = (d["shotResult"] == "Made").astype(np.int8)
    shot_type = [_cdn_shot_type(str(s), str(x or "")) for s, x in
                 zip(d["subType"].fillna(""), d["descriptor"].where(d["descriptor"].notna(), ""))]
    out = pd.DataFrame({
        "GRID_TYPE": "Shot Chart Detail", "GAME_ID": game_id, "GAME_EVENT_ID": d["actionNumber"].astype(np.int64),
        "PLAYER_ID": person, "PLAYER_NAME": name, "TEAM_ID": pd.to_numeric(d["teamId"]).astype(np.int64),
        "TEAM_NAME": d["teamTricode"], "TEAM_ABBREVIATION": d["teamTricode"], "PERIOD": d["period"].astype(int),
        "MINUTES_REMAINING": clock[0].to_numpy(), "SECONDS_REMAINING": clock[1].to_numpy(),
        "EVENT_TYPE": np.where(made == 1, "Made Shot", "Missed Shot"), "ACTION_TYPE": shot_type,
        "SHOT_TYPE": np.where(d["actionType"] == "3pt", "3PT Field Goal", "2PT Field Goal"),
        "SHOT_ZONE_BASIC": d["area"], "SHOT_ZONE_AREA": detail[1].map(CDN_ZONE_AREAS).to_numpy(),
        "SHOT_ZONE_RANGE": detail[0].map(CDN_ZONE_RANGES).to_numpy(),
        "SHOT_DISTANCE": pd.to_numeric(d["shotDistance"], errors="coerce").fillna(0).astype(int),
        "LOC_X": pd.to_numeric(d["xLegacy"], errors="coerce").fillna(0).astype(int),
        "LOC_Y": pd.to_numeric(d["yLegacy"], errors="coerce").fillna(0).astype(int),
        "SHOT_ATTEMPTED_FLAG": 1, "SHOT_MADE_FLAG": made,
        "GAME_DATE": pd.to_datetime(game_id.map(info["date"])).dt.strftime("%Y%m%d").astype(int).to_numpy(),
        "HTM": game_id.map(info["home"]).to_numpy(), "VTM": game_id.map(info["away"]).to_numpy(),
    })
    return out.reset_index(drop=True)


def load_shots(season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> pd.DataFrame:
    """Shot chart rows (zone, distance, action type, made) for one season."""
    path = download("shotdetail", season, data_dir, playoffs)
    df = pd.read_csv(path, low_memory=False)
    df["season"] = season
    return df
