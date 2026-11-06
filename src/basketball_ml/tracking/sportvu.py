"""Shot features from the raw 2015-16 tracking data.

For about 630 games of the first half of 2015-16, the raw SportVU data was
released: where the ten players and the ball were, 25 times a second. It is
mirrored in linouk23/NBA-Player-Movements, a 7z archive with one JSON file
per game (6 MB compressed, about 110 MB of JSON). Each shot of the shot
charts is found in the positions:

1. **Release.** Each time the ball reaches the rim area (within 3 feet of
   the basket's center, 7.5 to 14 feet high) between 6 seconds before and
   1 second after the scorer's time, the release is the last frame of the 3
   seconds before with the ball within 3 feet of the shooter. The release
   kept is the one closest to 2.1 seconds before the scorer's time, how far
   the scorer's clock usually runs behind. A shot that never reaches the
   rim (most blocked shots and airballs) is released the last time the
   ball was in the shooter's hands before its highest point.
2. **Features at the release** (:data:`RAW_FEATURES`): the closest
   defender's distance, the defenders within 4 and 6 feet, how much the
   closest defender closed in over the last 0.4 seconds, how long the
   shooter had held the ball, his speed and the shot clock.

The shot's distance from the basket in the positions is then within 1.3
feet of the shot chart's (median), and the closest defender's distances
have the spread of the official 2014-15 logs (:mod:`.shotlogs`).
"""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..shots.features import LOCATION_FEATURES

SEASON = 2016                 # 2015-16
LAST_DATE = 20160123          # the released games end here
RAW_URL = ("https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/data/"
           "2016.NBA.Raw.SportVU.Game.Logs/{}.7z")
FPS = 25
BASKETS = np.array([[5.25, 25.0], [88.75, 25.0]])   # feet, on the 94 x 50 court
IN_HAND = 3.0                 # the ball within this many feet of the shooter, on the floor plan
RIM_RADIUS = 3.0              # the ball within this many feet of the basket's center...
RIM_HEIGHT = (7.5, 14.0)      # ...and this high (feet): it reached the rim
LOOKBACK = 3                  # seconds from the release to the rim, at most
LAG = 2.1                     # usual seconds from the release to the scorer's time
WINDOW = (6.0, 1.0)           # seconds before and after the scorer's time to look in
HALF_COURT = 47.0
CLOSING_FRAMES = 10           # 0.4 seconds
SPEED_FRAMES = 5              # 0.2 seconds
FEATURE_COLUMNS = ["release_left", "lag", "sv_dist", "def_dist_sv", "defenders_4", "defenders_6", "closing",
                   "touch", "shooter_speed", "ball_z", "shot_clock_sv", "reached_rim"]
RAW_FEATURES = ["def_dist_sv", "defenders_4", "defenders_6", "closing", "touch", "shooter_speed", "shot_clock_sv"]
RAW_SETS = {
    "kind": [*LOCATION_FEATURES, "kind"],
    "kind + defender": [*LOCATION_FEATURES, "kind", "def_dist_sv"],
    "kind + all raw": [*LOCATION_FEATURES, "kind", *RAW_FEATURES],
}
RAW_NAMES = {
    "kind": "Location + game situation + shot kind",
    "kind + defender": "  + closest defender's distance",
    "kind + all raw": "  + defenders within 4 and 6 ft, closing in, touch time, shooter's speed, shot clock",
}


def archive_name(date: int | str, away: str, home: str) -> str:
    """The archive of one game in the release: 20151027, "DET", "ATL" -> "10.27.2015.DET.at.ATL"."""
    d = str(date)
    return f"{d[4:6]}.{d[6:8]}.{d[:4]}.{away}.at.{home}"


def season_games(raw: pd.DataFrame) -> pd.DataFrame:
    """``game_id`` and archive ``name`` of the 2015-16 games in the release, from shot chart rows."""
    games = raw[raw["GAME_DATE"] <= LAST_DATE].drop_duplicates("GAME_ID").sort_values("GAME_ID")
    names = [archive_name(d, a, h) for d, a, h in zip(games["GAME_DATE"], games["VTM"], games["HTM"])]
    return pd.DataFrame({"game_id": games["GAME_ID"].to_numpy(np.int64), "name": names})


def _loads(body: bytes):
    try:
        import orjson
    except ImportError:
        return json.loads(body)
    return orjson.loads(body)


def load_frames(path: str | Path) -> dict[str, np.ndarray]:
    """Every distinct moment of a game's JSON, in time order.

    ``period``, ``clock`` (seconds left in the period), ``shot_clock``,
    ``ball`` (x, y, height in feet), and for the ten players ``pid``, ``team``
    and ``xy``. Moments without the ball or ten players are skipped.
    """
    data = _loads(Path(path).read_bytes())
    rows, seen = [], set()
    for event in data["events"]:
        for moment in event["moments"]:
            period, stamp, clock, shot_clock, _, positions = moment
            if stamp in seen or len(positions) != 11 or positions[0][0] != -1 or clock is None:
                continue
            seen.add(stamp)
            rows.append((period, stamp, clock, np.nan if shot_clock is None else shot_clock, positions))
    rows.sort(key=lambda r: r[1])
    return {
        "period": np.array([r[0] for r in rows], np.int8),
        "clock": np.array([r[2] for r in rows], np.float32),
        "shot_clock": np.array([r[3] for r in rows], np.float32),
        "ball": np.array([r[4][0][2:5] for r in rows], np.float32).reshape(-1, 3),
        "pid": np.array([[p[1] for p in r[4][1:]] for r in rows], np.int64).reshape(-1, 10),
        "team": np.array([[p[0] for p in r[4][1:]] for r in rows], np.int64).reshape(-1, 10),
        "xy": np.array([[p[2:4] for p in r[4][1:]] for r in rows], np.float32).reshape(-1, 10, 2),
    }


def _starts(mask: np.ndarray) -> np.ndarray:
    """First index of each run of True in ``mask``."""
    idx = np.flatnonzero(mask)
    return idx[np.r_[True, np.diff(idx) > 1]] if len(idx) else idx


def shot_features(frames: dict[str, np.ndarray], shooter: int, period: int, clock_left: float) -> dict | None:
    """Tracking features of one shot at its release (see the module notes), or None if it isn't found.

    ``clock_left`` is the scorer's time of the shot: seconds left in ``period``.
    """
    clock = frames["clock"]
    sel = np.flatnonzero((frames["period"] == period) & (clock <= clock_left + WINDOW[0])
                         & (clock >= clock_left - WINDOW[1]))
    on = frames["pid"][sel] == shooter
    has = on.any(axis=1)
    sel, on = sel[has], on[has]
    if len(sel) < 10:
        return None
    k = on.argmax(axis=1)
    shooter_xy = frames["xy"][sel, k]
    shooter_team = frames["team"][sel, k]
    ball = frames["ball"][sel]
    in_hand = np.hypot(*(ball[:, :2] - shooter_xy).T) < IN_HAND
    best = None
    for b, basket in enumerate(BASKETS):
        to_rim = np.hypot(*(ball[:, :2] - basket).T)
        at_rim = (to_rim < RIM_RADIUS) & (ball[:, 2] > RIM_HEIGHT[0]) & (ball[:, 2] < RIM_HEIGHT[1])
        for s in _starts(at_rim):
            first = max(s - LOOKBACK * FPS, 0)
            held = np.flatnonzero(in_hand[first:s + 1])
            if not len(held):
                continue
            r = first + held[-1]
            if np.hypot(*(shooter_xy[r] - basket)) > HALF_COURT:
                continue
            score = abs(clock[sel[r]] - (clock_left + LAG))
            if best is None or score < best[0]:
                best = (score, r, b, True)
    if best is None:
        top = int(ball[:, 2].argmax())
        held = np.flatnonzero(in_hand[:top + 1])
        if not len(held):
            return None
        r = held[-1]
        best = (None, r, int(np.argmin(np.hypot(*(BASKETS - shooter_xy[r]).T))), False)
    _, r, b, reached = best
    i = sel[r]
    rivals = frames["team"][i] != shooter_team[r]
    if rivals.sum() != 5:
        return None
    gaps = np.hypot(*(frames["xy"][i][rivals] - shooter_xy[r]).T)
    j = max(r - CLOSING_FRAMES, 0)
    rivals_j = frames["team"][sel[j]] != shooter_team[j]
    earlier = (np.hypot(*(frames["xy"][sel[j]][rivals_j] - shooter_xy[j]).T).min()
               if rivals_j.sum() == 5 else np.nan)
    t = r
    while t > 0 and in_hand[t - 1]:
        t -= 1
    back = min(SPEED_FRAMES, r)
    speed = np.hypot(*(shooter_xy[r] - shooter_xy[r - back])) / (back / FPS) if back else np.nan
    return {"release_left": float(clock[i]), "lag": float(clock[i] - clock_left),
            "sv_dist": float(np.hypot(*(shooter_xy[r] - BASKETS[b]))), "def_dist_sv": float(gaps.min()),
            "defenders_4": int((gaps < 4).sum()), "defenders_6": int((gaps < 6).sum()),
            "closing": float(earlier - gaps.min()) if np.isfinite(earlier) else np.nan,
            "touch": float((r - t) / FPS), "shooter_speed": float(speed), "ball_z": float(ball[r, 2]),
            "shot_clock_sv": float(frames["shot_clock"][i]), "reached_rim": int(reached)}


def game_shot_features(frames: dict[str, np.ndarray], game_id: int, shots: pd.DataFrame) -> pd.DataFrame:
    """:func:`shot_features` of each shot chart row of one game, keyed by ``game_id`` and ``game_event_id``."""
    rows = []
    for s in shots.itertuples(index=False):
        found = shot_features(frames, s.PLAYER_ID, s.PERIOD, 60 * s.MINUTES_REMAINING + s.SECONDS_REMAINING)
        if found:
            rows.append({"game_id": game_id, "game_event_id": s.GAME_EVENT_ID, **found})
    return pd.DataFrame(rows, columns=["game_id", "game_event_id", *FEATURE_COLUMNS])


def _extract_json(archive: Path, dest: Path) -> Path:
    try:
        import py7zr
    except ImportError as err:
        raise SystemExit("The raw tracking archives are 7z files: pip install py7zr") from err
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(path=dest)
    found = sorted(dest.rglob("*.json"))
    if not found:
        raise ValueError(f"no JSON file in {archive.name}")
    return found[0]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["game_id", "game_event_id", *FEATURE_COLUMNS])


def game_features(game_id: int, name: str, shots: pd.DataFrame, cache_dir: str | Path) -> pd.DataFrame | None:
    """Features of each shot of one game (``shots``: its shot chart rows), cached as CSV in ``cache_dir``.

    The game's archive is downloaded and read once; None if it isn't in the
    release (some days are missing) or can't be read.
    """
    cache_dir = Path(cache_dir)
    out, missing = cache_dir / f"{game_id}.csv", cache_dir / f"{game_id}.missing"
    if out.exists():
        return _read(out)
    if missing.exists():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        archive = Path(tmp) / "game.7z"
        try:
            with urllib.request.urlopen(RAW_URL.format(name), timeout=300) as resp:
                archive.write_bytes(resp.read())
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            missing.write_text("not in the release\n")
            return None
        try:
            frames = load_frames(_extract_json(archive, Path(tmp) / "json"))
        except (ValueError, KeyError) as err:
            missing.write_text(f"unreadable: {err}\n")
            return None
    table = game_shot_features(frames, game_id, shots)
    table.to_csv(out, index=False)
    return table


def _game_job(job: tuple) -> pd.DataFrame | None:
    return game_features(*job)


def season_features(raw: pd.DataFrame, cache_dir: str | Path = "data/tracking/sportvu", games: int | None = None,
                    workers: int = 1, progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Tracking features of the 2015-16 shots in ``raw`` (shot chart rows), game by game.

    ``games`` limits the run to the first games of the release; each new game
    downloads about 6 MB, and ``workers`` games are read at once (each takes
    about 1 GB of memory while it is read).
    """
    todo = season_games(raw)
    if games:
        todo = todo.head(games)
    by_game = dict(tuple(raw[raw["GAME_ID"].isin(todo["game_id"])].groupby("GAME_ID")))
    jobs = [(int(g.game_id), g.name, by_game.get(g.game_id, raw.iloc[:0]), str(cache_dir))
            for g in todo.itertuples(index=False)]
    parts, done = [], 0
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for table in (pool.map(_game_job, jobs) if pool else map(_game_job, jobs)):
            done += 1
            if table is not None and len(table):
                parts.append(table)
            if progress and (done % 50 == 0 or done == len(jobs)):
                progress(f"  {done}/{len(jobs)} games read, {len(parts)} with tracking")
    finally:
        if pool:
            pool.shutdown()
    if not parts:
        return pd.DataFrame(columns=["game_id", "game_event_id", *FEATURE_COLUMNS])
    return pd.concat(parts, ignore_index=True)
