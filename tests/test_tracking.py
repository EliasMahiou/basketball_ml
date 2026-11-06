import json

import numpy as np
import pandas as pd
import pytest

from basketball_ml.tracking import (by_defender_distance, compare, contests, cross_validate, load_shot_logs,
                                    making_change, match_shots, split_half)
from basketball_ml.tracking import sportvu
from basketball_ml.tracking.shotlogs import elapsed, paired_difference

LOG_COLUMNS = ["GAME_ID", "MATCHUP", "LOCATION", "W", "FINAL_MARGIN", "SHOT_NUMBER", "PERIOD", "GAME_CLOCK",
               "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "SHOT_DIST", "PTS_TYPE", "SHOT_RESULT", "CLOSEST_DEFENDER",
               "CLOSEST_DEFENDER_PLAYER_ID", "CLOSE_DEF_DIST", "FGM", "PTS", "player_name", "player_id"]


def _log_row(game, period, clock, made, value, player=7, defender=9, dist=3.0):
    return [game, "", "H", "W", 1, 1, period, clock, 12.0, 1, -0.4, 20.0, value, "made" if made else "missed",
            "Doe, John", defender, dist, made, value * made, "a b", player]


def test_load_shot_logs_parses_the_clock_and_clips_touch_time(tmp_path):
    path = tmp_path / "logs.csv"
    pd.DataFrame([_log_row(21400001, 1, "1:09", 1, 2), _log_row(21400001, 5, "0:04", 0, 3)],
                 columns=LOG_COLUMNS).to_csv(path, index=False)
    logs = load_shot_logs(path)
    assert logs["seconds_left"].tolist() == [69, 4]
    assert (logs["touch_time"] == 0).all() and logs["value"].tolist() == [2, 3]
    assert elapsed([1, 5], [69, 4]).tolist() == [651, 2880 + 296]


def test_match_shots_pairs_each_log_with_the_next_charted_shot():
    shots = pd.DataFrame({"game_id": 1, "game_event_id": [10, 20, 30, 40], "player_id": 7, "raw_period": 1,
                          "seconds_left": [600, 500, 400, 300], "made": [1, 0, 1, 0], "value": [2, 2, 3, 2]})
    logs = pd.DataFrame({"game_id": 1, "player_id": 7, "period": 1,
                         "seconds_left": [603, 502, 501, 404, 320],    # the last one is too early
                         "made": [1, 0, 0, 0, 0], "value": [2, 2, 2, 3, 2],
                         "def_dist": [1.0, 2.0, 3.0, 4.0, 5.0], "defender_id": 9, "defender": "Doe, John",
                         "shot_clock": 10.0, "dribbles": 0.0, "touch_time": 1.0, "tracked_distance": 10.0})
    out = match_shots(shots, logs)
    assert out["game_event_id"].tolist() == [10, 20]       # shot 30's log disagrees on the result
    assert out["def_dist"].tolist() == [1.0, 3.0]          # shot 20 keeps the closer of its two logs


def _synthetic_shots(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0, 28, n)
    defender = rng.uniform(0, 10, n)
    p = 1 / (1 + np.exp(-(0.6 - 0.07 * dist + 0.25 * (defender - 4))))
    zones = np.where(dist < 8, "Restricted Area", "Mid-Range")
    return pd.DataFrame({
        "game_id": rng.integers(0, 300, n), "player_id": rng.integers(0, 40, n), "player": "p",
        "made": (rng.uniform(size=n) < p).astype(np.int8), "value": np.where(dist > 23.75, 3, 2),
        "distance": dist, "loc_x": rng.normal(0, 100, n), "loc_y": dist * 10, "angle": rng.uniform(-90, 90, n),
        "is_three": (dist > 23.75).astype(np.int8), "zone_basic": zones, "zone_area": "Center(C)",
        "zone_range": zones, "period": rng.integers(1, 5, n), "seconds_left": rng.integers(0, 720, n),
        "heave": 0, "margin": rng.normal(0, 8, n), "kind": "jump", "action": "jump shot",
        "def_dist": defender, "shot_clock": rng.uniform(0, 24, n), "dribbles": rng.integers(0, 5, n).astype(float),
        "touch_time": rng.uniform(0, 6, n), "defender_id": rng.integers(100, 110, n), "defender": "Doe, John",
    })


def test_tracking_features_improve_xfg_when_defense_matters():
    shots = _synthetic_shots()
    sets = {"kind": ["distance", "is_three", "kind"], "kind + defender": ["distance", "is_three", "kind", "def_dist"]}
    oof = cross_validate(shots, sets, folds=2)
    table = compare(shots, oof)
    assert table.loc["kind + defender", "log_loss"] < table.loc["kind", "log_loss"] - 0.01
    diff, lo, hi = paired_difference(shots, oof, "kind + defender", "kind", n_boot=200)
    assert hi < 0 and lo <= diff <= hi
    table = by_defender_distance(shots, oof["kind"].to_numpy())
    twos = table.loc["two"]
    assert twos.loc["6+ ft", "above"] > 0 > twos.loc["0-2 ft", "above"]


def test_player_and_defender_tables():
    shots = _synthetic_shots()
    stopper = shots["defender_id"] == 100
    shots.loc[stopper & (shots["def_dist"] < 6), "made"] = 0
    expected = np.full(len(shots), shots["made"].mean())
    table = contests(shots, expected, min_shots=50)
    assert table.index[0] == 100 and table["defender"].iloc[0] == "John Doe"
    change = making_change(shots, expected, expected, min_shots=50)
    assert np.allclose(change["change"], 0)
    assert -1 <= split_half(shots, shots["made"].to_numpy() - expected, min_shots=20) <= 1


def _court(n: int, ball: np.ndarray, shooter=(20.0, 25.0), on_court=True) -> dict:
    """``n`` frames of one possession: shooter 7 and four teammates (team 1), five defenders (team 2)."""
    players = [shooter, (30, 10), (30, 40), (40, 15), (40, 35),          # offense
               (23, 25), (20, 29), (60, 10), (60, 40), (70, 25)]          # defense: 3 ft and 4 ft from the shooter
    pid = [7 if on_court else 8, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    return {"period": np.ones(n, np.int8), "clock": (600 - np.arange(n) / sportvu.FPS).astype(np.float32),
            "shot_clock": np.full(n, 14.0, np.float32), "ball": ball.astype(np.float32),
            "pid": np.tile(pid, (n, 1)), "team": np.tile([1] * 5 + [2] * 5, (n, 1)),
            "xy": np.tile(np.array(players, np.float32), (n, 1, 1))}


def _flight(start, end, frames):
    t = np.arange(1, frames + 1)[:, None] / frames
    return np.asarray(start) + t * (np.asarray(end) - np.asarray(start))


def test_sportvu_finds_the_release_before_the_ball_reaches_the_rim():
    held = np.tile([20.0, 25.0, 7.0], (50, 1))                       # 2 s in the shooter's hands
    flight = _flight([20.0, 25.0, 7.0], [5.25, 25.0, 10.0], 25)       # 1 s to the left rim
    under = np.tile([5.25, 25.0, 3.0], (25, 1))
    frames = _court(100, np.vstack([held, flight, under]))
    f = sportvu.shot_features(frames, shooter=7, period=1, clock_left=596)
    # The ball counts as in hand while within 3 ft of the shooter: the first 5 frames of the flight.
    assert f["release_left"] == pytest.approx(600 - 54 / 25)
    assert f["touch"] == pytest.approx(54 / 25)
    assert f["sv_dist"] == pytest.approx(14.75)
    assert f["def_dist_sv"] == pytest.approx(3.0)
    assert (f["defenders_4"], f["defenders_6"], f["reached_rim"]) == (1, 2, 1)
    assert f["closing"] == pytest.approx(0.0) and f["shooter_speed"] == pytest.approx(0.0)
    assert f["shot_clock_sv"] == pytest.approx(14.0)
    assert sportvu.shot_features(_court(100, np.vstack([held, flight, under]), on_court=False), 7, 1, 596) is None


def test_sportvu_releases_a_short_shot_before_its_highest_point():
    held = np.tile([20.0, 25.0, 7.0], (50, 1))
    up = _flight([20.0, 25.0, 7.0], [15.0, 25.0, 12.0], 10)           # blocked: it never gets near the rim
    down = _flight([15.0, 25.0, 12.0], [25.0, 30.0, 1.0], 40)
    f = sportvu.shot_features(_court(100, np.vstack([held, up, down])), shooter=7, period=1, clock_left=596)
    assert f["reached_rim"] == 0
    assert f["release_left"] == pytest.approx(600 - 54 / 25)       # last frame within 3 ft before the top
    assert f["sv_dist"] == pytest.approx(14.75)                      # the closer basket


def test_sportvu_reads_moments_once_and_names_archives(tmp_path):
    def moment(stamp, clock, ball=True):
        players = [[1, 7, 20.0, 25.0, 0.0]] * 5 + [[2, 11, 23.0, 25.0, 0.0]] * 5
        return [1, stamp, clock, 14.0, None, ([[-1, -1, 20.0, 25.0, 7.0]] if ball else []) + players]
    events = {"events": [{"moments": [moment(2, 599.96), moment(1, 600.0)]},
                         {"moments": [moment(2, 599.96), moment(3, 599.92, ball=False)]}]}
    path = tmp_path / "game.json"
    path.write_text(json.dumps(events))
    frames = sportvu.load_frames(path)
    assert frames["clock"].tolist() == pytest.approx([600.0, 599.96])   # in time order, duplicate and no-ball dropped
    assert frames["xy"].shape == (2, 10, 2) and frames["ball"][0].tolist() == [20.0, 25.0, 7.0]
    assert sportvu.archive_name(20151027, "DET", "ATL") == "10.27.2015.DET.at.ATL"
    raw = pd.DataFrame({"GAME_ID": [21500002, 21500001, 21500001, 21500900], "VTM": ["CLE", "DET", "DET", "BOS"],
                        "HTM": ["CHI", "ATL", "ATL", "NYK"], "GAME_DATE": [20151027, 20151027, 20151027, 20160301]})
    assert sportvu.season_games(raw)["name"].tolist() == ["10.27.2015.DET.at.ATL", "10.27.2015.CLE.at.CHI"]
