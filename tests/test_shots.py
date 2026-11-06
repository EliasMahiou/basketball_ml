import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from basketball_ml.shots import (ShotModel, ZoneRate, arena_bias, calibration, consecutive, court_map,
                                 efficiency_shares, fill_zones, normalize_action, player_seasons, predict_next,
                                 prepare_shots, recenter, score_before, shot_kind, team_quality, walk_forward,
                                 year_to_year)
from basketball_ml.shots.making import ARENA_PRIOR_GAMES, CARRY_MIN_SHOTS


TEAMS = {100: "AAA", 200: "BBB", 300: "CCC"}
GAMES = [(100, 200), (200, 300), (300, 100)]   # (home, away), in rotation


def _raw_shots(n=3000, season=2001, seed=0, skill=None):
    """Synthetic shot chart rows: three teams of three players; makes depend on distance (and ``skill``).

    Players 1-3 play for team 100, 4-6 for 200 and 7-9 for 300.
    """
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0, 30, n)
    angle = np.radians(rng.uniform(-80, 80, n))
    x, y = np.round(10 * distance * np.sin(angle)), np.round(10 * distance * np.cos(angle))
    three = distance > 23.75
    game = np.arange(n) // 100
    home, away = (np.array([GAMES[g % 3][i] for g in game]) for i in (0, 1))
    team = np.where(rng.random(n) < 0.5, home, away)
    player = (team // 100 - 1) * 3 + rng.integers(1, 4, n)
    p = np.where(distance < 4, 0.65, np.where(three, 0.36, 0.42))
    if skill is not None:
        p = np.clip(p + np.vectorize(skill.get)(player), 0.01, 0.99)
    return pd.DataFrame({
        "season": season, "GAME_ID": 1000 * season + game, "GAME_EVENT_ID": np.arange(n) % 100 + 1,
        "PLAYER_ID": player, "PLAYER_NAME": [f"Player {i}" for i in player], "TEAM_ID": team,
        "HTM": [TEAMS[t] for t in home], "VTM": [TEAMS[t] for t in away],
        "PERIOD": rng.integers(1, 5, n), "MINUTES_REMAINING": rng.integers(0, 12, n),
        "SECONDS_REMAINING": rng.integers(0, 60, n),
        "ACTION_TYPE": np.where(distance < 4, "Driving Layup Shot", "Jump Shot"),
        "SHOT_TYPE": np.where(three, "3PT Field Goal", "2PT Field Goal"),
        "SHOT_ZONE_BASIC": np.where(distance < 4, "Restricted Area", np.where(three, "Above the Break 3", "Mid-Range")),
        "SHOT_ZONE_AREA": "Center(C)", "SHOT_ZONE_RANGE": np.where(distance < 8, "Less Than 8 ft.", "24+ ft."),
        "LOC_X": x, "LOC_Y": y, "SHOT_MADE_FLAG": (rng.random(n) < p).astype(int)})


def test_normalize_action_drops_bank_and_merges_renamed_labels():
    raw = pd.Series(["Jump Bank Shot", "Driving Floating Bank Jump Shot", "Turnaround Bank shot",
                     "Turnaround Fadeaway shot", "Slam Dunk Shot", "Tip Shot", "Jump Hook Shot", "Pullup Jump shot",
                     "No Shot", None])
    assert normalize_action(raw).tolist() == [
        "jump shot", "driving floating jump shot", "turnaround jump shot", "turnaround fadeaway jump shot",
        "dunk shot", "tip layup shot", "hook shot", "pullup jump shot", "no shot", "unknown"]
    kinds = shot_kind(pd.Series(["cutting dunk shot", "tip layup shot", "driving floating jump shot",
                                 "turnaround hook shot", "driving finger roll layup shot"]))
    assert kinds.tolist() == ["dunk", "tip", "jump", "hook", "layup"]


def test_score_before_leaves_out_the_shots_own_points():
    events = pd.DataFrame({
        "GAME_ID": [1] * 6 + [2] * 2, "EVENTNUM": [1, 2, 3, 4, 5, 6, 1, 2],
        "EVENTMSGTYPE": [12, 1, 2, 4, 1, 2, 12, 2],
        "SCORE": [None, "0 - 2", None, None, "3 - 2", None, None, None]})
    out = score_before(events).set_index(["GAME_ID", "GAME_EVENT_ID"])["home_margin"]
    # Home made a 2, away missed, away made a 3, home missed; a new game starts level.
    assert out.to_dict() == {(1, 2): 0.0, (1, 3): 2.0, (1, 5): 2.0, (1, 6): -1.0, (2, 2): 0.0}


def test_prepare_shots_builds_features_and_the_shooters_margin():
    raw = _raw_shots(n=600)
    # Game 0 is team 100 (AAA) at home against team 200 (BBB).
    raw.loc[0, ["LOC_X", "LOC_Y", "MINUTES_REMAINING", "SECONDS_REMAINING", "TEAM_ID"]] = [0, 400, 0, 1, 100]
    raw.loc[1, "TEAM_ID"] = 200
    margins = pd.DataFrame({"GAME_ID": raw["GAME_ID"], "GAME_EVENT_ID": raw["GAME_EVENT_ID"], "home_margin": 5.0})
    shots = prepare_shots(raw, margins)
    assert (shots["team"] == raw["TEAM_ID"].map(TEAMS)).all()
    assert (shots.loc[:1, "opponent"] == ["BBB", "AAA"]).all() and (shots["opponent"] != shots["team"]).all()
    assert shots.loc[0, "distance"] == 40.0 and shots.loc[0, "heave"] == 1 and shots.loc[0, "seconds_left"] == 1
    assert shots.loc[0, "margin"] == 5.0 and shots.loc[1, "margin"] == -5.0
    assert ((shots["value"] == 3) == (raw["SHOT_TYPE"] == "3PT Field Goal")).all()
    assert prepare_shots(raw)["margin"].isna().all()


class _Spy:
    """Records the seasons it is trained on."""
    seen: dict = {}

    def fit(self, shots):
        self.train = sorted(shots["season"].unique())
        return self

    def predict(self, shots):
        _Spy.seen[int(shots["season"].iloc[0])] = self.train
        return np.full(len(shots), 0.5)


def test_fill_zones_takes_the_zones_of_the_same_or_nearest_spot():
    reference = prepare_shots(_raw_shots(n=3000))
    shots = prepare_shots(_raw_shots(n=200, seed=1))
    blank = shots.assign(zone_basic=None, zone_area=None, zone_range=None)
    # A spot no reference shot came from: just off one that did.
    spot = reference[reference["is_three"] == 0].iloc[0]
    blank.loc[0, ["loc_x", "loc_y", "is_three"]] = [spot["loc_x"] + 1, spot["loc_y"], 0]
    filled = fill_zones(blank, reference)
    assert filled.loc[0, "zone_basic"] == spot["zone_basic"] and filled.loc[0, "zone_range"] == spot["zone_range"]
    # In the synthetic shots, zones follow from distance and value, as on a real court.
    rest = filled.iloc[1:]
    assert (rest["zone_basic"].to_numpy() == shots["zone_basic"].iloc[1:].to_numpy()).mean() > 0.95
    assert fill_zones(shots, reference) is shots


def test_walk_forward_trains_on_the_three_seasons_before():
    shots = pd.concat([prepare_shots(_raw_shots(n=300, season=s, seed=s)) for s in range(2001, 2007)])
    out = walk_forward(shots, range(2001, 2007), models={"spy": _Spy})
    assert sorted(out["season"].unique()) == [2004, 2005, 2006]
    assert _Spy.seen == {2004: [2001, 2002, 2003], 2005: [2002, 2003, 2004], 2006: [2003, 2004, 2005]}
    assert "xfg_spy" in out and len(out) == 900


def test_shot_model_learns_distance_and_groups_rare_labels():
    shots = prepare_shots(_raw_shots(n=4000))
    shots.loc[:9, "action"] = "rare label"
    test = prepare_shots(_raw_shots(n=1000, seed=1))
    with threadpool_limits(limits=1):   # a small fit; extra threads only slow it down on a busy machine
        model = ShotModel(min_action_shots=50, max_iter=40, early_stopping=False).fit(shots)
        p = model.predict(test)
        unseen = model.predict(test.assign(action="label never seen"))
    assert "rare label" not in model.categories_["action"] and "other" in model.categories_["action"]
    rim = test["distance"] < 4
    assert 0.55 < p[rim].mean() < 0.75 and 0.3 < p[~rim].mean() < 0.5
    assert np.isfinite(unseen).all() and ((unseen > 0) & (unseen < 1)).all()
    zone = ZoneRate().fit(shots).predict(test)
    assert abs(zone[rim].mean() - 0.65) < 0.05
    cal = calibration(test.assign(xfg=p))
    assert len(cal) == 10 and cal["shots"].sum() == len(test)


def _predictions(skill, seasons=(2001, 2002), n=6000):
    """Shots with a perfect shot-difficulty model as ``xfg``; shooters add ``skill`` to it."""
    parts = []
    for s in seasons:
        shots = prepare_shots(_raw_shots(n=n, season=s, seed=s, skill=skill))
        shots["xfg"] = np.where(shots["distance"] < 4, 0.65, np.where(shots["value"] == 3, 0.36, 0.42))
        parts.append(shots)
    return pd.concat(parts, ignore_index=True)


def test_shot_making_rewards_the_better_shooter_and_shrinks_small_samples():
    skill = {i: 0.0 for i in range(1, 10)} | {1: 0.15, 5: -0.15}
    shots = _predictions(skill)
    adjusted = recenter(shots)
    for _, g in shots.assign(p=adjusted).groupby(["season", "value"]):
        assert abs(g["made"].mean() - g["p"].mean()) < 1e-9
    table = player_seasons(shots, prior_min_shots=100).set_index(["season", "player_id"])
    best, worst = table.xs(1, level="player_id"), table.xs(5, level="player_id")
    assert (best["per_100"] > 10).all() and (worst["per_100"] < -10).all()
    assert (best["per_100_shrunk"].abs() <= best["per_100"].abs()).all()
    assert np.allclose(table["points_added"], table["pts"] - table["xpts"])
    assert table.attrs["tau"] > 0
    few = player_seasons(shots[(shots["player_id"] != 1) | (shots.index % 50 == 0)], prior_min_shots=100)
    small = few[few["player_id"] == 1].iloc[0]
    assert small["fga"] < 100 and abs(small["per_100_shrunk"]) < 0.6 * abs(small["per_100"])


def test_arena_bias_finds_and_removes_a_scorekeeper_effect():
    shots = _predictions({i: 0.0 for i in range(1, 10)}, seasons=(2001,), n=9000)
    arena = np.where(shots["home"] == 1, shots["team"], shots["opponent"])
    # AAA's scorekeeper logs twos too far out: they look 10 points harder than they are, for both teams.
    biased = (arena == "AAA") & (shots["value"] == 2)
    shots.loc[biased, "xfg"] -= 0.10
    bias = arena_bias(shots, recenter(shots, arenas=False))
    by_arena = bias[shots["value"] == 2].groupby(arena[shots["value"] == 2]).mean()
    # With three teams all of BBB's road games are at AAA, so BBB gets the mirror image
    # (with 30 teams that spillover is about 1/29 of the bias).
    assert np.allclose(by_arena[["AAA", "BBB", "CCC"]], [0.10, -0.10, 0.0], atol=0.025)
    assert (bias[shots["value"] == 3] == 0).all()
    fixed = (shots["made"] - recenter(shots))[biased].mean()
    unfixed = (shots["made"] - recenter(shots, arenas=False))[biased].mean()
    assert unfixed > 0.05 and abs(fixed) < 0.5 * unfixed


def test_a_season_in_progress_leans_on_last_seasons_arena_bias():
    shots = _predictions({i: 0.0 for i in range(1, 10)}, seasons=(2001, 2002), n=9000)
    arena = pd.Series(np.where(shots["home"] == 1, shots["team"], shots["opponent"]), index=shots.index)
    two = shots["value"] == 2
    # AAA's scorekeeper was off last season only.
    shots.loc[(arena == "AAA") & two & (shots["season"] == 2001), "xfg"] -= 0.10
    p = recenter(shots, arenas=False)
    own, blended = arena_bias(shots, p), arena_bias(shots, p, in_progress={2002})
    last_season = shots["season"] == 2001
    assert np.allclose(own[last_season], blended[last_season])
    by_arena = {name: b[two].groupby([shots["season"][two], arena[two]]).mean() for name, b in
                (("own", own), ("blended", blended))}
    games = shots.loc[~last_season, "game_id"].nunique()
    weight = games / (games + ARENA_PRIOR_GAMES)
    expected = weight * by_arena["own"].loc[2002] + (1 - weight) * by_arena["own"].loc[2001]
    assert np.allclose(by_arena["blended"].loc[2002], expected - expected.mean())
    assert by_arena["blended"].loc[(2002, "AAA")] > 0.06 > abs(by_arena["own"].loc[(2002, "AAA")])


def test_shot_making_of_a_season_in_progress_starts_from_last_season():
    skill = {i: 0.0 for i in range(1, 10)} | {1: 0.12, 5: -0.12}
    past = _predictions(skill, seasons=(2001, 2002, 2003))
    # Player 9 sits out 2003, so he has no last season to start from.
    past = past[(past["player_id"] != 9) | (past["season"] != 2003)]
    shots = pd.concat([past, _predictions(skill, seasons=(2004,), n=900)], ignore_index=True)
    table = player_seasons(shots, prior_min_shots=100, in_progress={2004})
    assert 0.5 < table.attrs["rho"] < 1.5 and 0 < table.attrs["s"] < table.attrs["tau"]
    now = table[table["season"] == 2004].set_index("player_id")
    before = table[table["season"] == 2003].set_index("player_id")["per_100_shrunk"]
    for player in (1, 5):
        # Between what last season predicts and this season's own making, much closer to the former.
        start, own = table.attrs["rho"] * before[player], now.loc[player, "per_100"]
        assert min(start, own) <= now.loc[player, "per_100_skill"] <= max(start, own)
        assert abs(now.loc[player, "per_100_skill"] - start) < abs(now.loc[player, "per_100_skill"] - own)
    assert now.loc[1, "per_100_skill"] > 10 > -10 > now.loc[5, "per_100_skill"]
    assert now.loc[9, "per_100_skill"] == now.loc[9, "per_100_shrunk"]
    assert (table.loc[table["season"] == 2001, "per_100_skill"] == table.loc[table["season"] == 2001,
                                                                             "per_100_shrunk"]).all()
    assert CARRY_MIN_SHOTS <= 200


def test_team_quality_splits_points_per_shot_into_quality_and_making():
    shots = _predictions({i: 0.0 for i in range(1, 10)} | {2: 0.2}, seasons=(2001, 2002, 2003))
    teams = team_quality(shots)
    assert set(teams["team"]) == {"AAA", "BBB", "CCC"}
    assert np.allclose(teams["off_pps"], teams["off_xpps"] + teams["off_making"])
    # Player 2 is on AAA: its shots go in more often than expected, from the same spots as everyone's.
    aaa = teams[teams["team"] == "AAA"]
    assert (aaa["off_making"] > 0.03).all() and np.allclose(aaa["off_xpps"], aaa["def_xpps"], atol=0.03)
    shares = efficiency_shares(teams)
    assert np.allclose(shares["quality_share"] + shares["making_share"], 1.0)


def test_persistence_detects_a_skill_that_carries_over():
    rng = np.random.default_rng(3)
    skill = dict(zip(range(1, 10), rng.normal(0, 0.06, 9)))
    players = player_seasons(_predictions(skill, seasons=(2001, 2002, 2003, 2004), n=8000), prior_min_shots=100)
    pairs = consecutive(players, min_shots=300)
    assert len(pairs) == 27 and set(pairs["season"]) == {2001, 2002, 2003}
    y2y = year_to_year(pairs, ["fg_above", "xfg_pct"])
    assert y2y.loc["fg_above", "r"] > 0.6
    result = predict_next(pairs, "fg_above", {"skill": ["fg_above"]})
    assert result.loc["skill", "r"] > 0.5


def test_court_map_saves_a_png(tmp_path):
    shots = prepare_shots(_raw_shots(n=2000)).assign(xfg=0.45)
    out = court_map(shots, tmp_path / "court.png", min_shots=1, title="test")
    assert out.exists() and out.stat().st_size > 10_000
