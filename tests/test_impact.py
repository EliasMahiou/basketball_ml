import pytest
import numpy as np
import pandas as pd

from basketball_ml.impact import fit_rapm, game_stints, player_columns, player_seconds, predict_margin
from basketball_ml.predictions import lineup_rapm

HOME = [1, 2, 3, 4, 5]
AWAY = [11, 12, 13, 14, 15]


def _event(period, kind, clock, p1=0, t1=None, p2=0, t2=None, p3=0, t3=None, score=None, home_text=None,
           away_text=None, action=0):
    return {"PERIOD": period, "EVENTMSGTYPE": kind, "EVENTMSGACTIONTYPE": action, "PCTIMESTRING": clock,
            "SCORE": score, "HOMEDESCRIPTION": home_text, "VISITORDESCRIPTION": away_text,
            "PLAYER1_ID": p1, "PLAYER1_TEAM_ABBREVIATION": t1, "PLAYER2_ID": p2,
            "PLAYER2_TEAM_ABBREVIATION": t2, "PLAYER3_ID": p3, "PLAYER3_TEAM_ABBREVIATION": t3}


def _game():
    """One period: every starter touches the ball, then player 6 replaces player 5."""
    ev = [_event(1, 12, "12:00"), _event(1, 10, "12:00", 1, "HHH", 11, "AAA", 2, "HHH")]
    for i, (h, a) in enumerate(zip(HOME[1:], AWAY[1:])):
        ev.append(_event(1, 2, f"11:{50 - i * 10}", h, "HHH", home_text="MISS shot"))
        ev.append(_event(1, 4, f"11:{48 - i * 10}", a, "AAA", away_text="REBOUND"))
    ev.append(_event(1, 1, "10:30", 5, "HHH", score="0 - 2", home_text="layup"))
    ev.append(_event(1, 8, "10:00", 5, "HHH", 6, "HHH", home_text="SUB: 6 FOR 5"))
    ev.append(_event(1, 1, "9:30", 11, "AAA", score="3 - 2", away_text="3PT jumper"))
    ev.append(_event(1, 1, "9:00", 6, "HHH", score="3 - 4", home_text="dunk"))
    ev.append(_event(1, 13, "0:00"))
    return pd.DataFrame(ev)


def test_game_stints_tracks_starters_and_substitutions():
    stints, used, skipped = game_stints(_game(), "HHH")
    assert (used, skipped) == (1, 0)
    assert len(stints) == 2
    first, second = stints
    assert first["home_players"] == (1, 2, 3, 4, 5) and first["away_players"] == (11, 12, 13, 14, 15)
    assert second["home_players"] == (1, 2, 3, 4, 6)
    assert (first["home_pts"], first["away_pts"]) == (2, 0)
    assert (second["home_pts"], second["away_pts"]) == (2, 3)
    assert first["home_fga"] == 5 and first["start"] - first["end"] == 120


def _synthetic_stints(n=4000, seed=0):
    """Random 5-on-5 stints among 30 players; player 0 adds 10 points per 100 possessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ids = rng.choice(30, 10, replace=False)
        home, away = sorted(ids[:5]), sorted(ids[5:])
        effect = 10.0 * ((0 in home) - (0 in away))
        poss = 10.0
        home_pts = rng.normal(110 + effect / 2, 30) * poss / 100
        away_pts = rng.normal(110 - effect / 2, 30) * poss / 100
        rows.append({**dict(zip(player_columns("home"), home)), **dict(zip(player_columns("away"), away)),
                     "home_pts": home_pts, "away_pts": away_pts, "home_poss": poss, "away_poss": poss,
                     "seconds": 120, "game_id": len(rows) // 20, "season": 2020})
    return pd.DataFrame(rows)


def test_rapm_finds_the_player_who_moves_the_score():
    stints = _synthetic_stints()
    ratings = fit_rapm(stints, alpha=100).sort_values("rapm", ascending=False)
    assert ratings.iloc[0]["player_id"] == 0
    assert 5 < ratings.iloc[0]["rapm"] < 15
    assert ratings.iloc[1:]["rapm"].abs().max() < 4
    predicted = predict_margin(stints, ratings, default=0.0)
    actual = 100 * (stints["home_pts"] - stints["away_pts"]) / stints["home_poss"]
    assert np.corrcoef(predicted, actual)[0, 1] > 0.1


def test_player_seconds_and_lineup_rapm():
    stints = _synthetic_stints(n=40)
    seconds = player_seconds(stints)
    assert seconds["seconds"].sum() == 10 * 120 * 40
    assert lineup_rapm({1: 600, 2: 1800}, {1: 4.0, 2: 0.0}, default=-1.5) == 5.0
    assert lineup_rapm({1: 600, 3: 600}, {1: 4.0}, default=-2.0) == 5.0


def test_add_impact_uses_the_latest_ratings_dated_before_each_game():
    from basketball_ml.predictions import add_impact

    games = pd.DataFrame({"game_id": [1, 2], "season": 2020, "home": "AAA", "away": "BBB",
                          "date": pd.to_datetime(["2020-01-01", "2020-02-01"])})
    seconds = pd.DataFrame({"game_id": [1, 1, 2, 2], "side": ["home", "away"] * 2,
                            "player_id": [7, 8, 7, 8], "seconds": [2400.0] * 4})
    ratings = pd.DataFrame({"season": 2020, "player_id": [7, 8, 7, 8], "rapm": [1.0, 0.0, 3.0, 0.0],
                            "as_of": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-15"] * 2)})
    out = add_impact(games, seconds, ratings)
    assert out["rapm_diff"].tolist() == [5.0, 15.0]


def test_rapm_shrinks_toward_the_box_score_prior():
    from basketball_ml.impact import box_prior

    stints = _synthetic_stints(n=600)
    # Player 0 is the strong one and also the most productive in the box score.
    rows = [{"season": 2020, "game_id": g, "player_id": pid, "production": 30.0 if pid == 0 else 10.0}
            for g in stints["game_id"].unique() for pid in range(30)]
    prior = box_prior(stints, pd.DataFrame(rows))
    assert prior.idxmax() == 0
    heavy = 1e7   # so much shrinkage that the ratings are the prior itself
    plain = fit_rapm(stints, alpha=heavy).set_index("player_id")["rapm"]
    shrunk = fit_rapm(stints, alpha=heavy, prior=prior).set_index("player_id")
    assert plain.abs().max() < 0.01
    assert np.allclose(shrunk["rapm"], prior.reindex(shrunk.index), atol=0.01)
    assert np.allclose(shrunk["orapm"], shrunk["drapm"], atol=0.01)


def test_box_scores_split_rebounds_and_skip_technicals():
    from basketball_ml.impact.box import box_scores

    ev = pd.DataFrame([
        _event(1, 2, "11:50", 1, "HHH", p3=11, t3="AAA", home_text="MISS 3PT jumper"),
        _event(1, 4, "11:48", 2, "HHH", home_text="REBOUND"),             # offensive: same team as the miss
        _event(1, 1, "11:40", 2, "HHH", 1, "HHH", score="0 - 2", home_text="layup (1 AST)"),
        _event(1, 2, "11:20", 12, "AAA", away_text="MISS jumper"),
        _event(1, 4, "11:18", 3, "HHH", home_text="REBOUND"),             # defensive
        _event(1, 5, "11:10", 3, "HHH", 13, "AAA", home_text="bad pass", away_text="STEAL"),
        _event(1, 6, "11:00", 13, "AAA", 3, "HHH", action=1, away_text="P.FOUL"),
        _event(1, 6, "11:00", 14, "AAA", action=11, away_text="T.FOUL"),  # technical: not a personal foul
        _event(1, 3, "11:00", 3, "HHH", score="0 - 3", home_text="Free Throw 1 of 2"),
        _event(1, 3, "11:00", 3, "HHH", home_text="MISS Free Throw 2 of 2"),
    ]).assign(GAME_ID=1)
    box = box_scores(ev).set_index("player_id")
    assert box.loc[1, ["fg3a", "fg3m", "ast", "pts"]].tolist() == [1, 0, 1, 0]
    assert box.loc[2, ["oreb", "dreb", "fg2m", "pts"]].tolist() == [1, 0, 1, 2]
    assert box.loc[3, ["dreb", "tov", "pfd", "fta", "ftm", "pts"]].tolist() == [1, 1, 1, 2, 1, 1]
    assert box.loc[11, "blk"] == 1 and box.loc[13, ["stl", "pf"]].tolist() == [1, 1]
    assert 14 not in box.index or box.loc[14, "pf"] == 0


def test_rapm_shrinks_toward_separate_offense_and_defense_priors():
    from basketball_ml.impact import BOX_STATS, box_prior_od

    stints = _synthetic_stints(n=600)
    # Player 0 is the strong one; he also blocks many more shots than anyone.
    rng = np.random.default_rng(1)
    rows = []
    for g in stints["game_id"].unique():
        for pid in range(30):
            row = {"season": 2020, "game_id": g, "team": "T", "player_id": pid}
            row.update({k: float(rng.poisson(2)) for k in BOX_STATS})
            row["blk"] = 12.0 if pid == 0 else float(rng.poisson(1))
            rows.append(row)
    prior = box_prior_od(stints, pd.DataFrame(rows))
    assert set(prior.columns) == {"orapm", "drapm"}
    assert (prior["orapm"] + prior["drapm"]).idxmax() == 0
    heavy = 1e7
    shrunk = fit_rapm(stints, alpha=heavy, prior=prior).set_index("player_id")
    assert np.allclose(shrunk["orapm"], prior["orapm"].reindex(shrunk.index), atol=0.01)
    assert np.allclose(shrunk["drapm"], prior["drapm"].reindex(shrunk.index), atol=0.01)


def test_luck_adjusted_rapm_ignores_opponent_three_point_luck():
    stints = _synthetic_stints(n=2000, seed=3)
    rng = np.random.default_rng(4)
    for side in ("home", "away"):
        stints[f"{side}_fg3a"] = 3.0
        stints[f"{side}_xfg3m"] = 1.08
        stints[f"{side}_fg3m"] = rng.binomial(3, 0.36, len(stints)).astype(float)
        stints[f"{side}_fta"] = 2.0
        stints[f"{side}_xftm"] = 1.5
        stints[f"{side}_ftm"] = rng.binomial(2, 0.75, len(stints)).astype(float)
    # Player 1's opponents get hot from three: 3 extra makes per stint that only the luck adjustment removes.
    for side, other in (("home", "away"), ("away", "home")):
        on = (stints[player_columns(side)] == 1).any(axis=1)
        stints.loc[on, f"{other}_fg3m"] += 3
        stints.loc[on, f"{other}_pts"] += 9
    plain = fit_rapm(stints, alpha=100).set_index("player_id")
    luck = fit_rapm(stints, alpha=100, luck=True).set_index("player_id")
    assert plain.loc[1, "drapm"] < -20
    assert abs(luck.loc[1, "drapm"]) < abs(plain.loc[1, "drapm"]) / 3
    assert np.isclose(luck.loc[0, "orapm"], plain.loc[0, "orapm"], atol=2.0)


def test_projection_learns_that_high_picks_and_rotation_players_are_better():
    from basketball_ml.impact import fit_projection, project

    rng = np.random.default_rng(5)
    bio = pd.DataFrame({"birth": pd.to_datetime("1998-06-01"), "height": 78.0,
                        "draft_year": 2019.0, "pick": np.arange(1, 31, dtype=float)},
                       index=pd.Index(range(30), name="player_id"))
    # Players 0-14 are rated before both seasons; 15-29 are newcomers whose value falls with the draft pick.
    rated = pd.DataFrame({"player_id": range(15), "rapm": 0.0, "orapm": 0.0, "drapm": 0.0,
                          "possessions": 4000.0})
    truth = {pid: (4.0 - 0.3 * pid if pid >= 15 else 0.0) for pid in range(30)}
    rows = []
    for season in (2021, 2022):
        for _ in range(3000):
            ids = rng.choice(30, 10, replace=False)
            home, away = sorted(ids[:5]), sorted(ids[5:])
            effect = sum(truth[p] for p in home) - sum(truth[p] for p in away)
            rows.append({**dict(zip(player_columns("home"), home)), **dict(zip(player_columns("away"), away)),
                         "home_pts": rng.normal(110 + effect / 2, 20) / 10, "away_pts": rng.normal(110 - effect / 2, 20) / 10,
                         "home_poss": 10.0, "away_poss": 10.0, "season": season})
    stints = pd.DataFrame(rows)
    coef = fit_projection(stints, {2021: rated, 2022: rated}, bio, [2021, 2022])
    assert coef["new_log_pick"] < -1
    out = project(rated, 2022, bio, coef, players=range(30)).set_index("player_id")
    assert len(out) == 30 and out.loc[15, "rapm"] > out.loc[29, "rapm"] + 2

    # The same projection, learned from earlier seasons only, added to every in-season refresh.
    from basketball_ml.impact import project_schedule, projection_coefficients

    coefficients = projection_coefficients(stints, {2021: rated, 2022: rated}, bio, [2021, 2022])
    assert list(coefficients.index) == [2022]
    schedule = pd.concat([rated.assign(season=s, as_of=pd.Timestamp(f"{s - 1}-11-01") + pd.Timedelta(days=d))
                          for s in (2021, 2022) for d in (0, 30)], ignore_index=True)
    projected = project_schedule(schedule, stints, bio, coefficients)
    assert len(projected[projected["season"] == 2021]) == 2 * 15     # no earlier season: unchanged
    late = projected[(projected["season"] == 2022) & (projected["as_of"] == projected["as_of"].max())]
    late = late.set_index("player_id")
    assert len(late) == 30 and late.loc[15, "rapm"] > late.loc[29, "rapm"] + 2


def test_bench_scores_minutes_weighted_team_ratings(tmp_path):
    from basketball_ml.impact.bench import score_ratings

    games = pd.DataFrame({"season": "2024-25", "game_id": ["1", "2", "3"], "home": ["AAA", "BBB", "AAA"],
                          "away": ["BBB", "AAA", "BBB"], "home_pts": [110, 100, 120], "away_pts": [100, 104, 100]})
    minutes = pd.DataFrame({"season": "2024-25", "game_id": ["1", "1", "2", "2", "3", "3"],
                            "team": ["AAA", "BBB", "BBB", "AAA", "AAA", "BBB"], "nba_id": [7, 8, 8, 7, 7, 8],
                            "min": 240.0})
    minutes = pd.concat([minutes, minutes.assign(season="2023-24", game_id=minutes["game_id"] + "x")])
    games.to_csv(tmp_path / "games.csv", index=False)
    minutes.to_csv(tmp_path / "minutes.csv", index=False)
    ratings = pd.DataFrame({"season": "2024-25", "nba_id": [7, 8], "rating": [2.0, -1.0]})
    result = score_ratings(ratings, tmp_path, seasons=("2024-25",))
    # Team AAA is rated 15 points better (5 x 2.0 against 5 x -1.0) and wins by 10, 4 and 20.
    expected = np.corrcoef([15, -15, 15], [10, -4, 20])[0, 1] ** 2
    assert result.loc["established", "all"] == pytest.approx(expected)
    # Last season's minutes are 480 for both players, so both rules use the same values here.
    assert result.loc["full forecast", "all"] == pytest.approx(expected)
