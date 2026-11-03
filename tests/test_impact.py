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
