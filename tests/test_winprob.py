import numpy as np
import pandas as pd

from basketball_ml.winprob import (SternModel, add_game_info, bootstrap_difference, credits, elapsed_seconds, evaluate,
                                   free_throw_details, game_states, model_frame, phase, possession_check,
                                   seconds_left, walk_forward, win_prob_changes)

HOME, AWAY = "HHH", "AAA"
HOME_ID, AWAY_ID = 1610612737, 1610612738


def _event(period, kind, clock, p1=0, t1=None, p2=0, t2=None, p3=0, t3=None, score=None, home_text=None,
           away_text=None, action=0):
    return {"GAME_ID": 1, "PERIOD": period, "EVENTMSGTYPE": kind, "EVENTMSGACTIONTYPE": action, "PCTIMESTRING": clock,
            "SCORE": score, "HOMEDESCRIPTION": home_text, "VISITORDESCRIPTION": away_text,
            "PLAYER1_ID": p1, "PLAYER1_TEAM_ABBREVIATION": t1, "PLAYER2_ID": p2, "PLAYER2_TEAM_ABBREVIATION": t2,
            "PLAYER3_ID": p3, "PLAYER3_TEAM_ABBREVIATION": t3}


def _states(events):
    s = game_states(pd.DataFrame(events), HOME)
    return s.set_index("row")


def test_possession_follows_makes_rebounds_and_turnovers():
    ev = [
        _event(1, 12, "12:00"),
        # The recorded tip goes to a home player, but the away team shoots first: trust the play.
        _event(1, 10, "12:00", 1, HOME, 11, AWAY, 2, HOME, home_text="Jump Ball"),
        _event(1, 2, "11:40", 11, AWAY, away_text="MISS jumper"),
        _event(1, 4, "11:38", 12, AWAY, away_text="REBOUND (Off:1 Def:0)"),            # offensive: keeps it
        _event(1, 1, "11:30", 12, AWAY, 11, AWAY, score="2 - 0", away_text="layup (11 AST)"),
        _event(1, 2, "11:10", 1, HOME, 0, None, 13, AWAY, home_text="MISS shot", away_text="BLOCK"),
        _event(1, 4, "11:08", 13, AWAY, away_text="REBOUND (Off:0 Def:1)"),            # defensive: switches
        _event(1, 5, "11:00", 13, AWAY, 2, HOME, away_text="Bad Pass Turnover", home_text="STEAL"),
        _event(1, 5, "10:40", HOME_ID, None, home_text="Shot Clock Team Turnover"),    # team event: side from text
        _event(1, 9, "10:40", AWAY_ID, None, away_text="Timeout"),
        _event(1, 1, "10:20", 14, AWAY, score="4 - 0", away_text="dunk"),
    ]
    s = _states(ev)
    assert s.loc[0, "poss"] == 0                       # before the opening tip
    assert s.loc[1, "poss"] == -1                      # jump ball: the away team plays first
    assert (s.loc[2, "poss"], s.loc[2, "loose"]) == (-1, 1)
    assert (s.loc[3, "poss"], s.loc[3, "loose"]) == (-1, 0)
    assert s.loc[4, "poss"] == 1 and s.loc[4, "margin"] == -2
    assert (s.loc[5, "poss"], s.loc[5, "loose"]) == (1, 1)
    assert s.loc[6, "poss"] == -1
    assert s.loc[7, "poss"] == 1 and s.loc[7, "p2"] == 2
    assert s.loc[8, "poss"] == -1 and s.loc[8, "side"] == 1
    assert 9 not in s.index                            # timeouts aren't states
    assert s.loc[10, "poss"] == 1 and s.loc[10, "margin"] == -4
    assert possession_check(s.reset_index()).min() == 1.0


def test_free_throw_trips():
    ev = [
        _event(1, 12, "12:00"),
        _event(1, 10, "12:00", 1, HOME, 11, AWAY, 2, HOME, home_text="Jump Ball"),
        _event(1, 6, "11:30", 11, AWAY, 1, HOME, away_text="S.FOUL", action=2),
        _event(1, 8, "11:30", 11, AWAY, 15, AWAY, away_text="SUB"),
        _event(1, 3, "11:30", 1, HOME, home_text="MISS Free Throw 1 of 2", action=11),
        _event(1, 4, "11:30", HOME_ID, None, home_text="Team Rebound"),                # dead ball: ignored
        _event(1, 3, "11:30", 1, HOME, score="0 - 1", home_text="Free Throw 2 of 2 (1 PTS)", action=12),
        _event(1, 6, "11:10", 1, HOME, 12, AWAY, home_text="T.FOUL", action=11),
        _event(1, 3, "11:10", 12, AWAY, score="1 - 1", away_text="Free Throw Technical (1 PTS)", action=16),
        _event(1, 6, "11:00", 2, HOME, 12, AWAY, home_text="FLAGRANT.FOUL", action=14),
        _event(1, 3, "11:00", 12, AWAY, score="2 - 1", away_text="flagrant Free Throw 1 of 2 (2 PTS)"),
        _event(1, 3, "11:00", 12, AWAY, away_text="MISS flagrant Free Throw 2 of 2"),
        _event(1, 1, "10:40", 12, AWAY, score="4 - 1", away_text="jumper"),
        _event(1, 1, "10:20", 1, HOME, score="4 - 3", home_text="layup"),               # and-one
        _event(1, 6, "10:20", 13, AWAY, 1, HOME, away_text="S.FOUL", action=2),
        _event(1, 3, "10:20", 1, HOME, home_text="MISS Free Throw 1 of 1", action=10),
        _event(1, 4, "10:18", 13, AWAY, away_text="REBOUND"),
    ]
    s = _states(ev)
    assert (s.loc[2, "poss"], s.loc[2, "fts"]) == (1, 2)   # two free throws known at the foul
    assert (s.loc[4, "poss"], s.loc[4, "fts"]) == (1, 1)
    assert 5 in s.index and s.loc[5, "fts"] == 1 and s.loc[5, "poss"] == 1
    assert (s.loc[6, "poss"], s.loc[6, "fts"], s.loc[6, "margin"]) == (-1, 0, 1)
    assert s.loc[8, "poss"] == -1 and s.loc[8, "margin"] == 0  # technical: the away team keeps the ball
    assert (s.loc[11, "poss"], s.loc[11, "loose"]) == (-1, 0)  # flagrant: ball back to the shooting team
    assert (s.loc[14, "poss"], s.loc[14, "fts"]) == (1, 1)     # and-one at the line
    assert (s.loc[15, "poss"], s.loc[15, "loose"]) == (1, 1)   # missed last free throw: loose ball
    assert s.loc[16, "poss"] == -1


def test_putback_rebound_order_and_new_quarter():
    ev = [
        _event(1, 12, "12:00"),
        _event(1, 10, "12:00", 1, HOME, 11, AWAY, 2, HOME, home_text="Jump Ball"),
        _event(1, 2, "11:40", 1, HOME, home_text="MISS layup"),
        _event(1, 1, "11:38", 3, HOME, score="0 - 2", home_text="Tip Layup"),
        _event(1, 4, "11:38", 3, HOME, home_text="REBOUND (Off:1 Def:0)"),            # recorded after its putback
        _event(1, 2, "11:20", 11, AWAY, away_text="MISS 3PT"),
        _event(1, 13, "0:00"),
        _event(2, 12, "12:00"),
        _event(2, 8, "12:00", 1, HOME, 6, HOME, home_text="SUB"),
        _event(2, 2, "11:45", 12, AWAY, away_text="MISS jumper"),
        _event(5, 12, "5:00"),
        _event(5, 10, "5:00", 1, HOME, 11, AWAY, 0, None, home_text="Jump Ball"),
        _event(5, 1, "4:40", 4, HOME, score="0 - 4", home_text="jumper"),
    ]
    s = game_states(pd.DataFrame(ev), HOME)
    assert s["row"].tolist()[:5] == [0, 1, 2, 4, 3]       # rebound moved before the made tip-in
    by_row = s.set_index("row")
    assert (by_row.loc[4, "poss"], by_row.loc[4, "margin"]) == (1, 0)
    assert by_row.loc[3, "poss"] == -1 and by_row.loc[3, "margin"] == 2
    assert by_row.loc[7, "poss"] == -1                       # Q2 starts with the team that shoots first
    assert by_row.loc[9, "subs"] == 1
    assert by_row.loc[10, "poss"] == 0 and by_row.loc[10, "seconds_left"] == 300
    assert by_row.loc[11, "poss"] == 1                       # unknown tip: the next shot tells
    assert by_row.loc[12, "seconds_left"] == 280


def test_clock_and_free_throw_text():
    assert seconds_left(1, 720).item() == 2880 and seconds_left(4, 30).item() == 30
    assert seconds_left(6, 100).item() == 100
    assert elapsed_seconds(1, 2880).item() == 0 and elapsed_seconds(4, 0).item() == 2880
    assert elapsed_seconds(6, 0).item() == 2880 + 600
    d = free_throw_details(pd.Series(["MISS Smith Free Throw Flagrant 2 of 2", "J. Doe technical Free Throw 1 of 1",
                                      "Free Throw Technical", "A. B clear-path Free Throw 1 of 2 (3 PTS)",
                                      "X away-from-play Free Throw 1 of 1"]), late=pd.Series([False] * 4 + [True]))
    assert d["n"].tolist() == [2, 1, 1, 1, 1] and d["m"].tolist() == [2, 1, 1, 2, 1]
    assert d["technical"].tolist() == [False, True, True, False, False]
    assert d["keeps"].tolist() == [True, False, False, True, True]
    assert d["missed"].tolist() == [True, False, False, False, False]


def _brownian_states(n_games=1500, seed=0, sigma=12.0):
    """Margins following a Brownian motion with drift = the pre-game spread; the home team wins if ahead at the end."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        spread = rng.normal(2.0, 5.0)
        times = np.sort(rng.uniform(0, 2880, 12))[::-1]
        elapsed = 2880 - times
        steps = np.diff(np.concatenate([[0.0], elapsed, [2880.0]]))
        path = np.cumsum(spread * steps / 2880 + rng.normal(0, sigma * np.sqrt(steps / 2880)))
        final = path[-1]
        for t, m in zip(times, path[:-1]):
            rows.append({"game_id": g, "season": 2020 + g % 2, "event": 2, "period": 4 - int(t // 720.01),
                         "seconds_left": int(t), "margin": int(round(m)), "poss": 0, "loose": 0, "fts": 0,
                         "elo_spread": spread, "elo_prob": 1 / (1 + np.exp(-spread / 7)), "home_win": int(final > 0)})
    return pd.DataFrame(rows)


def test_stern_model_recovers_the_brownian_noise():
    states = _brownian_states()
    model = SternModel().fit(model_frame(states), states["home_win"].to_numpy())
    params = model.describe()
    assert 9 < params["sigma"] < 15
    assert 0.6 < params["spread_weight"] < 1.5
    p = model.predict_proba(model_frame(states))
    late = states["seconds_left"] < 300
    assert ((p[late] > 0.5) == (states.loc[late, "margin"] > 0)).mean() > 0.9


def test_walk_forward_and_evaluation():
    states = _brownian_states(n_games=400)
    results = walk_forward(states, [2021], methods=("pregame", "stern", "logistic", "boosting"))
    assert set(results["season"]) == {2021} and {"p_stern", "p_logistic", "p_boosting"} <= set(results)
    table = evaluate(results)
    assert table.loc["All", ("log_loss", "stern")] < table.loc["All", ("log_loss", "pregame")]
    assert set(phase(results).unique()) <= set(table.index)
    d, lo, hi = bootstrap_difference(results, "stern", "pregame", n_boot=200)
    assert lo <= d <= hi < 0


def test_add_game_info_joins_strength_and_result():
    states = pd.DataFrame({"game_id": [1, 1, 2], "margin": [0, 2, 0]})
    games = pd.DataFrame({"game_id": [1, 2], "elo_prob": [0.6, 0.4], "elo_diff": [56.0, -28.0], "home_win": [1, 0]})
    out = add_game_info(states, games)
    assert out["elo_spread"].tolist() == [2.0, 2.0, -1.0] and out["home_win"].tolist() == [1, 1, 0]


def test_wpa_credit_shares():
    changes = pd.DataFrame({
        "season": 2026, "game_id": 1, "period": 4, "seconds_left": [200, 190, 180, 170, 160],
        "event": [12, 1, 2, 5, 6], "side": [0, 1, -1, -1, 1], "margin": [0, 2, 2, 2, 2],
        "p1": [0, 1, 11, 12, 3], "p2": [0, 2, 0, 4, 13], "p3": [0, 0, 5, 0, 0],
        "p_stern": [0.50, 0.60, 0.64, 0.70, 0.66]})
    c = credits(win_prob_changes(changes, "stern")).set_index(["player_id", "source"])["wpa"]
    assert np.isclose(c[(1, "shots")], 0.07) and np.isclose(c[(2, "assists")], 0.03)   # assisted make: 70/30
    assert np.isclose(c[(11, "shots")], -0.02) and np.isclose(c[(5, "defense")], 0.02)  # blocked miss: half each
    assert np.isclose(c[(12, "turnovers")], -0.03) and np.isclose(c[(4, "defense")], 0.03)
    assert np.isclose(c[(3, "fouls")], -0.04)                                          # fouler pays the free throws
    assert (13, "fouls") not in c.index                                                # fouled player: no credit
    changes["margin"] = [0, 2, 2, 9, 9]
    clutch = win_prob_changes(changes, "stern")["clutch"].tolist()
    assert clutch == [True, True, True, True, False]                                   # margin before the event
