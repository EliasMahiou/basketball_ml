import numpy as np
import pandas as pd

from basketball_ml import pbp
from basketball_ml.predictions import FEATURES, LeagueState, add_features, elo_win_prob, evaluate, walk_forward
from basketball_ml.predictions.games import ELO_HOME, ELO_START


def _fake_games(n_seasons=4, seed=0):
    """A strong team (AAA) beats everyone; others are coin flips."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows, gid = [], 0
    for season in range(2010, 2010 + n_seasons):
        day = pd.Timestamp(f"{season - 1}-11-01")
        for _ in range(60):
            home, away = rng.choice(teams, size=2, replace=False)
            strong = "AAA" in (home, away)
            home_wins = (home == "AAA") if strong else rng.random() < 0.55
            hp, ap = (110, 100) if home_wins else (100, 110)
            rows.append(dict(game_id=gid, date=day, season=season, home=home, away=away,
                             home_pts=hp, away_pts=ap))
            gid += 1
            day += pd.Timedelta(days=int(rng.integers(1, 3)))
    g = pd.DataFrame(rows)
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    return g


def test_archive_names_use_start_year():
    assert pbp.local_path("nbastats", 2025).name == "nbastats_2024.csv"
    assert pbp.local_path("shotdetail", 2025, playoffs=True).name == "shotdetail_po_2024.csv"


def test_first_game_features_and_elo_update():
    games, state = add_features(_fake_games(1))
    first = games.iloc[0]
    assert first["elo_diff"] == ELO_HOME
    assert np.isclose(first["elo_prob"], elo_win_prob(ELO_HOME))
    assert first["form_diff"] == 0 and first["rest_diff"] == 0
    assert state.elo["AAA"] > ELO_START


def test_back_to_back_flag():
    s = LeagueState()
    s.start_season(2020)
    d = pd.Timestamp("2020-01-01")
    s.update("AAA", "BBB", 100, 90, d)
    f = s.features("AAA", "CCC", d + pd.Timedelta(days=1))
    assert f["b2b_home"] == 1 and f["b2b_away"] == 0 and f["rest_diff"] == 1 - 4


def test_new_season_regresses_elo():
    s = LeagueState()
    s.start_season(2020)
    s.elo["AAA"] = ELO_START + 200
    s.start_season(2021)
    assert np.isclose(s.elo["AAA"], ELO_START + 150)


def test_walk_forward_learns_the_strong_team():
    games, _ = add_features(_fake_games())
    results = walk_forward(games, [2012, 2013])
    assert set(results["season"]) == {2012, 2013}
    table = evaluate(results)
    assert table.loc["logistic", "accuracy"] > table.loc["home_court", "accuracy"]
    assert set(FEATURES) <= set(games.columns)
