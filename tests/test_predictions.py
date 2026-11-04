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


def _fake_appearances(games):
    """AAA's star (id 1, 30 per game) sits out every 4th AAA game."""
    rows, aaa_games = [], 0
    for g in games.itertuples(index=False):
        for team in (g.home, g.away):
            base = {"AAA": 100, "BBB": 200, "CCC": 300, "DDD": 400}[team]
            players = [base + i for i in range(2, 8)]
            if team == "AAA":
                aaa_games += 1
                if aaa_games % 4:
                    players.append(1)
            for pid in players:
                rows.append({"season": g.season, "game_id": g.game_id, "team": team, "player_id": pid,
                             "player": f"P{pid}", "production": 30.0 if pid == 1 else 10.0})
    return pd.DataFrame(rows)


def test_availability_sees_the_missing_star():
    from basketball_ml.predictions import add_availability
    games, _ = add_features(_fake_games(1))
    out = add_availability(games, _fake_appearances(games))
    aaa_home = out[out.home == "AAA"].iloc[10:]
    lineup = _fake_appearances(games)
    star_out = ~aaa_home.game_id.isin(lineup[lineup.player_id == 1].game_id)
    assert (aaa_home.loc[star_out, "missing_home"] >= 30).all()
    assert (aaa_home.loc[~star_out, "strength_home"] > aaa_home.loc[star_out, "strength_home"].max()).all()


def test_expected_lineup_drops_players_marked_out():
    from basketball_ml.predictions import expected_lineup
    games, _ = add_features(_fake_games(1))
    apps = _fake_appearances(games)
    full, players = expected_lineup(apps, games, "AAA")
    without, _ = expected_lineup(apps, games, "AAA", out=["P1"])
    assert "P1" in set(players["player"])
    assert np.isclose(full - without, 30.0)


def test_calibration_bins_and_error():
    from basketball_ml.predictions import calibration, calibration_error
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    results = pd.DataFrame({"p_logistic": p, "home_win": (rng.uniform(size=p.size) < p).astype(int)})
    table = calibration(results)
    assert len(table) == 10 and table["games"].sum() == p.size
    assert calibration_error(results) < 0.02
    results["p_logistic"] = np.clip(p + 0.1, 0, 1)    # overconfident in home teams
    assert calibration_error(results) > 0.08


def test_no_vig_removes_the_bookmaker_margin():
    from basketball_ml.predictions.market import no_vig

    p = no_vig([-110, -200, 150, 5], [-110, 170, -180, 7])
    assert p[0] == 0.5
    assert abs(p[1] - (2 / 3) / (2 / 3 + 100 / 270)) < 1e-12
    assert 0.35 < p[2] < 0.4
    assert np.isnan(p[3])  # not a moneyline


def test_closing_lines_match_games_a_day_apart(tmp_path):
    import sqlite3

    from basketball_ml.predictions.market import load_lines, match_lines

    con = sqlite3.connect(tmp_path / "odds.sqlite")
    rows = [("2024-01-02", "Boston Celtics", "Los Angeles Lakers", -200, 170),
            ("2024-01-03", "LA Clippers", "Utah Jazz", 120, -140)]
    pd.DataFrame(rows, columns=["Date", "Home", "Away", "ML_Home", "ML_Away"]).to_sql("2023-24", con)
    # An older copy of a season stores dates in another format: skipped.
    pd.DataFrame([("2023-24-0102", "Boston Celtics", "Los Angeles Lakers", 500, -700)],
                 columns=["Date", "Home", "Away", "ML_Home", "ML_Away"]).to_sql("odds_2023-24", con)
    con.close()
    lines = load_lines(tmp_path / "odds.sqlite")
    assert len(lines) == 2 and set(lines["home"]) == {"BOS", "LAC"}
    games = pd.DataFrame({"game_id": [1, 2, 3], "home": ["BOS", "LAC", "BOS"], "away": ["LAL", "UTA", "LAL"],
                          "date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-03-01"])})
    p = match_lines(games, lines)
    assert p[1] > 0.6 and p[2] < 0.5 and np.isnan(p[3])


def test_market_backtest_blends_model_and_line():
    from basketball_ml.predictions.market import market_backtest

    games, _ = add_features(_fake_games())
    rng = np.random.default_rng(1)
    # A sharp line: right about the strong team, noise otherwise.
    p = np.where(games["home"] == "AAA", 0.9, np.where(games["away"] == "AAA", 0.1, rng.uniform(0.4, 0.7, len(games))))
    games["p_market"] = p
    games["market_logit"] = np.log(p / (1 - p))
    results = market_backtest(games, [2012, 2013], features=list(FEATURES))
    table = evaluate(results, ("market", "logistic", "blend"))
    assert len(results) == (games["season"] >= 2012).sum()
    assert table.loc["blend", "log_loss"] < table.loc["logistic", "log_loss"]


def _team(code, score, players=()):
    return {"teamTricode": code, "score": score, "players": list(players)}


def _player(pid, name, minutes, played="1", **stats):
    base = {"points": 0, "assists": 0, "reboundsTotal": 0, "steals": 0, "blocks": 0, "turnovers": 0}
    return {"personId": pid, "name": name, "played": played,
            "statistics": {**base, **stats, "minutes": minutes}}


def test_live_schedule_keeps_regular_season_games():
    from basketball_ml.predictions.live import season_games

    schedule = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": [{"games": [
        {"gameId": "0012600001", "gameDateEst": "2026-10-03T00:00:00Z", "gameStatus": 3,
         "homeTeam": _team("TOR", 105), "awayTeam": _team("MIA", 99)},
        {"gameId": "0022600001", "gameDateEst": "2026-10-20T00:00:00Z", "gameStatus": 3,
         "homeTeam": _team("DET", 121), "awayTeam": _team("BOS", 113)},
        {"gameId": "0022600002", "gameDateEst": "2026-11-04T00:00:00Z", "gameStatus": 1,
         "homeTeam": _team("NYK", 0), "awayTeam": _team("CHI", 0)}]}]}}
    games = season_games(schedule)
    assert list(games["game_id"]) == [22600001, 22600002]
    assert games["season"].eq(2027).all()
    assert games.loc[0, "home_win"] == 1 and np.isnan(games.loc[1, "home_win"])


def test_live_box_score_appearances():
    from basketball_ml.predictions.live import appearances_from_boxscore

    data = {"game": {"gameId": "0022600001", "gameStatus": 3,
                     "homeTeam": _team("DET", 121, [_player(1, "A", "PT33M15.00S", points=20, assists=5, turnovers=3),
                                                    _player(2, "B", "PT00M00.00S", played="0")]),
                     "awayTeam": _team("BOS", 113, [_player(3, "C", "PT02M30.00S", blocks=1)])}}
    apps = appearances_from_boxscore(data, 2027)
    assert list(apps["player_id"]) == [1, 3]
    assert list(apps["production"]) == [22, 1]
    assert list(apps["seconds"]) == [1995.0, 150.0]
    assert list(apps["side"]) == ["home", "away"]


def test_live_odds_take_the_median_book():
    from basketball_ml.predictions.live import odds_today

    def book(i, home, away):
        return {"id": i, "outcomes": [{"type": "home", "odds": home}, {"type": "away", "odds": away}]}

    odds = {"games": [{"gameId": "0022600173", "markets": [
        {"name": "2way", "books": [book("a", "1.50", "2.60"), book("b", "1.55", "2.45"), book("c", "1.60", "2.35")]},
        {"name": "spread", "books": [book("a", "1.90", "1.90")]}]}]}
    p = odds_today(odds)
    assert abs(p[22600173] - (1 / 1.55) / (1 / 1.55 + 1 / 2.45)) < 1e-12


def test_expected_seconds_switch_to_this_season_after_three_games():
    from basketball_ml.predictions.cli import expected_seconds

    seconds = pd.DataFrame({"season": [2026] * 3 + [2027] * 5, "game_id": range(8), "side": "home",
                            "player_id": [1, 1, 2, 1, 1, 1, 2, 3], "seconds": [600, 600, 900, 1800, 1800, 1800, 300, 60]})
    out = expected_seconds(seconds, 2027)
    assert out[1] == 1800        # three games this season
    assert out[2] == 900         # one game this season: last season's average
    assert out[3] == 60          # no history: this season's only game
