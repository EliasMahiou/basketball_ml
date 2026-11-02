import numpy as np
import pandas as pd

from basketball_ml.data import FEATURES
from basketball_ml.projections import TARGETS, backtest, build_pairs, project_season, score
from basketball_ml.twins import standardize_by_season

FEATS = list(FEATURES)


def _fake_league(n_players=40, first=2000, last=2010, seed=0):
    """Players whose BPM rises by exactly +1 every season, so the truth is known."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        start_age = 20 + i % 5
        profile = rng.normal(size=len(FEATS))
        for j, season in enumerate(range(first, last + 1)):
            row = dict(zip(FEATS, profile + rng.normal(scale=0.05, size=len(FEATS))))
            row.update(player=f"Player {i}", player_id=f"p{i}", season=season,
                       age=float(start_age + j), mp=1500.0, bpm=float(j))
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_pairs_deltas():
    seasons = _fake_league()
    pairs = build_pairs(seasons, standardize_by_season(seasons), horizon=2)
    assert np.allclose(pairs["bpm_delta"], 2.0)
    assert (pairs["bpm_later"] - pairs["bpm_now"] == 2.0).all()
    assert pairs["season"].max() == 2008


def test_projection_follows_comparables_and_has_no_lookahead():
    seasons = _fake_league()
    z = standardize_by_season(seasons)
    pairs = build_pairs(seasons, z)
    target_z = z[(z.player_id == "p0") & (z.season == 2006)].iloc[0]
    target_raw = seasons[(seasons.player_id == "p0") & (seasons.season == 2006)].iloc[0]
    proj, comps = project_season(target_z, target_raw, pairs, k=5)
    assert np.isclose(proj["bpm"], target_raw["bpm"] + 1.0)
    assert (comps["season"] + 1 <= 2006).all()
    assert (comps["age"] == int(target_z["age"])).all()
    assert "p0" not in set(comps["player_id"])


def test_backtest_scores_all_methods():
    seasons = _fake_league()
    pairs = build_pairs(seasons, standardize_by_season(seasons))
    results = backtest(pairs, range(2006, 2010), k=5)
    assert len(results) > 0
    table = score(results)
    assert set(table.columns) == {"comparables", "age_curve", "no_change", "range_80_coverage"}
    assert set(table.index) == set(TARGETS)
    # Every comparable improved by exactly +1 BPM, so the projection is exact.
    assert table.loc["bpm", "comparables"] < 1e-9
    assert np.isclose(table.loc["bpm", "no_change"], 1.0)


def test_weighted_quantile():
    from basketball_ml.projections import weighted_quantile
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.isclose(weighted_quantile(v, np.ones(5), 0.5), 2.0)
    # Nearly all the weight on 4 pulls the median there.
    assert weighted_quantile(v, np.array([1, 1, 1, 1, 100.0]), 0.5) > 3.9


def test_backtest_reports_range_coverage():
    seasons = _fake_league()
    pairs = build_pairs(seasons, standardize_by_season(seasons))
    table = score(backtest(pairs, range(2006, 2010), k=5))
    assert "range_80_coverage" in table.columns
    assert table.loc["bpm", "range_80_coverage"] == 1.0


def test_add_history_weights_recent_seasons_and_minutes():
    from basketball_ml.projections import add_history
    seasons = pd.DataFrame({"player_id": ["a", "a", "a"], "season": [2001, 2002, 2003],
                            "mp": [1000.0, 1000.0, 2000.0], "bpm": [0.0, 3.0, 6.0]})
    h = add_history(seasons, targets=["bpm"])
    # 2003: (5*2000*6 + 4*1000*3 + 3*1000*0) / (5*2000 + 4*1000 + 3*1000)
    assert np.isclose(h.loc[2, "bpm_hist"], (60000 + 12000) / 17000)
    assert h.loc[0, "bpm_hist"] == 0.0


def test_history_weight_moves_the_starting_point():
    from basketball_ml.projections import add_history
    seasons = add_history(_fake_league())
    z = standardize_by_season(seasons)
    pairs = build_pairs(seasons, z)
    tz = z[(z.player_id == "p0") & (z.season == 2006)].iloc[0]
    raw = seasons[(seasons.player_id == "p0") & (seasons.season == 2006)].iloc[0]
    now, _ = project_season(tz, raw, pairs, k=5, history_weight=0.0)
    mixed, _ = project_season(tz, raw, pairs, k=5, history_weight=0.5)
    # BPM rises every season, so the 3-season average sits below the current value.
    assert mixed["bpm"] < now["bpm"]
