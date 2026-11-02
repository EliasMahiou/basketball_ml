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
    assert set(table.columns) == {"comparables", "age_curve", "no_change"}
    assert set(table.index) == set(TARGETS)
    # Every comparable improved by exactly +1 BPM, so the projection is exact.
    assert table.loc["bpm", "comparables"] < 1e-9
    assert np.isclose(table.loc["bpm", "no_change"], 1.0)
