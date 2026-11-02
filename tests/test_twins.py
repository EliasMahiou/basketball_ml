import numpy as np
import pandas as pd
import pytest

from basketball_ml.data import FEATURES, _is_combined
from basketball_ml.twins import career_profiles, find_twins, resolve_player, standardize_by_season

FEATS = list(FEATURES)


def _fake_seasons():
    rng = np.random.default_rng(0)
    rows = []
    for season in (2020, 2021):
        for i in range(30):
            row = {f: rng.normal() for f in FEATS}
            row.update(player=f"Player {i}", player_id=f"p{i}", season=season, mp=1000 + i)
            rows.append(row)
    df = pd.DataFrame(rows)
    # p1 is a near-exact copy of p0, so it must come out first.
    for season in (2020, 2021):
        src = df[(df.player_id == "p0") & (df.season == season)][FEATS].to_numpy()
        df.loc[(df.player_id == "p1") & (df.season == season), FEATS] = src + 1e-3
    return df


def test_combined_team_codes():
    assert _is_combined("TOT") and _is_combined("2TM") and _is_combined("3TM")
    assert not _is_combined("LAL") and not _is_combined("OKC")


def test_standardize_by_season_is_zero_mean_unit_std():
    z = standardize_by_season(_fake_seasons())
    stats = z.groupby("season")[FEATS].agg(["mean", "std"])
    assert np.allclose(stats.xs("mean", axis=1, level=1), 0, atol=1e-9)
    assert np.allclose(stats.xs("std", axis=1, level=1), 1)


def test_season_twin_finds_the_copy_and_excludes_self():
    z = standardize_by_season(_fake_seasons())
    twins = find_twins(z, "p0", season=2020, k=5)
    assert twins.iloc[0]["player_id"] == "p1"
    assert twins.iloc[0]["similarity"] > 99
    assert "p0" not in set(twins["player_id"])
    assert twins["player_id"].is_unique


def test_career_twin_finds_the_copy():
    prof = career_profiles(standardize_by_season(_fake_seasons()))
    assert len(prof) == 30
    assert find_twins(prof, "p0", k=1).iloc[0]["player_id"] == "p1"


def test_resolve_player():
    df = _fake_seasons()
    assert resolve_player(df, "player 3") == "p3"
    with pytest.raises(KeyError):
        resolve_player(df, "Nobody")
