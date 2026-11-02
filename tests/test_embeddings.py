import numpy as np
import pandas as pd

from basketball_ml.embeddings import StyleAutoencoder, cluster, identification, summarize
from basketball_ml.embeddings.cli import normalize_name
from basketball_ml.embeddings.features import event_features, shot_features


def _shots():
    rows = []
    for zone, action, made in [("Restricted Area", "Driving Dunk Shot", 1), ("Restricted Area", "Cutting Layup Shot", 0),
                               ("Above the Break 3", "Jump Shot", 1), ("Above the Break 3", "Pullup Jump shot", 0)]:
        rows.append(dict(PLAYER_ID=1, SHOT_ZONE_BASIC=zone, ACTION_TYPE=action, SHOT_MADE_FLAG=made, SHOT_DISTANCE=10))
    rows.append(dict(PLAYER_ID=2, SHOT_ZONE_BASIC="Backcourt", ACTION_TYPE="Jump Shot", SHOT_MADE_FLAG=0, SHOT_DISTANCE=60))
    rows.append(dict(PLAYER_ID=2, SHOT_ZONE_BASIC="Mid-Range", ACTION_TYPE="Fadeaway Jump Shot", SHOT_MADE_FLAG=1, SHOT_DISTANCE=15))
    return pd.DataFrame(rows)


def test_shot_features_shares_and_backcourt_excluded():
    f = shot_features(_shots())
    assert f.loc[1, "fga"] == 4 and f.loc[2, "fga"] == 1
    assert np.isclose(f.loc[1, "share_rim"], 0.5) and np.isclose(f.loc[1, "share_above3"], 0.5)
    assert np.isclose(f.loc[1, "type_dunk"], 0.25) and np.isclose(f.loc[1, "type_catch_shoot"], 0.25)
    assert np.isclose(f.loc[2, "type_post_fade"], 1.0)


def _events():
    team = 1610612737
    e = [
        # made shot by 1 assisted by 2
        dict(GAME_ID=1, EVENTMSGTYPE=1, EVENTMSGACTIONTYPE=1, PLAYER1_ID=1, PLAYER2_ID=2, PLAYER3_ID=0,
             HOMEDESCRIPTION="A 3PT (2 AST)", VISITORDESCRIPTION=None, PLAYER1_NAME="One"),
        # miss by 1 blocked by 3, rebound by 2 (running totals in the text)
        dict(GAME_ID=1, EVENTMSGTYPE=2, EVENTMSGACTIONTYPE=1, PLAYER1_ID=1, PLAYER2_ID=0, PLAYER3_ID=3,
             HOMEDESCRIPTION="MISS", VISITORDESCRIPTION="BLOCK", PLAYER1_NAME="One"),
        dict(GAME_ID=1, EVENTMSGTYPE=4, EVENTMSGACTIONTYPE=0, PLAYER1_ID=2, PLAYER2_ID=0, PLAYER3_ID=0,
             HOMEDESCRIPTION="Two REBOUND (Off:1 Def:0)", VISITORDESCRIPTION=None, PLAYER1_NAME="Two"),
        dict(GAME_ID=1, EVENTMSGTYPE=4, EVENTMSGACTIONTYPE=0, PLAYER1_ID=2, PLAYER2_ID=0, PLAYER3_ID=0,
             HOMEDESCRIPTION="Two REBOUND (Off:1 Def:1)", VISITORDESCRIPTION=None, PLAYER1_NAME="Two"),
        # team rebound is ignored
        dict(GAME_ID=1, EVENTMSGTYPE=4, EVENTMSGACTIONTYPE=0, PLAYER1_ID=team, PLAYER2_ID=0, PLAYER3_ID=0,
             HOMEDESCRIPTION="Hawks Rebound", VISITORDESCRIPTION=None, PLAYER1_NAME=None),
        # bad pass turnover by 1, stolen by 3
        dict(GAME_ID=1, EVENTMSGTYPE=5, EVENTMSGACTIONTYPE=1, PLAYER1_ID=1, PLAYER2_ID=3, PLAYER3_ID=0,
             HOMEDESCRIPTION="Bad Pass", VISITORDESCRIPTION="STEAL", PLAYER1_NAME="One"),
    ]
    return pd.DataFrame(e)


def test_event_features_counts():
    f = event_features(_events())
    assert f.loc[2, "ast"] == 1 and f.loc[1, "fgm_assisted"] == 1
    assert f.loc[3, "blk"] == 1 and f.loc[3, "stl"] == 1
    assert f.loc[2, "oreb"] == 1 and f.loc[2, "dreb"] == 1
    assert f.loc[1, "tov"] == 1 and f.loc[1, "tov_bad_pass"] == 1
    assert 1610612737 not in f.index


def test_identification_finds_stable_players():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 6))
    rows, vecs = [], []
    for season in (2020, 2021):
        for i in range(30):
            rows.append({"player_id": i, "season": season})
            vecs.append(base[i] + rng.normal(scale=0.01, size=6))
    ranks = identification(pd.DataFrame(rows), np.array(vecs), [2020])
    assert summarize(ranks)["top1"] == 1.0


def test_autoencoder_and_clusters_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 10))
    ae = StyleAutoencoder(dim=3, max_iter=300).fit(X)
    assert ae.transform(X).shape == (200, 3)
    table = pd.DataFrame(X, columns=[f"rate_{i}" for i in range(10)])
    grouped, names = cluster(table, list(table.columns), k=4)
    assert grouped["archetype"].nunique() == 4 and len(names) == 4


def test_normalize_name():
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("Nikola Jokić") == "nikola jokic"
    assert normalize_name("De'Andre Hunter") == "deandre hunter"
