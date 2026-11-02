import pandas as pd

from basketball_ml import pbp


def _cdn_rows():
    base = {"gameId": 1, "period": 1, "subType": None, "descriptor": None, "personId": 0, "teamId": 0,
            "teamTricode": None, "scoreHome": 0, "scoreAway": 0, "shotResult": None, "assistPersonId": None,
            "stealPersonId": None, "blockPersonId": None, "foulDrawnPersonId": None, "playerNameI": None}
    rows = [
        dict(actionType="period", subType="start", clock="PT12M00.00S", description="Period Start"),
        dict(actionType="2pt", subType="Layup", clock="PT11M30.00S", personId=1, teamTricode="HOM", teamId=100,
             scoreHome=2, shotResult="Made", assistPersonId=2, description="A. One Layup (2 PTS) (B. Two 1 AST)",
             playerNameI="A. One"),
        dict(actionType="3pt", subType="Jump Shot", clock="PT11M10.00S", personId=11, teamTricode="AWY",
             teamId=200, scoreHome=2, shotResult="Missed", blockPersonId=3, description="MISS K. Eleven 3PT",
             playerNameI="K. Eleven"),
        dict(actionType="block", clock="PT11M10.00S", personId=3, teamTricode="HOM", teamId=100, scoreHome=2,
             description="C. Three BLOCK (1 BLK)", playerNameI="C. Three"),
        dict(actionType="rebound", subType="defensive", clock="PT11M08.00S", personId=2, teamTricode="HOM",
             teamId=100, scoreHome=2, description="B. Two REBOUND (Off:0 Def:1)", playerNameI="B. Two"),
        dict(actionType="turnover", subType="bad pass", clock="PT11M00.00S", personId=2, teamTricode="HOM",
             teamId=100, scoreHome=2, stealPersonId=12, description="B. Two bad pass TURNOVER (1 TO)",
             playerNameI="B. Two"),
        dict(actionType="steal", clock="PT11M00.00S", personId=12, teamTricode="AWY", teamId=200, scoreHome=2,
             description="L. Twelve STEAL (1 STL)", playerNameI="L. Twelve"),
        dict(actionType="substitution", subType="out", clock="PT10M50.00S", personId=1, teamTricode="HOM",
             teamId=100, scoreHome=2, description="SUB out: A. One", playerNameI="A. One"),
        dict(actionType="substitution", subType="in", clock="PT10M50.00S", personId=6, teamTricode="HOM",
             teamId=100, scoreHome=2, description="SUB in: F. Six", playerNameI="F. Six"),
        dict(actionType="3pt", subType="Jump Shot", clock="PT10M40.00S", personId=11, teamTricode="AWY",
             teamId=200, scoreHome=2, scoreAway=3, shotResult="Made", description="K. Eleven 3PT (3 PTS)",
             playerNameI="K. Eleven"),
        dict(actionType="turnover", subType="shot clock", clock="PT10M10.00S", teamTricode="HOM", teamId=100,
             scoreHome=2, scoreAway=3, description="HOM shot clock Team TURNOVER"),
        dict(actionType="period", subType="end", clock="PT00M00.00S", scoreHome=2, scoreAway=3,
             description="Period End"),
    ]
    out = pd.DataFrame([{**base, **r} for r in rows])
    out["orderNumber"] = range(len(out))
    out["actionNumber"] = range(1, len(out) + 1)
    return out


def test_from_cdn_matches_the_stats_nba_layout():
    events = pbp.from_cdn(_cdn_rows())
    assert set(pbp.PBP_COLUMNS) <= set(events.columns)
    assert events["EVENTMSGTYPE"].tolist() == [12, 1, 2, 4, 5, 8, 1, 5, 13]
    made, missed, rebound, turnover, sub, three, team_tov = (events.iloc[i] for i in range(1, 8))
    assert (made["PLAYER1_ID"], made["PLAYER2_ID"], made["SCORE"]) == (1, 2, "0 - 2")
    assert (missed["PLAYER3_ID"], missed["PLAYER3_TEAM_ABBREVIATION"]) == (3, "HOM")
    assert pd.isna(missed["SCORE"]) and missed["VISITORDESCRIPTION"].startswith("MISS")
    assert (turnover["PLAYER2_ID"], turnover["EVENTMSGACTIONTYPE"]) == (12, 1)
    assert (sub["PLAYER1_ID"], sub["PLAYER2_ID"]) == (1, 6)
    assert three["SCORE"] == "3 - 2" and three["PCTIMESTRING"] == "10:40"
    assert team_tov["PLAYER1_ID"] == 100 and pd.isna(team_tov["PLAYER1_TEAM_ABBREVIATION"])
    assert team_tov["HOMEDESCRIPTION"].startswith("HOM")
