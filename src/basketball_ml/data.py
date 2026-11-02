"""Load advanced stats in Basketball-Reference format.

Expected source: the ``Advanced.csv`` file from the Kaggle dataset
"NBA Stats (1947-present)" (a Basketball-Reference export), with one row per
player, season and team.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Style and efficiency stats, all independent of pace and playing time.
# Available for every season since 1979-80 (introduction of the 3-point line).
FEATURES: dict[str, str] = {
    "ts_percent": "TS%",
    "x3p_ar": "3PA rate",
    "f_tr": "FT rate",
    "orb_percent": "ORB%",
    "drb_percent": "DRB%",
    "ast_percent": "AST%",
    "stl_percent": "STL%",
    "blk_percent": "BLK%",
    "tov_percent": "TOV%",
    "usg_percent": "USG%",
    "obpm": "OBPM",
    "dbpm": "DBPM",
}

# "All teams combined" rows for traded players:
# "TOT" in older exports, "2TM", "3TM"... in recent ones.
_COMBINED_TEAM_PREFIXES = ("TOT",)
_COMBINED_TEAM_SUFFIX = "TM"


def _is_combined(team: str) -> bool:
    return team.startswith(_COMBINED_TEAM_PREFIXES) or (
        team.endswith(_COMBINED_TEAM_SUFFIX) and team[:-2].isdigit()
    )


def load_advanced(
    path: str | Path,
    min_season: int = 1980,
    min_minutes: int = 500,
    league: str = "NBA",
) -> pd.DataFrame:
    """Load ``Advanced.csv`` with one row per player and season.

    ``season`` is the year the season ends (2013 = the 2012-13 season).
    Traded players keep their combined row across all their teams.
    """
    df = pd.read_csv(path)
    df = df[(df["lg"] == league) & (df["season"] >= min_season)].copy()
    df["team"] = df["team"].astype(str)

    combined = df["team"].map(_is_combined)
    traded = df.loc[combined, ["season", "player_id"]]
    is_traded = df.set_index(["season", "player_id"]).index.isin(
        pd.MultiIndex.from_frame(traded)
    )
    df = df[combined | ~is_traded]

    df = df[df["mp"] >= min_minutes]
    df = df.dropna(subset=list(FEATURES))
    return df.reset_index(drop=True)
