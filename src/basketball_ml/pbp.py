"""Download and load NBA play-by-play and shot data.

Source: the public GitHub repository ``shufinskiy/nba_data``, which mirrors
stats.nba.com play-by-play (since 1996-97) and shot charts as one
``.tar.xz`` archive per season.

Seasons are named by the year they end, like everywhere else in this
package: ``2025`` is the 2024-25 season.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets"
KINDS = ("nbastats", "shotdetail")

PBP_COLUMNS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
    "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE", "SCOREMARGIN",
    "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION",
    "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ABBREVIATION",
]

# stats.nba.com EVENTMSGTYPE codes
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER, FOUL = 1, 2, 3, 4, 5, 6


def _archive_name(kind: str, season: int, playoffs: bool) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    # The source names archives by the year a season starts.
    return f"{kind}{'_po' if playoffs else ''}_{season - 1}"


def local_path(kind: str, season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> Path:
    return Path(data_dir) / f"{_archive_name(kind, season, playoffs)}.csv"


def download(kind: str, season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False,
             overwrite: bool = False) -> Path:
    """Download one season of ``kind`` ("nbastats" or "shotdetail") as CSV."""
    out = local_path(kind, season, data_dir, playoffs)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{_archive_name(kind, season, playoffs)}.tar.xz"
    with urllib.request.urlopen(url, timeout=300) as resp:
        payload = resp.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.isfile() and m.name.endswith(".csv"))
        with tar.extractfile(member) as src:
            out.write_bytes(src.read())
    return out


def load_pbp(season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> pd.DataFrame:
    """Play-by-play events for one season (downloaded on first use)."""
    path = download("nbastats", season, data_dir, playoffs)
    df = pd.read_csv(path, usecols=lambda c: c in PBP_COLUMNS, low_memory=False)
    df["season"] = season
    return df


def load_shots(season: int, data_dir: str | Path = "data/pbp", playoffs: bool = False) -> pd.DataFrame:
    """Shot chart rows (zone, distance, action type, made) for one season."""
    path = download("shotdetail", season, data_dir, playoffs)
    df = pd.read_csv(path, low_memory=False)
    df["season"] = season
    return df
