"""``basketball-ml impact`` and ``impact-backtest`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import pbp
from ..cache import cached_csv
from .rapm import DEFAULT_WINDOW, backtest, ratings_before
from .stints import build_stints


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last season loaded (end year)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    r = subparsers.add_parser("impact", help="Player impact ratings (RAPM) from who is on the court.")
    r.add_argument("player", nargs="?", help="Show one player instead of the leaderboard")
    r.add_argument("--season", type=int, help="Ratings through this season (default: the last loaded)")
    r.add_argument("--top", type=int, default=20, help="Players in the leaderboard")
    r.add_argument("--min-possessions", type=float, default=3000, help="Leaderboard minimum (both seasons)")
    _common(r)
    r.set_defaults(func=run_ratings)

    b = subparsers.add_parser("impact-backtest", help="Do the ratings explain next season's game margins?")
    b.add_argument("--test-from", type=int, default=2016, help="First season predicted")
    _common(b)
    b.set_defaults(func=run_backtest)


def load_stints(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Stints for seasons ``first`` to ``last``, cached in ``data_dir``."""
    data_dir = Path(data_dir)
    return cached_csv(data_dir / f"stints_{first}_{last}.csv", lambda: build_stints(range(first, last + 1), data_dir))


def load_production(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Box-score production per player and game (the RAPM prior's input), cached in ``data_dir``."""
    from ..predictions.availability import build_player_games

    data_dir = Path(data_dir)
    return cached_csv(data_dir / f"appearances_{first}_{last}.csv",
                      lambda: build_player_games(range(first, last + 1), data_dir))


def player_names(seasons, data_dir: str | Path) -> dict[int, str]:
    """Latest name of every player id seen in the play-by-play of ``seasons``."""
    names: dict[int, str] = {}
    for season in seasons:
        e = pbp.load_pbp(season, data_dir).dropna(subset=["PLAYER1_NAME"])
        ids = pd.to_numeric(e["PLAYER1_ID"], errors="coerce")
        names.update(dict(zip(ids.astype("Int64"), e["PLAYER1_NAME"])))
    return names


def run_ratings(args: argparse.Namespace) -> None:
    from ..twins.similarity import _normalize

    stints = load_stints(args.data_dir, args.first_season, args.last_season)
    production = load_production(args.data_dir, args.first_season, args.last_season)
    season = args.season or args.last_season
    table = ratings_before(stints, season + 1, production=production)
    seasons = list(range(season - DEFAULT_WINDOW + 1, season + 1))
    names = player_names(seasons, args.data_dir)
    table["player"] = table["player_id"].map(names)
    table["minutes"] = table["seconds"] / 60
    span = f"{seasons[0] - 1}-{str(seasons[0])[-2:]} to {season - 1}-{str(season)[-2:]}"
    cols = ["player", "orapm", "drapm", "rapm", "possessions", "minutes"]
    table = table.sort_values("rapm", ascending=False).reset_index(drop=True)
    table.index += 1
    if args.player:
        hit = table[table["player"].fillna("").map(_normalize).str.contains(_normalize(args.player))]
        if hit.empty:
            raise SystemExit(f"No player matching {args.player!r} in {span}")
        print(f"RAPM over {span} (rank among all {len(table)} players):\n")
        print(hit[cols].round(1).to_string())
        return
    board = table[table["possessions"] >= args.min_possessions].head(args.top)
    print(f"RAPM over {span}, points per 100 possessions vs an average player "
          f"({args.min_possessions:.0f}+ offensive possessions):\n")
    print(board[cols].round(1).to_string())


def run_backtest(args: argparse.Namespace) -> None:
    from .rapm import on_court_net

    stints = load_stints(args.data_dir, args.first_season, args.last_season)
    appearances = load_production(args.data_dir, args.first_season, args.last_season)

    def production(train: pd.DataFrame, season: int) -> pd.DataFrame:
        """Box-score production per game in each player's most recent season."""
        apps = appearances[appearances["season"].isin(set(train["season"]))]
        apps = apps[apps["season"] == apps.groupby("player_id")["season"].transform("max")]
        table = apps.groupby("player_id", as_index=False)["production"].mean()
        return table.rename(columns={"production": "rapm"})

    raters = {
        "RAPM, box-score prior": lambda train, t: ratings_before(train, t, production=appearances),
        "RAPM, no prior": lambda train, t: ratings_before(train, t),
        "Box-score production": production,
        "APM (no shrinkage)": lambda train, t: ratings_before(train, t, alpha=1.0),
        "On-court net rating": lambda train, t: on_court_net(train),
    }
    result = backtest(stints, range(args.test_from, args.last_season + 1), raters)
    print("Share of each game's final margin explained by the players on the court, "
          "rated from the two seasons before:\n")
    result.index = [f"{s - 1}-{str(s)[-2:]}" for s in result.index]
    print(result.round(3).to_string())
    print("\nAverage:")
    print(result.mean().round(3).to_string())
    wins = (result.idxmax(axis=1) == "RAPM, box-score prior").sum()
    print(f"\nRAPM with the box-score prior is best in {wins} of {len(result)} seasons.")
