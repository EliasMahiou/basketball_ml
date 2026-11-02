"""``basketball-ml twins`` subcommand."""

from __future__ import annotations

import argparse

import pandas as pd

from ..data import load_advanced
from .plot import add_percentiles, radar
from .similarity import career_profiles, find_twins, resolve_player, standardize_by_season


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("twins", help="Find the statistical twins of an NBA player.")
    p.add_argument("player", help='Player name, e.g. "LeBron James"')
    p.add_argument("--data", default="data/Advanced.csv", help="Path to Advanced.csv")
    p.add_argument("--season", type=int, default=None,
                   help="Season end year (2013 = 2012-13). Without this flag: career profile.")
    p.add_argument("-k", type=int, default=10, help="Number of twins to show")
    p.add_argument("--min-minutes", type=int, default=500, help="Minimum minutes per season")
    p.add_argument("--min-career-minutes", type=int, default=10000,
                   help="Minimum career minutes (career mode)")
    p.add_argument("--radar", default=None, help="Output PNG path for a player vs best twin radar chart")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    seasons = load_advanced(args.data, min_minutes=args.min_minutes)
    z = standardize_by_season(seasons)
    if args.season is None:
        table = career_profiles(z)
        table = table[table["mp"] >= args.min_career_minutes].reset_index(drop=True)
    else:
        table = z
    pid = resolve_player(table, args.player)

    twins = find_twins(table, pid, season=args.season, k=args.k)
    cols = ["player"] + (["season", "team", "age"] if args.season else ["first_season", "last_season"])
    pd.set_option("display.width", 120)
    print(twins[cols + ["similarity"]].round({"similarity": 1}).to_string(index=False))

    if args.radar:
        pct = add_percentiles(table)
        key = pct["player_id"] == pid
        if args.season is not None:
            key &= pct["season"] == args.season
        target = pct[key].iloc[0]
        best = twins.iloc[0]
        key2 = pct["player_id"] == best["player_id"]
        if args.season is not None:
            key2 &= pct["season"] == best["season"]
        out = radar([target, pct[key2].iloc[0]], args.radar,
                    title=f"{target['player']}'s twin: {best['player']} ({best['similarity']:.0f}/100)")
        print(f"Radar chart saved: {out}")
