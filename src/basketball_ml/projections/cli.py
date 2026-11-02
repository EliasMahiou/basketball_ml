"""``basketball-ml project`` and ``basketball-ml backtest`` subcommands."""

from __future__ import annotations

import argparse

import pandas as pd

from ..data import FEATURES, load_advanced
from ..twins.similarity import resolve_player, standardize_by_season
from .backtest import backtest, score
from .comparables import TARGETS, build_pairs, project_season

SHOWN = ["bpm", "obpm", "dbpm", "ts_percent", "usg_percent", "ast_percent", "x3p_ar"]
LABELS = {**FEATURES, "bpm": "BPM"}


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = load_advanced(args.data, min_minutes=args.min_minutes)
    if args.last_complete_season:
        seasons = seasons[seasons["season"] <= args.last_complete_season]
    return seasons, standardize_by_season(seasons)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default="data/Advanced.csv", help="Path to Advanced.csv")
    p.add_argument("-k", type=int, default=50, help="Number of comparables")
    p.add_argument("--min-minutes", type=int, default=500, help="Minimum minutes per season")
    p.add_argument("--last-complete-season", type=int, default=None,
                   help="Ignore seasons after this one (e.g. a season still in progress)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("project", help="Project a player's next seasons from his comparables.")
    p.add_argument("player", help='Player name, e.g. "Victor Wembanyama"')
    p.add_argument("--season", type=int, default=None, help="Starting season end year (default: his latest)")
    p.add_argument("--years", type=int, default=3, help="Number of seasons to project")
    _common(p)
    p.set_defaults(func=run_project)

    b = subparsers.add_parser("backtest", help="Measure projection error against simple baselines.")
    b.add_argument("--from-season", type=int, default=2011)
    b.add_argument("--to-season", type=int, default=2025)
    b.add_argument("--horizon", type=int, default=1, help="Seasons ahead")
    _common(b)
    b.set_defaults(func=run_backtest)


def run_project(args: argparse.Namespace) -> None:
    seasons, z = _load(args)
    pid = resolve_player(z, args.player)
    rows = z[z["player_id"] == pid]
    season = args.season or int(rows["season"].max())
    target_z = rows[rows["season"] == season].iloc[0]
    target_raw = seasons[(seasons["player_id"] == pid) & (seasons["season"] == season)].iloc[0]

    table = {f"{season - 1}-{str(season)[-2:]} (actual)": target_raw[SHOWN]}
    comps_first = None
    for h in range(1, args.years + 1):
        pairs = build_pairs(seasons, z, horizon=h)
        proj, comps = project_season(target_z, target_raw, pairs, horizon=h, k=args.k)
        s = season + h
        table[f"{s - 1}-{str(s)[-2:]} (projected)"] = proj[SHOWN]
        if comps_first is None:
            comps_first = comps

    out = pd.DataFrame(table).rename(index=LABELS).astype(float).round(3)
    print(f"{target_z['player']}, age {int(target_z['age'])} in {season - 1}-{str(season)[-2:]}\n")
    print(out.to_string())
    print("\nClosest comparables (season at the same age -> BPM change the next season):")
    top = comps_first.head(5)
    for _, c in top.iterrows():
        print(f"  {c['player']:<28} {int(c['season']) - 1}-{str(int(c['season']))[-2:]}"
              f"  similarity {c['similarity']:.0f}  BPM {c['bpm_now']:+.1f} -> {c['bpm_later']:+.1f}")


def run_backtest(args: argparse.Namespace) -> None:
    seasons, z = _load(args)
    pairs = build_pairs(seasons, z, horizon=args.horizon)
    results = backtest(pairs, range(args.from_season, args.to_season + 1), horizon=args.horizon, k=args.k)
    print(f"{len(results)} player-seasons projected {args.horizon} season(s) ahead, "
          f"{args.from_season} to {args.to_season}. Mean absolute error:\n")
    print(score(results).rename(index=LABELS).round(4).to_string())
