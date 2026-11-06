"""``basketball-ml tracking-shots`` and ``tracking-sportvu``: what player tracking adds to shot quality."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .. import pbp
from ..shots.features import LOCATION_FEATURES, build_margins, prepare_shots
from ..shots.model import ShotModel
from . import sportvu
from .shotlogs import (NAMES, SEASON, by_defender_distance, compare, contests, cross_validate, download_shot_logs,
                       load_shot_logs, making_change, match_shots, paired_difference, split_half)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    t = subparsers.add_parser("tracking-shots", help="What player tracking (2014-15 shot logs) adds to expected FG%%.")
    t.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    t.add_argument("--tracking-dir", default="data/tracking", help="Where the shot logs are cached")
    t.add_argument("--top", type=int, default=10, help="Players in each list")
    t.add_argument("--chart", metavar="PNG", help="Also save a chart of FG%% by the closest defender's distance")
    t.set_defaults(func=run_shots)

    s = subparsers.add_parser("tracking-sportvu",
                              help="Shot features from the raw 2015-16 tracking positions, and what they add to xFG.")
    s.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    s.add_argument("--tracking-dir", default="data/tracking", help="Where the shot logs and game features are cached")
    s.add_argument("--games", type=int, help="Only the first N games of the release (each downloads about 6 MB)")
    s.add_argument("--workers", type=int, default=3, help="Games read at once (about 1 GB of memory each)")
    s.set_defaults(func=run_sportvu)


def season_shots(data_dir: str | Path) -> pd.DataFrame:
    """2014-15 shot chart rows ready for the xFG models, with the unclipped period for matching."""
    raw = pbp.load_shots(SEASON, data_dir)
    shots = prepare_shots(raw, build_margins([SEASON], data_dir))
    shots["raw_period"] = raw["PERIOD"].to_numpy()
    return shots


def run_shots(args: argparse.Namespace) -> None:
    logs = load_shot_logs(download_shot_logs(args.tracking_dir))
    shots = season_shots(args.data_dir)
    matched = match_shots(shots, logs)
    print(f"{len(logs):,} tracked shots of 2014-15, {len(matched):,} matched to the shot charts "
          f"(same game, shooter, result and shot value) in {matched['game_id'].nunique()} games.\n")

    oof = cross_validate(matched)
    table = compare(matched, oof)
    table.index = [NAMES[n] for n in table.index]
    print("Expected FG% models, 5-fold cross-validation over games:\n")
    print(table.round(4).to_string())
    for a, b in (("kind + defender", "kind"), ("kind + tracking", "kind"), ("type + tracking", "type")):
        d, lo, hi = paired_difference(matched, oof, a, b)
        print(f"  {NAMES[a].strip()} vs {NAMES[b].strip()}: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")

    expected = oof["kind"].to_numpy()
    print("\nFG% by the closest defender's distance, against xFG without tracking:\n")
    print(by_defender_distance(matched, expected).round(3).to_string())

    change = making_change(matched, expected, oof["kind + tracking"].to_numpy())
    r = np.corrcoef(change["before"], change["after"])[0, 1]
    print(f"\nShot making (FG% above expected) of {len(change)} players with 250+ tracked shots: "
          f"correlation {r:.3f} between the models without and with tracking.")
    for name, col in (("without", "kind"), ("with", "kind + tracking")):
        rel = split_half(matched, matched["made"].to_numpy() - oof[col].to_numpy())
        print(f"  Odd vs even games, {name} tracking: r = {rel:.3f}")
    show = ["player", "shots", "fg", "before", "after", "defender"]
    print("\nBest shot makers once openness is accounted for:\n")
    print(change[show].head(args.top).round(3).to_string(index=False))
    print("\nWho lost the most (open shots):\n")
    print(change.sort_values("change")[show].head(args.top).round(3).to_string(index=False))

    contest = contests(matched, expected)
    rel = split_half(matched[matched["def_dist"] < 6],
                     (matched["made"] - expected)[matched["def_dist"] < 6].to_numpy(), key="defender_id")
    print(f"\nDefenders whose contests (within 6 ft) go in least, against xFG without tracking "
          f"(odd vs even games r = {rel:.3f}):\n")
    print(contest[["defender", "shots", "fg", "above", "distance"]].head(args.top).round(3).to_string(index=False))
    if args.chart:
        from .plot import defender_chart
        print(f"\nSaved {defender_chart(matched, expected, args.chart)}")


def run_sportvu(args: argparse.Namespace) -> None:
    raw = pbp.load_shots(sportvu.SEASON, args.data_dir)
    print("Reading the raw 2015-16 tracking, game by game (cached after the first run):")
    features = sportvu.season_features(raw, Path(args.tracking_dir) / "sportvu", games=args.games,
                                       workers=args.workers, progress=print)
    shots = prepare_shots(raw, build_margins([sportvu.SEASON], args.data_dir))
    found = shots.merge(features, on=["game_id", "game_event_id"], how="inner")
    if found.empty:
        raise SystemExit("No shot was found in the tracking data")
    found["def_dist"] = found["def_dist_sv"]
    gap = (found["distance"] - found["sv_dist"]).abs()
    print(f"\n{len(found):,} shots found in the positions of {found['game_id'].nunique()} games. Their distance "
          f"from the basket is {gap.median():.2f} ft from the shot chart's (median), correlation "
          f"{np.corrcoef(found['distance'], found['sv_dist'])[0, 1]:.3f}, {(gap < 6).mean():.1%} within 6 ft.")

    logs = load_shot_logs(download_shot_logs(args.tracking_dir))
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    spread = pd.DataFrame({"2015-16, from the positions": found["def_dist_sv"].quantile(q).to_numpy(),
                           "2014-15, NBA shot logs": logs["def_dist"].quantile(q).to_numpy()},
                          index=[f"{int(x * 100)}th percentile" for x in q]).T
    print("\nClosest defender's distance at the release (ft):\n")
    print(spread.round(2).to_string())

    oof = cross_validate(found, sportvu.RAW_SETS)
    table = compare(found, oof)
    table.index = [sportvu.RAW_NAMES[n] for n in table.index]
    print("\nExpected FG% models, 5-fold cross-validation over games of 2015-16:\n")
    print(table.round(4).to_string())
    for a in ("kind + defender", "kind + all raw"):
        d, lo, hi = paired_difference(found, oof, a, "kind")
        print(f"  {sportvu.RAW_NAMES[a].strip()}: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    rim = found[found["reached_rim"] == 1].reset_index(drop=True)
    rim_oof = cross_validate(rim, sportvu.RAW_SETS)
    print(f"\nThe same on the {len(rim):,} shots whose release was found from the ball reaching the rim:")
    for a in ("kind + defender", "kind + all raw"):
        d, lo, hi = paired_difference(rim, rim_oof, a, "kind")
        print(f"  {sportvu.RAW_NAMES[a].strip()}: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")

    print("\nEach other feature on top of the defender's distance (log loss change):")
    base = sportvu.RAW_SETS["kind + defender"]
    for feature in sportvu.RAW_FEATURES[1:]:
        extra = cross_validate(found, {feature: [*base, feature]})
        d, lo, hi = paired_difference(found, extra.assign(base=oof["kind + defender"].to_numpy()), feature, "base")
        print(f"  {feature:14s} {d:+.4f} ({lo:+.4f} to {hi:+.4f})")

    print("\nFG% by the closest defender's distance, against xFG without tracking:\n")
    print(by_defender_distance(found, oof["kind"].to_numpy()).round(3).to_string())

    train = match_shots(season_shots(args.data_dir), logs)
    y = found["made"].to_numpy()
    print("\nTrained on the 2014-15 shot logs, tested on 2015-16 with the distances from the positions:")
    for name, features in (("kind", [*LOCATION_FEATURES, "kind"]),
                           ("kind + defender", [*LOCATION_FEATURES, "kind", "def_dist"])):
        p = ShotModel(features=features).fit(train).predict(found)
        print(f"  {sportvu.RAW_NAMES[name].strip():40s} log loss {log_loss(y, p):.4f}, AUC {roc_auc_score(y, p):.3f}")
