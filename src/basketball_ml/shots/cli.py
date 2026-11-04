"""``basketball-ml xfg-backtest``, ``shot-making`` and ``shot-quality`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..cache import cached_csv
from .features import build_margins, build_shot_table
from .making import (MIN_SHOTS, consecutive, efficiency_shares, player_seasons, predict_next, team_quality,
                     year_to_year)
from .model import DEFAULT_WINDOW, NAMES, calibration, evaluate, prediction_columns, season_table, walk_forward


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last season loaded (end year)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("xfg-backtest", help="Expected FG%% models, each season predicted from the 3 before it.")
    b.add_argument("--chart", metavar="PNG", help="Also save a half-court map of expected FG%% to this path")
    _common(b)
    b.set_defaults(func=run_backtest)

    m = subparsers.add_parser("shot-making", help="Points scored above expected on the same shots, by player.")
    m.add_argument("player", nargs="?", help="Show one player's seasons instead of the leaderboard")
    m.add_argument("--season", type=int, help="Leaderboard season (default: the last loaded)")
    m.add_argument("--top", type=int, default=20, help="Players in the leaderboard")
    m.add_argument("--min-shots", type=int, default=MIN_SHOTS, help="Minimum attempts in a season")
    m.add_argument("--persistence", action="store_true",
                   help="Instead: how well each shooting number holds up from one season to the next")
    _common(m)
    m.set_defaults(func=run_making)

    q = subparsers.add_parser("shot-quality", help="Team shot quality and shot making, on offense and defense.")
    q.add_argument("--season", type=int, help="Season (default: the last loaded)")
    _common(q)
    q.set_defaults(func=run_quality)


def load_predictions(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Out-of-sample xFG of every shot from season ``first + 3`` to ``last``, cached in ``data_dir``.

    Each season is predicted by models trained on the 3 seasons before it;
    the score margins taken from the play-by-play are cached too.
    """
    data_dir = Path(data_dir)
    tag = f"{first}_{last}"

    def build() -> pd.DataFrame:
        margins = cached_csv(data_dir / f"xfg_margins_{tag}.csv", lambda: build_margins(range(first, last + 1), data_dir))
        shots = build_shot_table(range(first, last + 1), data_dir, margins)
        out = walk_forward(shots, range(first + DEFAULT_WINDOW, last + 1))
        out[["loc_x", "loc_y"]] = out[["loc_x", "loc_y"]].astype(int)
        cols = list(prediction_columns(out).values())
        out[cols] = out[cols].astype(float).round(4)
        return out

    return cached_csv(data_dir / f"xfg_predictions_{tag}.csv", build)


def _label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def _span(seasons) -> str:
    return f"{_label(min(seasons))} to {_label(max(seasons))}"


def run_backtest(args: argparse.Namespace) -> None:
    results = load_predictions(args.data_dir, args.first_season, args.last_season)
    seasons = sorted(results["season"].unique())
    print(f"Expected FG%, {_span(seasons)}: each season's shots predicted by models trained on the "
          f"{DEFAULT_WINDOW} seasons before it ({len(results):,} shots).\n")
    table = evaluate(results)
    table.index = [NAMES.get(m, m) for m in table.index]
    print(table.round(4).to_string())

    print("\nxFG model by season (fg and xfg: actual and average predicted FG%; "
          "auc_gain: AUC added by the shot type over the shot kind):\n")
    by_season = season_table(results)
    if "xfg_kind" in results:
        by_season["auc_gain"] = by_season["auc"] - season_table(results, "xfg_kind")["auc"]
    by_season.index = [_label(s) for s in by_season.index]
    print(by_season.round(4).to_string())

    cal = calibration(results)
    print("\nCalibration: shots in tenths by xFG, average xFG vs actual FG%:\n")
    print(cal.round(3).to_string())
    gap = (cal["predicted"] - cal["actual"]).abs()
    print(f"\nAverage gap {gap.mean():.1%}, largest {gap.max():.1%}.")
    if args.chart:
        from .plot import court_map

        recent = results[results["season"] > args.last_season - DEFAULT_WINDOW]
        title = f"Expected FG% by shot location, {_span(recent['season'])} ({len(recent):,} shots)"
        print(f"Saved {court_map(recent, args.chart, title=title)}")


def _board(table: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard columns, percentages in points."""
    out = table[["player", "team", "season", "fga"]].copy()
    out["season"] = out["season"].map(_label)
    out["fg%"] = 100 * table["fg_pct"]
    out["xfg%"] = 100 * table["xfg_pct"]
    out["efg%"] = 100 * table["efg_pct"]
    out["points_added"] = table["points_added"]
    out["per_100"] = table["per_100"]
    out["shrunk_per_100"] = table["per_100_shrunk"]
    return out


def run_making(args: argparse.Namespace) -> None:
    from ..twins.similarity import _normalize

    results = load_predictions(args.data_dir, args.first_season, args.last_season)
    players = player_seasons(results)
    if args.persistence:
        _persistence(players, args.min_shots)
        return
    note = (f"points per 100 shots shrunk toward 0 by each sample's noise; half of the observed spread "
            f"is skill at {players.attrs['half_shots']:.0f} attempts")
    if args.player:
        hit = players[players["player"].map(_normalize).str.contains(_normalize(args.player))].copy()
        if hit.empty:
            raise SystemExit(f"No player matching {args.player!r} in {_span(players['season'])}")
        qualified = players[players["fga"] >= args.min_shots]
        rank = qualified.groupby("season")["per_100_shrunk"].rank(ascending=False).astype(int)
        size = qualified.groupby("season")["per_100_shrunk"].transform("size")
        hit["rank"] = (rank.astype(str) + " of " + size.astype(str)).reindex(hit.index).fillna("-")
        print(f"Shot making by season: points above an average shooter on the same shots "
              f"(shrunk_per_100: {note}; rank among {args.min_shots}+ attempts):\n")
        board = _board(hit).assign(rank=hit["rank"])
        print(board.drop(columns="player").set_index("season").round(1).to_string())
        return
    season = args.season or int(players["season"].max())
    board = players[(players["season"] == season) & (players["fga"] >= args.min_shots)]
    board = board.sort_values("per_100_shrunk", ascending=False).head(args.top)
    out = _board(board).drop(columns="season").reset_index(drop=True)
    out.index += 1
    print(f"Best shot makers of {_label(season)} ({args.min_shots}+ attempts): points above an average shooter "
          f"taking the same shots\n(xFG set to the season's league level and corrected for each arena's "
          f"scorekeeper bias on twos).\nshrunk_per_100: {note}.\n")
    print(out.round(1).to_string())


def _persistence(players: pd.DataFrame, min_shots: int) -> None:
    pairs = consecutive(players, min_shots=min_shots)
    first, last = int(pairs["season"].min()), int(pairs["season"].max())
    span = f"from {_label(first)}/{_label(first + 1)} to {_label(last)}/{_label(last + 1)}"
    names = {"fg_pct": "FG%", "efg_pct": "eFG%", "xfg_pct": "xFG% (shot diet)", "xefg_pct": "xeFG% (shot diet)",
             "fg_above": "FG% above expected", "per_100": "Points added per 100 shots",
             "per_100_shrunk": "Points added per 100 shots, shrunk"}
    print(f"Year-to-year correlation, {len(pairs):,} players with {min_shots}+ attempts in two consecutive "
          f"seasons ({span}; r_lowest/r_highest: range over the season pairs):\n")
    y2y = year_to_year(pairs, list(names))
    y2y.index = [names[m] for m in y2y.index]
    print(y2y.drop(columns="pairs").round(3).to_string())

    if pairs["season"].nunique() < 2:
        return
    print("\nPredicting next season, linear fits on last season's numbers (each season pair predicted "
          "from the others):\n")
    rows = {}
    for target, label, sets in (
            ("fg_pct", "Next FG%", {"FG%": ["fg_pct"], "FG% above expected": ["fg_above"],
                                    "xFG% + FG% above expected": ["xfg_pct", "fg_above"]}),
            ("efg_pct", "Next eFG%", {"eFG%": ["efg_pct"], "Points added per 100 shots": ["per_100"],
                                      "xeFG% + points added per 100": ["xefg_pct", "per_100"]}),
            ("fg_above", "Next FG% above expected", {"FG%": ["fg_pct"], "FG% above expected": ["fg_above"]})):
        result = predict_next(pairs, target, sets)
        for name, row in result.iterrows():
            rows[(label, name)] = {"r": row["r"], "rmse (points)": 100 * row["rmse"]}
    table = pd.DataFrame(rows).T
    table.index.names = ["target", "from last season's"]
    print(table.round(3).to_string())


def run_quality(args: argparse.Namespace) -> None:
    results = load_predictions(args.data_dir, args.first_season, args.last_season)
    teams = team_quality(results)
    season = args.season or int(teams["season"].max())
    one = teams[teams["season"] == season].sort_values("net_pps", ascending=False).set_index("team")
    cols = ["off_xpps", "off_making", "off_pps", "def_xpps", "def_making", "def_pps", "net_xpps", "net_pps"]
    print(f"Team shots in {_label(season)}: expected points per shot (xpps, shot quality), points above that "
          f"(making) and actual points per shot (pps), on offense and allowed on defense\n(xFG set to the "
          f"season's league level and corrected for each arena's scorekeeper bias on twos):\n")
    print(one[cols].round(3).to_string())

    print(f"\nShare of the spread in points per shot explained by shot quality vs shot making "
          f"(within-season deviations, {_span(teams['season'])}; sd in points per shot):\n")
    print(efficiency_shares(teams).round(3).to_string())
    this = efficiency_shares(teams[teams["season"] == season])
    print(f"\n{_label(season)} alone: offense {this.loc['offense', 'quality_share']:.0%} quality / "
          f"{this.loc['offense', 'making_share']:.0%} making, defense {this.loc['defense', 'quality_share']:.0%} / "
          f"{this.loc['defense', 'making_share']:.0%}.")
    pairs = consecutive(teams, key="team", min_shots=0)
    metrics = {"off_xpps": "Offense shot quality", "off_making": "Offense shot making",
               "def_xpps": "Defense shot quality allowed", "def_making": "Defense shot making allowed"}
    y2y = year_to_year(pairs, list(metrics))
    y2y.index = [metrics[m] for m in y2y.index]
    print(f"\nYear-to-year correlation of team numbers ({len(pairs)} team pairs of consecutive seasons):\n")
    print(y2y.drop(columns="pairs").round(3).to_string())
