"""``basketball-ml xfg-backtest``, ``shot-making`` and ``shot-quality`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import pbp
from ..cache import cached_csv
from .features import build_margins, build_shot_table, fill_zones, prepare_shots, score_before
from .making import (MIN_SHOTS, consecutive, efficiency_shares, player_seasons, predict_next, team_quality,
                     year_to_year)
from .model import (DEFAULT_WINDOW, KEEP, NAMES, ShotModel, calibration, evaluate, prediction_columns, season_table,
                    walk_forward)

GAMES = 82   # per team in a season
TEAMS = 30


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last season loaded (end year)")


def _live(p: argparse.ArgumentParser) -> None:
    p.add_argument("--live", action="store_true",
                   help="Add the season in progress, from the play-by-play of the NBA's live feed")
    p.add_argument("--live-dir", default="data/live", help="Where files of finished games of the live feed are cached")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("xfg-backtest", help="Expected FG%% models, each season predicted from the 3 before it.")
    b.add_argument("--chart", metavar="PNG", help="Also save a half-court map of expected FG%% to this path")
    _common(b)
    b.set_defaults(func=run_backtest)

    m = subparsers.add_parser("shot-making", help="Points scored above expected on the same shots, by player.")
    m.add_argument("player", nargs="?", help="Show one player's seasons instead of the leaderboard")
    m.add_argument("--season", type=int, help="Leaderboard season (default: the last loaded)")
    m.add_argument("--top", type=int, default=20, help="Players in the leaderboard")
    m.add_argument("--min-shots", type=int,
                   help=f"Minimum attempts in a season (default {MIN_SHOTS}, in proportion to the games played "
                        f"for a season in progress)")
    m.add_argument("--persistence", action="store_true",
                   help="Instead: how well each shooting number holds up from one season to the next")
    _common(m)
    _live(m)
    m.set_defaults(func=run_making)

    q = subparsers.add_parser("shot-quality", help="Team shot quality and shot making, on offense and defense.")
    q.add_argument("--season", type=int, help="Season (default: the last loaded)")
    _common(q)
    _live(q)
    q.set_defaults(func=run_quality)


def _margins(data_dir: Path, first: int, last: int) -> pd.DataFrame:
    """Score margin before every shot from ``first`` to ``last`` (cached)."""
    return cached_csv(data_dir / f"xfg_margins_{first}_{last}.csv",
                      lambda: build_margins(range(first, last + 1), data_dir))


def load_predictions(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Out-of-sample xFG of every shot from season ``first + 3`` to ``last``, cached in ``data_dir``.

    Each season is predicted by models trained on the 3 seasons before it;
    the score margins taken from the play-by-play are cached too.
    """
    data_dir = Path(data_dir)

    def build() -> pd.DataFrame:
        shots = build_shot_table(range(first, last + 1), data_dir, _margins(data_dir, first, last))
        out = walk_forward(shots, range(first + DEFAULT_WINDOW, last + 1))
        out[["loc_x", "loc_y"]] = out[["loc_x", "loc_y"]].astype(int)
        cols = list(prediction_columns(out).values())
        out[cols] = out[cols].astype(float).round(4)
        return out

    return cached_csv(data_dir / f"xfg_predictions_{first}_{last}.csv", build)


def live_shots(games: pd.DataFrame, live_dir: str | Path = "data/live") -> pd.DataFrame:
    """Model-ready shots of every finished game of ``games`` (the season in progress), from the live feed.

    The feed's play-by-play gives each shot the shot chart's fields (see
    :func:`basketball_ml.pbp.shots_from_cdn`) but its zones, which
    :func:`fill_zones` adds, and the score before it; the box scores give the
    players' full names.
    """
    from ..predictions import live

    actions = live.season_actions(games, live_dir)
    finished = games[games["status"] == live.FINAL]
    names = live.season_appearances(finished, live_dir).drop_duplicates("player_id").set_index("player_id")["player"]
    raw = pbp.shots_from_cdn(actions, games, names).assign(season=int(games["season"].iloc[0]))
    return prepare_shots(raw, score_before(pbp.from_cdn(actions)))


def live_predictions(data_dir: str | Path, live_dir: str | Path, first: int, last: int) -> tuple[pd.DataFrame, dict]:
    """xFG of every shot of the season in progress, from the xFG model trained on the 3 seasons before it.

    Returns the shots (the columns of :func:`load_predictions`, with the main
    model only) and the season, its games so far and the date of the last.
    """
    from ..predictions import live

    games = live.season_games()
    season = int(games["season"].iloc[0])
    if season != last + 1:
        raise SystemExit(f"The live feed is on {_label(season)}: load past seasons through {_label(season - 1)} "
                         f"(--last-season {season - 1}).")
    if season - DEFAULT_WINDOW < first:
        raise SystemExit(f"The xFG model of {_label(season)} is trained on the {DEFAULT_WINDOW} seasons before it: "
                         f"load seasons from {_label(season - DEFAULT_WINDOW)} (--first-season {season - DEFAULT_WINDOW}).")
    data_dir = Path(data_dir)
    train = build_shot_table(range(season - DEFAULT_WINDOW, season), data_dir, _margins(data_dir, first, last))
    shots = fill_zones(live_shots(games, live_dir), train)
    out = shots[[c for c in KEEP if c in shots]].copy()
    out["xfg"] = ShotModel().fit(train).predict(shots).round(4)
    out[["loc_x", "loc_y"]] = out[["loc_x", "loc_y"]].astype(int)
    played = games[games["game_id"].isin(out["game_id"])]
    return out, {"season": season, "games": len(played), "through": played["date"].max()}


def _results(args: argparse.Namespace) -> tuple[pd.DataFrame, dict | None]:
    """Out-of-sample xFG of every shot, with the season in progress when ``--live`` (and its details)."""
    results = load_predictions(args.data_dir, args.first_season, args.last_season)
    if not getattr(args, "live", False):
        return results, None
    now, info = live_predictions(args.data_dir, args.live_dir, args.first_season, args.last_season)
    return pd.concat([results, now], ignore_index=True), info


def _in_progress(live: dict | None) -> set[int]:
    return {live["season"]} if live else set()


def _so_far(season: int, live: dict | None) -> str:
    """" so far, 121 games through Nov 5" for the season in progress."""
    if not live or season != live["season"]:
        return ""
    return f" so far, {live['games']} games through {live['through']:%b} {live['through'].day}"


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
    out["skill_per_100"] = table["per_100_skill"]
    return out


def _min_shots(args: argparse.Namespace, season: int, live: dict | None) -> int:
    """``--min-shots``, or :data:`MIN_SHOTS` in proportion to the games played for a season in progress."""
    if args.min_shots is not None:
        return args.min_shots
    if not live or season != live["season"]:
        return MIN_SHOTS
    return max(1, round(MIN_SHOTS * 2 * live["games"] / (TEAMS * GAMES)))


def run_making(args: argparse.Namespace) -> None:
    from ..twins.similarity import _normalize

    results, live = _results(args)
    in_progress = _in_progress(live)
    players = player_seasons(results, in_progress=in_progress)
    if args.persistence:
        _persistence(players[~players["season"].isin(in_progress)],
                     MIN_SHOTS if args.min_shots is None else args.min_shots)
        return
    # A finished season ranks players by its own shot making, a season in progress by the estimate from last season's.
    players["score"] = players["per_100_skill"].where(players["season"].isin(in_progress), players["per_100_shrunk"])
    note = (f"shrunk_per_100: points per 100 shots shrunk toward 0 by each sample's noise (half of the observed "
            f"spread is skill at {players.attrs['half_shots']:.0f} attempts).\nskill_per_100: shrunk toward "
            f"{players.attrs['rho']:.2f} x last season's instead (skill spreads {players.attrs['s']:.1f} points per "
            f"100 around that, {players.attrs['tau']:.1f} around 0); a season in progress is ranked by it.")
    if args.player:
        hit = players[players["player"].map(_normalize).str.contains(_normalize(args.player))].copy()
        if hit.empty:
            raise SystemExit(f"No player matching {args.player!r} in {_span(players['season'])}")
        least = players["season"].map(lambda s: _min_shots(args, s, live))
        qualified = players[players["fga"] >= least]
        rank = qualified.groupby("season")["score"].rank(ascending=False).astype(int)
        size = qualified.groupby("season")["score"].transform("size")
        hit["rank"] = (rank.astype(str) + " of " + size.astype(str)).reindex(hit.index).fillna("-")
        among = f"{_min_shots(args, 0, None)}+ attempts"
        if live and live["season"] in set(hit["season"]):
            among += f", {_min_shots(args, live['season'], live)}+ in {_label(live['season'])} so far"
        print(f"Shot making by season: points above an average shooter on the same shots (rank among {among}).\n"
              f"{note}\n")
        board = _board(hit).assign(rank=hit["rank"])
        print(board.drop(columns="player").set_index("season").round(1).to_string())
        return
    season = args.season or int(players["season"].max())
    least = _min_shots(args, season, live)
    board = players[(players["season"] == season) & (players["fga"] >= least)]
    board = board.sort_values("score", ascending=False).head(args.top)
    out = _board(board).drop(columns="season").reset_index(drop=True)
    out.index += 1
    print(f"Best shot makers of {_label(season)}{_so_far(season, live)} ({least}+ attempts): points above an "
          f"average shooter taking the same shots\n(xFG set to the season's league level and corrected for each "
          f"arena's scorekeeper bias on twos).\n{note}\n")
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
    results, live = _results(args)
    teams = team_quality(results, in_progress=_in_progress(live))
    season = args.season or int(teams["season"].max())
    one = teams[teams["season"] == season].sort_values("net_pps", ascending=False).set_index("team")
    cols = ["off_xpps", "off_making", "off_pps", "def_xpps", "def_making", "def_pps", "net_xpps", "net_pps"]
    print(f"Team shots in {_label(season)}{_so_far(season, live)}: expected points per shot (xpps, shot quality), "
          f"points above that (making) and actual points per shot (pps), on offense and allowed on defense\n(xFG "
          f"set to the season's league level and corrected for each arena's scorekeeper bias on twos):\n")
    print(one[cols].round(3).to_string())

    done = teams[~teams["season"].isin(_in_progress(live))]
    print(f"\nShare of the spread in points per shot explained by shot quality vs shot making "
          f"(within-season deviations, {_span(done['season'])}; sd in points per shot):\n")
    print(efficiency_shares(done).round(3).to_string())
    this = efficiency_shares(teams[teams["season"] == season])
    print(f"\n{_label(season)} alone: offense {this.loc['offense', 'quality_share']:.0%} quality / "
          f"{this.loc['offense', 'making_share']:.0%} making, defense {this.loc['defense', 'quality_share']:.0%} / "
          f"{this.loc['defense', 'making_share']:.0%}.")
    pairs = consecutive(done, key="team", min_shots=0)
    metrics = {"off_xpps": "Offense shot quality", "off_making": "Offense shot making",
               "def_xpps": "Defense shot quality allowed", "def_making": "Defense shot making allowed"}
    y2y = year_to_year(pairs, list(metrics))
    y2y.index = [metrics[m] for m in y2y.index]
    print(f"\nYear-to-year correlation of team numbers ({len(pairs)} team pairs of consecutive seasons):\n")
    print(y2y.drop(columns="pairs").round(3).to_string())
