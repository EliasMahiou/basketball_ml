"""``basketball-ml games-backtest``, ``ratings`` and ``predict`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..cache import cached_csv
from .availability import add_availability, add_impact, build_player_games, expected_lineup, lineup_rapm
from .games import IMPACT_FEATURES, add_features, build_games
from .model import calibration, calibration_error, evaluate, make_model, walk_forward


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last complete season loaded (end year)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("games-backtest", help="Evaluate game win-probability models season by season.")
    b.add_argument("--test-from", type=int, default=2018, help="First season predicted out of sample")
    b.add_argument("--calibration", metavar="PNG", help="Also save a reliability chart to this path")
    _common(b)
    b.set_defaults(func=run_backtest)

    r = subparsers.add_parser("ratings", help="Team Elo ratings at the end of the last season loaded.")
    _common(r)
    r.set_defaults(func=run_ratings)

    p = subparsers.add_parser("predict", help="Home win probability for a matchup, from the latest ratings.")
    p.add_argument("home", help="Home team abbreviation, e.g. BOS")
    p.add_argument("away", help="Away team abbreviation, e.g. LAL")
    p.add_argument("--next-season", action="store_true",
                   help="Treat the game as the start of the next season (ratings regressed, form reset)")
    p.add_argument("--out-home", default="", help='Home players who will not play, e.g. "Jayson Tatum,Jrue Holiday"')
    p.add_argument("--out-away", default="", help="Away players who will not play")
    _common(p)
    p.set_defaults(func=run_predict)


def _games(args: argparse.Namespace, lineups: bool = True):
    """Games with pre-game features, the league state after the last one, and lineup data."""
    from ..impact import player_seconds, rating_schedule, ratings_before
    from ..impact.cli import load_stints

    seasons = range(args.first_season, args.last_season + 1)
    tag = f"{args.first_season}_{args.last_season}"
    data_dir = Path(args.data_dir)
    raw = cached_csv(data_dir / f"games_{tag}.csv", lambda: build_games(seasons, data_dir), parse_dates=["date"])
    games, state = add_features(raw)
    if not lineups:
        return games, state, None
    appearances = cached_csv(data_dir / f"appearances_{tag}.csv", lambda: build_player_games(seasons, data_dir))
    games = add_availability(games, appearances)
    stints = load_stints(data_dir, args.first_season, args.last_season)

    dates = games.set_index("game_id")["date"]
    ratings = cached_csv(data_dir / f"rapm_schedule_{tag}.csv", parse_dates=["as_of"], build=lambda: rating_schedule(
        stints, dates, range(args.first_season + 1, args.last_season + 1)))
    seconds = player_seconds(stints)
    games = add_impact(games, seconds, ratings)
    latest = cached_csv(data_dir / f"rapm_latest_{tag}.csv", lambda: ratings_before(stints, args.last_season + 1))
    return games, state, {"appearances": appearances, "seconds": seconds, "latest_ratings": latest}


def run_backtest(args: argparse.Namespace) -> None:
    games, _, _ = _games(args)
    results = walk_forward(games, range(args.test_from, args.last_season + 1))
    print(f"{len(results)} games from {args.test_from - 1}-{str(args.test_from)[-2:]} to "
          f"{args.last_season - 1}-{str(args.last_season)[-2:]}, each season predicted from earlier ones only:\n")
    print(evaluate(results).round(4).to_string())
    print("\nCalibration of the logistic model (share of home wins by predicted probability):\n")
    print(calibration(results).round(3).to_string())
    print(f"\nAverage gap between predicted and actual: {calibration_error(results):.1%}")
    if args.calibration:
        from .plot import calibration_plot
        print(f"Saved {calibration_plot(results, args.calibration)}")


def run_ratings(args: argparse.Namespace) -> None:
    _, state, _ = _games(args, lineups=False)
    table = pd.DataFrame({"team": list(state.elo), "elo": list(state.elo.values())})
    table = table.sort_values("elo", ascending=False).reset_index(drop=True)
    table.index += 1
    print(table.round(0).to_string())


def run_predict(args: argparse.Namespace) -> None:
    from ..impact import NEWCOMER

    games, state, lineup_data = _games(args)
    if args.next_season:
        state.start_season(args.last_season + 1)
    home, away = args.home.upper(), args.away.upper()
    if home not in state.elo or away not in state.elo:
        raise SystemExit(f"Unknown team: use abbreviations like {', '.join(sorted(state.elo)[:5])}...")
    feats = state.features(home, away)
    seconds = lineup_data["seconds"]
    seconds = seconds[seconds["season"] == args.last_season].groupby("player_id")["seconds"].mean()
    latest = lineup_data["latest_ratings"]
    rating = dict(zip(latest["player_id"], latest["rapm"]))
    lineups = {}
    for side, team, out in (("home", home, args.out_home), ("away", away, args.out_away)):
        names = [n.strip() for n in out.split(",") if n.strip()]
        strength, regulars = expected_lineup(lineup_data["appearances"], games, team, out=names)
        expected = {pid: float(seconds.get(pid, 0.0)) for pid in regulars["player_id"]}
        lineups[side] = (strength, regulars, lineup_rapm(expected, rating, NEWCOMER))
    feats["strength_diff"] = lineups["home"][0] - lineups["away"][0]
    feats["rapm_diff"] = lineups["home"][2] - lineups["away"][2]
    model = make_model("logistic").fit(games[IMPACT_FEATURES], games["home_win"])
    p = float(model.predict_proba(pd.DataFrame([feats])[IMPACT_FEATURES])[0, 1])
    print(f"{home} (home) vs {away}: {home} {p:.0%}, {away} {1 - p:.0%}")
    print(f"  Elo {state.elo[home]:.0f} vs {state.elo[away]:.0f}, "
          f"last-10 margin difference {feats['form_diff']:+.1f}, "
          f"lineup strength {lineups['home'][0]:.0f} vs {lineups['away'][0]:.0f}, "
          f"lineup impact {lineups['home'][2]:+.1f} vs {lineups['away'][2]:+.1f} per 100 possessions")
    for side, team in (("home", home), ("away", away)):
        top = lineups[side][1].head(5)
        print(f"  {team} top players counted: {', '.join(top['player'])}")
