"""``basketball-ml games-backtest``, ``ratings`` and ``predict`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .availability import add_availability, build_player_games, expected_lineup
from .games import LINEUP_FEATURES, add_features, build_games
from .model import evaluate, make_model, walk_forward


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2025, help="Last complete season loaded (end year)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("games-backtest", help="Evaluate game win-probability models season by season.")
    b.add_argument("--test-from", type=int, default=2018, help="First season predicted out of sample")
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


def _cached(path: Path, build, parse_dates=()):
    """Read a derived table from ``path``, building and saving it on first use."""
    if path.exists():
        return pd.read_csv(path, parse_dates=list(parse_dates))
    table = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def _games(args: argparse.Namespace, lineups: bool = True):
    seasons = range(args.first_season, args.last_season + 1)
    tag = f"{args.first_season}_{args.last_season}"
    data_dir = Path(args.data_dir)
    raw = _cached(data_dir / f"games_{tag}.csv", lambda: build_games(seasons, data_dir), parse_dates=["date"])
    games, state = add_features(raw)
    appearances = None
    if lineups:
        appearances = _cached(data_dir / f"appearances_{tag}.csv", lambda: build_player_games(seasons, data_dir))
        games = add_availability(games, appearances)
    return games, state, appearances


def run_backtest(args: argparse.Namespace) -> None:
    games, _, _ = _games(args)
    results = walk_forward(games, range(args.test_from, args.last_season + 1))
    print(f"{len(results)} games from {args.test_from - 1}-{str(args.test_from)[-2:]} to "
          f"{args.last_season - 1}-{str(args.last_season)[-2:]}, each season predicted from earlier ones only:\n")
    print(evaluate(results).round(4).to_string())


def run_ratings(args: argparse.Namespace) -> None:
    _, state, _ = _games(args, lineups=False)
    table = pd.DataFrame({"team": list(state.elo), "elo": list(state.elo.values())})
    table = table.sort_values("elo", ascending=False).reset_index(drop=True)
    table.index += 1
    print(table.round(0).to_string())


def run_predict(args: argparse.Namespace) -> None:
    games, state, appearances = _games(args)
    if args.next_season:
        state.start_season(args.last_season + 1)
    home, away = args.home.upper(), args.away.upper()
    if home not in state.elo or away not in state.elo:
        raise SystemExit(f"Unknown team: use abbreviations like {', '.join(sorted(state.elo)[:5])}...")
    feats = state.features(home, away)
    lineups = {}
    for side, team, out in (("home", home, args.out_home), ("away", away, args.out_away)):
        names = [n.strip() for n in out.split(",") if n.strip()]
        lineups[side] = expected_lineup(appearances, games, team, out=names)
    feats["strength_diff"] = lineups["home"][0] - lineups["away"][0]
    model = make_model("logistic").fit(games[LINEUP_FEATURES], games["home_win"])
    p = float(model.predict_proba(pd.DataFrame([feats])[LINEUP_FEATURES])[0, 1])
    print(f"{home} (home) vs {away}: {home} {p:.0%}, {away} {1 - p:.0%}")
    print(f"  Elo {state.elo[home]:.0f} vs {state.elo[away]:.0f}, "
          f"last-10 margin difference {feats['form_diff']:+.1f}, "
          f"lineup strength {lineups['home'][0]:.0f} vs {lineups['away'][0]:.0f}")
    for side, team in (("home", home), ("away", away)):
        top = lineups[side][1].head(5)
        print(f"  {team} top players counted: {', '.join(top['player'])}")
