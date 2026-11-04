"""``basketball-ml winprob-backtest``, ``winprob-game`` and ``wpa`` subcommands."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..cache import cached_csv
from .model import METHODS, NAMES, bootstrap_difference, calibration_error, endgame, evaluate, \
    evaluate_at_time_left, trainable, walk_forward
from .states import add_game_info, possession_check, season_states

BEST = "stern"  # tied for the lowest out-of-sample log loss in the backtest, and the best calibrated
TEST_FROM = 2016
STATE_DTYPES = {"game_id": "int32", "season": "int16", "row": "int16", "event": "int8", "period": "int8",
                "seconds_left": "int16", "margin": "int16", "poss": "int8", "loose": "int8", "fts": "int8",
                "side": "int8", "p1": "int32", "p2": "int32", "p3": "int32", "subs": "int16"}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last season loaded (end year)")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("winprob-backtest", help="Evaluate live win-probability models season by season.")
    b.add_argument("--test-from", type=int, default=TEST_FROM, help="First season predicted out of sample")
    b.add_argument("--calibration", metavar="PNG", help="Also save a reliability chart to this path")
    _common(b)
    b.set_defaults(func=run_backtest)

    g = subparsers.add_parser("winprob-game", help="Chart a game's live win probability.")
    g.add_argument("game_id", nargs="?", type=int, help="stats.nba.com game id, e.g. 22500123")
    g.add_argument("--biggest-comeback", action="store_true", help="Chart the season's biggest comeback")
    g.add_argument("--season", type=int, help="Season of --biggest-comeback (default: the last loaded)")
    g.add_argument("--model", default=BEST, choices=[m for m in METHODS if m != "pregame"],
                   help=f"Whose probabilities to use (default: {BEST})")
    g.add_argument("--out", metavar="PNG", help="Where to save the chart (default: outputs/winprob_<game>.png)")
    _common(g)
    g.set_defaults(func=run_game)

    w = subparsers.add_parser("wpa", help="Win probability added: season leaderboard of players.")
    w.add_argument("--season", type=int, help="Season (default: the last loaded)")
    w.add_argument("--clutch", action="store_true",
                   help="Only the last 5 minutes of the 4th quarter and overtime with the margin within 5")
    w.add_argument("--on-court", action="store_true",
                   help="Rank by the team's win probability added while each player is on the court")
    w.add_argument("--top", type=int, default=20, help="Players in the leaderboard")
    w.add_argument("--model", default=BEST, choices=[m for m in METHODS if m != "pregame"],
                   help=f"Whose probabilities to use (default: {BEST})")
    _common(w)
    w.set_defaults(func=run_wpa)


def load_games(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Games with pre-game Elo (``predictions.add_features``), cached in ``data_dir``."""
    from ..predictions import add_features, build_games

    data_dir = Path(data_dir)
    raw = cached_csv(data_dir / f"games_{first}_{last}.csv", lambda: build_games(range(first, last + 1), data_dir),
                     parse_dates=["date"])
    return add_features(raw)[0]


def load_states(data_dir: str | Path, seasons, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Game states of ``seasons`` (cached one season per file), with pre-game strength and result if ``games``."""
    data_dir = Path(data_dir)
    parts = []
    for season in seasons:
        table = cached_csv(data_dir / f"winprob_states_{season}.csv", lambda s=season: season_states(s, data_dir))
        parts.append(table.astype(STATE_DTYPES))
    states = pd.concat(parts, ignore_index=True)
    return add_game_info(states, games) if games is not None else states


def load_results(data_dir: str | Path, first: int, last: int, test_from: int = TEST_FROM,
                 seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Out-of-sample win probabilities of every state from ``test_from`` to ``last`` (or of ``seasons`` only).

    Each season is predicted by models fit on the seasons from ``first`` to
    the one before it, and cached in ``data_dir``.
    """
    data_dir = Path(data_dir)
    games = load_games(data_dir, first, last)
    methods = [m for m in METHODS if m != "pregame"]
    tests = range(test_from, last + 1)
    paths = {s: data_dir / f"winprob_probs_{first}_{s}.csv" for s in tests}
    wanted = [s for s in (tests if seasons is None else seasons) if s in paths]
    missing = [s for s in tests if not paths[s].exists()]
    states = load_states(data_dir, range(first, last + 1) if missing else wanted, games)
    states = states[trainable(states)]
    for season in missing:
        res = walk_forward(states[states["season"] <= season], [season], methods)
        cached_csv(paths[season], lambda res=res: res[["game_id", "row", *[f"p_{m}" for m in methods]]])
    if not wanted:
        return states.iloc[:0].assign(**{f"p_{m}": [] for m in METHODS})
    probs = pd.concat([pd.read_csv(paths[s]) for s in wanted], ignore_index=True)
    results = states[states["season"].isin(wanted)].merge(probs, on=["game_id", "row"], how="inner")
    results["p_pregame"] = results["elo_prob"]
    return results


def _season(s: int) -> str:
    return f"{s - 1}-{str(s)[-2:]}"


def run_backtest(args: argparse.Namespace) -> None:
    results = load_results(args.data_dir, args.first_season, args.last_season, args.test_from)
    games = results["game_id"].nunique()
    print(f"{len(results):,} game states ({games:,} games) from {_season(args.test_from)} to "
          f"{_season(args.last_season)}, each season predicted by models fit on earlier seasons only.\n")
    table = evaluate(results)
    for metric, label in (("log_loss", "Log loss"), ("brier", "Brier score")):
        show = pd.DataFrame({"states": table[("states", "")].astype(int)})
        for m in METHODS:
            show[NAMES[m]] = table[(metric, m)].astype(float)
        print(f"{label} by game phase (lower is better; every event counts once, and events of a game "
              "are correlated):\n")
        print(show.round(4).to_string(), "\n")

    seasons = {_season(s): evaluate(g)[[("log_loss", m) for m in METHODS]].loc["All"].to_numpy()
               for s, g in results.groupby("season")}
    print("Log loss by test season:\n")
    print(pd.DataFrame(seasons, index=[NAMES[m] for m in METHODS]).T.astype(float).round(4).to_string())

    print("\nLog loss at fixed minutes left in regulation (one state per game):\n")
    print(evaluate_at_time_left(results).round(4).to_string())

    print("\nDifference in log loss, with a 95% interval from resampling whole games:")
    for a, b in (("boosting", "logistic"), ("boosting", "stern"), ("logistic", "stern")):
        for label, mask in (("all states", None), ("last 2 minutes", endgame(results))):
            d, lo, hi = bootstrap_difference(results, a, b, mask=mask)
            print(f"  {a} - {b}, {label}: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")

    print("\nCalibration (state-weighted average gap between predicted and actual):")
    for m in METHODS[1:]:
        print(f"  {NAMES[m]}: all states {calibration_error(results, m):.1%}, "
              f"last 2 minutes {calibration_error(results, m, mask=endgame(results)):.1%}")

    from .model import MAX_TRAIN_ROWS, SternModel, model_frame

    games_table = load_games(args.data_dir, args.first_season, args.last_season)
    train = load_states(args.data_dir, range(args.first_season, args.last_season), games_table)
    train = train[trainable(train)]
    train = train.sample(min(MAX_TRAIN_ROWS, len(train)), random_state=0)
    stern = SternModel().fit(model_frame(train), train["home_win"].to_numpy()).describe()
    print(f"\nBrownian model fit on {_season(args.first_season)} to {_season(args.last_season - 1)}: "
          f"final margin noise sigma = {stern['sigma']:.1f} points per 48 minutes (+{stern['extra_seconds']:.0f} s); "
          f"the ball is worth {stern['ball']:+.2f} points to the team that has it, a rebound pending after its "
          f"miss {stern['rebound']:+.2f}, each free throw to shoot {stern['free_throw']:+.2f}; "
          f"pre-game Elo spread weight {stern['spread_weight']:.2f}.")

    states = load_states(args.data_dir, range(args.test_from, args.last_season + 1))
    check = possession_check(states[trainable(states)])
    print("\nPossession inference: the team inferred to have the ball is the one that "
          f"shoots {check['shots']:.2%} of the time, turns it over {check['turnovers']:.2%}; "
          f"{check['rebounds after a loose ball']:.2%} of rebounds follow a missed shot or free throw.")
    if args.calibration:
        from .plot import calibration_plot
        print(f"Saved {calibration_plot(results, args.calibration)}")


def run_game(args: argparse.Namespace) -> None:
    from .plot import game_plot
    from .wpa import biggest_comebacks, game_curve

    games = load_games(args.data_dir, args.first_season, args.last_season)
    if args.biggest_comeback:
        season = args.season or args.last_season
    elif args.game_id:
        hit = games.loc[games["game_id"] == args.game_id, "season"]
        if hit.empty:
            raise SystemExit(f"Unknown game {args.game_id}")
        season = int(hit.iloc[0])
    else:
        raise SystemExit("Give a GAME_ID or --biggest-comeback")
    results = load_results(args.data_dir, args.first_season, args.last_season, seasons=[season])
    if args.biggest_comeback:
        top = biggest_comebacks(results, games, args.model)
        if top.empty:
            raise SystemExit(f"No out-of-sample probabilities for {_season(season)}")
        print(f"Biggest comebacks of {_season(season)} (lowest win probability of the eventual winner):\n")
        print(top.head(10).to_string(index=False))
        game_id = int(top.iloc[0]["game_id"])
    else:
        game_id = args.game_id
    curve = game_curve(results, games, game_id, args.model, args.data_dir)
    out = args.out or f"outputs/winprob_{game_id}.png"
    print(f"\nSaved {game_plot(curve, games[games['game_id'] == game_id].iloc[0], out)}")


def run_wpa(args: argparse.Namespace) -> None:
    from ..impact.cli import player_names
    from .wpa import on_court_wpa, player_wpa

    season = args.season or args.last_season
    results = load_results(args.data_dir, args.first_season, args.last_season, seasons=[season])
    if results.empty:
        raise SystemExit(f"No out-of-sample probabilities for {_season(season)}")
    games = load_games(args.data_dir, args.first_season, args.last_season)
    names = player_names([season], args.data_dir)
    if args.on_court:
        from ..impact.cli import load_stints

        stints = load_stints(args.data_dir, args.first_season, args.last_season)
        table = on_court_wpa(results, games, stints[stints["season"] == season], args.model)
        sort = "on_court_wpa"
    else:
        table = player_wpa(results, games, args.model)
        sort = "clutch_wpa" if args.clutch else "wpa"
    table["player"] = table["player_id"].map(names)
    table = table.sort_values(sort, ascending=False).head(args.top).reset_index(drop=True)
    table.index += 1
    what = {"wpa": "Win probability added", "clutch_wpa": "Clutch win probability added (last 5 minutes "
            "of the 4th quarter or overtime, margin within 5)", "on_court_wpa": "Team win probability added "
            "while on the court"}[sort]
    print(f"{what}, {_season(season)}, in wins:\n")
    cols = ["player", "team", "games", sort] + [c for c in table.columns if c.startswith("from_") and sort == "wpa"]
    if sort != "wpa" and "wpa" in table:
        cols.insert(4, "wpa")
    print(table[[c for c in cols if c in table]].round(2).to_string())
