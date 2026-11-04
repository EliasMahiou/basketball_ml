"""``basketball-ml games-backtest``, ``ratings``, ``predict`` and ``tonight`` subcommands."""

from __future__ import annotations

import argparse
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import cached_csv
from .availability import (DEFAULT_SECONDS, MIN_GAMES, add_availability, add_impact, build_player_games,
                           expected_lineup, lineup_rapm)
from .games import IMPACT_FEATURES, add_features, build_games
from .model import calibration, calibration_error, evaluate, make_model, walk_forward

GAME_COLUMNS = ["game_id", "date", "home", "away", "home_pts", "away_pts", "season", "margin", "home_win"]
APPEARANCE_COLUMNS = ["season", "game_id", "team", "player_id", "player", "production"]
SECONDS_COLUMNS = ["season", "game_id", "side", "player_id", "seconds"]


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--bio-dir", default="data/bref", help="Where draft and birth date tables are cached")
    p.add_argument("--first-season", type=int, default=2013, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last complete season loaded (end year)")


def _live(p: argparse.ArgumentParser) -> None:
    p.add_argument("--live-dir", default="data/live", help="Where box scores of the season in progress are cached")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    b = subparsers.add_parser("games-backtest", help="Evaluate game win-probability models season by season.")
    b.add_argument("--test-from", type=int, default=2018, help="First season predicted out of sample")
    b.add_argument("--calibration", metavar="PNG", help="Also save a reliability chart to this path")
    b.add_argument("--market", action="store_true",
                   help="Also compare with closing betting lines (downloaded to --odds-dir on first use)")
    b.add_argument("--odds-dir", default="data/odds", help="Where the betting-line file is cached")
    _common(b)
    b.set_defaults(func=run_backtest)

    r = subparsers.add_parser("ratings", help="Team Elo ratings at the end of the last season loaded.")
    _common(r)
    r.set_defaults(func=run_ratings)

    p = subparsers.add_parser("predict", help="Home win probability for a matchup, from the latest ratings.")
    p.add_argument("home", help="Home team abbreviation, e.g. BOS")
    p.add_argument("away", help="Away team abbreviation, e.g. LAL")
    p.add_argument("--live", action="store_true",
                   help="Bring the ratings up to date with the season in progress, from the NBA's live feed")
    p.add_argument("--next-season", action="store_true",
                   help="Treat the game as the start of the next season (ratings regressed, form reset)")
    p.add_argument("--out-home", default="", help='Home players who will not play, e.g. "Jayson Tatum,Jrue Holiday"')
    p.add_argument("--out-away", default="", help="Away players who will not play")
    _common(p)
    _live(p)
    p.set_defaults(func=run_predict)

    t = subparsers.add_parser("tonight", help="Win probabilities for one day of the season in progress, "
                                              "next to the sportsbooks' odds, from the NBA's live feed.")
    t.add_argument("--date", help="Day of the games, YYYY-MM-DD (default: the next day with games to play)")
    t.add_argument("--out", default="", help='Players who will not play, e.g. "Jayson Tatum,Joel Embiid"')
    _common(t)
    _live(t)
    t.set_defaults(func=run_tonight)


def _games(args: argparse.Namespace, lineups: bool = True):
    """Games with pre-game features, the league state after the last one, and lineup data."""
    from ..impact import player_seconds
    from ..impact.cli import load_forecasts, load_schedule, load_stints

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
    ratings = load_schedule(data_dir, args.bio_dir, args.first_season, args.last_season, dates)
    seconds = player_seconds(stints)
    games = add_impact(games, seconds, ratings)
    forecasts = load_forecasts(data_dir, args.bio_dir, args.first_season, args.last_season)
    latest = forecasts[forecasts["season"] == args.last_season + 1].drop(columns="season")
    return games, state, {"appearances": appearances, "seconds": seconds, "latest_ratings": latest}


def _label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def run_backtest(args: argparse.Namespace) -> None:
    games, _, _ = _games(args)
    results = walk_forward(games, range(args.test_from, args.last_season + 1))
    print(f"{len(results)} games from {_label(args.test_from)} to {_label(args.last_season)}, "
          f"each season predicted from earlier ones only:\n")
    print(evaluate(results).round(4).to_string())
    print("\nCalibration of the logistic model (share of home wins by predicted probability):\n")
    print(calibration(results).round(3).to_string())
    print(f"\nAverage gap between predicted and actual: {calibration_error(results):.1%}")
    if args.calibration:
        from .plot import calibration_plot
        print(f"Saved {calibration_plot(results, args.calibration)}")
    if args.market:
        from .market import add_market, market_backtest

        lined = market_backtest(add_market(games, args.odds_dir), range(args.test_from, args.last_season + 1))
        table = evaluate(lined, ("market", "logistic", "blend"))
        table.index = ["closing line", "model", "model + closing line"]
        print(f"\nAgainst the betting market, on the {len(lined)} of these games with a closing moneyline:\n")
        print(table.round(4).to_string())


def run_ratings(args: argparse.Namespace) -> None:
    _, state, _ = _games(args, lineups=False)
    table = pd.DataFrame({"team": list(state.elo), "elo": list(state.elo.values())})
    table = table.sort_values("elo", ascending=False).reset_index(drop=True)
    table.index += 1
    print(table.round(0).to_string())


def expected_seconds(seconds: pd.DataFrame, season: int) -> dict:
    """Each player's usual time on the court per game: this season's average after 3 games, else last season's."""
    this = seconds[seconds["season"] == season].groupby("player_id")["seconds"].agg(["mean", "size"])
    last = seconds[seconds["season"] == season - 1].groupby("player_id")["seconds"].mean()
    out = this["mean"].to_dict()
    few = set(this.index[this["size"] < MIN_GAMES])
    out.update({pid: value for pid, value in last.items() if pid not in out or pid in few})
    return out


def matchup(home: str, away: str, state, appearances: pd.DataFrame, games: pd.DataFrame, seconds: dict,
            rating: dict, out: tuple[list[str], list[str]] = ([], []), date: pd.Timestamp | None = None):
    """Pre-game features of ``home`` vs ``away`` after the last game in ``games``, with the regulars expected to play.

    ``out`` lists the home and away players who will not play. Returns the
    features and, per side, (lineup strength, regulars, lineup impact).
    """
    from ..impact import NEWCOMER

    feats = state.features(home, away, date)
    lineups = {}
    for side, team, names in (("home", home, out[0]), ("away", away, out[1])):
        strength, regulars = expected_lineup(appearances, games, team, out=names)
        minutes = {pid: seconds.get(pid, DEFAULT_SECONDS) for pid in regulars["player_id"]}
        lineups[side] = (strength, regulars, lineup_rapm(minutes, rating, NEWCOMER))
    feats["strength_diff"] = lineups["home"][0] - lineups["away"][0]
    feats["rapm_diff"] = lineups["home"][2] - lineups["away"][2]
    return feats, lineups


def live_ratings(args: argparse.Namespace, season: int, played: pd.DataFrame, dates: pd.Series,
                 newcomers: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player ratings of the season in progress: its 30-day refreshes so far, and ratings through its last game.

    The season's play-by-play comes from the live feed and is added to the
    archived seasons; both tables get the forecast layer (see
    :func:`basketball_ml.impact.projection.project_schedule`), with
    draft-pick and age values for ``newcomers``.
    """
    from ..impact import box_scores, project, project_schedule, rating_schedule, ratings_as_of, season_stints
    from ..impact.cli import load_before, load_box, load_projection, load_stints
    from ..impact.projection import load_bio
    from . import live

    bio = load_bio(args.bio_dir)
    coefficients = load_projection(args.data_dir, args.bio_dir, args.first_season, args.last_season)
    newcomers = set(newcomers) | set(bio.index[bio["draft_year"] == season - 1])
    if played.empty:
        current = project(load_before(args.data_dir, args.first_season, args.last_season)[season], season, bio,
                          coefficients.loc[season], sorted(newcomers))
        return current.assign(season=season, as_of=pd.NaT), current
    events = live.season_events(played, args.live_dir)
    stints = pd.concat([load_stints(args.data_dir, args.first_season, args.last_season),
                        season_stints(season, events=events)], ignore_index=True)
    box = pd.concat([load_box(args.data_dir, args.first_season, args.last_season),
                     box_scores(events).assign(season=season)], ignore_index=True)
    schedule = project_schedule(rating_schedule(stints, dates, [season], box=box, luck=True), stints, bio,
                                coefficients)
    latest = ratings_as_of(stints, dates, season, played["date"].max() + pd.Timedelta(days=1), box=box, luck=True)
    newcomers |= set(stints.loc[stints["season"] == season, [f"{s}_{i}" for s in ("home", "away")
                                                              for i in range(1, 6)]].to_numpy().ravel())
    current = project(latest, season, bio, coefficients.loc[season], sorted(newcomers))
    return schedule, current


def live_season(args: argparse.Namespace) -> dict:
    """The game model and the season in progress, brought up to its last finished game.

    Past seasons come from the cached archives (through ``--last-season``);
    the season in progress from the NBA's live feed: Elo, form, rest and
    lineups from its games and box scores, player ratings refit from its
    play-by-play (see :func:`live_ratings`).
    """
    from . import live

    history, _, data = _games(args)
    model = make_model("logistic").fit(history[IMPACT_FEATURES], history["home_win"])
    now = live.season_games()
    season = int(now["season"].iloc[0])
    if season != args.last_season + 1:
        raise SystemExit(f"The live feed is on {_label(season)}: load past seasons through {_label(season - 1)} "
                         f"(--last-season {season - 1}).")
    played = now[now["status"] == live.FINAL]
    apps_now = live.season_appearances(played, args.live_dir) if len(played) else pd.DataFrame(
        columns=list(dict.fromkeys([*APPEARANCE_COLUMNS, *SECONDS_COLUMNS])))
    dates = pd.concat([history.set_index("game_id")["date"], played.set_index("game_id")["date"]])
    schedule, current = live_ratings(args, season, played, dates, set(apps_now["player_id"]))

    games, state = add_features(pd.concat([history[GAME_COLUMNS], played[GAME_COLUMNS]], ignore_index=True))
    recent = games[games["season"] >= season - 1]
    old_apps, old_seconds = data["appearances"], data["seconds"]
    appearances = pd.concat([old_apps.loc[old_apps["season"] == season - 1, APPEARANCE_COLUMNS],
                             apps_now[APPEARANCE_COLUMNS]], ignore_index=True)
    seconds = pd.concat([old_seconds.loc[old_seconds["season"] == season - 1, SECONDS_COLUMNS],
                         apps_now[SECONDS_COLUMNS]], ignore_index=True)
    recent = add_availability(recent, appearances)
    recent = add_impact(recent, seconds, schedule.dropna(subset=["as_of"]))
    return {"model": model, "schedule": now, "season": season, "state": state, "games": recent,
            "so_far": recent[recent["season"] == season], "appearances": appearances,
            "seconds": expected_seconds(seconds, season), "rating": dict(zip(current["player_id"], current["rapm"]))}


def _names(text: str) -> list[str]:
    return [n.strip() for n in text.split(",") if n.strip()]


def run_predict(args: argparse.Namespace) -> None:
    if args.live:
        ctx = live_season(args)
        model, state, games, appearances = ctx["model"], ctx["state"], ctx["games"], ctx["appearances"]
        seconds, rating = ctx["seconds"], ctx["rating"]
    else:
        games, state, data = _games(args)
        model = make_model("logistic").fit(games[IMPACT_FEATURES], games["home_win"])
        appearances = data["appearances"]
        seconds = expected_seconds(data["seconds"], args.last_season)
        rating = dict(zip(data["latest_ratings"]["player_id"], data["latest_ratings"]["rapm"]))
        if args.next_season:
            state.start_season(args.last_season + 1)
    home, away = args.home.upper(), args.away.upper()
    if home not in state.elo or away not in state.elo:
        raise SystemExit(f"Unknown team: use abbreviations like {', '.join(sorted(state.elo)[:5])}...")
    feats, lineups = matchup(home, away, state, appearances, games, seconds, rating,
                             out=(_names(args.out_home), _names(args.out_away)))
    p = float(model.predict_proba(pd.DataFrame([feats])[IMPACT_FEATURES])[0, 1])
    print(f"{home} (home) vs {away}: {home} {p:.0%}, {away} {1 - p:.0%}")
    print(f"  Elo {state.elo[home]:.0f} vs {state.elo[away]:.0f}, "
          f"last-10 margin difference {feats['form_diff']:+.1f}, "
          f"lineup strength {lineups['home'][0]:.0f} vs {lineups['away'][0]:.0f}, "
          f"lineup impact {lineups['home'][2]:+.1f} vs {lineups['away'][2]:+.1f} per 100 possessions")
    for side, team in (("home", home), ("away", away)):
        top = lineups[side][1].head(5)
        print(f"  {team} top players counted: {', '.join(top['player'])}")


def _scores(y: np.ndarray, p: np.ndarray) -> str:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return f"accuracy {np.mean((p > 0.5) == y):.1%}, log loss {loss:.3f}"


def run_tonight(args: argparse.Namespace) -> None:
    from . import live

    ctx = live_season(args)
    model, schedule, season = ctx["model"], ctx["schedule"], ctx["season"]
    so_far = ctx["so_far"]
    if len(so_far):
        y = so_far["home_win"].to_numpy(float)
        p = model.predict_proba(so_far[IMPACT_FEATURES])[:, 1]
        print(f"{_label(season)} so far, {len(so_far)} games, each predicted from the games before it "
              f"(with the players who played): model {_scores(y, p)}; "
              f"Elo alone {_scores(y, so_far['elo_prob'].to_numpy())}.\n")
    upcoming = schedule[schedule["status"] != live.FINAL]
    if upcoming.empty:
        raise SystemExit("No games left to play this season.")
    date = pd.Timestamp(args.date) if args.date else upcoming["date"].min()
    day = upcoming[upcoming["date"] == date]
    if day.empty:
        raise SystemExit(f"No games to play on {date:%Y-%m-%d}.")
    try:
        market = live.odds_today()
    except (urllib.error.URLError, KeyError, ValueError):
        market = pd.Series(dtype=float)
    out = _names(args.out)
    rows = []
    for g in day.itertuples(index=False):
        feats, lineups = matchup(g.home, g.away, ctx["state"], ctx["appearances"], ctx["games"], ctx["seconds"],
                                 ctx["rating"], out=(out, out), date=date)
        p = float(model.predict_proba(pd.DataFrame([feats])[IMPACT_FEATURES])[0, 1])
        rows.append({"game": f"{g.away} @ {g.home}", "home win": f"{p:.0%}",
                     "sportsbooks": f"{market[g.game_id]:.0%}" if g.game_id in market.index else "-",
                     "elo": f"{feats['elo_diff']:+.0f}", "lineup impact": f"{feats['rapm_diff']:+.1f}",
                     "rest": f"{feats['rest_diff']:+d}"})
    table = pd.DataFrame(rows).set_index("game")
    print(f"{date:%A %B} {date.day}, {date.year}: home win probability, next to the sportsbooks' current odds with their margin "
          f"removed\n(elo: home Elo edge including home court; lineup impact: home minus away, points per 100 "
          f"possessions; rest: home minus away days):\n")
    print(table.to_string())
    if out:
        print(f"\nCounted as out: {', '.join(out)}.")
