"""``basketball-ml impact``, ``impact-backtest``, ``impact-export`` and ``impact-bench`` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pbp
from ..cache import cached_csv
from .rapm import DEFAULT_WINDOW, backtest, rating_schedule, ratings_before
from .stints import build_stints


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--bio-dir", default="data/bref", help="Where draft and birth-date tables are cached")
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

    e = subparsers.add_parser("impact-export", help="Write preseason forecast ratings (season, nba_id, rating).")
    e.add_argument("--out", default="outputs/impact_forecasts.csv", help="CSV to write")
    _common(e)
    e.set_defaults(func=run_export)

    h = subparsers.add_parser("impact-bench", help="Score the forecasts on the Hoop Impact Bench game-margin test.")
    h.add_argument("--bench-dir", default="data/bench", help="Where the benchmark's games and minutes are cached")
    _common(h)
    h.set_defaults(func=run_bench)


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


def load_box(data_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Per-game box scores of every player (the offense/defense prior's input), cached in ``data_dir``."""
    from .box import build_box

    data_dir = Path(data_dir)
    return cached_csv(data_dir / f"box_{first}_{last}.csv", lambda: build_box(range(first, last + 1), data_dir))


def player_names(seasons, data_dir: str | Path) -> dict[int, str]:
    """Latest name of every player id seen in the play-by-play of ``seasons``."""
    names: dict[int, str] = {}
    for season in seasons:
        e = pbp.load_pbp(season, data_dir).dropna(subset=["PLAYER1_NAME"])
        ids = pd.to_numeric(e["PLAYER1_ID"], errors="coerce")
        names.update(dict(zip(ids.astype("Int64"), e["PLAYER1_NAME"])))
    return names


def load_before(data_dir: str | Path, first: int, last: int) -> dict[int, pd.DataFrame]:
    """Ratings fitted before each season from ``first + 2`` to ``last + 1``, cached in ``data_dir``.

    Each uses the two seasons before it, the offense/defense box-score prior and the luck adjustment.
    """
    stints = load_stints(data_dir, first, last)
    box = load_box(data_dir, first, last)

    def build() -> pd.DataFrame:
        seasons = range(first + 2, last + 2)
        return pd.concat([ratings_before(stints, t, box=box, luck=True).assign(season=t) for t in seasons],
                         ignore_index=True)

    table = cached_csv(Path(data_dir) / f"rapm_before_{first}_{last}.csv", build)
    return {int(t): g.drop(columns="season").reset_index(drop=True) for t, g in table.groupby("season")}


def load_projection(data_dir: str | Path, bio_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Projection coefficients for every season from ``first + 3`` to ``last + 1``, cached in ``data_dir``."""
    from .projection import load_bio, projection_coefficients

    def build() -> pd.DataFrame:
        stints = load_stints(data_dir, first, last)
        before = load_before(data_dir, first, last)
        return projection_coefficients(stints, before, load_bio(bio_dir), range(first + 3, last + 2)).reset_index()

    return cached_csv(Path(data_dir) / f"rapm_projection_{first}_{last}.csv", build).set_index("season")


def load_forecasts(data_dir: str | Path, bio_dir: str | Path, first: int, last: int) -> pd.DataFrame:
    """Preseason forecast ratings for every season from ``first + 3`` to ``last + 1``, cached in ``data_dir``."""
    from .projection import load_bio, preseason_ratings

    def build() -> pd.DataFrame:
        stints = load_stints(data_dir, first, last)
        before = load_before(data_dir, first, last)
        return preseason_ratings(stints, before, load_bio(bio_dir), range(first + 3, last + 2),
                                 coefficients=load_projection(data_dir, bio_dir, first, last))

    return cached_csv(Path(data_dir) / f"rapm_forecast_{first}_{last}.csv", build)


def load_schedule(data_dir: str | Path, bio_dir: str | Path, first: int, last: int, dates: pd.Series) -> pd.DataFrame:
    """In-season forecast ratings of every season from ``first + 1`` to ``last``, refreshed every 30 days.

    Each refresh is fitted from the games before it (with the offense/defense
    box-score prior and the luck adjustment) and the projection is added (see
    :func:`basketball_ml.impact.projection.project_schedule`). ``dates`` maps
    game ids to dates. The ratings are cached in ``data_dir``.
    """
    from .projection import load_bio, project_schedule

    stints = load_stints(data_dir, first, last)
    schedule = cached_csv(Path(data_dir) / f"rapm_schedule_{first}_{last}.csv", parse_dates=["as_of"], build=lambda: (
        rating_schedule(stints, dates, range(first + 1, last + 1), box=load_box(data_dir, first, last), luck=True)))
    return project_schedule(schedule, stints, load_bio(bio_dir), load_projection(data_dir, bio_dir, first, last))


def _span(first: int, last: int) -> str:
    return f"{first - 1}-{str(first)[-2:]} to {last - 1}-{str(last)[-2:]}"


def run_ratings(args: argparse.Namespace) -> None:
    from ..twins.similarity import _normalize

    season = args.season or args.last_season
    table = load_before(args.data_dir, args.first_season, args.last_season)[season + 1].copy()
    for col in ("orapm", "drapm", "rapm"):      # on the forecast's scale: the average possession is 0
        table[col] -= np.average(table[col], weights=table["possessions"])
    forecasts = load_forecasts(args.data_dir, args.bio_dir, args.first_season, args.last_season)
    forecast = forecasts[forecasts["season"] == season + 1].set_index("player_id")["rapm"]
    table["forecast"] = table["player_id"].map(forecast)
    seasons = list(range(season - DEFAULT_WINDOW + 1, season + 1))
    names = player_names(seasons, args.data_dir)
    table["player"] = table["player_id"].map(names)
    table["minutes"] = table["seconds"] / 60
    span = _span(seasons[0], season)
    nxt = f"{season}-{str(season + 1)[-2:]}"
    cols = ["player", "orapm", "drapm", "rapm", "forecast", "possessions", "minutes"]
    table = table.sort_values("rapm", ascending=False).reset_index(drop=True)
    table.index += 1
    if args.player:
        hit = table[table["player"].fillna("").map(_normalize).str.contains(_normalize(args.player))]
        if hit.empty:
            raise SystemExit(f"No player matching {args.player!r} in {span}")
        print(f"RAPM over {span} and {nxt} forecast (rank among all {len(table)} players):\n")
        print(hit[cols].round(1).to_string())
        return
    board = table[table["possessions"] >= args.min_possessions].head(args.top)
    print(f"RAPM over {span}, points per 100 possessions vs an average player "
          f"({args.min_possessions:.0f}+ offensive possessions), and the {nxt} forecast:\n")
    print(board[cols].round(1).to_string())


def run_backtest(args: argparse.Namespace) -> None:
    from .rapm import on_court_net

    stints = load_stints(args.data_dir, args.first_season, args.last_season)
    appearances = load_production(args.data_dir, args.first_season, args.last_season)
    box = load_box(args.data_dir, args.first_season, args.last_season)
    forecasts = load_forecasts(args.data_dir, args.bio_dir, args.first_season, args.last_season)

    def production(train: pd.DataFrame, season: int) -> pd.DataFrame:
        """Box-score production per game in each player's most recent season."""
        apps = appearances[appearances["season"].isin(set(train["season"]))]
        apps = apps[apps["season"] == apps.groupby("player_id")["season"].transform("max")]
        table = apps.groupby("player_id", as_index=False)["production"].mean()
        return table.rename(columns={"production": "rapm"})

    best = "Forecast: RAPM + rookies, playing time, age"
    raters = {
        best: lambda train, t: forecasts[forecasts["season"] == t],
        "RAPM, O/D box prior, luck-adjusted": lambda train, t: ratings_before(train, t, box=box, luck=True),
        "RAPM, O/D box prior": lambda train, t: ratings_before(train, t, box=box),
        "RAPM, net box prior": lambda train, t: ratings_before(train, t, production=appearances),
        "RAPM, no prior": lambda train, t: ratings_before(train, t),
        "Box-score production": production,
        "APM (no shrinkage)": lambda train, t: ratings_before(train, t, alpha=1.0),
        "On-court net rating": lambda train, t: on_court_net(train),
    }
    result = backtest(stints, range(args.test_from, args.last_season + 1), raters)
    print("Share of each game's final margin explained by the players on the court, "
          "rated from the two seasons before:\n")
    result.index = [f"{s - 1}-{str(s)[-2:]}" for s in result.index]
    print(result.round(3).T.to_string())
    print("\nAverage:")
    print(result.mean().round(3).to_string())
    wins = (result.idxmax(axis=1) == best).sum()
    print(f"\nThe forecast is best in {wins} of {len(result)} seasons.")


def bench_format(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Forecasts as ``season`` ("2024-25"), ``nba_id`` and ``rating``, the Hoop Impact Bench format."""
    return pd.DataFrame({"season": forecasts["season"].map(lambda t: f"{t - 1}-{str(t)[-2:]}"),
                         "nba_id": forecasts["player_id"].astype(int), "rating": forecasts["rapm"].round(3)})


def run_export(args: argparse.Namespace) -> None:
    forecasts = load_forecasts(args.data_dir, args.bio_dir, args.first_season, args.last_season)
    out = bench_format(forecasts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    seasons = sorted(forecasts["season"].unique())
    print(f"Wrote {len(out)} forecasts for {_span(seasons[0], seasons[-1])} to {args.out}")


def run_bench(args: argparse.Namespace) -> None:
    from .bench import score_ratings

    forecasts = load_forecasts(args.data_dir, args.bio_dir, args.first_season, args.last_season)
    result = score_ratings(bench_format(forecasts), args.bench_dir)
    print("Hoop Impact Bench game-margin test, preseason forecasts, share of game margins explained (R2):\n")
    print(result.round(3).to_string())
