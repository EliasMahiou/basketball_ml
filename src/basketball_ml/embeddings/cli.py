"""``basketball-ml style-twins``, ``archetypes`` and ``style-backtest`` subcommands."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..twins.similarity import resolve_player, similarity_score
from .archetypes import cluster, style_map
from .features import build_style_table, style_columns
from .model import StyleAutoencoder, embed, identification, summarize


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data/pbp", help="Where play-by-play files are cached")
    p.add_argument("--first-season", type=int, default=2014, help="First season loaded (end year)")
    p.add_argument("--last-season", type=int, default=2026, help="Last season loaded (end year)")
    p.add_argument("--min-fga", type=int, default=300, help="Minimum shot attempts in a season")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    t = subparsers.add_parser("style-twins", help="Players with the closest play-by-play style.")
    t.add_argument("player", help='Player name, e.g. "Stephen Curry"')
    t.add_argument("--season", type=int, default=None, help="Season end year (default: his latest)")
    t.add_argument("-k", type=int, default=10, help="Number of twins to show")
    _common(t)
    t.set_defaults(func=run_twins)

    a = subparsers.add_parser("archetypes", help="Group one season's players into style archetypes.")
    a.add_argument("--season", type=int, default=2026)
    a.add_argument("-k", type=int, default=8, help="Number of archetypes")
    a.add_argument("--map", default=None, help="Output PNG path for the style map")
    _common(a)
    a.set_defaults(func=run_archetypes)

    b = subparsers.add_parser("style-backtest",
                              help="How often a style vector finds the same player the next season.")
    b.add_argument("--advanced", default=None, help="Optional Advanced.csv to compare with box-score stats")
    _common(b)
    b.set_defaults(func=run_backtest)


def load_table(args: argparse.Namespace) -> pd.DataFrame:
    """Style table, cached as CSV next to the play-by-play files."""
    cache = Path(args.data_dir) / f"style_{args.first_season}_{args.last_season}_{args.min_fga}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    table = build_style_table(range(args.first_season, args.last_season + 1), args.data_dir, args.min_fga)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


def _label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def run_twins(args: argparse.Namespace) -> None:
    table = load_table(args).assign(mp=lambda d: d["touches"])
    cols = style_columns(table)
    pid = resolve_player(table, args.player)
    rows = table[table["player_id"] == pid]
    season = args.season or int(rows["season"].max())
    target = rows[rows["season"] == season]
    if target.empty:
        raise SystemExit(f"No {season} season with {args.min_fga}+ attempts for {args.player}")
    x = target[cols].to_numpy(float)[0]
    cands = table[table["player_id"] != pid].copy()
    cands["distance"] = np.linalg.norm(cands[cols].to_numpy(float) - x, axis=1)
    cands["similarity"] = similarity_score(cands["distance"].to_numpy(), len(cols))
    cands = cands.sort_values("distance").drop_duplicates("player_id").head(args.k)
    print(f"Closest play-by-play styles to {target['player'].iloc[0]} in {_label(season)}:\n")
    for _, c in cands.iterrows():
        print(f"  {c['player']:<26} {_label(int(c['season']))}  similarity {c['similarity']:.1f}")


def run_archetypes(args: argparse.Namespace) -> None:
    table = load_table(args)
    cols = style_columns(table)
    season = table[table["season"] == args.season].reset_index(drop=True)
    grouped, names = cluster(season, cols, k=args.k)
    for _, n in names.iterrows():
        members = grouped[grouped["archetype"] == n["archetype"]].sort_values("fga", ascending=False)
        print(f"{n['archetype']}. {n['traits']} ({n['players']} players)")
        print(f"   {', '.join(members['player'].head(8))}")
    if args.map:
        # Label the highest-volume shooter of each archetype, then the next biggest names.
        by_volume = grouped.sort_values("fga", ascending=False)
        stars = by_volume.drop_duplicates("archetype")["player"].tolist()
        stars += [p for p in by_volume["player"] if p not in stars][:12 - len(stars)]
        out = style_map(grouped, cols, args.map, names=names, highlight=stars,
                        title=f"NBA style map, {_label(args.season)} (from play-by-play)")
        print(f"\nStyle map saved: {out}")


def normalize_name(name: str) -> str:
    """Lowercase, no accents, punctuation or suffixes: "Jaren Jackson Jr." -> "jaren jackson"."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return " ".join(n.split())


def run_backtest(args: argparse.Namespace) -> None:
    table = load_table(args)
    cols = style_columns(table)
    vectors = {"Play-by-play style (34 features)": cols}
    if args.advanced:
        from ..data import FEATURES, load_advanced
        from ..twins.similarity import standardize_by_season

        adv = standardize_by_season(load_advanced(args.advanced, min_minutes=0))
        adv = adv.assign(key=adv["player"].map(normalize_name))
        box = {f: f"box_{f}" for f in FEATURES}
        table = table.assign(key=table["player"].map(normalize_name)).merge(
            adv[["key", "season", *FEATURES]].rename(columns=box), on=["key", "season"])
        table = table.drop_duplicates(["player_id", "season"]).reset_index(drop=True)
        vectors["Box-score advanced stats (12)"] = list(box.values())
        vectors["Both"] = cols + list(box.values())
    seasons = range(args.first_season, args.last_season)
    train = (table["season"] <= args.first_season + (args.last_season - args.first_season) // 2).to_numpy()
    rows = {name: summarize(identification(table, table[c].to_numpy(float), seasons)) for name, c in vectors.items()}
    for method in ("pca", "autoencoder"):
        v = embed(table, cols, method, dim=8, train_mask=train)
        rows[f"Style compressed to 8 numbers ({method})"] = summarize(identification(table, v, seasons))
    out = pd.DataFrame(rows).T
    out["players"] = out["players"].astype(int)
    print(f"Each player's season is compared with every player's next season "
          f"({int(out['players'].iloc[0])} player-seasons). top1 = his own next season comes out closest.\n")
    print(out.round(3).to_string())
    ae = StyleAutoencoder(dim=8).fit(table.loc[train, cols].to_numpy(float))
    print(f"\nAutoencoder: 8 numbers rebuild {ae.reconstruction_r2(table.loc[~train, cols].to_numpy(float)):.0%} "
          f"of the style features' variance on held-out seasons.")
