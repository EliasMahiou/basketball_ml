# Odds archive

Snapshots of the NBA's public odds feed (`odds_todaysGames.json` on the `nba-prod-us-east-1-mediaops-stats` bucket), saved several times a day by the `odds archive` workflow on `main`. The feed only shows the odds of today's and the next few games, so this branch keeps them: each game's last odds before tip-off score the live predictions against the sportsbooks over the season.

- `odds/<UTC time>.json.gz`: the feed as it was at that time, gzipped.
- `odds/index.txt`: every snapshot, oldest first.

`basketball_ml.predictions.live.odds_snapshots` reads them (cached in `data/live/odds/`).
