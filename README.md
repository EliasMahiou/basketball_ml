# basketball_ml

Machine learning on NBA data. Each project lives in its own module of the `basketball_ml` package and shares the same data loaders.

| Module | Status | What it does |
| --- | --- | --- |
| [`twins`](#statistical-twins) | Ready | Finds a player's "statistical twin" for a season or a whole career, across every era |
| [`projections`](#career-projections) | Ready | Projects a player's next seasons from what his twins did at the same age |
| `predictions` | Planned | Predicts game outcomes |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
```

### Data

Download the Kaggle dataset [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) and put `Advanced.csv` in `data/`.

```bash
# with the Kaggle CLI (kaggle.json configured)
kaggle datasets download sumitrodatta/nba-aba-baa-stats -f Advanced.csv -p data --unzip
```

## Statistical twins

![LeBron James's twin](docs/lebron_career.png)

### Method

1. **Data**: Basketball-Reference advanced stats, one row per player and season, since 1979-80. Only seasons with at least 500 minutes are kept.
2. **12 style and efficiency stats**, independent of pace and playing time: TS%, 3-point attempt rate, free throw rate, ORB%, DRB%, AST%, STL%, BLK%, TOV%, USG%, OBPM, DBPM.
3. **Per-season standardization**: each stat becomes a z-score computed within its season, so a 2025 3-point rate can be compared with a 1995 one.
4. **Career profile**: the mean of a player's season z-scores, weighted by minutes played (10,000 career minutes minimum).
5. **Nearest neighbors**: Euclidean distance between profiles, converted into a score out of 100. A score of 100 means an identical profile; an average gap of half a standard deviation per stat gives about 61.

### Usage

```bash
# Career twins
basketball-ml twins "LeBron James" --radar outputs/lebron.png

# Single-season twins (2013 = the 2012-13 season)
basketball-ml twins "LeBron James" --season 2013 -k 5
```

Example output (career):

```
                 player  first_season  last_season  similarity
            Luka Dončić          2019         2026        60.5
           James Harden          2010         2026        53.7
            Dwyane Wade          2004         2019        53.2
Shai Gilgeous-Alexander          2019         2026        50.8
          Magic Johnson          1980         1996        49.3
```

The closest twin to LeBron's 2012-13 season is Michael Jordan's 1988-89 season (57.5/100), followed by Kevin Durant's 2013-14 (55.5/100).

From Python:

```python
from basketball_ml.data import load_advanced
from basketball_ml.twins import standardize_by_season, career_profiles, find_twins, resolve_player

z = standardize_by_season(load_advanced("data/Advanced.csv"))
careers = career_profiles(z)
find_twins(careers, resolve_player(careers, "Victor Wembanyama"), k=5)
```

## Career projections

To project a player, look at the players who looked most like him at the same age and see what they did next.

### Method

1. **Comparables**: the 50 player-seasons closest to the target season (same z-score profile as the twins module), among players of the **same age** whose later season is already known. Only seasons that ended before the target season can be used, so nothing leaks from the future.
2. **What happened next**: for each comparable, the raw change of each stat between that season and the one `h` years later (for example +1.2 BPM).
3. **Projection**: the target's current value plus the similarity-weighted average of those changes, for 1 to `h` seasons ahead.
4. **80% range**: the 10th and 90th weighted percentiles of the same changes, which show how differently players like him turned out.

### Usage

```bash
# Next three seasons (--last-complete-season skips a season still in progress)
basketball-ml project "Victor Wembanyama" --last-complete-season 2025

# Backtest against simple baselines
basketball-ml backtest --last-complete-season 2025
```

```
Victor Wembanyama, age 21 in 2024-25

          2024-25 (actual)  2025-26 (projected)  2026-27 (projected)  2027-28 (projected)
BPM                  6.500                6.904                7.510                7.928
OBPM                 3.600                4.057                4.490                4.669
DBPM                 2.900                2.864                3.033                3.266
TS%                  0.594                0.601                0.608                0.615
USG%                30.400               31.144               31.983               31.598

BPM 80% range (10th to 90th percentile of the comparables' outcomes):
  2025-26: +4.6 to +9.0
  2026-27: +5.0 to +9.9
  2027-28: +5.7 to +9.9
```

### Backtest

Each of the 3,958 player-seasons from 2010-11 to 2024-25 was projected one season ahead and compared with what actually happened. The comparables beat both baselines on all 13 stats. The baselines are "no change", which repeats the current season, and the "age curve", which adds the average change for that age.

| Mean absolute error | Comparables | Age curve | No change | Actual inside the 80% range |
| --- | --- | --- | --- | --- |
| BPM | **1.37** | 1.43 | 1.47 | 77% |
| OBPM | **1.16** | 1.21 | 1.25 | 79% |
| DBPM | **0.64** | 0.66 | 0.66 | 78% |
| TS% | **0.0315** | 0.0335 | 0.0337 | 76% |
| USG% | **2.10** | 2.14 | 2.19 | 76% |
| AST% | **2.91** | 2.94 | 2.97 | 74% |

The ranges are slightly too narrow (77% instead of 80% for BPM), and the 3-point attempt rate is the worst at 70%, because its league-wide rise is not in older comparables.

Limits: only players who stayed in the league with 500+ minutes have an outcome, so comparables are biased toward careers that continued. League-wide trends (like the rise of the 3-point shot) are only partly captured, because comparables come from earlier seasons.

## Project layout

```
src/basketball_ml/
├── data.py          # shared loaders (Basketball-Reference advanced stats)
├── cli.py           # basketball-ml entry point, one subcommand per module
├── twins/
│   ├── similarity.py  # per-season z-scores, career profiles, nearest neighbors
│   ├── plot.py        # percentile radar charts
│   └── cli.py         # basketball-ml twins
└── projections/
    ├── comparables.py # same-age comparables and weighted projections
    ├── backtest.py    # error against no-change and age-curve baselines
    └── cli.py         # basketball-ml project / backtest
tests/
```

## Tests

```bash
pytest
```

## Roadmap

- **Better projections**: regress toward the mean, add minutes and availability, and adjust for league-wide trends.
- **Style embeddings** learned from play-by-play data, as in [NBA2Vec](https://arxiv.org/abs/2302.13386), to replace aggregated stats.
- **Game predictions** from recent form, rest, injuries and Elo-style ratings.
- **Tracking stats** (drives, catch & shoot, rim protection) via `nba_api`.
