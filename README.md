# basketball_ml

Machine learning on NBA data. Each project lives in its own module of the `basketball_ml` package and shares the same data loaders.

| Module | Status | What it does |
| --- | --- | --- |
| [`twins`](#statistical-twins) | Ready | Finds a player's "statistical twin" for a season or a whole career, across every era |
| `projections` | Planned | Projects a player's next seasons from what his twins did at the same age |
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

## Project layout

```
src/basketball_ml/
├── data.py          # shared loaders (Basketball-Reference advanced stats)
├── cli.py           # basketball-ml entry point, one subcommand per module
└── twins/
    ├── similarity.py  # per-season z-scores, career profiles, nearest neighbors
    ├── plot.py        # percentile radar charts
    └── cli.py         # basketball-ml twins
tests/
```

## Tests

```bash
pytest
```

## Roadmap

- **Career projections from comparables**: predict a player's next seasons from his twins at the same age, in the spirit of FiveThirtyEight's CARMELO.
- **Style embeddings** learned from play-by-play data, as in [NBA2Vec](https://arxiv.org/abs/2302.13386), to replace aggregated stats.
- **Game predictions** from recent form, rest, injuries and Elo-style ratings.
- **Tracking stats** (drives, catch & shoot, rim protection) via `nba_api`.
