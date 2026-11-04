# basketball_ml

Machine learning on NBA data. Each project lives in its own module of the `basketball_ml` package and shares the same data loaders.

| Module | Status | What it does |
| --- | --- | --- |
| [`twins`](#statistical-twins) | Ready | Finds a player's "statistical twin" for a season or a whole career, across every era |
| [`projections`](#career-projections) | Ready | Projects a player's next seasons from what his twins did at the same age |
| [`predictions`](#game-predictions) | Ready | Home win probability for any matchup, from Elo, recent form, rest and who plays, benchmarked against closing betting lines, and live for the season in progress |
| [`embeddings`](#play-by-play-style) | Ready | Style fingerprints from play-by-play: style twins, archetypes and a style map |
| [`impact`](#player-impact-rapm) | Ready | How many points each player adds per 100 possessions (RAPM), and a next-season forecast that beats public metrics on a public benchmark |
| [`shots`](#shot-quality-expected-fg) | Ready | Expected FG% of every shot, shot making by player, and shot quality by team |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
```

### Data

Download `Advanced.csv` from the Kaggle dataset [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats), or from its GitHub source, and put it in `data/`:

```bash
curl -L -o data/Advanced.csv https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data/Advanced.csv
```

Play-by-play and shot charts come from the public GitHub repository [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data), which covers every season since 1996-97. Game predictions, player impact and play-by-play style use these files. They download automatically into `data/pbp/` the first time they are needed (about 140 MB per season). The 2025-26 play-by-play only exists there in the cdn.nba.com format, so it is converted to the stats.nba.com layout on download. On 2024-25, where both formats exist, the converted data gives the same final score in 1,229 of 1,230 games and box-score totals within 0.3%.

Other public sources, also downloaded on first use:

- **Draft picks, birth dates and player ids**: [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets) and the NBA.com to Basketball-Reference id table of [gabriel1200/site_Data](https://github.com/gabriel1200/site_Data), into `data/bref/` (impact forecast).
- **Hoop Impact Bench**: games and minutes of [lumahoops/hoop-impact-bench](https://github.com/lumahoops/hoop-impact-bench) (CC BY 4.0), into `data/bench/`.
- **Closing betting lines**: [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting), into `data/odds/` (market benchmark).
- **The season in progress**: the NBA's live feed (schedule, box scores and today's odds, on the public bucket `nba-prod-us-east-1-mediaops-stats`), into `data/live/`.

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
3. **Starting point**: half the current season, half a 3-season average weighted 5/4/3 by recency and by minutes (as in the Marcel projection system). One noisy season counts less.
4. **Projection**: the starting point plus the similarity-weighted average of the comparables' changes, for 1 to `h` seasons ahead.
5. **80% range**: the 10th and 90th weighted percentiles of the same changes, which show how differently players like him turned out.

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
BPM                  6.500                6.563                7.169                7.587
OBPM                 3.600                3.611                4.044                4.223
DBPM                 2.900                2.969                3.138                3.371
TS%                  0.594                0.593                0.601                0.607
USG%                30.400               31.616               32.455               32.071

BPM 80% range (10th to 90th percentile of the comparables' outcomes):
  2025-26: +4.2 to +8.7
  2026-27: +4.7 to +9.6
  2027-28: +5.4 to +9.5
```

### Backtest

Each of the 3,958 player-seasons from 2010-11 to 2024-25 was projected one season ahead and compared with what actually happened. The comparables beat both baselines on all 13 stats. The baselines are "no change", which repeats the current season, and the "age curve", which adds the average change for that age.

| Mean absolute error | Comparables | Age curve | No change | Actual inside the 80% range |
| --- | --- | --- | --- | --- |
| BPM | **1.29** | 1.43 | 1.47 | 79% |
| OBPM | **1.11** | 1.21 | 1.25 | 80% |
| DBPM | **0.61** | 0.66 | 0.66 | 80% |
| TS% | **0.0296** | 0.0335 | 0.0337 | 78% |
| USG% | **2.07** | 2.14 | 2.19 | 76% |
| AST% | **2.81** | 2.94 | 2.97 | 75% |

What helped and what didn't:

| Change | BPM error | Kept |
| --- | --- | --- |
| Current season as the starting point | 1.37 | |
| **Half current season, half 3-season average** | **1.29** | Yes. The weight was picked on 2011-2017 and holds on 2018-2025 (1.36 → 1.29). |
| Blending the projection with the age curve | 1.37 or worse at every weight | No |
| Matching comparables on play-by-play style instead of box-score stats (2014+) | 1.35 vs 1.33 | No. Style identifies a player but doesn't predict his development better. |

Limits: the 3-point attempt rate range only holds 71% of the time, because its league-wide rise isn't in older comparables. Also, only players who stayed in the league with 500+ minutes have an outcome, so comparables are biased toward careers that continued.

## Game predictions

### Method

1. **Games**: final scores are the highest score reached in each game's play-by-play, and dates and home and away teams come from the shot charts. Regular season only, from 2012-13 to 2025-26 (16,895 games). The season in progress comes from the NBA's live feed (see [below](#the-season-in-progress)).
2. **Pre-game features**, computed only from earlier games:
    - **Elo** with a margin-of-victory multiplier and 25% regression to the mean between seasons, as in FiveThirtyEight's model. Its 100-point home edge is lowered to 60, because home teams win less often than they used to (about 59% of games until 2018-19, about 55% since).
    - **Form**: average margin over each team's last 10 games
    - **Season**: season-to-date average margin
    - **Rest**: days since each team's previous game, and back-to-back flags
    - **Lineup strength**: who actually played. Each player's production per game (points + assists + rebounds + steals + blocks - turnovers, from the play-by-play) is averaged over the season so far, falling back to last season until he has 3 games. A team's strength is the sum over its 10 best players in the game. Who plays is known shortly before tip-off from injury reports, so the backtest treats it as known.
    - **Lineup impact**: the [RAPM](#player-impact-rapm) ratings of the players who play, weighted by their usual minutes. Ratings are refit every 30 days during the season from the games already played plus the two previous seasons, with the offense and defense box-score prior, the luck adjustment and the forecast layer (rookies by draft pick and age, playing time, aging).
3. **Models**: logistic regression and gradient boosting on those features, trained on all earlier seasons and tested on the next one.

### Usage

```bash
basketball-ml tonight                         # today's games of the season in progress, next to the sportsbooks
basketball-ml tonight --date 2026-11-06 --out "Joel Embiid"   # another day, without injured players
basketball-ml predict SAS OKC --live          # any matchup, with the season in progress
basketball-ml predict SAS OKC                 # SAS (home) vs OKC at the end of 2025-26: SAS 56%, OKC 44%
basketball-ml predict SAS OKC --out-home "Victor Wembanyama"   # SAS drops to 44%
basketball-ml predict WAS OKC --next-season   # season opener: ratings regressed, form reset
basketball-ml ratings                         # Elo table after the last season loaded
basketball-ml games-backtest --market --calibration docs/calibration.png   # walk-forward evaluation
```

### Backtest

Out of sample from 2017-18 to 2025-26 (10,746 games), with each season predicted by a model trained only on the seasons before it:

| Model | Accuracy | Log loss | Brier score |
| --- | --- | --- | --- |
| Always the home team | 55.9% | 0.687 | 0.247 |
| Elo alone | 65.1% | 0.622 | 0.217 |
| Logistic regression, without lineups | 65.6% | 0.620 | 0.216 |
| Logistic regression, with lineup strength | 66.2% | 0.612 | 0.212 |
| **Logistic regression, with lineup strength and impact** | **67.4%** | **0.603** | **0.208** |
| Gradient boosting, same features | 67.3% | 0.605 | 0.209 |

Lower log loss and Brier score are better. Knowing who plays is the biggest gain after Elo, and the two lineup features add up: box-score strength alone gives a log loss of 0.6117, RAPM impact alone 0.6040, both 0.6027. The better player ratings of the [impact module](#player-impact-rapm) (offense and defense prior, luck adjustment, forecast layer) cut the log loss from 0.6058 to 0.6027 and raised accuracy from 67.1% to 67.4%. Refreshing the impact ratings every month, instead of only using previous seasons, lets rookies and players whose role changed move away from their starting value; on 2017-18 to 2024-25 it raised accuracy from 66.3% to 66.6%. 2025-26 was the most predictable of the nine test seasons (70.7% accuracy, log loss 0.573), and 2020-21 the least accurate (65.0%). A "missing production" feature (the summed production of regulars who sat out) helped less than lineup strength on 2017-18 to 2024-25 (log loss 0.619) and added nothing on top of it, so it isn't used. Neither did travel: the distance each team flew since its previous game, time zones crossed, consecutive road games and the altitude of Denver and Utah together moved the log loss by 0.0002.

### Calibration

A win probability is only useful if it means what it says: games predicted at 70% should be won about 70% of the time.

![Calibration chart](docs/calibration.png)

Both models track the diagonal, with an average gap of 1.6 points between the predicted and the actual win rate. On 2017-18 to 2024-25, Elo with FiveThirtyEight's 100-point home edge was off by 6 points on average, because it kept favoring home teams as if it were still 2013. Lowering the edge to 60 points fixed its calibration and its log loss (0.635 to 0.626). The logistic model still overrates home teams by about 1.5 points, since each season's drop in home advantage can't be learned from the seasons before it. Weighting recent seasons more (half-life of 3 seasons) cut that bias to 1.0 point but barely moved the log loss (0.6058 to 0.6050), so it isn't used.

### Against the betting market

Closing betting lines are the hardest benchmark: they know everything public before tip-off, injuries included. `games-backtest --market` scores the model against the closing moneylines of Sportsbook Review, from the public file of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting), with the bookmaker's margin removed. They cover 10,054 of the 10,746 test games (2025-26 only until January 7):

| | Accuracy | Log loss | Brier score |
| --- | --- | --- | --- |
| Closing line | 67.5% | 0.6007 | 0.2070 |
| Model | 67.0% | 0.6072 | 0.2098 |
| Model given the closing line as one more feature | 67.6% | 0.5999 | 0.2066 |

The market is better in every season, by 0.0066 in log loss (paired standard error 0.0014). When the two disagree by more than 10 points, the market is almost always right: in the 781 games where the model gave the home team 10+ points more than the line (58% against 44% on average), the home team won 45%. Most of those gaps are likely what a box score can't see, such as minutes limits, rest decisions and late scratches. The model does add a little to the line: with the line's log-odds as a feature, it beats the line alone in 7 of 9 seasons, by 0.0007 in log loss (standard error 0.0006), too small to be sure of.

### The season in progress

The play-by-play archives are published after each season, so the season being played comes from the NBA's public live feed: the JSON files of its own sites, on a public S3 bucket (league schedule with scores, each finished game's box score, and today's odds from several sportsbooks). `tonight` and `predict --live` bring Elo, form, rest and the lineups up to the last game played, and rate players with the preseason forecast of the [impact module](#player-impact-rapm), with draft-pick and age values for every newcomer. Box scores are cached in `data/live/`.

Through November 2, 2026, the 107 games of 2026-27 predicted that way (from the games before each one, with the players who played) went 66.4% right with a log loss of 0.582; Elo alone was 60.7% right (log loss 0.634). On November 4, 2026:

```
          home win sportsbooks   elo lineup impact rest
game
PHI @ DET      69%         56%  +189          +2.7   +0
ATL @ WAS      34%         42%  -157          -5.7   +0
MIL @ BOS      89%         80%  +362          +9.9   +0
CHI @ NYK      84%         82%  +295          +6.8   +0
DAL @ SAS      89%         83%  +305         +12.0   +0
LAL @ SAC      15%         17%  -186         -12.0   +0
MIA @ LAC      43%         34%   +53          -4.9   +0
```

The model can't see injuries announced before a game: list players who won't play with `--out`.

## Play-by-play style

![NBA style map](docs/style_map_2026.png)

Box scores say how much a player produces. Play-by-play shows **how** he plays: where he shoots from, which kind of shots, how often his makes are assisted, how he turns the ball over.

### Method

1. **34 style features** per player-season (300+ shot attempts), all shares or rates, so they don't depend on playing time:
    - **Touches**: field goal attempts, free throws, assists, offensive and defensive rebounds, steals, blocks, turnovers, fouls and fouls drawn, each as a share of the player's total involvement
    - **Shot diet**: share of attempts at the rim, in the paint, from mid-range, from the corners and above the break, plus dunks, layups, putbacks, floaters, hooks, post fadeaways, pull-ups, drives, cuts, lobs and catch-and-shoot jumpers
    - **Shot making**: field goal percentage by zone, shrunk toward the league average
    - **Creation**: share of made shots that were assisted, share of turnovers that were bad passes, average shot distance
2. **Per-season z-scores**, as in the twins module.
3. **Archetypes**: k-means clusters, each named by its three most distinctive traits.
4. **Style map**: the two main axes of the features (PCA). One runs from perimeter to interior, the other from self-created to spot-up. The map labels the highest-volume shooter of each archetype plus the next biggest names.
5. **Compact embedding**: an autoencoder squeezes the 34 features into 8 numbers.

### Usage

```bash
basketball-ml style-twins "Stephen Curry"         # closest styles since 2013-14
basketball-ml archetypes --season 2026 --map docs/style_map_2026.png
basketball-ml style-backtest --last-season 2025 --advanced data/Advanced.csv
```

```
Closest play-by-play styles to Stephen Curry in 2025-26:

  Bradley Beal               2016-17  similarity 55.0
  Evan Fournier              2019-20  similarity 53.7
  Dennis Schroder            2019-20  similarity 52.7
```

### Does it capture a player's identity?

Here's the test: take a player's season, then look among every player's **next** season for the closest one. The question is whether his own next season comes out first. There are 2,254 player-seasons from 2013-14 to 2023-24, each picked from about 270 candidates.

| Vector | Own next season found first | In the top 5 |
| --- | --- | --- |
| Box-score advanced stats (12, from the twins module) | 47.4% | 74.9% |
| **Play-by-play style (34)** | **65.8%** | **87.9%** |
| Both | 69.9% | 90.2% |
| Style compressed to 8 numbers (PCA) | 44.7% | 76.6% |
| Style compressed to 8 numbers (autoencoder) | 41.3% | 72.2% |

Play-by-play style is a much stronger fingerprint than box-score rates. Compressing it to 8 numbers keeps 78% of its variance, but it loses what makes each player unique. That's why style twins use the full 34 features, while the 8-number embedding is kept for maps and clustering.

## Player impact (RAPM)

How many points does a player add to his team per 100 possessions, once you account for who he shares the court with? Regularized adjusted plus-minus (RAPM) answers it from the play-by-play, helped by the box score, the draft and age.

### Method

1. **Who is on the court.** The play-by-play lists substitutions but not who starts each quarter, so the starters of a quarter are the players whose first appearance in it isn't being subbed in. This works for 99.7% of quarters; the rest (a player who never touches the ball for a whole quarter) are skipped. Events are kept in file order, because event numbers are often out of order around substitutions.
2. **Stints.** Each stretch with the same ten players gives the points each team scored and its possessions (`FGA + 0.44 FTA - OREB + TOV`). That's 853,000 stints from 2012-13 to 2025-26.
3. **Ridge regression.** Each stint is two observations, one per team on offense: points per 100 possessions = league average + home edge + the 5 offensive players' O ratings + the 5 defenders' D ratings, weighted by possessions. Ridge shrinks every rating toward a prior, so players with few minutes or always sharing the court with the same teammates don't get wild values. Ratings use two seasons, the older one at 71% weight.
4. **Offense and defense box-score priors.** Each player is shrunk toward a box-score estimate of his offensive and of his defensive rating instead of toward average. Box scores come from the play-by-play (rebounds split into offensive and defensive by the team of the last miss). His per-100-possession rates of 2- and 3-point makes and attempts, free throws, assists, offensive rebounds, turnovers and fouls drawn on offense, and of defensive rebounds, steals, blocks and fouls on defense, plus his minutes per game, are mapped to the rating scale by regressing plain offensive and defensive RAPM on them (every rate is available to both). With a better prior, RAPM leans on it more (`alpha = 8000` instead of 4000). Shot zones and the share of assisted baskets added nothing.
5. **Luck adjustment.** Free throws count as the shooter's expected makes, and for the defensive ratings the opponents' threes count as expected makes too: a defense has little control over whether open threes go in, and opponent 3-point percentage takes thousands of attempts to mean anything. Each shooter's percentages come from the other half of the season's games, padded with 450 league-average threes and 24 free throws. Offense keeps its actual threes, since shooting is a skill. This is the recipe of [OpenRAPM](https://github.com/bbstats/openrapm).
6. **Forecast for the next season.** Ratings describe the past. To forecast, three corrections are learned each season from the earlier ones (the margin of every stint that last season's ratings leave unexplained, regressed on player features): players without a rating get a value from their draft pick and age, players rated from few possessions are moved down (`possessions / (possessions + 1000)`), and young players up and old ones down. Learned through 2025-26: a 19-year-old first pick is worth -0.7, a 20-year-old 10th pick -2.3 and an undrafted 23-year-old -2.6, but an undrafted 27-year-old (often a veteran of another league) -0.6; a player rated from 300 possessions loses 1.6 points and one rated from 8,000 gains 0.6; a 21-year-old gains 1.3 points on a 33-year-old. Draft picks, birth dates and NBA.com-to-Basketball-Reference ids come from public GitHub mirrors ([sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets), [gabriel1200/site_Data](https://github.com/gabriel1200/site_Data)), downloaded into `data/bref/` on first use.
7. **Tuning.** The shrinkage, the window, the prior's features and shrinkage (300 possessions), the share of 3-point luck removed and the rating of unrated players (-1.5) were chosen on 2015-16 to 2018-19, by how well each season's ratings explain the next season's game margins. The forecast's features were picked after looking at every season, so its edge in the tables below is a little optimistic.

### Usage

```bash
basketball-ml impact                      # leaderboard over the last two seasons, with next season's forecast
basketball-ml impact "Victor Wembanyama"  # one player, with his rank
basketball-ml impact --season 2016 --top 5
basketball-ml impact-backtest             # every rating method, season by season
basketball-ml impact-export               # forecasts for every season, as season,nba_id,rating
basketball-ml impact-bench                # score the forecasts on the Hoop Impact Bench test
```

2024-25 and 2025-26, players with 3,000+ offensive possessions:

| | Player | Offense | Defense | RAPM | 2026-27 forecast |
| --- | --- | --- | --- | --- | --- |
| 1 | Nikola Jokić | +7.3 | +1.2 | +8.4 | +8.5 |
| 2 | Shai Gilgeous-Alexander | +6.7 | +1.6 | +8.3 | +8.8 |
| 3 | Victor Wembanyama | +2.3 | +5.0 | +7.3 | +8.3 |
| 4 | Giannis Antetokounmpo | +5.3 | +0.7 | +6.0 | +5.9 |
| 5 | Kawhi Leonard | +4.6 | +1.2 | +5.8 | +5.4 |
| 6 | Luka Dončić | +5.1 | +0.3 | +5.4 | +5.9 |
| 7 | Donovan Mitchell | +3.9 | +0.7 | +4.6 | +4.8 |
| 8 | Tyrese Haliburton | +4.1 | +0.4 | +4.5 | +4.9 |
| 9 | Chet Holmgren | +1.4 | +3.1 | +4.5 | +5.2 |
| 10 | Jarrett Allen | +2.6 | +1.8 | +4.3 | +4.7 |

Defensive players (Wembanyama, Holmgren, Derrick White, Alex Caruso) rank much higher than in box-score metrics, which is the point of plus-minus models; the flip side is that a role player on a great team can get credit for the stars around him, and two seasons is still a small sample. The forecast moves young players up (Wembanyama, Holmgren, Ausar Thompson) and older ones down (Stephen Curry, Jimmy Butler, Kawhi Leonard).

### Does it work?

Ratings from the two previous seasons, applied to the players actually on the court in each stint, explain this share of the variance of every game's final margin (2015-16 to 2025-26, 13,206 games):

| Rating | Margin explained | Best in |
| --- | --- | --- |
| **Forecast: RAPM + rookies, playing time and age** | **17.8%** | 9 of 11 seasons |
| RAPM, offense/defense box-score prior, luck-adjusted | 16.8% | 2 |
| RAPM, offense/defense box-score prior | 16.4% | |
| RAPM, one box-score production prior (previous version) | 15.4% | |
| RAPM shrunk toward average | 14.6% | |
| Box-score production per game | 12.0% | |
| Raw on-court net rating | 11.3% | |
| Adjusted plus-minus without shrinkage | 11.1% | |

Each step helps in most seasons: the offense/defense prior in 10 of 11, the luck adjustment in 9 of 11 and the forecast in 9 of 11. The forecast's corrections are learned from earlier seasons only, so they are weak in the first test seasons (learned from one to three seasons) and worth about 2 points from 2020-21 on.

**Against public metrics.** [Hoop Impact Bench](https://github.com/lumahoops/hoop-impact-bench) scores public impact metrics on the same test: preseason ratings weighted by the minutes players actually played, 7,230 games from 2020-21 to 2025-26. Under its "established players" rule, everyone with fewer than 250 minutes the season before counts as -2.0; the "full forecast" uses each metric's own values for rookies too:

| Metric | Established players | Full forecast |
| --- | --- | --- |
| **basketball_ml forecast** | **.182** | .188 |
| LOKI | .170 | **.196** |
| Pulse | .171 | .177 |
| OpenRAPM | .162 | .171 |
| xRAPM | .161 | .169 |
| LEBRON | .159 | .154 |
| EPM | .157 | .178 |
| DARKO | .155 | .159 |
| BPM (Basketball-Reference) | .154 | .154 |
| Last season's plain RAPM | .141 | .133 |

Other metrics' numbers are the benchmark's published v1.1.0 results. Ours come from `impact-bench`, a re-implementation of its game-margin test on its own game and minutes files, which gives exactly its published numbers for the metrics it ships (LOKI, OpenRAPM, ARC). The benchmark calls gaps under about .006 noise, so on established players the forecast is ahead of every published metric; with rookies counted, LOKI's newcomer values are better than ours.

## Shot quality (expected FG%)

![Expected FG% by location](docs/xfg_court.png)

Two players can both shoot 47%, one on layups and the other on step-back threes. Expected FG% (xFG) is the chance that an average NBA shooter makes a shot, judged from the shot alone. Actual points minus expected points is shot making.

### Method

1. **Shots**: every field goal attempt in the shot charts since 2012-13, 2.9 million in all.
2. **Features**: only the shot, never the shooter.
    - Distance, location and angle, 2 or 3, and the three NBA zone fields.
    - The shot type (`ACTION_TYPE`); labels with fewer than 200 attempts are grouped as "other".
    - Period, seconds left, and an end-of-period heave flag (30+ feet with 3 seconds or less left).
    - The shooting team's lead before the shot, from the play-by-play (matched for 99.98% of shots).
3. **No result in the labels**: "bank" is dropped from shot types, because scorers mostly call a shot a bank shot when it goes in. Bank shots from 3 to 25 feet go in 11 to 19 points more often than other shots from the same distance. Labels renamed in 2015-16 ("Slam Dunk", "Tip Shot") are merged.
4. **Model**: gradient boosting with native categorical splits. Each season from 2015-16 to 2025-26 is predicted by a model trained on the 3 seasons before it, so every xFG is out of sample.
5. **Shot making**: points per shot minus expected points per shot (xFG × 2 or 3).
    - xFG is first set to each season's league level, separately for twos and threes, because 2-point accuracy has risen almost every season.
    - Twos are also corrected for each arena's scorekeeper bias. Some arenas log shots farther from the rim than they were, so twos there go in more often for both teams. The bias, estimated from home-minus-road gaps, ranges from -7 to +6 FG% points in 2025-26; Utah, Golden State and Sacramento are highest.
    - Each player's rate is shrunk toward 0 by the noise of his sample (empirical Bayes). Half of the observed spread is skill at about 270 attempts.
6. **Teams**: shot quality is the average expected points per shot (xPPS) of a team's attempts, and, on defense, of its opponents' attempts.

### Usage

```bash
basketball-ml xfg-backtest --chart docs/xfg_court.png   # walk-forward evaluation, calibration, chart
basketball-ml shot-making                                # best shot makers of 2025-26
basketball-ml shot-making "Stephen Curry"                # one player, season by season
basketball-ml shot-making --persistence                  # is shot making a skill?
basketball-ml shot-quality --season 2026                 # teams, offense and defense
```

The first run takes about 10 minutes. It caches the out-of-sample predictions for all 2.3 million shots in `data/pbp/xfg_predictions_2013_2026.csv`; after that every command takes a few seconds.

2025-26, players with 300+ attempts:

| | Player | FGA | FG% | xFG% | Points added | Per 100 shots (shrunk) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Nikola Jokić | 1,132 | 56.9% | 46.6% | +243 | +17.5 |
| 2 | Luke Kennard | 443 | 53.3% | 43.3% | +115 | +14.9 |
| 3 | Shai Gilgeous-Alexander | 1,321 | 55.3% | 47.1% | +230 | +14.6 |
| 4 | Kevin Durant | 1,376 | 52.0% | 44.3% | +239 | +14.5 |
| 5 | Jakob Poeltl | 310 | 70.0% | 59.8% | +63 | +13.0 |
| 6 | Jamal Murray | 1,360 | 48.3% | 42.3% | +213 | +12.9 |
| 7 | Cam Spencer | 556 | 47.3% | 40.0% | +112 | +12.5 |
| 8 | Luka Dončić | 1,457 | 47.6% | 41.8% | +184 | +10.4 |
| 9 | Bobby Portis | 750 | 48.8% | 43.6% | +105 | +10.0 |
| 10 | Kon Knueppel | 1,084 | 47.5% | 42.9% | +142 | +10.0 |

Teams: Oklahoma City had the best net in 2025-26, scoring 1.12 points per shot on shots worth 1.10 and allowing 1.04 on shots worth 1.07. Denver had the best offense (1.15 points per shot) on below-average shots (1.07 expected), thanks to the most shot making in the league (+0.08 per shot). Across 2015-16 to 2025-26, about a quarter of the spread in teams' points per shot comes from shot quality and three quarters from shot making, on offense (24%) and on defense (26%).

### Does it work?

Out of sample, 2015-16 to 2025-26 (2.3 million shots), each season predicted from the three before it:

| Model | Log loss | Brier score | AUC |
| --- | --- | --- | --- |
| League FG% (no model) | 0.691 | 0.249 | 0.507 |
| League FG% by zone | 0.660 | 0.234 | 0.630 |
| Logistic regression, distance + 3-point flag | 0.664 | 0.236 | 0.634 |
| Boosting, location + game situation | 0.650 | 0.229 | 0.644 |
| + shot kind (dunk, layup, hook, tip, jumper) | 0.641 | 0.226 | 0.651 |
| **+ shot type (xFG model)** | **0.635** | **0.223** | **0.663** |

Lower log loss and Brier score are better. The xFG model is the best in every season. Sorted into tenths by xFG, predicted and actual FG% are 0.8 points apart on average, and 0.4 points in 2024-25 and 2025-26. The score margin adds nothing measurable (log loss 0.6368 with it, 0.6366 without, in 2025-26).

The shot type looks more useful in 2015-16 and 2016-17 (AUC 0.68) than it is. Before 2017-18, scorers gave made shots the detailed labels (pull-up, driving, step back) more often than misses. In 2012-13, 83% of missed twos from 10+ feet were plain "Jump Shot", against 70% of makes; by 2022-23 the two shares were within a point. The detail adds 0.04 AUC over the coarse shot kind in 2015-16, but only 0.003 to 0.005 since 2022-23. Models trained on those early seasons also overrate shots by 1.5 to 2.4 points through 2018-19, so shot making before 2019-20 is less reliable.

**Is shot making a skill?** 2,034 players with 300+ attempts in two consecutive seasons, 2015-16 to 2025-26:

| Year-to-year correlation | r |
| --- | --- |
| FG% | 0.823 |
| eFG% | 0.635 |
| xFG% (shot diet) | 0.917 |
| FG% above expected (shot making) | 0.634 |
| Points added per 100 shots | 0.602 |

Shot making holds up from one season to the next as well as eFG% does: it is a skill, not luck. On its own, though, it predicts next season's FG% poorly (r 0.26, against 0.82 for last season's FG%). Most of a player's FG% is his shot diet, and that barely changes. The two parts work best together:

- Last season's xFG% and shot making, weighted separately, predict next season's FG% better than last season's FG% (error 3.28 vs 3.40 points, each pair of seasons predicted from the others).
- Last season's shot making predicts next season's shot making much better than FG% does (r 0.63 vs 0.28).

Without tracking data, shot making also includes how open the shots were. Free throws are left out, so players get no credit for drawing fouls.

## Project layout

```
src/basketball_ml/
├── data.py          # Basketball-Reference advanced stats loader
├── pbp.py           # play-by-play and shot chart downloader
├── cache.py         # derived tables cached as CSV
├── cli.py           # basketball-ml entry point, one subcommand per module
├── twins/
│   ├── similarity.py  # per-season z-scores, career profiles, nearest neighbors
│   ├── plot.py        # percentile radar charts
│   └── cli.py         # basketball-ml twins
├── projections/
│   ├── comparables.py # same-age comparables, weighted projections and 80% ranges
│   ├── backtest.py    # error against no-change and age-curve baselines
│   └── cli.py         # basketball-ml project / backtest
├── predictions/
│   ├── games.py       # game results, Elo and pre-game features
│   ├── availability.py # who played in each game, lineup strength and impact
│   ├── model.py       # win-probability models, walk-forward evaluation and calibration
│   ├── market.py      # closing betting lines, matched to games, and the market benchmark
│   ├── live.py        # the season in progress from the NBA's live feed: schedule, box scores, odds
│   ├── plot.py        # calibration chart
│   └── cli.py         # basketball-ml predict / tonight / ratings / games-backtest
├── embeddings/
│   ├── features.py    # 34 style features from play-by-play and shot charts
│   ├── model.py       # autoencoder, PCA and the identification test
│   ├── archetypes.py  # k-means archetypes and the style map
│   └── cli.py         # basketball-ml style-twins / archetypes / style-backtest
├── impact/
│   ├── stints.py      # who is on the court, stint by stint, with expected threes and free throws
│   ├── box.py         # per-game box scores from the play-by-play
│   ├── rapm.py        # ridge-regression ratings, box-score priors, luck adjustment and backtest
│   ├── projection.py  # next-season forecast: rookies by draft pick and age, playing time, aging
│   ├── bench.py       # the Hoop Impact Bench game-margin test
│   └── cli.py         # basketball-ml impact / impact-backtest / impact-export / impact-bench
└── shots/
    ├── features.py    # shot features, label cleaning, score margin from the play-by-play
    ├── model.py       # xFG models, baselines, walk-forward evaluation and calibration
    ├── making.py      # league re-centering, arena correction, shrinkage, team tables, persistence
    ├── plot.py        # half-court map of expected FG%
    └── cli.py         # basketball-ml xfg-backtest / shot-making / shot-quality
tests/
```

## Tests

```bash
pytest
```

## Roadmap

- **Better projections**: regress toward the mean, add minutes and availability, and adjust for league-wide trends.
- **Lineup embeddings** learned from who is on the court, as in [NBA2Vec](https://arxiv.org/abs/2302.13386).
- **Better rookie values** for the impact forecast (college or international stats, height), where LOKI is ahead.
- **Tracking stats** (drives, catch & shoot, rim protection) via `nba_api`.
