# Elo benchmark arena

**56 datasets, 47 entrants.** The consolidated board combines 18 regression
tasks and 38 classification tasks. The November 21, 2026 update puts the
v0.4.11 single-tree `ModelTree(all)` at **1,290.1 Elo, rank 9**, with finite
scores on **56 of 56 datasets**. [Overall leaderboard](elo_overall.md)

| View | What you can compare |
| --- | --- |
| [Overall arena](elo_overall.md) | Rank, Elo, dataset coverage, and reported runtime across all 56 tasks |
| [Regression arena](elo_board.md) | Test RMSE on each of the 18 regression datasets |
| [Classification arena](elo_cla_board.md) | Test accuracy and log loss on 38 datasets, with binary and multiclass views |
| [Dataset sources](datasets.md) | OpenML sources, targets, and feature metadata |

## What changed in this update

The v0.4.11 rerun scored all 56 tasks and completed QSAR-TID-11 and
Superconductivity, which timed out in the earlier round. The original Diamonds
split overwrote its `y` feature and left `z` in the target position. We restored
the nine features, used `price` as the target, and rescored all 35 Diamonds
entrants. Other peer scores come from the archived arena.

| Peer | Diamonds RMSE | ModelTree regression W–T–L | ModelTree classification W–T–L |
| --- | ---: | ---: | ---: |
| CatBoost(200) | 537.58 | 0–0–18 | 10–1–27 |
| LightGBM(200) | 539.08 | 2–0–16 | 13–1–24 |
| ChimeraBoost | 520.64 | 0–0–18 | 10–2–26 |

ModelTree scored 761.10 RMSE on corrected Diamonds. The head-to-head counts
use test RMSE for regression and test accuracy for classification. Overall
Elo depends on each entrant's matches against the full field, which explains
how ModelTree can rank above LightGBM while losing more of their direct matches.

## How the local board works

- Regression uses a seeded random 80/20 split. Classification uses a seeded
  stratified 80/20 split. Both use seed 7. The suite combines 51 TabArena-v0.1
  datasets with five additional regression datasets.
- The overall board starts ratings at 1,000 and applies round-robin updates
  with K = 32. It processes regression RMSE first and classification accuracy
  second. The classification board also ranks test log loss separately.
- The dataset column counts available score rows. The saved regression
  snapshot includes one infinite RMSE for the historical `ModelTree(opt)`
  on `fc2008`. It ranks below finite RMSE for that task.
- The ModelTree rerun used a 180-second budget per dataset. Diamonds peer
  reruns used 300 seconds. Reported runtimes combine separate runs and are
  best read as context for those scores.
- The [public TabArena leaderboard](https://tabarena.ai/) uses its own folds,
  metrics, and entrant pool. Its Elo values describe that tournament.

## Download or rebuild the tables

[Regression score CSV](scores/v0.4.11/elo_scores.csv) ·
[Classification score CSV](scores/v0.4.11/elo_cla_scores.csv) ·
[Classification task types](scores/v0.4.11/classification_tasks.json)

Run this from the repository root to rebuild all three pages from these saved
inputs.

```bash
uv run --no-sync python benchmarks/arena/elo_table.py
```
