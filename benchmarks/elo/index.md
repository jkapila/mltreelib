# ELO

One scoreboard, one protocol, every model scored on the **same prepared data**.

## The protocol

- **Data**: the regression sets below, one 80/20 split (seed 7), identical numeric encodings for every scorer (prepared once by `examples/elo_prepare.py` — no per-model preprocessing luck).
- **Matchups**: per dataset, every unordered pair of models "plays" once; the lower test RMSE wins (tie = draw).
- **Ratings**: standard Elo, K=32, everyone starts at 1000, deterministic round-robin order. Ratings are field-relative — adding models moves numbers around, so compare within one run.

Reproduce: `uv run python examples/elo_report.py` (reads `benchmarks/elo_scores.csv`, written incrementally by the R, Python and Rust scorers).

## The board

```
== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==
+-----------+---------------------+--------------------------+-------------+
|    ELO    |    FIT TIME (±sd)   |          MODEL           |     DATA    |
+-----------+---------------------+--------------------------+-------------+
|   1576.5  |   15.41 s (±24.59)  |         grf_llf          |    17/18    |
|   1524.9  |     0.67 s (±0.55)  |      LightGBM(200)       |    18/18    |
|   1479.7  |     0.36 s (±0.31)  |      CatBoost(200)       |    18/18    |
|   1462.9  |    6.50 s (±13.12)  |     perpetual (Rust)     |    18/18    |
|   1329.9  |     0.50 s (±0.29)  |      forust (Rust)       |    18/18    |
|   1228.0  |   11.47 s (±17.45)  |          grf_rf          |    18/18    |
|   1214.8  |     0.23 s (±0.37)  |         CTBoost          |    18/18    |
|   1001.9  |   17.93 s (±35.31)  |      ModelTree(all)      |    17/18    |
|   982.5   |   13.89 s (±27.18)  |          lmtree          |    16/18    |
|   973.1   |     1.19 s (±1.53)  |          cubist          |    18/18    |
|   956.5   |   10.39 s (±19.45)  |         glmtree          |    15/18    |
|   881.4   |   17.77 s (±44.25)  |    modeltrees lmtree     |    17/18    |
|   823.3   |  143.31 s (±463.47) |    MARS (mars-earth)     |    12/18    |
|   752.2   |     1.46 s (±2.38)  |          ctree           |    18/18    |
|   745.9   |     0.26 s (±0.37)  |          earth           |    17/18    |
|   715.1   |     3.14 s (±4.71)  |    ModelTree(winnow)     |    17/18    |
|   678.5   |     0.13 s (±0.20)  |       sklearn tree       |    18/18    |
|   628.0   |     0.14 s (±0.19)  |     ModelTree(mean)      |    18/18    |
|   539.8   |     0.22 s (±0.32)  |          rpart           |    18/18    |
|   505.1   |     0.89 s (±1.25)  |     ModelTree(path)      |    18/18    |
+-----------+---------------------+--------------------------+-------------+
```

Reading it:

- **Stock configs only** — nothing tuned per dataset. The top of the board is *forests and boosted ensembles with strong feature handling* (grf's local-linear forest leads; forust/perpetual are boosted trees).
- **`ModelTree(all)` at 1001.9 now sits above the whole classical model-tree and CART pack** — lmtree (982.5), cubist (973.1), glmtree (956.5), ctree (752.2), sklearn tree (678.5), rpart (539.8) — the plain single tree with ridge-linear leaves on all 18 official sets. The gap to the boosting tier is the remaining mission (committees/forests, see `references/research/knob-gap-review.md`).
- `FIT mean±sd` = fit seconds across all scored datasets (mean ± 1 sd); `DATA` = datasets where the model has a score — matchups only happen where both models have one. Shortfalls are honest failures, not skates: mars-earth hits the 180s-per-algorithm budget on wide sets (12/18), and grf\_llf/earth skip fc2008's NAs.

TabArena numbers are an **anchor, not a comparison** — their protocol (tuned+ensembled, different splits) differs; shown only for scale: `1955.0 Kumo-Tabular (best overall)`, `1411.0 LightGBM tuned+ens (best tree-based)`.

## The wide table (model × dataset)

Every score in the field — RMSE per model per dataset, best per column bolded. Click any column header to sort (the table is interactive):

| ELO | FIT TIME (±sd) | MODEL | abalone | boston | concrete | fc2008 | ozone | airfoil | ta\_concrete | diamonds | fiat500 | food\_delivery | healthcare | houses | miami | protein | qsar\_fish | qsar\_tid11 | superconductivity | wine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1577 | 15.41±24.59s | **grf\_llf** 17/18 | 2.31 | 3.75 | 6.29 | · | **3.77** | 2.03 | 5.07 | 536.61 | 695.74 | 7.55 | 4564.69 | 0.23 | 102864.92 | **3.59** | 0.97 | **0.86** | **8.96** | 0.64 |
| 1525 | 0.67±0.55s | **LightGBM(200)** 18/18 | 2.43 | 3.47 | 5.85 | 196.57 | 4.09 | **1.62** | 4.29 | **0.00** | 716.88 | **7.42** | 5121.26 | 0.22 | 93946.87 | 3.92 | 0.97 | 0.90 | 9.64 | **0.63** |
| 1480 | 0.36±0.31s | **CatBoost(200)** 18/18 | 2.33 | **2.54** | 5.74 | **168.89** | 3.82 | 1.66 | **4.05** | **0.00** | **678.25** | 7.44 | 4664.68 | 0.22 | **89876.80** | 3.92 | **0.95** | 0.91 | 10.24 | 0.65 |
| 1463 | 6.50±13.12s | **perpetual (Rust)** 18/18 | 2.29 | 3.44 | 5.78 | 220.86 | 3.88 | 2.35 | 4.75 | 0.00 | 702.46 | 7.47 | 4533.55 | **0.21** | 91669.36 | 3.78 | 0.96 | 1.02 | 9.61 | 0.66 |
| 1330 | 0.50±0.29s | **forust (Rust)** 18/18 | 2.45 | 3.14 | **5.59** | 173.06 | 3.94 | 1.73 | 4.43 | 0.00 | 744.09 | 7.44 | 5113.39 | 0.22 | 93768.85 | 3.96 | 1.01 | 0.92 | 9.87 | 0.66 |
| 1228 | 11.47±17.45s | **grf\_rf** 18/18 | 2.36 | 3.27 | 8.26 | 228.61 | 4.44 | 2.78 | 6.49 | 599.99 | 698.28 | 7.54 | 4614.00 | 0.25 | 124261.13 | 4.00 | 0.98 | 1.01 | 11.01 | 0.69 |
| 1215 | 0.23±0.37s | **CTBoost** 18/18 | 2.48 | 4.01 | 6.38 | 192.08 | 4.54 | 2.69 | 4.91 | 0.01 | 703.96 | 7.45 | **4508.58** | 0.23 | 107014.09 | 4.19 | 0.99 | 1.02 | 11.41 | 0.69 |
| 1002 | 17.93±35.31s | **ModelTree(all)** 17/18 | 2.33 | 4.75 | 6.82 | 218.18 | 5.76 | 2.49 | 5.81 | **0.00** | 2116.18 | 7.46 | 4639.92 | 0.36 | 1490016.74 | 4.85 | 1.13 | 1.09 | · | 0.71 |
| 983 | 13.89±27.18s | **lmtree** 16/18 | 2.32 | 4.05 | 7.64 | 237.92 | 4.27 | 2.87 | 6.64 | 876.88 | 718.84 | 7.51 | 4791.58 | 0.24 | 112822.63 | · | 1.04 | 1.12 | · | 0.71 |
| 973 | 1.19±1.53s | **cubist** 18/18 | 2.31 | 3.98 | 6.81 | 204.32 | 4.57 | 2.43 | 6.14 | 608.85 | 739.27 | 7.53 | 4535.60 | 0.23 | 114392.73 | 4.43 | 1.02 | 1.11 | 12.01 | 0.72 |
| 957 | 10.39±19.45s | **glmtree** 15/18 | 2.31 | 4.05 | 7.42 | 237.92 | 4.27 | 3.31 | 6.13 | 641.43 | 718.84 | 7.51 | 4791.58 | 0.24 | · | · | 1.03 | 1.12 | · | 0.71 |
| 881 | 17.77±44.25s | **modeltrees lmtree** 17/18 | 2.32 | 4.05 | 7.64 | 237.92 | 4.27 | 2.87 | 6.64 | 0.02 | 718.84 | 7.51 | 4791.58 | 0.24 | · | 4.85 | 1.04 | 1.12 | 81.75 | 0.71 |
| 823 | 143.31±463.47s | **MARS (mars-earth)** 12/18 | 2.27 | 4.75 | 7.11 | · | 4.10 | 4.31 | 6.38 | **0.00** | 713.60 | · | 5853.26 | · | · | · | 0.99 | 1.30 | · | 0.73 |
| 752 | 1.46±2.38s | **ctree** 18/18 | 2.43 | 4.02 | 9.76 | 276.40 | 4.26 | 3.58 | 7.10 | 603.51 | 788.02 | 7.87 | 4850.64 | 0.30 | 140881.71 | 4.59 | 1.08 | 1.15 | 13.60 | 0.75 |
| 746 | 0.26±0.37s | **earth** 17/18 | **2.26** | 4.51 | 7.06 | · | 4.23 | 4.26 | 6.20 | 1094.57 | 714.05 | 7.99 | 5724.24 | 0.29 | 145755.85 | 5.03 | 1.00 | 1.30 | 18.59 | 0.73 |
| 715 | 3.14±4.71s | **ModelTree(winnow)** 17/18 | 2.33 | 4.75 | 6.82 | 231.47 | 5.58 | 2.39 | 5.81 | 0.05 | 989.82 | 7.86 | 4788.86 | 0.34 | 3399823.59 | 5.01 | 1.13 | 1.12 | · | 0.73 |
| 679 | 0.13±0.20s | **sklearn tree** 18/18 | 3.16 | 4.87 | 7.93 | 197.74 | 5.46 | 2.66 | 5.66 | **0.00** | 841.83 | 10.61 | 6404.11 | 0.32 | 146810.51 | 5.07 | 1.27 | 1.11 | 11.36 | 0.83 |
| 628 | 0.14±0.19s | **ModelTree(mean)** 18/18 | 2.54 | 4.01 | 9.77 | 262.32 | 4.62 | 4.67 | 10.29 | 0.23 | 789.31 | 7.88 | 4865.40 | 0.38 | 188893.02 | 5.36 | 1.13 | 1.27 | 18.06 | 0.76 |
| 540 | 0.22±0.32s | **rpart** 18/18 | 2.59 | 4.31 | 9.07 | 279.94 | 4.71 | 4.43 | 9.79 | 1496.89 | 823.83 | 8.03 | 5088.41 | 0.38 | 192132.40 | 5.46 | 1.12 | 1.29 | 18.91 | 0.77 |
| 505 | 0.89±1.25s | **ModelTree(path)** 18/18 | 2.34 | 4.40 | 11.38 | 270.69 | 5.78 | 4.03 | 8.65 | 0.27 | 1073.65 | 7.86 | 6573.19 | 0.38 | 1540449.25 | 36.25 | 1.14 | 1.29 | 18.98 | 0.75 |

## Development rounds (ours vs the frozen field)

The external field (R packages, LightGBM, CatBoost, CTBoost, forust, perpetual, MARS, sklearn, modeltrees) is scored **once** and frozen in `benchmarks/elo_baselines.csv` — *where we started*. Every development round re-scores only the ModelTree variants on the same prepared splits:

```
uv run python examples/elo_round.py                 # all datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # a subset
uv run python examples/elo_round.py --freeze        # deliberately re-anchor
```

Each round prints `started (frozen baseline) -> reached` Elo for every ModelTree flavor, with per-fit timings in the scores file — so progress across iterations is visible instead of silently overwritten. The full from-scratch rebuild (all models, groups of 5 datasets, shareable intermediate state after each group) is `bash examples/elo_run_groups.sh`.

The field's raw scores (per dataset RMSE/MAE/fit time) live in [`benchmarks/elo_scores.csv`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/elo_scores.csv>); the longer research log is in `docs/field-notes.md`.
