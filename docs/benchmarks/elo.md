# ELO

One scoreboard, one protocol, every model scored on the **same prepared data**.

## The protocol

- **Data**: the regression sets below, one 80/20 split (seed 7), identical
  numeric encodings for every scorer (prepared once by
  `examples/elo_prepare.py` — no per-model preprocessing luck).
- **Matchups**: per dataset, every unordered pair of models "plays" once; the
  lower test RMSE wins (tie = draw).
- **Ratings**: standard Elo, K=32, everyone starts at 1000, deterministic
  round-robin order. Ratings are field-relative — adding models moves numbers
  around, so compare within one run.

Reproduce: `uv run python examples/elo_report.py` (reads
`benchmarks/elo_scores.csv`, written incrementally by the R, Python and Rust
scorers).

## The board

```text
== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==
+--------+------------------+------------------------+--------------+
|    ELO |FIT mean±sd       |MODEL                   |DATA          |
+--------+------------------+------------------------+--------------+
| 1576.5 |15.41±24.59s      |grf_llf                 |17/18         |
| 1524.9 |0.67±0.55s        |LightGBM(200)           |18/18         |
| 1479.7 |0.36±0.31s        |CatBoost(200)           |18/18         |
| 1462.9 |6.50±13.12s       |perpetual (Rust)        |18/18         |
| 1329.9 |0.50±0.29s        |forust (Rust)           |18/18         |
| 1228.0 |11.47±17.45s      |grf_rf                  |18/18         |
| 1214.8 |0.23±0.37s        |CTBoost                 |18/18         |
| 1001.9 |17.93±35.31s      |ModelTree(all)          |17/18         |
|  982.5 |13.89±27.18s      |lmtree                  |16/18         |
|  973.1 |1.19±1.53s        |cubist                  |18/18         |
|  956.5 |10.39±19.45s      |glmtree                 |15/18         |
|  881.4 |17.77±44.25s      |modeltrees lmtree       |17/18         |
|  823.3 |143.31±463.47s    |MARS (mars-earth)       |12/18         |
|  752.2 |1.46±2.38s        |ctree                   |18/18         |
|  745.9 |0.26±0.37s        |earth                   |17/18         |
|  715.1 |3.14±4.71s        |ModelTree(winnow)       |17/18         |
|  678.5 |0.13±0.20s        |sklearn tree            |18/18         |
|  628.0 |0.14±0.19s        |ModelTree(mean)         |18/18         |
|  539.8 |0.22±0.32s        |rpart                   |18/18         |
|  505.1 |0.89±1.25s        |ModelTree(path)         |18/18         |
+--------+------------------+------------------------+--------------+
```

Reading it:

- **Stock configs only** — nothing tuned per dataset. The top of the board is
  *forests and boosted ensembles with strong feature handling* (grf's
  local-linear forest leads; forust/perpetual are boosted trees).
- **`ModelTree(all)` at 1001.9 now sits above the whole classical model-tree
  and CART pack** — lmtree (982.5), cubist (973.1), glmtree (956.5), ctree
  (752.2), sklearn tree (678.5), rpart (539.8) — the plain single tree with
  ridge-linear leaves on all 18 official sets. The gap to the boosting tier
  is the remaining mission (committees/forests, see
  `references/research/knob-gap-review.md`).
- `FIT mean±sd` = fit seconds across all scored datasets (mean ± 1 sd);
  `DATA` = datasets where the model has a score — matchups only
  happen where both models have one. Shortfalls are honest failures, not
  skates: mars-earth hits the 180s-per-algorithm budget on wide sets (12/18),
  and grf_llf/earth skip fc2008's NAs.

TabArena numbers are an **anchor, not a comparison** — their protocol
(tuned+ensembled, different splits) differs; shown only for scale:
`1955.0 Kumo-Tabular (best overall)`, `1411.0 LightGBM tuned+ens (best
tree-based)`.

## The wide table (model × dataset)

Every score in the field — RMSE per model per dataset, best per column
bolded. Click any column header to sort (the table is interactive):

{% include-markdown "benchmarks/elo_board.md" %}

## Development rounds (ours vs the frozen field)

The external field (R packages, LightGBM, CatBoost, CTBoost, forust,
perpetual, MARS, sklearn, modeltrees) is scored **once** and frozen in
`benchmarks/elo_baselines.csv` — *where we started*. Every development round
re-scores only the ModelTree variants on the same prepared splits:

```bash
uv run python examples/elo_round.py                 # all datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # a subset
uv run python examples/elo_round.py --freeze        # deliberately re-anchor
```

Each round prints `started (frozen baseline) -> reached` Elo for every
ModelTree flavor, with per-fit timings in the scores file — so progress across
iterations is visible instead of silently overwritten. The full from-scratch
rebuild (all models, groups of 5 datasets, shareable intermediate state after
each group) is `bash examples/elo_run_groups.sh`.

The field's raw scores (per dataset RMSE/MAE/fit time) live in
[`benchmarks/elo_scores.csv`](https://github.com/jkapila/mltreelib/blob/main/benchmarks/elo_scores.csv);
the longer research log is in `docs/field-notes.md`.
