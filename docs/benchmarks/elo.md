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
 1576.5  grf_llf  (17/18 datasets)
 1524.9  LightGBM(200)  (18/18 datasets)
 1479.7  CatBoost(200)  (18/18 datasets)
 1462.9  perpetual (Rust)  (18/18 datasets)
 1329.9  forust (Rust)  (18/18 datasets)
 1228.0  grf_rf  (18/18 datasets)
 1214.8  CTBoost  (18/18 datasets)
 1001.9  ModelTree(all)  (17/18 datasets)
  982.5  lmtree  (16/18 datasets)
  973.1  cubist  (18/18 datasets)
  956.5  glmtree  (15/18 datasets)
  881.4  modeltrees lmtree  (17/18 datasets)
  823.3  MARS (mars-earth)  (12/18 datasets)
  752.2  ctree  (18/18 datasets)
  745.9  earth  (17/18 datasets)
  715.1  ModelTree(winnow)  (17/18 datasets)
  678.5  sklearn tree  (18/18 datasets)
  628.0  ModelTree(mean)  (18/18 datasets)
  539.8  rpart  (18/18 datasets)
  505.1  ModelTree(path)  (18/18 datasets)
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
- `(n/18 datasets)` = datasets where the model has a score; matchups only
  happen where both models have one. Shortfalls are honest failures, not
  skates: mars-earth hits the 180s-per-algorithm budget on wide sets (12/18),
  and grf_llf/earth skip fc2008's NAs.

TabArena numbers are an **anchor, not a comparison** — their protocol
(tuned+ensembled, different splits) differs; shown only for scale:
`1955.0 Kumo-Tabular (best overall)`, `1411.0 LightGBM tuned+ens (best
tree-based)`.

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
