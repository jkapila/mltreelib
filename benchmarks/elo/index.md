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
 1343.6  forust (Rust)  (5/10 datasets)
 1342.9  perpetual (Rust)  (5/10 datasets)
 1341.1  CatBoost(200)  (10/10 datasets)
 1297.6  grf_llf  (4/10 datasets)
 1241.4  LightGBM(200)  (10/10 datasets)
 1103.9  earth  (4/10 datasets)
 1046.7  CTBoost  (10/10 datasets)
 1014.2  MARS (mars-earth)  (8/10 datasets)
  983.9  glmtree  (5/10 datasets)
  979.1  cubist  (5/10 datasets)
  956.8  lmtree  (5/10 datasets)
  934.2  grf_rf  (5/10 datasets)
  933.1  ctree  (5/10 datasets)
  868.6  modeltrees lmtree  (10/10 datasets)
  820.1  ModelTree(mean)  (10/10 datasets)
  811.5  sklearn tree  (10/10 datasets)
  704.7  rpart  (5/10 datasets)
  655.0  ModelTree(all)  (10/10 datasets)
  621.5  ModelTree(path)  (10/10 datasets)
```

Reading it:

- **Stock configs only** — nothing tuned per dataset. The top of the board is *ensembles with strong feature handling* (forust/perpetual are boosted trees, CatBoost and LightGBM boost, grf\_llf is a forest of local-linear trees).
- **Our `ModelTree` rows are the plain single tree** — its honest baseline position, exactly what the [default bake-off](<https://jkapila.github.io/mltreelib/reference/data-model/index.md>) optimizes against.
- `(n/10 datasets)` = datasets where the model has a score; matchups only happen where both models have one.

TabArena numbers are an **anchor, not a comparison** — their protocol (tuned+ensembled, different splits) differs; shown only for scale: `1955.0 Kumo-Tabular (best overall)`, `1411.0 LightGBM tuned+ens (best tree-based)`.

## Development rounds (ours vs the frozen field)

The external field (R packages, LightGBM, CatBoost, CTBoost, forust, perpetual, MARS, sklearn, modeltrees) is scored **once** and frozen in `benchmarks/elo_baselines.csv` — *where we started*. Every development round re-scores only the ModelTree variants on the same prepared splits:

```
uv run python examples/elo_round.py                 # all datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # a subset
uv run python examples/elo_round.py --freeze        # deliberately re-anchor
```

Each round prints `started (frozen baseline) -> reached` Elo for every ModelTree flavor, with per-fit timings in the scores file — so progress across iterations is visible instead of silently overwritten. The full from-scratch rebuild (all models, groups of 5 datasets, shareable intermediate state after each group) is `bash examples/elo_run_groups.sh`.

The field's raw scores (per dataset RMSE/MAE/fit time) live in [`benchmarks/elo_scores.csv`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/elo_scores.csv>); the longer research log is in `docs/field-notes.md`.
