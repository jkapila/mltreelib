# ELO Benchmark Arenas

One scoreboard, one protocol, every model scored on the **same prepared data**.
The arena evaluates models across **Rust, Python, and R** under a strict 300s process-group timeout.

Detailed runbook & ModelTree experimentation guide: [`benchmarks/README.md`](https://github.com/jkapila/mltreelib/blob/main/benchmarks/README.md).

---

## The Protocols

### 1. Regression Arena
- **Data**: 18 regression benchmark datasets, one stratified 80/20 split (seed 7), prepared once by `examples/elo_prepare.py`.
- **Matchups**: Per dataset, every unordered pair of models "plays" once; lower test RMSE wins (tie = draw).
- **Scores**: Evaluates 36 models across Rust, Python, and R (`benchmarks/elo_scores.csv`).
- **Reproduce**: `uv run python examples/elo_report.py` (or `bash examples/elo_run_resume.sh`).

### 2. Classification Arena
- **Data**: 24 TabArena datasets (20 binary, 4 multiclass) from OpenML suite 457, stratified 80/20 split within class (seed 7), prepared once by `examples/elo_prepare_cla.py` with raw strings for categoricals and `<ds>.meta.json`.
- **Matchups**: Per dataset, round-robin pairwise comparisons based on test **log loss** (cross-entropy), K=32, starting rating 1000.
- **Scores**: Evaluates 28 models across Python, Rust, and R (`benchmarks/elo_cla_scores.csv`), including RuleQuest `C5.0 (tree)` and `C5.0 (rules)` in R, `HistGradientBoosting`, `sklearn GBM`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `ranger`, `bart`, `glmtree`. *(Note: ModelTree classification is currently undergoing internal development in Rust and will be attached once complete).*
- **Reproduce**: `uv run python examples/elo_report_cla.py` (or `bash examples/elo_run_cla.sh`).

---

## Metadata Taxonomy

Every score tracked in the arena includes two core architectural dimensions:
- **`lang`**: Model implementation language — **`Rust`**, **`Python`**, or **`R`**.
- **`cat_handling`**:
  - **`internal`** (*native / as-is*): Direct native consumption of categoricals without driver encoding (`ModelTree`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `HistGradientBoosting`, `modeltrees`, R factors like `rpart`, `cubist`, `C5.0`, `ctree`, `glmtree`).
  - **`external`** (*driver-encoded*): Requires driver-side ordinal encoding fit on train (`sklearn RF`, `sklearn GBM`, `ExtraTrees`, `sklearn tree`, `forust`, `perpetual`, `MARS`).
  - **`none`** (*numeric-only*): Strictly requires numeric matrices; refuses categorical columns (`statsmodels`, `trust-free`, R `grf`).

---

## The Regression Board

```text
== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==
+-----------+---------------------+------------------------+--------+------------+-----------+
|    ELO    |    FIT TIME (±sd)   |         MODEL          |  LANG  |    CAT     |    DATA   |
+-----------+---------------------+------------------------+--------+------------+-----------+
|   1690.0  |   12.01 s (±17.76)  |      ChimeraBoost      | Python |  internal  |   18/18   |
|   1622.3  |   18.24 s (±23.58)  |         ranger         |   R    |  internal  |   16/18   |
|   1618.1  |   19.20 s (±33.35)  |       sklearn RF       | Python |  external  |   18/18   |
|   1568.1  |   13.02 s (±33.28)  |       ExtraTrees       | Python |  external  |   18/18   |
|   1532.3  |   23.17 s (±31.44)  |        grf_llf         |   R    |    none    |   16/18   |
|   1470.6  |     0.35 s (±0.22)  |     LightGBM(200)      | Python |  internal  |   18/18   |
|   1439.3  |    7.20 s (±23.69)  |    perpetual (Rust)    |  Rust  |  external  |   17/18   |
|   1425.5  |     1.04 s (±1.10)  |     CatBoost(200)      | Python |  internal  |   18/18   |
|   1350.1  |     0.95 s (±1.91)  |  HistGradientBoosting  | Python |  internal  |   17/18   |
|   1319.3  |   65.08 s (±68.98)  |          EBM           | Python |  internal  |   15/18   |
|   1299.2  |     1.22 s (±0.86)  |     forust (Rust)      |  Rust  |  external  |   18/18   |
|   1210.8  |     0.76 s (±1.83)  |        CTBoost         | Python |  internal  |   17/18   |
|   1168.4  |   27.91 s (±39.58)  |          bart          |   R    |  internal  |   17/18   |
|   1166.7  |   18.06 s (±23.80)  |         grf_rf         |   R    |    none    |   18/18   |
|   1050.1  |   13.77 s (±20.73)  |     ModelTree(all)     |  Rust  |  internal  |   16/18   |
|   1030.9  |    7.00 s (±13.24)  |      sklearn GBM       | Python |  external  |   18/18   |
|   976.4   |   20.22 s (±47.48)  |         lmtree         |   R    |  internal  |   14/18   |
|   972.9   |   15.29 s (±51.54)  |   modeltrees lmtree    | Python |  internal  |   14/18   |
|   958.1   |   29.02 s (±57.11)  |        glmtree         |   R    |  internal  |   14/18   |
|   928.3   |     5.95 s (±8.90)  |   ModelTree(winnow)    |  Rust  |  internal  |   16/18   |
|   892.7   |    6.23 s (±15.19)  |         cubist         |   R    |  internal  |   17/18   |
|   873.1   |   20.49 s (±31.78)  |   MARS (mars-earth)    | Python |  external  |   10/18   |
|   761.5   |   20.60 s (±51.49)  |         mboost         |   R    |  internal  |   15/18   |
|   697.6   |     0.33 s (±0.68)  |    statsmodels OLS     | Python |    none    |    8/18   |
|   690.7   |   10.19 s (±27.50)  |         earth          |   R    |  internal  |   17/18   |
|   681.8   |     0.09 s (±0.10)  |          KNN           | Python |  external  |   18/18   |
|   655.7   |   21.17 s (±59.09)  |         ctree          |   R    |  internal  |   17/18   |
|   611.2   |    14.37 s (±7.33)  |       trust-free       | Python |    none    |    4/18   |
|   584.5   |     0.09 s (±0.23)  |     sklearn Linear     | Python |  external  |   18/18   |
|   547.3   |   10.80 s (±21.74)  |    ModelTree(path)     |  Rust  |  internal  |   18/18   |
|   539.8   |     0.34 s (±0.57)  |      sklearn tree      | Python |  external  |   18/18   |
|   527.1   |   17.48 s (±43.02)  |    statsmodels RLM     | Python |    none    |    8/18   |
|   482.0   |    5.34 s (±20.14)  |    ModelTree(mean)     |  Rust  |  internal  |   18/18   |
|   368.3   |   22.52 s (±59.15)  |  statsmodels QuantReg  | Python |    none    |    7/18   |
|   289.3   |     1.34 s (±3.71)  |         rpart          |   R    |  internal  |   17/18   |
+-----------+---------------------+------------------------+--------+------------+-----------+
```

TabArena anchor (different protocol - reference only):
```

Reading it:

- **Stock configs only** — nothing tuned per dataset. The top of the board is
  *forests and boosted ensembles with strong feature handling* (grf's
  local-linear forest leads; forust/perpetual are boosted trees).
- **Our flavors reflect the current defaults** (the split grid at 0.001,
  ridge-linear leaves): `ModelTree(all)` 914.8, `winnow` 650.7, `mean`
  649.7, `path` 545.8. The field anchor predates the split-grid default, so
  the live comparison is the round deltas — see **Running a round** below
  and `benchmarks/elo_round.log`: the grid trades differently per flavor
  (`path`/`mean` gain, `all`/`winnow` pay), and the manifests in
  `benchmarks/rounds/` record exactly which settings produced each move.
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

## Running a round (repeatable, with provenance)

One command re-scores **our four flavors** against the frozen field:

```bash
uv run python examples/elo_round.py                    # all 18 datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # subset
```

Every round records what created its numbers:

- `benchmarks/rounds/<stamp>/settings.json` — the **exact settings dict**
  each flavor was fit with (single-sourced: the manifest records the same
  dict `variants()` builds from, so record and reality cannot drift), the
  git SHA, the dataset list, every result row, and Elo before → after.
- `benchmarks/elo_round.log` — stdlib `logging` output (file + console), one line per round: `timestamp, stamp, sha,
  dataset count, per-model Elo before->after`. This is the first place to
  look for "what did we change and what did it move".
- `benchmarks/elo_scores.csv` — the live board (our rows upserted);
  `benchmarks/elo_baselines.csv` — the frozen anchor ("where we started").

Tuned settings from `examples/hpo_optuna.py` print in the same dict shape —
paste them into `SETTINGS` in `elo_round.py` to score a tuned flavor.
