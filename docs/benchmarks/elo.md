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
+-----------+---------------------+--------------------------+-------------+
|    ELO    |    FIT TIME (±sd)   |          MODEL           |     DATA    |
+-----------+---------------------+--------------------------+-------------+
|   1577.1  |   15.41 s (±24.59)  |         grf_llf          |    17/18    |
|   1525.9  |     0.67 s (±0.55)  |      LightGBM(200)       |    18/18    |
|   1480.5  |     0.36 s (±0.31)  |      CatBoost(200)       |    18/18    |
|   1463.7  |    6.50 s (±13.12)  |     perpetual (Rust)     |    18/18    |
|   1331.3  |     0.50 s (±0.29)  |      forust (Rust)       |    18/18    |
|   1229.5  |   11.47 s (±17.45)  |          grf_rf          |    18/18    |
|   1217.3  |     0.23 s (±0.37)  |         CTBoost          |    18/18    |
|   980.1   |   13.89 s (±27.18)  |          lmtree          |    16/18    |
|   977.2   |     1.19 s (±1.53)  |          cubist          |    18/18    |
|   954.2   |   10.39 s (±19.45)  |         glmtree          |    15/18    |
|   914.8   |   70.56 s (±238.75) |      ModelTree(all)      |    18/18    |
|   891.5   |   17.77 s (±44.25)  |    modeltrees lmtree     |    17/18    |
|   823.5   |  143.31 s (±463.47) |    MARS (mars-earth)     |    12/18    |
|   765.6   |     1.46 s (±2.38)  |          ctree           |    18/18    |
|   763.5   |     0.26 s (±0.37)  |          earth           |    17/18    |
|   691.7   |     0.13 s (±0.20)  |       sklearn tree       |    18/18    |
|   650.7   |   56.32 s (±225.98) |    ModelTree(winnow)     |    18/18    |
|   649.7   |     0.15 s (±0.20)  |     ModelTree(mean)      |    18/18    |
|   566.3   |     0.22 s (±0.32)  |          rpart           |    18/18    |
|   545.8   |     0.86 s (±1.22)  |     ModelTree(path)      |    18/18    |
+-----------+---------------------+--------------------------+-------------+

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
