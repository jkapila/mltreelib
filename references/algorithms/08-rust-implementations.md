# 08 — Rust implementations (what to steal from / port to)

These are not new algorithms — they are the Rust codebases surveyed early on
(crates.io pass) and later. Canonical pseudocode lives in files 01–07; entries
here give only the deltas. House position: we build our own core, but these
are the reference implementations and the competition.

## forust-ml (`repos/`-worthy; crate `forust-ml`)

**Delta over #18 (XGBoost):** implements the **same algorithm as XGBoost**
(second-order gradient boosting) in pure Rust with a Python wrapper
(`py-forust`); "nearly identical results" to XGBoost on the same settings.
No GPU; smaller surface (no ranking objectives).
**Steal:** its histogram/quantile split loop as a Rust reference when we
implement the fast trees.

## perpetual — PerpetualBooster (crate `perpetual`)

**Delta over #18:** self-generalizing GBM that **replaces hyperparameter
search with a `budget` parameter** — growth/regularization scale with budget;
a generalization mechanism prevents overfitting (their paper is WIP; blog
explains). Natively handles categoricals, **learnable missing-value splits**,
monotone and feature-interaction constraints. Rust core, zero-copy
Polars/Arrow, Python + R bindings. TabArena entrant (09).
**Steal:** budget-driven growth as our "no HPO" mode; missing-split learning
is exactly our house requirement #2.

## gbdt (mesalock-linux/gbdt-rs)

Classic **GBDT regression** (squared loss, second-order updates) in
minimal Rust — the cleanest small reference implementation to read before
writing our own boosting loop.

## bosk

Pure-Rust **inference** for LightGBM/CatBoost/ONNX model files (parses the
text model format, no FFI). Deployment-side entry: shows the model-file
parsers we may want for interop (train in LGBM, serve in Rust).

## smartcore

Rust ML library: **CART decision trees, RandomForest, ExtraTrees** with
standard impurity criteria — the tree basics already exist in Rust; our
differentiators are native categoricals, missing values and model leaves.

## linfa

`linfa-trees`: **linear decision trees**; `linfa-ensemble`: bagging, random
forest, AdaBoost. Scientific-Rust style APIs; useful reference for trait
design (`Predict`, `Fit`).

## miniboosts

Collection of boosting algorithms (AdaBoost family and friends) in Rust —
readable implementations of the classic boosting zoo (#16 lineage).

## sooboost-core

GBDT in pure Rust: `no_unsafe`, Arrow zero-copy, deterministic/reproducible —
the engineering constraints match ours (we have the same determinism rule).

## biosphere

"Simple, fast random forests" — small focused RF crate; benchmark reference
for our forest speed targets.

## sklears-tree

Trees with **CHAID and oblique (hyperplane) splits**, isolation forests,
multi-output/multi-label trees, incremental/streaming variants. The Rust
oblique-split reference (#5 CHAID + oblique).

## treeboost

Hybrid engine combining **linear models + gradient boosted trees + random
forest robustness**, zero-copy Rust. Close in spirit to the mixing recipes
(#32 LinearForest) — watch as competitor.

## affinitree

Distills faithful **decision trees from piecewise-linear neural networks**
(knowledge distillation) — the "explain the net with a tree" direction; useful
later for FM-to-tree distillation experiments.

## gbt-quantile

Pure-Rust **quantile GBT** (pinball loss) — the Rust side of #23.

## Binding crates (not separate algorithms)

`xgb` / `xgboost-rs`, `lightgbm3` / `lgbm`, `catboost-rs` / `catboost-rust`
wrap the C++ giants (#18–20). For comparisons we use the Python giants; the
bindings matter only if we benchmark Rust-to-Rust.

## Adjacent: rpart (`repos/rpart`, R)

CART as R ships it: cost-complexity pruning **plus surrogate splits** for
missing data (best surrogate = most correlated split — the classic missing
value mechanism our v1 default directions should know about).

## What this means for us

- Nothing here combines: native categoricals + missing values + model leaves
  + structure-recovery measurement. That gap is the package.
- For the build: forust/perpetual/gbdt are the split-loop references;
  smartcore/linfa are the trait-design references; bosk points at model-file
  interop; perpetual's learnable missing splits are the missing-value spec.
