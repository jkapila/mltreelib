# mltreelib — Machine Learning with Tree Based Library

Tree-based machine learning with a **Rust core** and thin **Python** (and, next, **R**) frontends. Evolved from the attempt to make the *right* kind of **Decision Trees**, after Hastie, Tibshirani, Friedman, Quinlan, Loh, Chaudhari.

The bet: tree/ensemble methods are under-studied relative to tabular foundation models, and the information hidden in **categorical, grouped and hierarchical structure** is exactly what flat learners lose. We build the classics properly in Rust (CART → C4.5 → C5.0 → Cubist → GritBot), measure them honestly on data where that structure matters, and only then mix and extend.

## Layout

```
crates/mltreelib-core/   Rust core: data layer, generator, metrics, the model tree
crates/mltreelib-py/     PyO3 bindings (the mltreelib Python surface)
python/mltreelib/        Python package: datasets loader + optional plots layer
examples/                runnable examples (all docs output comes from these)
docs/                    the documentation site (zensical; benchmarks + API + guides)
tests/                   Python behavior tests (the Rust seam lives in the core crate)
benchmarks/              the Elo field: prepared splits, scores, attempts, round manifests
specs/                   implementation specs (published as GitHub issues)
playground/              local marimo playground for edge-case hunting (not in git)
```

## Setup

Requires Rust and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                        # .venv + builds the Rust extension (maturin)
uv run --with pandas --with pyarrow python examples/quickstart.py
```

Optional extras: `plotting` (matplotlib/pygraphviz/seaborn), `tuning` (optuna).

## The model tree today

Typed ingest — categoricals enter as **levels** (a distinct value is a level;
the split search partitions level sets), NAs route as-is, nothing is encoded
for the tree:

```python
from mltreelib import ModelTree, Metrics

m = ModelTree(
    max_depth=4, min_rows=10, leaf="linear",   # leaf: linear|mean|median|quantile|huber
    l2=0.1,                                    # ridge on leaf coefficients
    gain="sse",                                # regression: sse|friedman_mse|absolute_error|huber|poisson|sdr
                                               # class: entropy|gini|gain_ratio|twoing | callable
    split_precision=0.001,                     # candidate grid (None = exact scan)
    cat_order="frequency_desc", cat_search="cumulative",
    alpha=0.0, imbalance_penalty=0.0,          # grf-style split balance
    prune="none",                              # none|bic|mdl (+ cp family landing)
    split_test="none",                         # none|chow — significance-gated splitting
    split_alpha=0.05, bonferroni=False,        # gate level (+ selection correction)
    col_sample=None, seed=42,                  # per-node random column subset
    leaf_digest=True,                          # per-leaf t-digest: any tau post-fit
)
m.fit(x, y, dtypes=dtypes)      # dtypes per column: "numeric" | "categorical"
m.predict(xte)                  # NAs never block a prediction
m.quantile(x, 0.9)              # conditional quantiles from the leaf digests
m.training_log()                # candidate counts, dtype search, timings (verbose)
```

Growth knobs under active development (spec `#3`, all default-off until a
round measures them): `honesty`, NA **surrogate splits**, the pruning
family (cost-complexity / C5 pessimistic / cost-matrix),
`minsplit/minbucket/minprob`, EFB-style pre-selection, and an experimental
purity gate. Significance-gated splitting has **landed** (`split_test="chow"`
+ `split_alpha` + `bonferroni`: the scan's best split must beat the pooled
node model in a Chow F test, else the node stays a leaf). Distributional
leaves (`leaf_family`: binomial/poisson →
probit/tobit/tweedie/softmax, `predict_proba`) are spec `#4`.

## Dataset component

The dataset component is the heart of the library: an adjustable benchmark
generator whose knobs control how much signal lives in categorical, grouped and
hierarchical structure, plus a data layer that takes any data in.

```python
from mltreelib import generate, IndexMapper, grouped_transform, summarize

g = generate(
    n_rows=10_000, n_numeric=5, cardinalities=[3, 8, 50, 200],
    hierarchy_levels=3, branching=[4, 10, 25],
    group_effect_strength=0.6, hierarchy_effect_strength=0.3,
    interaction_strength=0.2, noise=0.1, seed=42,
)
g["columns"]                 # num_*, cat_*, hier_* columns (None = NA)
g["target"]                  # regression target (labels for task="classification")
g["truth"]["variance"]       # realized variance share per component
g["truth"]["categorical_effects"]  # ground-truth level effects
```

Because the ground truth is known, every learner can be scored on *structure
recovery*, not just accuracy — including tail behavior (`tail_behavior="bathtub"`
reproduces the DAR bathtub edges) and missingness.

## Measuring: the Elo field

`benchmarks/` holds a living field: 18 prepared datasets, 30+ models (our
flavors; rpart/ctree/lmtree/glmTree/cubist/grf/mboost/ranger/BART-family/
TRUST/earth in R; LightGBM/CatBoost/CTBoost/ChimeraBoost/EBM/ExtraTrees/
statsmodels-linear/KNN/trust-free/XBART in Python), round-robin Elo on
per-dataset RMSE. Every round records a **provenance manifest** (git SHA +
the exact settings dict + all results) in `benchmarks/rounds/`, and every fit
runs under strict time tiers: 30s = slow (the TabArena-toppers line), 180s =
critical, 300s = hard kill. All run events go through stdlib `logging` into
`benchmarks/elo_run.log` / `elo_round.log`.

Docs: [API reference](https://jkapila.github.io/mltreelib/reference/api/),
[ELO board](https://jkapila.github.io/mltreelib/benchmarks/elo/),
[guides and examples](https://jkapila.github.io/mltreelib/examples/walkthrough/).

## Dependency policy

Lean but **pragmatic — no wheel reinvention**: fast data handling and fast
tree algorithms, with research on top. Scikit-learn stays out of the library
(API shape only, never imported); established packages earn their place where
they replace real complexity — `pyarrow` (parquet), `tdigest` (leaf quantiles),
`rayon` (parallel scans), `faer` (dense solves in the split hot path). The
field/benchmark tooling may use anything it measures (that is its job).

## Research inputs

Grounded research lives with the team (references/ is local-only): the
split-criteria survey, NGBoost leaf/training deep-dive, leaf-purity
literature review, and the knob-gap reviews against rpart/ctree/cubist/
partykit/grf. Specs are published as GitHub issues (`ready-for-agent`).

## Roadmap

1. [x] Data & feature layer (builder, encoders, transforms, generator, datasets loader)
2. [x] Model tree in Rust (typed ingest, criterion family, leaf kinds, quantiles,
      pruning (bic/mdl), balance knobs, budgets, training log)
3. [ ] **Model-tree capability knobs** (issue #3) — significance splits, honesty,
      NA surrogates, pruning family, minsplit/minbucket/minprob, EFB, purity gate
4. [ ] **Distributional leaves** (issue #4) — `leaf_family`: binomial/poisson →
      probit/tobit/tweedie/softmax, `predict_proba`, IRLS on faer
5. [ ] **Forests & committees** — honesty + averaging (grf-lesson), Cubist-style
      committees + neighbor adjustment; the ensembles currently top the board
6. [ ] C5.0 rulesets + committees; GritBot surprising-subgroup mining
7. [ ] Mixing: LinearForest-style ensembles, sparse leaves (TRUST lineage)

Ground rule: **every knob is default-off until a measured round decides
otherwise** — the round manifests exist so that no number moves for free.
