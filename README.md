# mltreelib — Machine Learning with Tree Based Library

Tree-based machine learning with a **Rust core** and thin **Python** (and, next, **R**) frontends. Evolved from the attempt to make the *right* kind of **Decision Trees**, after Hastie, Tibshirani, Friedman, Quinlan, Loh, Chaudhari.

The bet: tree/ensemble methods are under-studied relative to tabular foundation models, and the information hidden in **categorical, grouped and hierarchical structure** is exactly what flat learners lose. We build the classics properly in Rust (CART → C4.5 → C5.0 → Cubist → GritBot), measure them honestly on data where that structure matters, and only then mix and extend. Research grounding lives in [`references/`](references/) (TabArena analysis, methods map, dataset component design).

## Layout

```
crates/mltreelib-core/   Rust core: dataset component (data layer + adjustable
                         benchmark generator); next: feature transforms, tree learners
crates/mltreelib-py/    PyO3 bindings, exposed as mltreelib._core
python/mltreelib/       Python package — scikit-learn-shaped API, no sklearn dependency
tests/                  Python behavior tests
datasets/               real benchmark datasets + MANIFEST (sourced, size-capped)
references/             research notes, benchmarks, source material
legacy/                 previous Python/nbdev implementation (kept as reference oracle)
```

## Setup

Requires Rust and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # creates .venv and builds the Rust extension (maturin)
uv run pytest tests/ -q  # behavior tests
```

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

## Roadmap / TODO

Order matters: foundations first, then honest replication, then novelty.

### 1. Data & feature layer  ← current
- [x] Core columnar `Dataset`/`Column` with NA slots and categorical vocabularies
- [x] `n_category` schema rule (absolute levels or %-of-rows)
- [x] `IndexMapper` (multi-column categorical codes + exact inverse)
- [x] Grouped variable transformations (`fit`/`transform`, mean/std/count/quantiles)
- [x] Adjustable generator + ground-truth manifest (Rust + Python)
- [ ] `DatasetBuilder` ingest of any data: NA policy (allow/drop/impute), digit rounding, intercept column, dtype reduction
- [ ] Calendar/time features (year, month, week, day-of-week, hour…)
- [ ] Trended transforms at user granularity (year-month, year-week, …) — featuregen lineage
- [ ] Decomposed transforms (series − trend − seasonality)
- [ ] Aggregation features (grouped sums/means/counts over windows)
- [ ] Wire `datasets/` real-data suite into the test/harness loaders

### 2. Tree learners in Rust (replicate the giants first)
- [ ] CART (regression + classification) — oracle: `legacy/mltreelib/tree/models.py`
- [ ] C4.5 (information gain, gain ratio)
- [ ] C5.0: rule sets, boosting committees, winnowing — `references/see5-unix.md`
- [ ] Cubist: model trees + committees + neighbor correction — `references/cubist-unix.md`
- [ ] GritBot: surprising-subgroup mining — `references/gritbot-unix.md`

### 3. Honest comparisons (proof-driven, not blind experimentation)
- [ ] Harness vs LightGBM / XGBoost / CatBoost / EBM on `datasets/` + generator sweeps
- [ ] Vs tabular foundation models (TabPFN-class) on the same splits — see `references/tree-vs-foundation-tabular.md`
- [ ] Structure-recovery scoring: does each method capture the known group/hierarchy signal?
- [ ] Equal tuning budgets for trees (the classic weak-baseline trap)

### 4. Mixing & differentiating features
- [ ] CatBoost-style statistical features (ordered target statistics) for high-cardinality categoricals
- [ ] EBM-style consolidation (additive shape functions) as a feature layer
- [ ] NGBoost-style distributional heads; DAR-style target-space routing
- [ ] Stacking/blending across families (trees + foundation models)

### 5. Packaging
- [ ] R bindings (extendr over `mltreelib-core`)
- [ ] Python API parity with the sklearn shape (`fit`/`predict`/`transform`), still no sklearn dependency
- [ ] Docs + reproducible experiment reports

## License

Apache-2.0
