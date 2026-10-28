# mltreelib — Machine Learning with Tree Based Library

Tree-based machine learning with a **Rust core** and thin **Python** (and, next, **R**) frontends. Evolved from the attempt to make the *right* kind of **Decision Trees**, after Hastie, Tibshirani, Friedman, Quinlan, Loh, Chaudhari.

The bet: tree/ensemble methods are under-studied relative to tabular foundation models, and the information hidden in **categorical, grouped and hierarchical structure** is exactly what flat learners lose. We build the classics properly in Rust (CART → C4.5 → C5.0 → Cubist → GritBot), measure them honestly on data where that structure matters, and only then mix and extend. Research grounding lives in [`references/research/`](references/research/) (TabArena analysis, methods map, dataset component design).

## Layout

```
crates/mltreelib-core/   Rust core: dataset component (data layer + adjustable
                         benchmark generator); next: feature transforms, tree learners
crates/mltreelib-py/    PyO3 bindings, exposed as mltreelib._core
python/mltreelib/       Python package — scikit-learn-shaped API, no sklearn dependency
tests/                  Python behavior tests
datasets/               real benchmark datasets + MANIFEST (sourced, size-capped)
references/             research/ notes · docs/ vendor manuals · data/ benchmarks ·
                        sources/ archives · repos/ cloned projects
legacy/                 previous Python/nbdev implementation (kept as reference oracle)
```

## Setup

Requires Rust and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # creates .venv and builds the Rust extension (maturin)
uv run pytest tests/ -q  # behavior tests
```

## Dependency policy

Lean but **pragmatic — no wheel reinvention**: the goal is fast data handling and
fast tree algorithms, with research on top. Scikit-learn stays out (API shape
only, never import it); established core packages are welcome where they replace
real complexity — `pyarrow` for parquet (used by `mltreelib.datasets`), and
arrow-rs (`arrow`/`parquet`) or Polars in Rust when frames/IO grow beyond the
in-house `frame` module. Every other dependency must earn its place.

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
- [x] `DatasetBuilder` ingest of any data: NA policy (allow/drop/impute), digit rounding, intercept column, dtype reduction
- [x] Calendar/time features (year, halfyear, quarter, month, ISO week, weekday, day-of-year, season, weekend)
- [x] Lag features by whole days within a key (featuregen `LagFeatures` lineage)
- [x] Trended transforms at user granularity (`year_month`, `year_week`, `month_weekday`, …) — featuregen lineage
- [x] Decomposed transforms — statsmodels `seasonal_decompose` convention (trend, seasonal, residual)
- [x] Windowed aggregations (trailing sums/means/counts over day windows per key)
- [x] Per-column variable transformations (`VariableScaler`, legacy `VariableTransformation`)
- [x] Missing data is first-class: NA **and NaN** = missing end-to-end (missing in → missing out);
      `summarize` shows `n_missing` by default; imputation is optional (allow/drop/impute);
      time-series transforms drop incomplete rows and warn
- [x] Built-in categorical encoders: ordinal (`IndexMapper`), one-hot, hashing, base-N/binary,
      count, sum contrast, smoothed target + M-estimate, leave-one-out, CatBoost-style
      **ordered target statistics** (`fit_transform` leakage-safe), quantile, James-Stein,
      class-level `TfidfEncoder`, `WoEEncoder`, `EmbeddingEncoder` (reusable `nn` trainer;
      ReLU/linear output by target sign) — *research in
      `references/research/categorical-encoding-research.md`*
- [ ] cat×cat combinations (inside the tree split search); deferred encoders: GLMM,
      Helmert/Polynomial/BackwardDifference contrasts, Gray, MultiHot/RankHot, CountTarget
- [ ] Featuretools-style automated features (Cartesian products, sub-level %-by-class, deep feature synthesis lite) — *later, same reason*
- [x] `mltreelib.datasets` loader (pyarrow) — suite loads into `DatasetBuilder`-shaped dicts
- [ ] Harness integration of `datasets/` + generator sweeps

### 2. Tree learners in Rust (replicate the giants first)

Research is consolidated — build order below (see
`references/research/tree-research-consolidation.md` for the evidence):

1. [ ] **Simple model tree v1** — leaf-local linear models, split gain = SSE
   reduction from fitting leaf models on both sides (linear-tree recipe);
   leaf OLS via faer; native missing values + native categorical splits in the
   first cut; per-leaf `Metrics` reports from day one
2. [ ] CART + C4.5 with constant leaves (oracle: `legacy/mltreelib/tree/models.py`)
3. [ ] **Cubist**: model trees + committees + kNN corrections
4. [ ] C5.0 rulesets + committees; GritBot surprising-subgroup mining
5. [ ] Second split criterion: MOB instability tests (partykit lineage);
   third: gradient improvement (TAO path)
6. [ ] Mixing: LinearForest-style ensembles, sparse leaves (TRUST), uncertainty (BART-style)

Design requirements for every learner (the sklearn pipeline is the anti-pattern:
no forced imputation, no forced numeric encoding):
- [ ] **Native missing values** — missing-aware split search + learned default
      directions (XGBoost/LightGBM style); surrogate splits where the classic
      calls for them (C5.0 lineage)
- [ ] **Native categorical splits** — high-cardinality aware, no one-hot tax
- [ ] CART (regression + classification) — oracle: `legacy/mltreelib/tree/models.py`
- [ ] C4.5 (information gain, gain ratio)
- [ ] C5.0: rule sets, boosting committees, winnowing — `references/docs/see5-unix.md`
- [ ] Cubist: model trees + committees + neighbor correction — `references/docs/cubist-unix.md`
- [ ] GritBot: surprising-subgroup mining — `references/docs/gritbot-unix.md`

### 3. Honest comparisons (proof-driven, not blind experimentation)
- [ ] Harness vs LightGBM / XGBoost / CatBoost / EBM on `datasets/` + generator sweeps
- [ ] Vs tabular foundation models (TabPFN-class) on the same splits — see `references/research/tree-vs-foundation-tabular.md`
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

### 6. Research & publication (the end goal)
A research-backed package **and a published paper**: the claim is that tree/ensemble
methods with structural awareness of categorical/group/hierarchical signal recover
what flat learners lose — measured as structure recovery on generated ground truth
plus TabArena-v0.1 real data.
- [ ] Reproducible experiment reports (fixed seeds, uv lockfile, per-dataset win/loss)
- [ ] Figures/tables generated from the harness, not hand-made
- [ ] Paper draft — **stack chosen at paper time, no pre-commitment** (survey 2026-10):
      Typst for modern typesetting; reproducible-manuscript pipelines (Quarto-style)
      can emit Typst/LaTeX; LaTeX where venues demand it; Rχiv-Maker-style
      automation is the pattern to evaluate when we get there
- [ ] Research speed tooling: `autogluon/tabarena` harness, OpenML (`datasets/`),
      uv for envs, arXiv workflow; agent skills for artifacts (hallmark for pages,
      docx/pptx/pdf for documents, research skills for sources)

## Documentation

Built with **Zensical** (`variant: modern` — the maintained successor to
mkdocs-material; it reads the same `mkdocs.yml`, and `mkdocs build` remains as a
safety net):

```sh
uv run zensical serve     # live docs at http://127.0.0.1:8000
uv run zensical build     # static site in site/
```

- **Docs site** — quickstart, demo walkthrough (with figures), datasets, API reference
  (generated from the live docstrings by `scripts/gen_api_docs.py`; Zensical natives:
  `minify`, `llmstxt` → `llms.txt` for LLM consumption)
- **[AGENTS.md](AGENTS.md)** — project rules and conventions (also [CLAUDE.md](CLAUDE.md))
- **Rust API**: `cargo doc -p mltreelib-core --no-deps --open` (doc-comments are the reference)
- **Runnable walkthroughs**: `uv run python examples/data_layer_demo.py [dataset ...]` (data &
  feature layer) and `uv run python examples/encoding_demo.py [dataset ...]` (every encoder
  on real data)

## License

Apache-2.0
