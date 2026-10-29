# Pre-selection & expert-mixture thesis — the frame for the next phase

**Date:** 2026-10-28. **Source:** Jitin's direction (consolidated this session).
**Status:** frame + starting points, *not* a build task yet. **Constraint
(standing):** classical tree structures only — no neural components in this
line of work.

---

## 1. The arc so far — and the two questions that ARE a tree

1. **Single tree**, grown to its best: CART → C4.5 → C5.0 → Cubist (model
   leaves). One tree, enhanced maximally.
2. **Boosting**: the first tree's error is propagated — what gets subtracted
   becomes the next tree's target. (Friedman; XGBoost/LightGBM lineage.)
3. **Cost functions**: impurity → squared-error reduction → likelihood gains;
   the split objective became the field's obsession.
4. Distilled: the core hypothesis of **any** tree is exactly two decisions —
   (a) **which variable** to split on, (b) **at which point / in which
   direction** of that variable to split. Everything else — pruning, leaves,
   boosting, ensembles — hangs off these two. Any real improvement must land
   on one of them.

## 2. Pre-selection of variables — shrink the search space BEFORE the search

**Hypothesis (Jitin):** pre-selecting variables for tree building is doubly
good — it reduces the search space *and* makes the tree itself better (fewer
spurious splits, less high-cardinality noise, better splits on survivors).

### 2.1 What the giants do

- **C5.0 winnowing** (RuleQuest; grounded in `../docs/see5-unix.md:1110-1256`,
  option `-w`): after analyzing the training cases and **before** the tree is
  built, C5.0 discards "unhelpful" attributes ("winnowing… wheat from chaff").
  Winnowed attributes behave exactly like names-file exclusions — but they
  may still appear in *derived* attribute definitions. The output then lists
  the survivors by estimated importance. Recommended by the vendor for large
  applications (100k+ cases) with many marginally relevant attributes.
- **LightGBM** (Ke et al., NeurIPS 2017) — two complementary reductions:
  - **GOSS** (Gradient-based One-Side Sampling): *row-side* — keep all
    large-gradient instances, randomly sample small-gradient ones and
    rescale their contribution; the search sees fewer rows without losing
    information.
  - **EFB** (Exclusive Feature Bundling): *column-side* — mutually exclusive
    sparse features (one-hot families, rare indicators) are bundled into
    single columns via graph coloring; the search sees fewer columns.
  - plus the trivial pre-filter (`feature_pre_filter`) dropping
    zero/constant-gain columns.

### 2.2 Other approaches of the same shape (the survey)

| method | screen | granularity | what it removes |
|---|---|---|---|
| C5.0 winnowing | attribute utility vs class | features | weak attributes (pre-build) |
| LightGBM GOSS | gradient magnitude | rows | low-information rows |
| LightGBM EFB | exclusivity/sparsity | features | redundant bundles |
| Boruta | RF importance vs shadow (permuted) copies | features | statistically irrelevant features |
| ReliefF (mlxtend suite) | neighborhood-difference relevance | features | weak features |
| SIS (Fan & Lv 2008) | marginal correlation screen | features | ultra-low-signal features (ultra-high-dim case) |
| mboost / glmboost | coefficient paths + early stopping | features | late/never-entered features (selection *is* the path) |
| ctree / MOB gating | association tests at split time | features *during* growth | biased/insignificant split candidates |
| variance / NZV filter | dispersion | columns | constants, near-constants |

(`boruta-fru` is already in `../sources/`; ReliefF is in the mlxtend catalog;
the mboost source is in `../sources/extracted/mboost_2.9-14/`.)

### 2.3 The preprocessing steps each performs — our starting-point ingredients

- **winnowing**: whole-set utility analysis of each attribute vs the target
  → survivor set + importance ranking.
- **GOSS**: per-row gradient audit → row weighting/rescaling instead of
  deletion.
- **EFB**: feature-feature conflict graph → bundled columns.
- **Boruta/ReliefF/SIS**: per-feature relevance score → threshold cut.
- **mboost**: sequential fit → path position = selection.
- **common shape**: *score every column (and sometimes rows) cheaply, before
  any split search; hand the tree only the survivors.* That is the layer to
  build (Starting point B).

## 3. Expert-mixture thesis — heterogeneous trees

- Almost all literature predicts from a **homogeneous** tree: one perspective
  over the data; prediction = the leaf value.
- **Garcin, Dimitrakakis & Faltings, "Personalized News Recommendation with
  Context Trees" (RecSys 2013)** — *read this one*: https://arxiv.org/abs/1303.0665
  — builds **experts per node** (standard / popularity / freshness views),
  mixed along the active path by learned weights. Prediction is a **mixture
  of perspectives**, not a lookup. The tree handles "how much history", the
  experts handle "which lens".
- **Jitin's idea:** bake expert behavior **into the tree itself** — different
  regions/views of the data get different expert machinery, and the tree
  *automatically decides* which expert governs where. The math is ours to
  design later (weights, gating, counts — all classical).
- Why experts can dissolve the open gaps (survey, `11-sequential-rec-trees.md`):
  - **beam-pruning error propagation** → an internal node can *hedge* with a
    mixture instead of committing to one route;
  - **static-tree rigidity** → expert weights can drift/re-adapt without
    restructuring the hierarchy;
  - **popularity vs recency conflation** → separate experts for the two
    mechanisms — the causal separation the survey asks for;
  - hyperbolic-geometry gap stays out of scope (that line is geometric/NN).
- Constraint honored: mixtures of counts, weights, tests, and classical leaf
  models — nothing neural.

## 4. Starting point A — one single tree that extrapolates to multiple things

- The model-tree v1 (build TODO item 1), restated by this thesis: a single
  tree whose **leaves are logistic or linear regression** (per task), which
  through structure-recovery scoring can stand as the base for boosting,
  experts, and everything above it.
- **Rust infra needed:**
  - linear leaves: least squares — QR via `nalgebra` (or `faer` if we want
    the speed; decide at build time).
  - logistic leaves: IRLS/Newton iterations reusing the same linear solver,
    + optional ridge for separation safety.
  - the `NodeModel` contract is already designed in
    `../research/dataset-component-design.md` (fit/predict per node).

## 5. Starting point B — the pre-selection layer

- A `preselect` stage before split search: score columns/rows with the 2.3
  menu (start: variance/NZV + winnowing-style utility + mboost-style path
  later), hand only survivors to the split search.
- Measure both promised effects with `Metrics` + structure recovery:
  search-space size/time, and whether screening keeps the planted causal
  columns while dropping injected junk.

## 6. When we pick this up — ordered next moves

1. Pin C5.0's exact winnowing semantics (importance measure) from
   `../docs/see5-unix.md` — done at doc level, refine at design level.
2. Implement the linear + logistic leaf solvers in Rust (A's dependency).
3. Design the `preselect` module; wire candidate-column lists into the split
   search (B).
4. Design the expert-mixture math (Garcin-style weights over classical
   experts; how experts map to regions/leaves).
5. Then: the single tree that extrapolates, built from all of the above.
