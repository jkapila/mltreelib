# Dataset component design — "the heart" (draft spec, 2026-10-28)

Grounded in: `mltreelib/data/*.py` (existing Python, nbdev-style), `featuregen`, Contrec Prototype V3.ipynb, DAR/CrossTab-Sparsity/TrieKNN essays. Constraints from Jitin: **no sklearn dependency — sklearn-style API shape only**; Rust core, Python + R exposed; scope = **generator + data layer**.

## What exists today (Python, `mltreelib/`)

| Module | Content | Rust fate |
|---|---|---|
| `data/data.py` | `Data` adapter: reduce_datatype, encode_category, add_intercept (model-tree ready), na_treatment, digits, `n_category` (level-count or %-of-rows category inference), dtype ranges | Port semantics → `mltreelib-data` |
| `data/utility.py` | `IndexMapper` (multi-col categorical index + inverse), `summary_func`, `VariableTransformation`, `GroupedVariableTransformation` | Port; TODOs in file become v1: **trended transforms at time granularity (year, year-month, year-week...)**, ungrouped transform, decomposed (trend+seasonality) transform |
| `data/datasets.py` | `get_dataset` (adult, palmerpenguins, california, titanic), `show_data`; TODOs: statsmodels-style dataset metadata, Rdatasets, use for comparisons | Port + extend with generated datasets |
| `base/tree.py` | `Node`, `DecisionNode`, `BaseDecisionTree` (objectives: class/reg/survival/poisson) | Reference oracle for Rust trees |
| `tree/models.py` | `Tree`, `CARTRegressionTree`, `CART`, `C45` | Replicate in Rust (phase 3 baselines) |
| `utils/` | entropy/gini/MSE/MAE/RSS, `divide_on_feature`, kernels, metrics | Port as needed |

## Design: two halves in one crate family

### A. Generator (adjustable benchmark data with known ground truth)

Purpose: produce datasets where categorical/group/hierarchical signal strength is a **knob**, so "trees capture group behavior that LightGBM/FMs lose" becomes measurable, falsifiable, and reproducible (seeded).

Proposed knobs (defaults in brackets):
- size: `n_rows` [10_000], `n_numeric` [5]
- categoricals: `cardinalities` [[3, 8, 50, 200]] (mixed low/high-card)
- hierarchy: `hierarchy_levels` [3] (e.g. region→city→store), `branching` [[4, 10, 25]]
- signal: `group_effect_strength` [0.5] (variance share from group random effects), `hierarchy_effect_strength` [0.3], `interaction_strength` [0.2] (cat×numeric), `noise` [0.1]
- target: `task` [regression|classification], `tail_behavior` [gaussian|heavy|bathtub] — the DAR motivation: signal weak at distribution ends
- realism: `missingness` [0.0], `drift` [none|group_shift|temporal] (BeyondArena alignment), `seed`
- output: feature table + **ground-truth manifest** (true signal per feature/group, hierarchy tree, noise floor) → lets every method be scored on *recovery*, not just accuracy

### B. Data layer (any data in → model-ready)

- `Dataset` schema inference: numeric / ordinal / categorical / text-like / datetime; category vs int via `n_category` rule (port from `Data`)
- `IndexMapper` semantics: fit/transform/inverse_transform, unseen-level policy
- grouped & trended transforms (featuregen semantics; time granularity user-set)
- NA policy (`allow`/`drop`/`impute-*`), dtype reduction, rounding, intercept
- sklearn-style API surface (`fit`, `transform`, `fit_transform`, `predict` on models) without importing sklearn; Arrow/Polars zero-copy at the Rust boundary (cf. `perpetual`'s interop pattern)

## Why the generator encodes the hypothesis

Contrec notebook's thesis: feeding category vectors to a flat layer loses "global behaviour" and "relative behavior of data from overall features". Same claim as the tree hypothesis. With group_effect/hierarchy_effect knobs we can draw the *phase diagram*: at what group-signal strength / cardinality / N does each method (CART, C4.5, C5.0, Cubist, GritBot-style mining, LightGBM, XGB, CatBoost, EBM, TabPFN-class) capture or lose the structure. CrossTab Sparsity becomes the structural metric on generated data; TrieKNN-style methods become contenders.

## Phasing (matches Jitin's stated order)

1. Research comparisons (TabArena brief done; experiment harness next) ✔ started
2. Dataset component (this spec) — **priority**
3. Bring in **featuregen** (temporal features) and complete its in-pipeline transforms (trended/decomposed) — Rust side
4. Replicate in Rust: CART, C4.5, then C5.0, Cubist, GritBot (RuleQuest docs in `references/`); compare against Python `tree/models.py` as oracle and against giants
5. Mixing: stacking (The_Stacker/Blender lineage), CatBoost-style statistical features, EBM-style consolidation, NGBoost-style distributional heads, DAR-style routing

## Open items awaiting Jitin

- More repos/method links are coming before build ("share their links as well ... and then we can build on stuff")
- Generator knob defaults above are proposed, not confirmed
- R binding order (extendr) vs Python-first
