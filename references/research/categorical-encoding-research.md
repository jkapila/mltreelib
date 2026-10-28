# Categorical encoding research — CatBoost mechanism and encoder effort

**Date:** 2026-10-28 (companion to `jkapila-methods-map.md`)
**Decision this informs:** where categorical encoding belongs in mltreelib
(data layer vs tree learner) and what the encoder suite must implement.

## Where CatBoost encodes categoricals: data level, computed at split time, per row

Sources: CatBoost docs, "Transforming categorical features to numerical
features" (catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic);
Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features",
NeurIPS 2018; arXiv:1810.11363.

The docs state: *"Before each split is selected in the tree, categorical
features are transformed to numerical."* Pipeline:

1. Shuffle rows (multiple random permutations generated).
2. Labels -> integers (regression labels are quantized into buckets).
3. Per-row CTR / target statistic: `(countInClass + prior) / (totalCount + 1)`,
   counting **only objects earlier in the shuffled order** (prefix statistics).
4. The resulting numerics replace the category **per row**; splits are then
   ordinary numeric splits. The same category gets different values in
   different rows — that row-dependence IS the anti-leakage mechanism
   (the "ordered target statistics" of the NeurIPS paper).
5. Combinations: cat x cat (and cat x numeric) features are built greedily at
   split construction, then CTR'd the same way.
6. `one_hot_max_size`: low-cardinality features get one-hot instead of CTRs.

Answers to the design question:

- **Not root-node-level**: encoding is not a one-off tree-top decision; it is
  consulted whenever a split is chosen.
- **Not leaf-node-level**: leaf values are standard Newton/gradient steps with
  nothing categorical.
- **Data level, but lazily recomputed at split time** with ordered prefix
  statistics — a hybrid unique to CatBoost.

Comparative picture:

| System | Where categoricals enter |
|---|---|
| CatBoost | data level (ordered CTR per row, recomputed before splits) + greedy combinations |
| LightGBM | split level (gradient-statistics category-partition search) |
| XGBoost | split level (native categorical partitions, optional) or external encoding |
| mltreelib (plan) | data-level encoder suite (this module) **and** native categorical splits in the learners |

## Encoder suite — effort and status

| Encoder | Effort | Status |
|---|---|---|
| Ordinal | done | `IndexMapper` |
| One-hot | ~1 session | implemented (`encode.rs`) |
| Hashing | ~1 session | implemented (`encode.rs`) |
| Target (smoothed) | ~1 session | implemented — leaky variant, documented |
| CatBoost-style ordered TS | 1-2 sessions | implemented (`fit_transform` = ordered; `transform` = fitted means) |
| Count, Base-N/binary, Sum contrast | ~1 session | implemented (`encode.rs`) |
| Leave-one-out, M-estimate | ~1 session | implemented (`fit_transform` leakage-aware for LOO) |
| Quantile, James-Stein | ~1 session | implemented (`encode.rs`) |
| Class-level TF-IDF, WoE | ~1 session | implemented (classification family) |
| Embedding (NN per feature) | 1-2 sessions | implemented; trainer extracted to `nn.rs` (reusable), ReLU/linear output by target sign |
| cat x cat combinations | 2+ sessions | deferred to tree building (belongs in split search) |
| GLMM, Helmert/Polynomial/BackwardDifference, Gray, MultiHot/RankHot, CountTarget | varies | deferred (heavy machinery or careful contrast definitions) |

Design rules carried over from the data layer: fit/transform, missing in ->
missing out, unseen levels -> documented fallback (global mean / zeros),
never panic; supervised encoders must offer a leakage-safe mode.

## Notes for the tree work

- Native categorical splits (LightGBM-style partition search) remain a tree
  requirement; the encoder suite does not replace it.
- Ordered target statistics computed *inside* boosting (ordered boosting,
  different permutation per tree) is what makes CatBoost unbiased — a
  candidate differentiator for our learners later.
