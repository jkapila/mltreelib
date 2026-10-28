# Jitin's methods map — thought process behind the tree research program

**Date:** 2026-10-28 (companion to `tree-vs-foundation-tabular.md`)
**Purpose:** Ground the Rust tree-research plan in the methods already built, so we extend rather than reinvent.

## Sources studied

- `references/featuregen/` — feature-generation package (Python): `GroupedVariableTransformation(key, target)`, aggregates, transformation, **time_features.py (22KB, largest module)**, metrics, IndexMapper. Group/hierarchy-aware feature engineering. Docs mention "digital twin" components.
- `references/pygemodels/` — Growth & Epidemiology model fitting: Logistic, Richard, Gompertz, Bass CDF, Weibull; SIR/SIS/SIRS/SEIR/SEQIAHR (pipeline). `fit/predict/summary/plot` interfaces. **Temporal dynamics package.**
- `references/paper-codebase/` — long-running research notebooks & code: generic `Tree`/`TreeNode` builder ("A generic tree builder" — split/agg/predict function injection), Novelty Trees, K-NN Tree, Rounding k-NN, Evidence Tree Clustering, The_Stacker_V1, Blender, stochastic_ml(_blend), lookalike, oneclass, DriverFatigueness (lab project with data/), VAE, repeat-purchase.
- `references/ngboost/` — Stanford NGBoost: natural gradient boosting, probabilistic/distributional prediction, sklearn-based, Apache-2.0. Field-proven.
- Essays: **DAR / Adaptive Regression** (arXiv:1805.01618 — MAPE bathtub curve by target deciles; split target distribution; per-bucket regressors + KNN routing at scoring), **CrossTab Sparsity** (arXiv:1810.03419 — contingency-table sparsity/segregation metric over feature-bins × clusters; categorical group structure quantified), **TrieKNN** (trie over categorical combinations + leaf-local KNN on numerics; weighted traversal).

## The through-line (my reading — correct me where wrong)

1. **Categorical/group/hierarchical structure is signal that mainstream tools flatten.** TrieKNN encodes it structurally (trie), CrossTab Sparsity measures it, `featuregen.GroupedVariableTransformation` engineers it. LightGBM/XGBoost ("LEDGPM" read as LightGBM) reduce categoricals to splits/target-stats and lose group/hierarchy behavior.
2. **Distribution-level failures hide behind single metrics.** DAR fixes per-target-region underfitting via bucketing + routing — analogous to Cubist's kNN correction and to C5.0's committees.
3. **Trees have been under-studied relative to foundation models.** People wrap C5.0/Cubist as-is; the algorithms themselves (rule committees, winnowing, model trees + neighbors, GritBot surprising-subgroup mining) deserve modern reimplementation + research. GritBot ≈ systematic Crosstab-Sparsity-style surprising-pattern discovery.
4. **Method:** replicate the classics in Rust first (CART, C5.0, Cubist, GritBot) → compare honestly against them and the giants (LightGBM/XGB/CatBoost/EBM/TFMs) → then mix/extend (stacking à la The_Stacker/Blender, NGBoost-style distributional heads, DAR-style routing, TrieKNN-style categorical structure). Proof-driven, not blind experimentation.

## Mapping to the build plan (as stated)

| Phase | Content | Existing assets |
|---|---|---|
| 1. Research | Tree vs FM in categorical/group/hierarchical settings | `tree-vs-foundation-tabular.md`, TabArena CSVs |
| 2. Dataset component (the heart) | Rust, "more adjustable", Python/R exposed, any data in, experiments + viz | designed pandas dataset from ML lab (**location unconfirmed**) |
| 3. Temporal package | bring in the temporal package shared below the comment | pygemodels? featuregen time_features? (**unconfirmed**) |
| 4. Classic replications | Cubist, C5.0, GritBot, simple CART in Rust → compare | RuleQuest docs in `references/`; generic Tree builder notebook |
| 5. Mixing & features | ensembles, statistical features (CatBoost-style), EBM-style consolidation, NGBoost heads | The_Stacker, Blender, DAR, ngboost |

## EBM note

Read "the method which consolidates the zero parity" as **EBM = Explainable Boosting Machine** (GA2M: additive per-feature shape functions + selected pairwise interactions; on the TabArena board at 1191–1254 Elo). CatBoost's contribution read as: injecting statistically-derived variables (ordered target statistics) into tree building. Correct me if "zero parity" meant something else.

## Not found / unresolved

- **The ML-lab pandas dataset** designed to carry categorical/group/hierarchical signal — not located in the three repos (closest: `fatigue/DriverFatigueness/data`, notebook-embedded frames). Location?
- **Which package is "the temporal things"** to bring in: pygemodels (growth/epi curves) or featuregen's time_features, or another?
- Rust workspace naming/layout preferences (new monorepo vs extend mltreelib).
