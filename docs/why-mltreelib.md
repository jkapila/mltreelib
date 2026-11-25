---
title: Why mltreelib
description: What mltreelib does differently from scikit-learn trees, gradient boosting libraries and R tree packages, and where to check the evidence.
---

# Why mltreelib

mltreelib is a single-tree-first library: one readable tree, fitted fast, that you can audit. It is not trying to beat gradient boosting at raw leaderboard accuracy. It targets the cases where you must explain, constrain or trust the model.

## What is different

| You need | mltreelib | Where to read more |
| :--- | :--- | :--- |
| Missing data without imputation | `None` and `NaN` are normalized at ingestion. Missing in gives missing out, with surrogate-split routing. It never panics. | [Missing-data routing](concepts/missing-data-routing.md) |
| Linear or spline leaves, not constants | `ModelTree(leaf="linear")` fits a regularized linear model per leaf. `"mars"`, `"huber"`, `"quantile"` and `"knn"` leaves are also available. | [Quickstart](quickstart.md), [Piecewise linear trees](concepts/piecewise-linear-and-cubist.md) |
| Rules you can read | Extract C5.0-style rules, print rpart-style structure, per-leaf `summary(lm)` tables. | [Inspection, rules and stumps](guides/model-inspection-and-rules.md) |
| Distributions, not just means | Gamma, Poisson, Tweedie, Tobit and heteroscedastic leaves, plus survival and competing-risk leaves. | [Distributional leaves](concepts/distributional-leaves.md), [Survival](guides/survival.md) |
| Constraints and honesty | Monotone splits and slopes, interaction isolation, and honest (split/estimate) sampling. | [Constraints and honesty](concepts/constraints-and-honesty.md) |
| One API across tasks | `ModelTree`, `ClusterModel`, `AnomalyModel`, `ComponentModel` and `RuleModel` share a `fit` / `predict` shape. | [Guides](guides/index.md) |
| A small runtime | A Rust core with a thin Python layer and no scikit-learn dependency. | [Installation](installation.md) |

## How it compares

- **scikit-learn trees**: CART with constant leaves. mltreelib adds model-tree leaves, distributional criteria, survival and causal trees, and rule extraction. The `fit`/`predict` shape is familiar, but it is not a drop-in subclass.
- **XGBoost, LightGBM, CatBoost**: boosted ensembles optimized for predictive accuracy. Use them when accuracy is the only goal. Use mltreelib when you need one auditable tree, constraints, or a distribution at the leaf.
- **R `rpart`, Cubist, C5.0, GritBot**: the Quinlan and CART lineage. mltreelib reimplements these ideas in Rust and checks `rpart` behavior against R.

## Where the evidence is

Claims on this page are qualitative. For measured results, see the [Elo arena](benchmarks/elo.md), which compares mltreelib against peer libraries on 56 datasets, and the [regression methods study](benchmarks/regression-methods.md). Both show how to reproduce the numbers.
