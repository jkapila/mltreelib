---
title: Why mltreelib
description: Why mltreelib exists, what it is for, and where it is not the right tool.
---

# Why mltreelib

mltreelib is an end-to-end library for building models you can explain. It is built around one idea: a single decision tree, built as far as it will go, is the model you can defend to the people who have to act on it.

## The problem

Most data scientists today meet modelling through a handful of library conventions, and those conventions leave gaps.

- **A class per task, and metrics somewhere else.** Regression, classification, survival, uplift and boosting each live in a different package with a different API. Metrics are separate again. Moving between tasks means learning a new surface each time.
- **Summaries went missing.** A statistical summary of a fitted model (coefficients, standard errors, tests) is the first tool for troubleshooting data. It is common in R and statsmodels and rare in the code newer practitioners learn first.
- **Deployment tools answer the engineer, not the business.** Pipelines and serving make the engineer's job easier. When the business asks why a model stopped working, or why it drifted, there is often no answer to give.
- **One global model rarely fits.** A single linear regression over the whole dataset usually fails. The data has regions that behave differently, and that is what a tree finds.

## What mltreelib does about it

- **One surface across tasks.** `ModelTree`, `ClusterModel`, `AnomalyModel`, `ComponentModel` and `RuleModel` share the same `fit` and `predict` shape, so moving from regression to survival to uplift changes the options, not the library. Evaluation, calibration and diagnostics live in the same package.
- **A single tree, built brutally far.** Split, grow, prune and fit a leaf are four separate choices, each a knob. The leaf is where the tree becomes expressive: linear, quantile, distributional, survival. That is divide and conquer: instead of one regression that does not work, many local models, each of which can be inspected and justified to a business reader.
- **Summaries you can read.** Each leaf can report coefficients, standard errors and tests in the style of `summary(lm)`, and a tree can be printed, drawn, and turned into rules.
- **Missing data stays in the data.** Missing values are never imputed by the library. The tree routes them with surrogate splits and a learned direction, so the pattern of missingness stays visible.
- **A small footprint.** A Rust core with a thin Python layer, no scikit-learn dependency, and a deliberate effort to keep dependencies down. If something can be built inside the library, it is built inside.

## What it is not for

- **Maximum accuracy on large tabular benchmarks.** Boosted ensembles such as LightGBM, CatBoost and XGBoost are generally more accurate than a single tree, and the arena shows that. mltreelib is built for the cases where you must explain, constrain or defend the model. Ensembles (forest, boosting, meta) are planned and come after the single tree is settled.
- **Everything already covered by interpretable boosting or GAM libraries.** Interpretable boosting (for example EBM) is a strong alternative when you want additive explanations. mltreelib is the better fit when you want rules and per-leaf models in one tree.
- **A finished product.** The package is at an early stage: APIs may change or break between releases, and drift monitoring, data tools and a metrics area are planned, not shipped. See the [roadmap](https://github.com/jkapila/mltreelib/blob/main/specs/README.md) and raise an issue before proposing a new feature.

## How it compares

- **scikit-learn trees** are CART with constant leaves. mltreelib adds model-tree leaves, distributional criteria, survival and causal trees, native missing data and rule extraction. The `fit` / `predict` shape is familiar, but it is not a drop-in subclass and there is no scikit-learn dependency.
- **XGBoost, LightGBM, CatBoost** are boosted ensembles optimized for predictive accuracy. Use them when accuracy is the only goal.
- **R `rpart`, Cubist, C5.0, GritBot** are the Quinlan and CART lineage. mltreelib reimplements these ideas in Rust, and checks `rpart` behaviour against R.

## Where to look next

- [Concepts](concepts/index.md): how the tree, its splits and its leaves work, in reading order.
- [Guides](guides/index.md): task-by-task workflows on real data.
- [Benchmarks](benchmarks/elo.md): measured results, with reproduce commands.
