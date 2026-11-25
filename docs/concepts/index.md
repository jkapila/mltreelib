---
title: Concepts
description: "How mltreelib works, in reading order: the tree, its splits and leaves, missing data, and the task-specific trees built on top."
---

# Concepts

Concepts explain how and why the library works. Read them in order the first time: each page uses what the previous one set up. To do a task end to end, use the [Guides](../guides/index.md) instead.

A decision tree is built from four choices: how to **split** a node, how to **grow** the tree, how to **prune** it, and what to fit in each **leaf**. Parts 1 to 3 cover the first three and the safeguards around them. Parts 4 to 6 cover the leaf. The rest are task-specific trees and tools built on the same pieces.

| # | Page | You learn | Why it comes next |
| :-: | :--- | :--- | :--- |
| 1 | [Tree architecture and growth knobs](core-tree-architecture.md) | The split, grow, prune and leaf model, and the two controls that hold the knobs | Everything else refers to these knobs |
| 2 | [Growth knobs in practice](tree-architecture-and-knobs.md) | What depth, minimum rows and pruning do on real data, and what `cp` means | Turns the knob list in 1 into behaviour you can see |
| 3 | [Constraints and honesty](constraints-and-honesty.md) | Monotone and interaction constraints, honest sample splitting | Makes a grown tree trustworthy before you look at leaves |
| 4 | [Native missing-data routing](missing-data-routing.md) | How missing values are routed with surrogate splits and a learned direction | Real data has gaps; the tree must handle them without imputation |
| 5 | [Data ingestion in practice](data-matrix-and-routing.md) | Typed ingestion, missing routing and categorical encoders on example data | Applies 4 to a concrete input pipeline |
| 6 | [Statistical and metric splitting](statistical-and-metric-splitting.md) | Alternative split criteria and gates: MOB, CTree, fairness, oblique and more | Splitting rules are the other half of the tree after the leaf |
| 7 | [Distributional and censored leaves](distributional-leaves.md) | Leaves that model a distribution: Gamma, Poisson, Tweedie, Tobit and others | The leaf is where a single tree gets expressive |
| 8 | [Piecewise linear trees and Cubist rules](piecewise-linear-and-cubist.md) | Linear leaves, M5 and Cubist-style rules | The most common non-constant leaf |
| 9 | [Leaf retraining and micro-models](hierarchical-retrain.md) | Refit leaves on new data without changing the partition | Keeps a deployed tree current |
| 10 | [Survival and competing risks](survival-trees.md) | Trees for time-to-event outcomes with censoring | Uses the leaf and split ideas for a new outcome type |
| 11 | [Time series forecasting](timeseries-forecasting.md) | Calendar, lag and window features with trees | Another task built on the same tree |
| 12 | [Clustering trees](clustering-trees.md) | White-box segments with predictive clustering trees and density trees | Trees without a labelled target |
| 13 | [Anomaly trees](anomaly-trees.md) | Isolation and density trees for outliers | Same, for outliers |
| 14 | [Components and rotations](components-and-rotations.md) | PCA, factor analysis and rotations | Reduces or reshapes features before or beside trees |
| 15 | [Solvers and optimizers](solvers-and-optimizers.md) | The bounded optimizers used by leaves and for tuning | The numerical layer under the leaves |
