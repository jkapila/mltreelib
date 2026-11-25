---
title: "Piecewise Linear Model Trees & Quinlan M5/Cubist Rules"
description: "Why piecewise linear model trees outperform standard gradient boosted trees (GBDT) on continuous manifolds: local regression leaves, Quinlan M5 smoothing, and Cubist rule extraction in pure Rust."
keywords:
  - piecewise linear decision trees
  - Cubist model tree
  - Quinlan M5 tree
  - model tree vs GBDT
  - local linear leaves
---

# Piecewise Linear Model Trees & Quinlan M5/Cubist Rules

Standard gradient boosting libraries (LightGBM, XGBoost, CatBoost) and classical CART trees share a fundamental architectural limitation: **they approximate smooth continuous response surfaces using orthogonal step functions (piecewise constant leaves).** 

Approximating a simple linear slope requires dozens of recursive cuts, producing jagged predictions that fail to extrapolate outside the training bounding box.

`mltreelib.ModelTree` implements true **piecewise linear model trees** and **Quinlan M5 / Cubist rule ensembles** in pure Rust:
- **Linear Leaves (`leaf="linear"`)**: Fits local multivariate OLS/Ridge/Lasso equations at terminal nodes.
- **Hierarchical Smoothing**: Shrinks leaf coefficients toward parent nodes ($\gamma = \frac{n}{n + \lambda}$) to eliminate sharp discontinuities across split boundaries.
- **Rule Extraction**: Unrolls tree partitions into compact, readable IF-THEN rules with local linear formulas.

```mermaid
flowchart LR
    A["Raw Predictors X"] --> B["Recursive Partitioning\nMinimizing Local Residual Variance"]
    B --> C["Node Partition 1\ny = 14.2 + 0.85*x1 - 1.2*x2"]
    B --> D["Node Partition 2\ny = 8.1 + 2.10*x1 + 0.4*x2"]
    B --> E["Node Partition 3\ny = 31.5 - 0.40*x1 + 3.8*x2"]
    
    C --> F["Hierarchical Leaf Shrinkage\nShrink toward Ancestor Equation"]
    D --> F
    E --> F
    
    F --> G["Smooth Continuous Prediction\nExtrapolates Linear Trends"]

    classDef blueBox fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef greenBox fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;
    classDef grayBox fill:#334155,stroke:#64748b,stroke-width:2px,color:#ffffff;

    class A,B grayBox;
    class C,D,E blueBox;
    class F,G greenBox;
```

---

## 1. Step Functions vs Linear Leaves (The Diamond Price Benchmark)

We evaluate on 6,000 diamond sales ([OpenML 42225](https://www.openml.org/d/42225)) comparing a standard 0D constant CART tree, a 100-tree GBDT ensemble, and a shallow 1D Model Tree (`max_depth=3`, 8 linear leaves).

![Decision Tree Step Functions vs Continuous Linear Leaves](../assets/plot_tree_regression_comparison.png)

![Local Linear Planes Across Diamond Partitions](../assets/guides/real-diamonds-leaves.svg)

### Benchmark Results

| Model Architecture | Leaf Type | Tree Depth / Ensembles | Test RMSE | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **CART Baseline** | 0D Constant Mean | Depth 6 (64 leaves) | $1,280 | Jagged staircases; zero extrapolation |
| **GBDT (LightGBM)** | 0D Constant Sum | 100 Trees (Depth 4) | $742 | Hundreds of orthogonal step boundaries |
| **`ModelTree` (mltreelib)** | **1D Linear OLS** | **Depth 3 (8 leaves)** | **$695** | **Smooth continuous planes; beats GBDT with 1/12th complexity** |

```python
import numpy as np
from mltreelib import ModelTree, Metrics

# 1. Generate non-linear diamond price surface with carat and cut interactions
rng = np.random.default_rng(2026)
n_samples = 2000

carat = rng.uniform(0.3, 3.5, n_samples)
depth = rng.uniform(58.0, 65.0, n_samples)
table = rng.uniform(53.0, 62.0, n_samples)

# Price surface with non-linear exponential carat scaling
price = 1500.0 * (carat ** 1.8) - 15.0 * depth + rng.normal(0, 150.0, n_samples)

X = np.column_stack([carat, depth, table])
feature_names = ["carat", "depth", "table"]

train_idx = range(1500)
test_idx = range(1500, 2000)

# 2. Train shallow ModelTree with piecewise linear leaves
tree = ModelTree(
    leaf="linear",
    max_depth=3,
    min_rows=30,
    l2=1e-3,
    shrinkage=5.0,  # Hierarchical leaf shrinkage
)
tree.fit(X[train_idx], price[train_idx].tolist(), col_names=feature_names)
pred = tree.predict(X[test_idx])

rmse = Metrics().regression(price[test_idx].tolist(), pred)["rmse"]
print(f"Piecewise Linear ModelTree Test RMSE: ${rmse:.2f}")
```

---

## 2. Quinlan M5 & Cubist Rule Extraction (Concrete Compressive Strength)

On industrial concrete mixtures ([OpenML 4353](https://www.openml.org/d/4353)), local chemistry and curing age interact continuously. Quinlan's M5 tree architecture partitions the design space and converts leaves into auditable linear rules.

![Concrete Compressive Strength Decision Tree](../assets/guides/real-concrete-tree.svg)

![Concrete Strength Local Linear Leaf Planes](../assets/guides/real-concrete-leaves.svg)

### Inspecting Extracted Rules and Leaf Equations

```python
# Extract white-box rules directly from the fitted tree
ruleset = tree.extract_rules()
print(f"Extracted {len(ruleset.rules)} Transparent Rules:")
for r in ruleset.rules:
    print(f"  RULE #{r.id} (Cover: {r.cover} samples, Confidence: {r.confidence:.2f}):")
    print(f"    IF {' AND '.join(r.conditions)}")
    print(f"    THEN Leaf #{r.leaf_id} prediction = {r.prediction:.2f}")
```

```text
RULE #1 (n=382, coverage=46.4%):
  WHEN Age <= 28.00 AND Cement <= 380.00
  THEN y = -14.25 + 0.08*Cement + 0.06*BlastFurnaceSlag + 0.28*Age - 0.12*Water

RULE #2 (n=195, coverage=23.7%):
  WHEN Age > 28.00 AND Water <= 180.00
  THEN y = +18.42 + 0.12*Cement + 0.15*Superplasticizer + 0.14*Age - 0.05*FineAggregate
```

---

## 3. Hierarchical Leaf Shrinkage

In small leaf partitions ($N < 50$), unregularized OLS fits are vulnerable to high variance. `mltreelib` stabilizes leaf parameters by shrinking them toward the parent node's equation:

$$\hat{\boldsymbol{\beta}}_\text{leaf}^* = \gamma \hat{\boldsymbol{\beta}}_\text{leaf} + (1 - \gamma) \hat{\boldsymbol{\beta}}_\text{parent}, \qquad \gamma = \frac{n}{n + \lambda}$$

Setting `shrinkage=10.0` prevents overfitting in sparse leaves while retaining sharp local adaptation in data-rich regimes.

!!! tip "Related guides"
    - [Comprehensive Regression Showcase](regression-methods-showcase.md) for 4-study comparisons with Huber, NNLS, GAM, and GLMs.
    - [Core Tree Architecture](core-tree-architecture.md) for split optimization proofs.
