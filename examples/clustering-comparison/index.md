# Clustering models comparison — Crosstab Sparsity (CSP)

Comparison of all tree-based clustering paradigms in `mltreelib` on the `credit-g` dataset using Jitin Kapila's **Crosstab Sparsity (CSP)** metric.

Script: [`examples/clustering_comparison.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/clustering_comparison.py>) (`uv run python examples/clustering_comparison.py`).

## Evaluated models

1. **Predictive Clustering Trees (PCT / CLUS)**: Multi-objective variance reduction across numerical targets while partitioning mixed attributes.
2. **HDBSCAN Condensed Tree**: Unsupervised density-based hierarchical clustering with Excess of Mass (EOM) cluster selection.
3. **CLTree**: Density-based hyper-rectangle clustering contrasting true samples against synthetic background noise.
4. **Unsupervised Random Forest (URF)**: Affinity proximity matrix derived from synthetic label discrimination with Lance-Williams hierarchical clustering.

## Crosstab Sparsity (CSP) metric

The Crosstab Sparsity metric measures how well clustering partitions separate attributes:

$$\\text{CSP} = \\sum\_{j} \\left\[ \\frac{n\_v}{\\max(l, k)} \\times \\ln \\left( \\frac{N}{l \\cdot k} \\right) \\right\]$$

- **Segregation factor**: measures mass concentration in contingency table cells above the median cell count.
- **Explanation factor**: measures information density gained over a uniform null spread.

## Code snippet

```
import pandas as pd
from mltreelib import (
    CLTree,
    HDBSCAN,
    PredictiveClusteringTree,
    UnsupervisedRandomForest,
    load_dataset,
)
from examples.clustering_comparison import crosstab_sparsity

# Load credit-g dataset
data = load_dataset("credit-g")
df = pd.DataFrame(data["data"], columns=data["columns"])

# Fit Predictive Clustering Tree
pct = PredictiveClusteringTree(
    max_depth=5,
    min_leaf_samples=15,
    min_variance_reduction=0.01,
)
pct.fit(
    x_matrix,
    target_indices=[0, 1, 3],
    col_names=feature_names,
    dtypes=dtypes,
)
labels_pct = pct.predict_cluster(x_matrix)

# Evaluate Crosstab Sparsity
csp_score, summary = crosstab_sparsity(df, labels_pct)
print(f"PCT Total Crosstab Sparsity: {csp_score:.3f}")
```

## Benchmark results

| Algorithm | Paradigm | Clusters Formed | Noise Points | Crosstab Sparsity (CSP) |
| :--- | :--- | :--- | :--- | :--- |
| **Predictive Clustering Tree (PCT)** | Multi-Objective Variance Reduction | 11 | 0 | **68.505** |
| **HDBSCAN** | Density Stability &amp; Excess of Mass | 2 | 267 | **53.655** |
| **CLTree** | Hyper-Rectangle Noise Discrimination | 7 | 0 | **49.859** |
| **Unsupervised Random Forest (URF)** | Proximity Co-occurrence | 5 | 0 | **48.540** |

## Takeaway

PCT achieves the highest Crosstab Sparsity score because its splitting criterion directly optimizes multi-target variance reduction along numerical dimensions (`duration_months`, `credit_amount`, `age_years`), maximizing separation across attribute combinations.
