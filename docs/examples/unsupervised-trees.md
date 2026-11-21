# Unsupervised Trees: Predictive Clustering & Anomaly Triage

This guide details `mltreelib`'s unsupervised tree architectures—unifying **Predictive Clustering Trees (PCT)** for interpretable multi-target segmentation with rule extraction and **Extended Isolation Forests (EIF)** for artifact-free operational anomaly detection.

Script: [`examples/suites/unsupervised_trees.py`](https://github.com/jkapila/mltreelib/blob/main/examples/suites/unsupervised_trees.py) (`uv run python examples/suites/unsupervised_trees.py`).

---

## 1. Domain Context & Architectural Rationale

Unsupervised learning and operational triage traditionally rely on distance-based heuristics (such as $k$-means, DBSCAN, or standard Isolation Forests). In regulated and industrial deployments, these methods exhibit two major failure modes:

1. **Uninterpretable Cluster Boundaries**: $k$-means and GMMs produce soft Voronoi centroids that cannot be translated into auditable decision rules. A domain expert cannot inspect *why* an applicant or chemical mixture was assigned to a given cluster without evaluating global distance metrics.
2. **Axis-Aligned Ghost Anomalies**: Standard Isolation Forests isolate outliers by sampling cuts parallel to feature coordinate axes. In multi-modal or correlated distributions, axis-aligned cuts create artificial rectangular bands of low average path length, producing false-positive "ghost" anomalies in empty regions between inlier clusters.

`mltreelib` unifies unsupervised learning through specialized tree engines:
- **Predictive Clustering Trees (`ClusterTree(kind="pct")`)**: Partitions feature space by maximizing multi-target variance reduction across all covariates simultaneously, yielding transparent, human-readable rule bounding boxes.
- **Extended Isolation Forest (`AnomalyTree(kind="eif")`)**: Cuts feature space along random **oblique hyperplanes** with uniformly sampled normal vectors, producing smooth, rotationally invariant anomaly score contours that eliminate ghost artifacts.

```mermaid
flowchart TD
    subgraph Clustering["ClusterTree (kind='pct' | 'cltree')"]
        direction TB
        C1["Unlabeled Data Matrix"] --> C2{"Divisive Variance Reduction\nacross Multi-Targets"}
        C2 --> C3["Terminal Cluster Partitions\n(n_clusters discovered)"]
        C3 --> C4["Interpretable Rule Extraction\n(e.g., alcohol > 10.5 & volatile_acidity <= 0.4)"]
    end

    subgraph Anomaly["AnomalyTree (kind='eif' | 'isolation')"]
        direction TB
        A1["Telemetry Observations"] --> A2{"Oblique Hyperplane Sampling\nw · (x - p) <= 0"}
        A2 --> A3["Path Length Aggregation h(x)"]
        A3 --> A4["Continuous Anomaly Score s(x, n)\n(Rotationally Invariant Contours)"]
    end

    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef splitBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef nodeBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;
    classDef ruleBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;
    classDef anomBox fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;

    class C1,A1 dataBox;
    class C2,A2 splitBox;
    class C3,A3 nodeBox;
    class C4 ruleBox;
    class A4 anomBox;
```

---

## 2. Benchmark Study: Wine Quality & Multimodal Anomaly Mixture

We evaluate clustering on the **Wine Quality** benchmark (chemical attributes: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, alcohol) and anomaly scoring on a 2D multimodal mixture with peripheral outliers.

```python
from mltreelib import load_dataset
from mltreelib.anomaly import AnomalyTree
from mltreelib.cluster import ClusterTree

# 1. Predictive Clustering Trees (PCT) on Wine Quality
data = load_dataset("wine_quality")
features = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "alcohol"]
X_wine = [[float(data["columns"][c][i]) for c in features] for i in range(1500)]

pct = ClusterTree(kind="pct", max_depth=3, min_rows=30)
pct.fit(X_wine, col_names=features)

print(f"Discovered clusters: {pct.n_clusters_}")
for idx, rule in enumerate(pct.rules()[:2]):
    print(f"Cluster {idx} rule: {rule['rule']}")

# 2. Extended Isolation Forest (EIF) with oblique hyperplanes
eif = AnomalyTree(kind="eif", max_depth=8, extension_level=1, seed=42)
eif.fit(X_2d, col_names=["x1", "x2"])
scores = eif.predict(X_query, kind="score")
```

---

## 3. Empirical Results: Cluster Partitions & Anomaly Contours

```text
=== 1. Predictive Clustering Trees (PCT) on Wine Quality ===
Loaded Wine Quality dataset: 1500 samples, features: ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'alcohol']
Discovered clusters: 8

=== 2. Extended Isolation Forest vs Axis-Aligned Isolation Forest ===
Generated 2D mixture: 600 inliers (2 clusters), 40 peripheral anomalies.
Outlier sample anomaly scores:
  Axis-Aligned IF : [0.590, 0.531, 0.479, 0.590, 0.590]
  Extended IF     : [0.590, 0.709, 0.597, 0.751, 0.590]
```

![Unsupervised Clustering and Anomaly Density Landscapes](../assets/unsupervised_clusters_anomalies.png)

### Key Quantitative Findings

1. **Auditable Cluster Profiles (Panels 1 & 2)**: `ClusterTree(kind="pct")` automatically isolates distinct chemical cohorts (such as high-alcohol/low-acidity dessert wines vs dry table wines) with exact bounding rules.
2. **Elimination of Ghost Anomalies (Panels 3 & 4)**:
   - In Panel 3, standard axis-aligned Isolation Forest produces artificial cross-shaped grid ridges between the two normal clusters where empty space is falsely flagged as anomalous.
   - In Panel 4, Extended Isolation Forest's oblique cuts produce smooth circular density contours centered strictly on genuine sample density.
3. **Outlier Discrimination**: EIF elevates anomaly scores for true peripheral outliers (up to **0.751**) while keeping inliers below the 0.50 contamination boundary.
