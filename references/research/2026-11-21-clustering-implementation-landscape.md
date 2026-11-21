# Clustering Implementation Landscape for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Clustering methods implementable in current `mltreelib` Rust core, with emphasis on mixed data, grouped structure, high dimension, hierarchy, and interpretability.  
**Decision informed:** Which clustering capabilities to add next and which solver ideas support them.  
**Confidence:** High for repository fit and classical methods. Medium for recent optimal-tree clustering because evidence comes from small or solver-dependent studies.

## 1. Executive Summary

Current core already exports four distinct clustering engines:

- `PredictiveClusteringTree` (PCT / CLUS): multi-target variance-reduction clustering.
- `ClTree`: clustering through decision-tree construction.
- `HdbscanTree`: density hierarchy, condensed tree, and GLOSH outlier scores.
- `UnsupervisedProximityForest` (URF): synthetic-null forest proximities and clustering.

This is a strong tree-native base. Missing portfolio pieces are:

1. **General-purpose partition clustering** — especially mixed numeric/categorical data.
2. **Direct explainability** — clusters defined by rules, not labels explained afterward.
3. **Large-scale nonconvex clustering** — clustering quality beyond greedy k-means.
4. **Hierarchical and convex cluster paths** — global order and path-based diagnostics.
5. **Objective-based hierarchy evaluation**.

Recommended sequence:

1. **Explainable k-means / k-medians threshold tree** — strongest missing capability and strongest fit with package thesis.
2. **Gower k-medoids and k-prototypes** — closes mixed-type and prototype gaps.
3. **Convex clustering with tree-guided weights** — high-value hierarchy and global-optimum story.
4. **Evidence accumulation ensemble** — combines existing diverse clusterers.
5. **Spectral clustering on URF proximity** — useful non-Euclidean option after eigensolver integration.
6. **Optimal multi-way clustering trees** — research-grade or optional backend because core formulation needs binary linear optimization.

Do not add another greedy tree clustering engine before closing these portfolio gaps.

---

## 2. Current Package Baseline

| Module | What it does | Portfolio role |
|---|---|---|
| `cluster/pct.rs` | Splits on variance reduction over predictive targets | Rule-defined predictive clustering |
| `cluster/cltree.rs` | Histogram-based tree clustering | Efficient supervised-by-clustering tree |
| `cluster/hdbscan_tree.rs` | Density hierarchy, stability selection, GLOSH | Arbitrary shapes, noise, hierarchy |
| `cluster/urf.rs` | Real-versus-permuted forest and proximity clustering | Mixed-type proximity and ensemble structure |

Existing `references/research/2026-11-01-tree-based-clustering-frontiers.md` remains valuable for URF, PCT, evidence trees, random projection trees, density level sets, and BIRCH. This note extends that survey with implementation gaps, objective-based evaluation, explainable partitioning, mixed-type clustering, and recent convex/solver-based clustering.

Current public Python surface exists in `python/mltreelib/cluster.py`. Any new estimator must preserve the package's `fit` / `transform` / `predict` shape and expose cluster labels, stability, and rule explanations where applicable.

---

## 3. Method Portfolio

### 3.1 Explainable k-means and k-medians

**Explainable K-Means and K-Medians** construct a decision tree whose leaves define clusters. Each internal node selects one threshold on one feature. Objective guarantees connect tree complexity and clustering quality.

Moshkovitz et al. (ICML 2020) show $O(k)$ approximation for k-medians and $O(k^2)$ for k-means, with a lower bound of $\Omega(\log k)$ for any such tree. Makarychev and Shan (ICML 2021) improve the guarantees to near-optimal $\widetilde O(\log k)$ for $\ell_1$ k-medians and $\widetilde O(k)$ for k-means.

**Fit:** excellent.

- Reuses threshold scans, split statistics, and rule-description code.
- Produces direct cluster rules.
- Supports categorical thresholds through existing split representation.
- Needs a center/medoid objective separate from PCT's variance objective.
- Can combine k-means assignment with threshold-tree search without exposing optimization internals.

**Recommendation:** highest-priority clustering addition.

### 3.2 Mixed-type clustering

**Gower distance** compares numeric and categorical features by scale-normalized numeric differences and categorical mismatch indicators. It provides one dissimilarity across mixed frames.

**PAM / k-medoids** minimizes medoid dissimilarities and can use Gower distance. PAM is more robust to outliers than k-means, but exact PAM is expensive. CLARA-style sampling and k-medoids++ initialization can reduce cost.

**k-prototypes** extends Lloyd-style clustering with numeric centroids and categorical modes. Huang's 1998 algorithm is simple, but its weighting of numeric and categorical components needs explicit sensitivity analysis.

**FAMD / PCAmix** provide mixed-data factorization before Euclidean clustering. They add linear algebra and preprocessing complexity, so they rank below direct Gower methods.

**Recommendation:** implement one common dissimilarity layer, then expose Gower k-medoids and k-prototypes as distinct estimators. Avoid multiple private distance implementations.

### 3.3 Hierarchical clustering

Classical agglomerative clustering remains useful for small and medium datasets and for explicit dendrograms. Ward linkage is efficient but Euclidean. Complete and average linkage support arbitrary dissimilarities but require distance matrices.

**Hierarchical Clustering Cost (HAC)** evaluates a hierarchy using a global objective rather than local merge quality. It fits `mltreelib` as a diagnostic for URF, PCT, CLTree, and HDBSCAN-generated hierarchies.

**Package fit:** medium for new estimators, high for evaluation utilities. Use `faer` for dense eigendecomposition when needed.

### 3.4 Spectral clustering

Spectral clustering optimizes

\[
\max_{U^\top U=I}\operatorname{tr}(U^\top L U)
\]

for a normalized graph Laplacian $L$. It captures non-convex cluster shapes and works with custom similarities.

On URF proximity, spectral clustering turns tree co-occurrence into a robust graph clustering method. This composes existing code instead of duplicating a new forest.

**Cost:** dense eigendecomposition is $O(n^3)$ and memory $O(n^2)$. Sparse eigensolvers or landmark approximations are needed for scale.

**Recommendation:** add after common proximity/dissimilarity interface exists. Start with small exact solver and clearly report memory ceiling.

### 3.5 Density and anomaly-aware clustering

HDBSCAN-tree already covers density hierarchy, cluster stability, and GLOSH scores. DBSCAN, OPTICS, and DBCAN add little without stronger incremental or distribution-shift requirements.

**Cluster-based local outliers (CBLOF)** can use existing cluster assignments and cluster-level distributions. It remains useful as a plug-in score over any clusterer.

**INCAD** jointly models clustering and anomaly probabilities for streaming data. It prevents anomalies from distorting cluster formation, but adds a probabilistic streaming model.

**Recommendation:** expose anomaly scores from existing clusterers first. Do not duplicate INCAD until streaming requirements justify its model.

### 3.6 Evidence accumulation and ensemble clustering

Evidence accumulation clusters many base partitions and aggregates pairwise co-association evidence. It generalizes URF's one forest and can combine PCT, CLTree, HDBSCAN, and future partitioners.

**Cost:** $O(Bn^2)$ dense co-association memory. Sparse representation or bounded sample subsets are necessary.

**Package fit:** high conceptually. This is a strong ensemble primitive because current package already has diverse clusterers.

### 3.7 Convex clustering

Convex clustering solves

\[
\min_\Theta \frac12\sum_i \|y_i-\theta_i\|_2^2
+\lambda\sum_{i<j}w_{ij}\|\theta_i-\theta_j\|_q.
\]

It gives a hierarchy and avoids k-means local minima when solved globally.

**Tree-Guided $L_1$-Convex Clustering (TGCC)** uses minimum-spanning-tree weights. It reduces each coordinate to a tree-structured fused-lasso problem and uses dynamic programming. The 2025 paper reports complete cluster paths for $10^6$ two-dimensional points in about 60 seconds on a laptop and avoids cluster splits in its constrained formulation.

**Package fit:** strong but non-trivial. Tree-structured fused lasso is a natural match for `mltreelib` tree theme and avoids general $O(n^2)$ weights. Initial implementation should remain numeric and small-to-medium scale.

**Recommendation:** Tier 1 research implementation after explainable partitioning. It also becomes a concrete solver benchmark for ADMM, dynamic programming, and proximal methods.

### 3.8 Random-projection and isolation-based clustering

Random projection trees adapt partitions to intrinsic dimension. Isolation-kernel clustering uses data-dependent isolation trees and a kernel that grows clusters from seeds.

Both are pure-Rust compatible, but both overlap existing EIF/RCT and tree-proximity mechanisms. Their payoff depends on benchmark evidence in current package datasets, not novelty.

**Recommendation:** implement only with demonstrated accuracy or scaling gain.

### 3.9 Optimal interpretable clustering trees

Recent **Interpretable Clustering via Optimal Multi-way Trees (ICOMT)** enumerates paths on a feature graph and selects clusters through binary linear optimization. One-dimensional k-means binning preserves skew and multimodality better than quantile binning. The paper reports higher ARI than ICOT, PCT, and two-step clustering on four datasets, but runtime reaches minutes on one dataset because Gurobi solves the path-selection model.

**Package fit:** medium-low for default core. Candidate path enumeration fits Rust. Exact path selection needs a binary linear solver.

**Recommendation:** split research prototype from product API:

- Rust: candidate path enumeration, path costs, coverage constraints.
- Optional backend: binary linear solver.
- Bounded fallback: greedy path selection for large instances, clearly labeled non-optimal.

---

## 4. Implementation-Fit Matrix

| Method | New dependencies | Runtime shape | Memory | Mixed types | Interpretability | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| Explainable k-means tree | None | $O(dn\log n)$ typical | $O(n)$ | Yes | Very high | Build first |
| Gower k-medoids | None | $O(n^2)$ distance or sampled | $O(n^2)$ or $O(n)$ | Yes | Medium | Build second |
| k-prototypes | None | $O(nk\,iter)$ | $O(nk)$ | Yes | Medium | Build with Gower layer |
| Convex clustering / TGCC | None | $O(n)$ to $O(n\log n)$ per path under tree weights | $O(n)$ | Numeric first | High | High-value research feature |
| Evidence accumulation | None | $O(Bn^2)$ naive | $O(Bn^2)$ dense | Via base clusterers | Low-medium | Use sparse co-association |
| Spectral on URF graph | None | $O(n^3)$ exact | $O(n^2)$ | Via proximity | Low | Small exact first |
| Hierarchical cost | None | $O(n^2)$ naive | $O(n^2)$ | Via metric | Diagnostic | Add evaluation utility |
| ICOMT | Optional MILP/BLO solver | Path enumeration plus solve | Candidate-path heavy | Yes | Very high | Optional research path |
| RP-tree / isolation kernel | None | Moderate | Moderate | Extension-dependent | Low-medium | Benchmark before build |

---

## 5. Recommended Architecture

### 5.1 Shared abstractions

Introduce or document these seams before adding estimators:

- `ClusteringResult`: labels, medoids/centers, stability, scores, optional rules.
- `Dissimilarity`: numeric, categorical, and mixed-type distance policies.
- `ClusterLinkage`: for hierarchical and agglomerative estimators.
- `ClusterSelection`: fixed $k$, stability, threshold, or path cut.
- `ClusterExplanation`: leaf rules or feature summaries.

This prevents each estimator from inventing incompatible label, score, and explanation formats.

### 5.2 Phase sequence

**Phase 1 — Explainable partitioning**

- Implement k-medians threshold tree.
- Add k-means objective variant.
- Support numeric and categorical thresholds.
- Compare against k-means, k-medoids, PCT, and CLTree.
- Measure rule size, ARI/NMI, silhouette, and stability.

**Phase 2 — Mixed-type portfolio**

- Add Gower distance in one shared module.
- Add Gower k-medoids with deterministic k-medoids++ seeding.
- Add k-prototypes using same mixed-feature normalization policy.
- Preserve missing values through explicit missing-aware comparisons.

**Phase 3 — Hierarchy and ensembles**

- Add evidence accumulation over existing clusterers.
- Add convex clustering / TGCC numeric prototype.
- Add hierarchical cost evaluation.
- Add spectral URF clustering only if memory-safe small mode is useful.

**Phase 4 — Exact research paths**

- Implement ICOMT candidate enumeration in Rust.
- Keep binary solver optional.
- Compare greedy, bounded exact, and external-solver results.
- Publish optimality gap when solver returns one.

---

## 6. Evaluation Protocol

Clustering has no universal metric. Every experiment should report:

- External metrics: ARI, NMI, homogeneity/completeness, V-measure.
- Internal metrics: silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn.
- Stability: bootstrap or subsample assignment agreement.
- Explanation: depth, number of rules, and feature usage.
- Scalability: fit time, score time, peak memory, and rows/features per second.
- Sensitivity: seed, $k$, contamination, feature scale, missingness, and category count.
- Comparison fairness: same preprocessing and same metric direction.

For generated data, use known latent groups and known structured anomalies. For public data, separate descriptive validity from predictive usefulness.

---

## 7. Risks and Required Fixes

1. **Missing-value contract.** Audit report says Python `ClusterTree` converted `None` to `0.0` for CLTree, HDBSCAN, and URF. This must be fixed before adding mixed-type methods. Zero is not missing.
2. **Method naming.** Use `k-medoids` and `k-prototypes`, not “Gower clustering.” State exact dissimilarity and objective.
3. **No silent approximations.** Exact and sampled/greedy variants need explicit configuration and result metadata.
4. **Quadratic memory.** PAM, spectral, and evidence accumulation need sample or sparse limits.
5. **Interpretability inflation.** A rule tree after clustering is not direct interpretability unless cluster formation used the rules.
6. **Optimality claims.** Greedy or bounded solvers cannot claim global optimum.
7. **Categorical order.** Arbitrary category codes cannot enter Euclidean centers without explicit treatment.

---

## 8. Rejected or Deferred Options

| Option | Decision | Reason |
|---|---|---|
| Another greedy tree clusterer | Reject | Four tree-native clusterers already exist |
| DBSCAN/OPTICS rewrite | Reject | HDBSCAN-tree covers density hierarchy |
| DET level-set clustering | Defer | Small wrapper over DET, weaker portfolio value |
| BIRCH CF-tree | Defer | Streaming case exists, but interpretability and mixed-type support weaker |
| Deep embedding clustering | Reject for core | Neural dependency and opaque clusters |
| Full dense spectral clustering at scale | Reject | $O(n^2)$ memory blocks package users |
| ICOMT as only production path | Reject | Requires external exact solver and has narrow evidence base |

---

## 9. Primary Sources and Indexes

1. Moshkovitz, M., Dasgupta, S., Rashtchian, C., and Frost, H. (2020). “Explainable K-Means and K-Medians Clustering.” *ICML*. [arXiv:2002.12538](https://arxiv.org/abs/2002.12538).
2. Makarychev, K. and Shan, Y. (2021). “Approximation Algorithms for Explainable Clustering.” *ICML*. [arXiv:2107.00798](https://arxiv.org/abs/2107.00798).
3. Gower, J. C. (1971). “A General Coefficient of Similarity and Some of Its Properties.” *Biometrics*. [DOI](https://doi.org/10.2307/2528823).
4. Huang, T. (1998). “Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values.” *Data Mining and Knowledge Discovery*. [DOI](https://doi.org/10.1023/A:1009769707641).
5. Kaufman, L. and Rousseeuw, P. J. (1990). “Finding Groups in Data: An Introduction to Cluster Analysis.” Wiley.
6. Ng, A., Jordan, M., and Weiss, Y. (2001). “On Spectral Clustering: Analysis and an Algorithm.” *NeurIPS*. [Paper](https://proceedings.neurips.cc/paper/2001/hash/9e3f2ba2f96d02f9193fa5f9ce33884b-Abstract.html).
7. Fred, T. L. and Jain, D. (2005). “Evidence Accumulation Clustering.” *IEEE TPAMI*. [DOI](https://doi.org/10.1109/TPAMI.2005.159).
8. Zhang, B. and Terada, Y. (2025). “Tree-Guided $L_1$-Convex Clustering.” [arXiv:2503.24012](https://arxiv.org/abs/2503.24012).
9. Suzuki, H., Ikeda, S., Nishimura, N., and Takano, Y. (2026). “Interpretable Clustering via Optimal Multi-way Decision Trees.” [arXiv:2602.13586](https://arxiv.org/abs/2602.13586).
10. Bertsimas, D., Orfanoudaki, A., and Wiberg, H. (2021). “Interpretable Clustering: An Optimization Approach.” *Machine Learning* 110. [DOI](https://doi.org/10.1007/s10994-020-05908-3).
11. Murtagh, F. (1983). “A Survey of Recently Published Algorithms for Hierarchical Clustering.” *ACM Computing Surveys*. [DOI](https://doi.org/10.1145/358950.358955).
12. Dasgupta, S. (2016). “The Cost of Hierarchical Clustering.” *STOC*. [arXiv:1510.05043](https://arxiv.org/abs/1510.05043).
13. Lampainen, J., Joki, K., Karmitsa, N., and Mäkelä, M. M. (2025). “Clust-Splitter: An Efficient Nonsmooth Optimization-Based Algorithm for Clustering Large Datasets.” [arXiv:2505.04389](https://arxiv.org/abs/2505.04389).
14. Sampaio, R. A., Garcia, J. D., Poggi, M., and Vidal, T. (2023). “Regularization and Global Optimization in Model-Based Clustering.” [Papers With Code](https://paperswithcode.com/paper/regularization-and-global-optimization-in).
15. “Large-Scale Spectral Clustering.” [Papers With Code method page](https://paperswithcode.com/method/large-scale-spectral-clustering).
16. “Clustering by Transitive Propagation.” [Papers With Code](https://paperswithcode.com/paper/clustering-by-transitive-propagation).

## 10. Source Notes

- alphaXiv supplied recent mixed-type surveys, convex clustering, nonsmooth large-scale clustering, and optimal multi-way tree evidence.
- Papers With Code supplied historical method and implementation metadata for regularized model-based clustering and spectral clustering. Primary papers remain source of truth.
- treg catalog exposed Perplexity, Brave, and Scholar endpoints. Live calls failed because saved treg authentication expired, so no failed response supports any claim.
