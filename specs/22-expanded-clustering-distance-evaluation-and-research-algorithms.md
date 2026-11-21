# Spec 22: Expanded Clustering, Shared Distances, Clustering Evaluation, and Research Algorithms

Status: proposed

## Problem Statement

`mltreelib` already contains four distinct clustering engines:

- `PredictiveClusteringTree` for predictive, rule-defined clustering.
- `ClTree` for density-based tree clustering.
- `HdbscanTree` for hierarchical density clustering, stability selection, and GLOSH outlier scores.
- `UnsupervisedRandomForest` for proximity-based clustering.

The existing base is credible, but the package currently lacks several capabilities required for credible clustering research:

1. **No shared distance contract.** Distance logic is duplicated between the KNN leaf module, HDBSCAN, and general pairwise metrics. Supported sets differ. HDBSCAN accepts only Euclidean, Manhattan, and Chebyshev. KNN exposes ten metrics, while the general pairwise helper exposes a different eight. Gower in the KNN path is numeric-only despite the package's mixed-type thesis.
2. **Distances lack validation.** Existing tests mostly assert that distance-enabled models run and return finite values. They do not establish metric identities, symmetry, zero diagonals, scale behavior, known reference values, invariances, degenerate cases, or mixed-type semantics. Several current implementations apply missing-value rescaling or return zero when no coordinates overlap. Such behavior can silently turn incomparable rows into identical rows.
3. **Clustering evaluation is fragmented and mostly external.** The unified benchmark script implements Adjusted Rand Index in Python and reports only ARI and fit time. Internal validity, stability, noise behavior, objective quality, cluster balance, prediction behavior, and rule quality are missing. There is no shared clustering report comparable to the package's regression and classification reports.
4. **HDBSCAN core-distance computation is quadratic.** Every point scans every other point, then Prim's algorithm performs another quadratic distance scan. Exact DBSCAN and OPTICS would inherit the same scaling limit.
5. **Tree-based similarity and metric neighborhoods are not separated.** Existing tree similarity through URF leaf co-occurrence is useful, but it is not automatically interchangeable with a metric neighborhood required by DBSCAN or OPTICS. The package needs one neighbor-search seam that keeps exact metric neighborhoods, accelerated metric neighborhoods, and approximate tree-similarity graphs explicit.
6. **Streaming clustering is incomplete.** BIRCH is not implemented. Existing engines fit a batch and do not expose a general incremental update contract.
7. **Mixed-type clustering is incomplete.** Numeric-only conversion remains common in the Python facade. A native Gower path, categorical-aware dissimilarities, prototypes, and mixed-type centroids are absent from clustering.
8. **Convex clustering is absent.** Convex clustering provides a global hierarchy without k-means local minima. Tree-guided $L_1$ convex clustering aligns strongly with the package's tree thesis.
9. **Recent embedding-clustering research is absent.** e-means uses exponential moving-average centroid updates, a cosine-annealed step, automatic batch and step laws, and dead-cluster revival. o-means adds two-level centroid-cell assignment. Both target very large embedding workloads. They are useful research references for efficient assignment and mini-batch updates, but current evidence includes counterexamples to mass-based revival and separate distance-aware revival work.
10. **The historical K-NN Tree idea is unfinished.** A notebook under the historical paper codebase describes an online distance-confidence tree with centroid nodes, reduced reference storage, and distance-based neighborhood expansion. It is prototype code, not a validated algorithm. Its relationship to current TrieKNN and URF proximity is unclear without experiments.

The package needs a coherent expanded clustering scope with one distance registry, one neighbor-search seam, one evaluation report, and clearly separated production algorithms and research prototypes.

## Solution

Expand clustering along four coordinated tracks:

1. **Shared distance and dissimilarity registry.** Consolidate all clustering distances behind one typed, feature-aware contract. Validate each distance against mathematical properties, known values, invariances, mixed-type semantics, and reference implementations.
2. **Scalable density clustering.** Add a reusable neighbor-search seam, accelerate HDBSCAN core distances, add pure DBSCAN and OPTICS, and implement BIRCH CF-tree clustering for incremental and streaming workflows.
3. **Mixed-type, convex, and explainable clustering.** Add native Gower dissimilarity, Gower k-medoids and k-prototypes, explainable k-means/k-medians trees, and tree-guided convex clustering.
4. **Research prototypes and evaluation.** Add e-means and o-means-inspired tree-accelerated prototypes, revive the historical K-NN Tree idea as a measured prototype, and establish one clustering evaluation and benchmark program.

Production algorithms must use validated exact semantics. Research prototypes must be labeled as such and must not weaken correctness contracts of existing engines.

### Required architecture

The clustering module gains four deep shared abstractions:

- **Distance registry:** one canonical distance/dissimilarity enumeration, one typed mixed-data distance service, fit-time scale learning, and explicit support declarations.
- **Neighbor search:** one seam for exact scan, spatial-index exact search, and approximate tree-similarity graph search. Search results report mode, exactness, and evaluated-neighbor count.
- **Clustering result:** labels, noise assignment, centers or prototypes, stability, objective value, per-point scores or distances, rules, and algorithm diagnostics.
- **Clustering report:** external validity, internal validity, stability, balance, noise rate, runtime, memory, and algorithm-specific diagnostics in one result.

The existing `ClusterTree` facade becomes the stable public entry point for both existing and new engines. The facade must not silently drop categorical columns or coerce missing values to zero.

## User Stories

### Shared distance and dissimilarity registry

1. As a clustering user, I want one canonical distance name accepted by every clustering engine, so that `metric="manhattan"` means the same thing everywhere.
2. As a clustering user, I want unsupported engine and distance combinations to fail before fitting, so that data is never silently coerced or ignored.
3. As a clustering user, I want Gower distance to work across numeric and categorical columns, so that mixed-type frames use their declared types rather than encoded integers.
4. As a clustering user, I want Gower numeric ranges and categorical mismatch weights learned from training data only, so that test data cannot influence scaling.
5. As a clustering user, I want missing values excluded pairwise from the Gower average, so that one missing feature does not dominate a comparison.
6. As a clustering user, I want a configurable minimum overlap for pairwise distance, so that rows with insufficient shared observations can be reported as incomparable instead of assigned distance zero.
7. As a clustering user, I want distance results to state whether they are metrics, semi-metrics, or general dissimilarities, so that I know which neighborhood and triangle properties are guaranteed.
8. As a clustering user, I want every distance to be symmetric, so that neighbor relationships do not depend on argument order.
9. As a clustering user, I want every valid distance to have zero diagonal for identical observations, so that self-distance never creates artificial neighbors.
10. As a clustering user, I want every distance to be non-negative, so that radius and threshold semantics remain valid.
11. As a clustering user, I want genuine metrics to satisfy the triangle inequality on reference cases, so that radius neighborhoods behave predictably.
12. As a clustering user, I want semi-metrics and dissimilarities to be labeled when they do not satisfy the triangle inequality, so that algorithms do not claim stronger guarantees than available.
13. As a user, I want Euclidean distance to match closed-form and SciPy reference values, so that my baseline is trustworthy.
14. As a user, I want Manhattan, Chebyshev, and Minkowski distances to match reference values, so that norm choices behave correctly.
15. As a user, I want cosine and correlation distances to match reference values, so that directional and standardized comparisons are validated.
16. As a user, I want Canberra and Bray-Curtis distances to match reference values, so that bounded and ecological-style dissimilarities are validated.
17. As a user, I want diagonal Mahalanobis distance to match reference values, so that variance weighting is correct.
18. As a user, I want Gower distance to match reference values on mixed numeric and categorical data, so that mixed-type clustering has a sound basis.
19. As a user, I want Hamming and Jaccard distances to match reference values on categorical or binary data, so that categorical clustering can use them explicitly.
20. As a user, I want scale invariance documented for every distance, so that I can preprocess data consistently.
21. As a user, I want distance computation to avoid unnecessary allocation, so that large benchmark runs remain practical.
22. As a library maintainer, I want one distance implementation per mathematical definition, so that HDBSCAN, KNN, URF, and future engines cannot drift.
23. As a library maintainer, I want fit-time feature scales stored with fitted clustering state, so that prediction uses training-time semantics.
24. As a library maintainer, I want distance support declarations checked in Rust, so that Python cannot request an invalid combination.

### HDBSCAN acceleration and neighbor search

25. As a clustering user, I want HDBSCAN core distances computed with an indexed neighbor search when possible, so that medium-sized datasets do not require quadratic scans.
26. As a clustering user, I want HDBSCAN to produce the same clusters under exact scan and accelerated exact search on reference data, so that acceleration changes runtime only.
27. As a clustering user, I want the HDBSCAN result to report whether neighbor search was exact or approximate, so that I know the quality guarantee.
28. As a clustering user, I want neighbor search to report the number of distance evaluations, so that speedups can be audited.
29. As a clustering user, I want an exact spatial-index path for supported numeric metrics, so that Euclidean and Manhattan neighborhoods avoid full scans.
30. As a clustering user, I want a fallback to exact scan when an index cannot guarantee correctness, so that optimization never changes semantics.
31. As a clustering user, I want tree-based similarity available as an explicit approximate mode, so that URF-style proximity can accelerate exploratory clustering without being presented as metric distance.
32. As a clustering user, I want approximate tree-similarity results to expose similarity values separately from distances, so that radius and distance thresholds are not confused.
33. As a clustering user, I want DBSCAN to support fixed radius neighborhoods, so that classic density clustering becomes available.
34. As a clustering user, I want OPTICS to return reachability ordering and cluster extraction at multiple densities, so that I can inspect cluster structure without committing to one radius.
35. As a clustering user, I want DBSCAN and OPTICS to support the same validated distance registry as HDBSCAN, so that engine choice does not change metric semantics.
36. As a clustering user, I want DBSCAN and OPTICS noise labels represented consistently with HDBSCAN, so that downstream workflows can consume all density engines uniformly.
37. As a clustering user, I want OPTICS extraction methods such as Xi and steepest-density reachable sets, so that I can choose extraction semantics.
38. As a library maintainer, I want core-distance, mutual-reachability, and neighborhood code shared among HDBSCAN, DBSCAN, and OPTICS, so that density semantics cannot diverge.
39. As a library maintainer, I want one neighbor-search seam used by KNN leaves and clustering, so that index support and exactness have one source of truth.
40. As a library maintainer, I want approximate search to remain behind an explicit mode, so that production defaults preserve exact behavior.

### BIRCH and incremental clustering

41. As a streaming user, I want BIRCH clustering, so that I can cluster data larger than memory through a compact summary tree.
42. As a streaming user, I want CF-tree nodes to maintain sufficient statistics needed for clustering, so that raw points need not be retained for every cluster.
43. As a streaming user, I want a configurable radius threshold controlling when a CF-tree leaf splits, so that summary granularity matches workload scale.
44. As a streaming user, I want configurable branching and leaf-capacity limits, so that memory use has an explicit bound.
45. As a streaming user, I want incremental point updates, so that clusters evolve as new observations arrive.
46. As a streaming user, I want centroid and medoid summaries, so that I can choose the representation that fits the objective.
47. As a streaming user, I want an optional micro-cluster representative set for final assignment, so that the compact summary can be refined without raw history.
48. As a streaming user, I want update statistics such as points seen, splits, merges, and memory estimate, so that streaming behavior is observable.
49. As a streaming user, I want order-sensitive and order-robust evaluation, so that I know whether input order affects the result.
50. As a library maintainer, I want BIRCH behind the same clustering result and report contracts, so that streaming does not create a separate API.
51. As a library maintainer, I want streaming update operations separated from the initial fit operation, so that partial fit and incremental update semantics remain distinct.

### Gower and mixed-type clustering

52. As a clustering user, I want Gower k-medoids, so that mixed numeric and categorical data can cluster without numeric encoding.
53. As a clustering user, I want Gower k-prototypes, so that numeric centers and categorical modes use an efficient Lloyd-style update.
54. As a clustering user, I want mixed-type cluster prototypes to report their numeric component and categorical modes, so that clusters remain interpretable.
55. As a clustering user, I want deterministic mixed-type initialization, so that repeated runs with one seed agree.
56. As a clustering user, I want missing values handled explicitly during mixed-type assignment, so that incomplete rows are not assigned through zero imputation.
57. As a clustering user, I want the relative weight of numeric and categorical components configurable, so that mixed-type objectives can be tuned and compared.
58. As a clustering user, I want Gower k-medoids to support sampled initialization for large data, so that practical scale remains reachable.
59. As a clustering user, I want Gower results to report medoids and pairwise objective value, so that quality can be audited.
60. As a clustering user, I want existing numeric-only engines to reject categorical columns explicitly, so that they never silently discard or encode them.
61. As a clustering user, I want Gower-based engines available through the same Python facade and snapshot format, so that mixed-type clustering is a first-class capability.

### Explainable and convex clustering

62. As a clustering user, I want explainable k-means where leaves define clusters through feature thresholds, so that every cluster has a direct rule.
63. As a clustering user, I want explainable k-medians using medoid representatives and threshold leaves, so that robust mixed-type clustering remains interpretable.
64. As a clustering user, I want explainable clustering to report rule depth and rule count, so that interpretability cost is measurable.
65. As a clustering user, I want explainable clustering compared against post-hoc rule extraction, so that direct and explained clustering remain distinct.
66. As a clustering user, I want tree-guided $L_1$ convex clustering, so that I receive a hierarchy without k-means local minima.
67. As a clustering user, I want a convex cluster path over regularization strength, so that I can inspect how clusters merge.
68. As a clustering user, I want convex clustering to use tree-structured weights efficiently, so that complete graph pairwise weights are not required.
69. As a clustering user, I want convex clustering to report objective value, path nodes, and merge events, so that the hierarchy can be audited.
70. As a clustering user, I want ADMM or equivalent convex optimization available as a baseline solver for convex clustering, so that tree-guided dynamic programming can be validated.
71. As a library maintainer, I want the convex solver behind the general optimization interface from Spec 21, so that clustering does not create a second optimizer stack.
72. As a library maintainer, I want convex clustering initially restricted to numeric data with explicit type checks, so that categorical semantics remain honest.

### e-means, o-means, and tree-accelerated assignment

73. As a research user, I want an e-means-inspired prototype using exponential moving-average centroid updates, so that I can evaluate the recent fast embedding-clustering method inside `mltreelib`.
74. As a research user, I want the e-means prototype to support cosine-annealed step schedules, so that the published update behavior can be reproduced.
75. As a research user, I want the e-means prototype to expose automatic batch-size and peak-step laws, so that I can compare against the published recipe rather than hand-tuned settings.
76. As a research user, I want last-epoch mean finalization and slow-EMA finalization available separately, so that training and generalization behavior can be compared.
77. As a research user, I want dead-cluster revival disabled by default in the research prototype, so that known rare-cluster failure modes do not silently remove valid clusters.
78. As a research user, I want distance-aware revival available as an alternative, so that small isolated clusters are not removed solely because of count.
79. As a research user, I want the e-means prototype to include the published rare-cluster counterexample as a regression case, so that mass-based revival failure is visible and documented.
80. As a research user, I want an o-means-inspired prototype with hierarchical centroid cells and approximate assignment, so that large-k assignment cost can be evaluated.
81. As a research user, I want the o-means prototype to report nearest-centroid miss rate and distance-evaluation ratio, so that approximate assignment quality is measurable.
82. As a research user, I want the o-means prototype to rebuild centroid cells during training, so that moving centroids do not leave stale partitions.
83. As a research user, I want the o-means prototype to compare exact final assignment with approximate training assignment, so that reported labels and centroids have explicit semantics.
84. As a research user, I want a tree-accelerated e-means variant that uses the package's tree indexes to prune impossible centroid comparisons, so that the tree package can contribute to embedding clustering rather than only tabular clustering.
85. As a research user, I want a tree-accelerated variant to fall back to exact assignment when pruning cannot guarantee a bound, so that approximate search is always visible.
86. As a research user, I want e-means and o-means results compared against Lloyd k-means on equal wall-clock and equal-epoch budgets, so that speed claims have context.
87. As a research user, I want e-means and o-means labeled experimental and excluded from production default routing, so that recent methods do not displace stable defaults without evidence.
88. As a research user, I want Gower-aware prototype clustering, so that the e-means update concept can be evaluated for mixed numeric and categorical data without pretending a Euclidean mean is meaningful.
89. As a research user, I want Gower-aware prototypes to use medoids or prototype summaries, so that categorical modes and numeric summaries follow defensible semantics.

### Historical K-NN Tree prototype

90. As a research user, I want the historical K-NN Tree idea revived as a labeled prototype, so that its distance-confidence regions and centroid calibration can be tested.
91. As a research user, I want the K-NN Tree prototype to support incremental updates, so that its claimed online behavior can be measured.
92. As a research user, I want the K-NN Tree prototype compared with TrieKNN, URF proximity, and plain k-nearest neighbors, so that its distinct value is established or rejected.
93. As a research user, I want K-NN Tree calibration and pruning behavior exposed as parameters, so that the notebook's unfinished policy can be tested rather than copied implicitly.
94. As a research user, I want K-NN Tree to report retained reference size and distance-evaluation count, so that claimed space and runtime reductions can be checked.
95. As a research user, I want the K-NN Tree prototype excluded from production default routing until it beats established baselines, so that prototype complexity does not become public complexity.

### Clustering evaluation and benchmark program

96. As a clustering user, I want one clustering report containing external validity, internal validity, stability, balance, noise rate, runtime, memory, and algorithm diagnostics, so that I can compare methods without assembling metrics myself.
97. As a clustering user, I want Adjusted Rand Index implemented once in Rust and reused by Python, so that benchmark results do not depend on ad hoc Python implementations.
98. As a clustering user, I want Normalized Mutual Information, homogeneity, completeness, and V-measure, so that partition agreement has standard alternatives.
99. As a clustering user, I want silhouette, Calinski-Harabasz, Davies-Bouldin, and Dunn indices, so that internal compactness and separation are visible.
100. As a clustering user, I want stability measured across bootstrap or subsample refits using ARI across assignments, so that a partition's reproducibility is quantified.
101. As a clustering user, I want cluster-size balance and noise rate reported, so that pathological tiny or empty clusters remain visible.
102. As a clustering user, I want algorithm-specific objective values reported, so that k-means inertia, Gower dispersion, mutual reachability, and convex objectives remain comparable only within compatible algorithms.
103. As a clustering user, I want evaluation metrics to ignore noise rows consistently, so that density clustering is not penalized for correctly labeling outliers.
104. As a clustering user, I want external metrics to state their assumption that labels are ground truth, so that unsupervised partitions are not confused with supervised targets.
105. As a clustering user, I want the report to include the exact label convention and cluster count used, so that results are reproducible.
106. As a clustering user, I want prediction assignments included in evaluation, so that fit quality and out-of-sample behavior are separate.
107. As a clustering user, I want objective quality, validity, stability, and runtime plotted as a Pareto view, so that no single metric dominates method selection.
108. As a benchmark author, I want one canonical clustering benchmark entry point, so that duplicated scripts do not drift.
109. As a benchmark author, I want synthetic benchmarks with known Gaussian clusters, anisotropic clusters, non-convex shapes, hierarchical groups, rare isolated groups, mixed types, and missingness, so that each failure mode has a controlled case.
110. As a benchmark author, I want real tabular clustering data with supervised labels used only for external validity, so that descriptive and predictive validity remain separate.
111. As a benchmark author, I want the benchmark to compare current engines, new production engines, and research prototypes in separate groups, so that experimental methods do not distort stable baselines.
112. As a benchmark author, I want every method run under equal distance-evaluation and wall-clock budgets where meaningful, so that speed and quality comparisons are interpretable.
113. As a benchmark author, I want benchmark output written as machine-readable JSON plus a concise Markdown or plot summary, so that results can be compared across revisions.
114. As a benchmark author, I want benchmark records to include package version, platform, data hash, metric, parameters, seed, and missing-value policy, so that historical results remain interpretable.
115. As a benchmark author, I want a tiny-budget smoke run for every engine in routine tests, so that interface regressions surface without running the full benchmark.
116. As a benchmark author, I want the full clustering benchmark to run separately from routine unit tests, so that the normal suite remains fast.
117. As a library maintainer, I want existing clustering benchmarks and docs updated to use the shared report, so that old and new evidence use one metric definition.

### Public facade and persistence

118. As a Python user, I want all clustering engines available through one `ClusterTree` facade, so that I do not manage separate APIs.
119. As a Python user, I want explicit engine names for existing and new methods, so that configuration files remain readable.
120. As a Python user, I want one metric parameter whose accepted values depend on the engine, so that unsupported combinations produce actionable errors.
121. As a Python user, I want one evaluation method returning the shared clustering report, so that evaluation is consistent.
122. As a Python user, I want fitted clustering state to round-trip through snapshots, so that clusters, scales, indexes, and reports remain reproducible.
123. As a Python user, I want fitted distance scales and neighbor indexes to survive snapshot round-trips, so that prediction does not silently change.
124. As a Python user, I want experimental engines clearly marked, so that I know which methods have production stability guarantees.
125. As a Python user, I want unknown or unsupported parameters rejected with suggestions, so that configuration mistakes fail early.

## Implementation Decisions

### Decision 1: One canonical distance registry

The Rust core owns one canonical distance registry used by clustering, KNN leaves, and general pairwise-distance helpers. It replaces the separate HDBSCAN metric enum and duplicate metric switches.

The registry distinguishes:

- True metrics: Euclidean, Manhattan, Chebyshev, Minkowski with valid power, and diagonal Mahalanobis under positive scales.
- Semi-metrics or dissimilarities: cosine, correlation, Canberra, Bray-Curtis, Hamming, Jaccard, and Gower where triangle inequality is not guaranteed.
- Binary or categorical distances: Hamming and Jaccard require explicit binary or categorical feature declarations.

The registry reports, per distance:

- Canonical name and accepted aliases.
- Required feature types.
- Whether it is a metric, semi-metric, or general dissimilarity.
- Fit-time scale requirements.
- Missing-value behavior.
- Whether an exact spatial index can support it.
- Whether it supports mixed numeric and categorical data.

Existing public names remain accepted. Unsupported combinations fail in Rust with an actionable message naming valid distances for the selected engine.

### Decision 2: Typed mixed-data distance service

The distance service consumes typed columns rather than a matrix of opaque numbers. Numeric columns contribute numeric distance. Categorical columns contribute mismatch indicators. Ordinal columns remain out of scope until their ordering semantics are specified.

Gower distance follows the standard pairwise form: average normalized numeric difference and categorical mismatch over jointly observed features. Numeric scales are training ranges. Categorical mismatches use a unit contribution unless an explicit weight is supplied.

Rows with no jointly observed feature are incomparable. The distance service returns an explicit incomparable result rather than zero. Minimum overlap is configurable and defaults to one feature for dissimilarity use, while metric engines can require stronger coverage.

The fitted model stores scales and categorical state. Test-time distance uses training scales.

### Decision 3: Distance validation is a first-class deliverable

A Rust distance-validation suite checks every registry entry against:

- Non-negativity.
- Symmetry.
- Zero self-distance for identical valid rows.
- Reference values for hand-computed and public benchmark cases.
- Scale and invariance expectations.
- Degenerate cases: zero vectors, constant columns, empty overlap, single feature, and extreme magnitudes.
- Triangle inequality for entries classified as true metrics, within floating-point tolerance.

Generic pairwise-distance helpers reuse the registry rather than maintaining a separate switch. KNN leaf code reuses the same semantics.

### Decision 4: One neighbor-search seam

The neighbor-search seam exposes one operation for k-nearest neighbors, radius neighbors, and adaptive neighbors where applicable. It returns neighbor indices, distances or similarities, evaluated-neighbor count, and an exactness status.

Search modes are explicit:

- Exact brute-force scan. Correct fallback for every supported distance.
- Exact indexed search. Used only when the index guarantees metric semantics for that distance and data type.
- Approximate tree-similarity graph. Uses URF-style tree co-occurrence or another explicit tree similarity. Results are labeled similarity, not metric distance.

HDBSCAN core distances use this seam. Exact indexed and brute-force paths must produce equivalent reference neighborhoods. DBSCAN and OPTICS use metric search only.

### Decision 5: HDBSCAN acceleration preserves exact semantics

HDBSCAN's mutual-reachability graph, condensed tree, cluster extraction, and GLOSH scores remain unchanged. Only core-distance and MST construction move behind the neighbor seam.

For supported numeric metrics, an exact spatial index accelerates neighbor discovery. The implementation falls back to exact scan whenever the index cannot guarantee the same neighborhoods.

The result reports search mode and distance-evaluation count. Existing HDBSCAN default output remains semantically unchanged.

### Decision 6: DBSCAN and OPTICS share density machinery

Pure DBSCAN and OPTICS are new production engines built on shared core-distance, neighborhood, and mutual-reachability code.

DBSCAN exposes radius, minimum samples, distance registry, and exact or indexed search. Noise is represented consistently with HDBSCAN.

OPTICS returns the reachability ordering, extraction tree or hierarchy metadata, and a configurable cluster extraction method. Initial support includes Xi extraction and steepest-density reachable sets.

Approximate tree similarity is not accepted by DBSCAN or OPTICS because both require metric radius semantics.

### Decision 7: BIRCH CF-tree engine

BIRCH is implemented as a separate production engine with an explicit streaming update surface.

Each CF-tree node maintains count, linear sum, and sum of squares for numeric summaries. The tree uses configurable branching factor, leaf capacity, and radius threshold. Leaf splitting and merging follow the published CF-tree design.

Initial production support is numeric data. Mixed-type extensions use prototype summaries rather than Euclidean CF statistics and are deferred until the mixed-type distance contract is stable.

The engine exposes batch fit, incremental update, final representatives, final assignments, and diagnostics. Snapshot persistence stores the CF tree or sufficient retained state required for deterministic future updates.

### Decision 8: Gower clustering engines

Gower k-medoids and k-prototypes are production engines.

Gower k-medoids uses medoid representatives and Gower objective. Large inputs use deterministic sampled initialization. Gower k-prototypes maintains numeric centers and categorical modes with an explicit numeric-versus-categorical weight.

Both engines use the shared mixed-data distance service and preserve missingness during assignment. They report prototypes and objective value through the common clustering result.

### Decision 9: Explainable clustering is direct, not post-hoc

Explainable k-means and k-medians construct trees whose leaves define clusters. Each internal node selects one feature threshold. Cluster rules come directly from the tree used for cluster formation.

The engine reports rule depth, rule count, feature usage, leaf sizes, and cluster representative statistics. Post-hoc rule extraction from arbitrary cluster labels remains a separate capability and does not count as explainable clustering.

### Decision 10: Convex clustering uses the shared optimization module

Tree-guided $L_1$ convex clustering is a production research-grade engine. Numeric data is the initial supported domain.

The optimization uses the general solver interface from Spec 21. ADMM with sparse graph weights is the general baseline. Tree-structured fused-lasso dynamic programming is the scalable tree-weighted path.

The engine returns a cluster path, merge events, objective values, and assignments at a requested cut. It does not claim global optimality when using an approximate or bounded solver.

### Decision 11: e-means and o-means are labeled research prototypes

e-means and o-means-inspired implementations live under an experimental clustering namespace. They do not replace stable defaults in the public facade.

The e-means prototype implements the published update semantics:

- Per-cluster count and sum maintained as exponential moving averages, with centroid equal to sum divided by count.
- Cosine-annealed step from a peak value down to a tenth of that peak over the planned update count.
- Automatic batch-size and peak-step scaling laws expressed as power laws in cluster count, row count, batch count, and epoch count, with documented constants, so the prototype needs no manual hyperparameter sweep. Manual override remains available.
- Initialization from sampled rows, with optional stronger initialization.
- Finalization options: last-epoch mean for training loss, and a slower moving-average centroid for out-of-sample generalization.
- Revival policies: off, mass-based proportional split, and distance-aware revival that keeps isolated small clusters.

Mass-based revival is disabled by default because published counterexamples show it can destroy a globally optimal solution on rare-cluster data. The published rare-cluster counterexample is a required regression case, and the prototype must report when a revival policy removes a cluster.

The o-means prototype implements:

- Coarse centroid cells built from the centroids themselves, with the centroid table kept grouped by cell.
- Limited cell probing per row, so each row compares against only a subset of centroids.
- Periodic cell rebuilding while centroids move, with a distinct early-rebuild path during the first pass when revival is active.
- Split lineage recorded so rows near a donor can reach centroids created from it.
- Recency-weighted closing mean for runs shorter than two epochs, and last-epoch mean otherwise.
- Exact final assignment separated from approximate training assignment, with nearest-centroid miss rate and distance-evaluation ratio reported.

Both prototypes expose Lloyd k-means as a baseline and compare equal-epoch and equal-wall-clock quality. Neither prototype claims FAISS-scale GPU throughput; the `mltreelib` versions target CPU tabular and moderate-embedding workloads.

### Decision 12: Tree-accelerated e-means and Gower variants are bounded research prototypes

Tree-accelerated assignment reuses the tree index and neighbor-search seam. Pruning must use admissible bounds. If a bound cannot guarantee correctness, the search falls back to exact assignment. Every approximate run reports mode, pruning statistics, and exactness.

Gower-aware prototypes use medoids or prototype summaries. They do not use Euclidean means for categorical components. Their objective and update semantics are documented as experimental.

### Decision 13: Historical K-NN Tree remains a prototype until proven

The notebook in the historical paper codebase is not ported wholesale. Its useful concepts are extracted: distance-confidence regions, centroid representatives, incremental updates, bounded retained references, calibration, and pruning.

The prototype lives under the experimental namespace and compares itself with TrieKNN, URF proximity clustering, and plain k-nearest neighbors. It must report retained-reference size, distance evaluations, prediction quality, and clustering validity.

No prototype enters stable default routing without a benchmark win and passing the shared regression suite.

### Decision 14: Common clustering result and report

The clustering result contains:

- Labels using one documented convention, with noise represented consistently.
- Centers, medoids, prototypes, or tree leaves as applicable.
- Per-point distance or similarity.
- Algorithm objective value.
- Stability values.
- Optional direct rules.
- Diagnostics such as search mode, evaluated-neighbor count, splits, merges, and revival events.

The clustering report contains:

- External validity: ARI, NMI, homogeneity, completeness, V-measure.
- Internal validity: silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn.
- Stability: ARI across bootstrap or subsample refits.
- Cluster-size balance and noise rate.
- Fit time, evaluation time, and peak memory when available.
- Objective value and algorithm-specific diagnostics.
- Label convention, cluster count, metric, and missing-value policy.

Metrics live in Rust core and are exposed through Python. Existing Python-only ARI implementations are replaced by the shared report.

### Decision 15: Evaluation treats noise and ground truth explicitly

External metrics are computed only when reference labels exist. Density-clustering noise rows are excluded by default and the count is reported. A report can request inclusion of noise rows for sensitivity analysis, but default values remain fixed.

Internal metrics that require distances are computed over clustered rows and report the evaluated subset. Algorithms with incompatible objectives do not present objective values as directly comparable.

### Decision 16: One canonical clustering benchmark

The repository keeps one maintained clustering benchmark program. It covers:

- Existing PCT, CLTree, HDBSCAN, and URF.
- New production DBSCAN, OPTICS, BIRCH, Gower, explainable, and convex engines when available.
- Experimental e-means, o-means, tree-accelerated, Gower-aware, and K-NN Tree prototypes in a separate results group.

Benchmarks include controlled synthetic data with known structure and real tabular data. Results are machine-readable JSON plus concise Markdown or plots. Benchmark metadata include package version, platform, data hash, parameters, seed, metric, and missing-value policy.

### Decision 17: Missing-value correctness is a dependency, not duplicated work

The missing-value conversion defect is being fixed by another workstream. This specification consumes the corrected behavior and must not introduce a parallel fix.

Acceptance still requires clustering engines to preserve missing values, report incomparable pairs explicitly, and never treat zero as a universal missing sentinel. Clustering-specific tests verify that behavior after the dependency lands.

### Decision 18: Public facade and persistence

The Python `ClusterTree` facade gains explicit engine names for all production methods and a separate experimental namespace for prototypes. The facade validates metric and engine compatibility in Rust.

Fitted clustering state persists through the existing snapshot mechanism. Persisted state includes distance scales, categorical state, centers or prototypes, labels, rules where applicable, and index state needed for deterministic prediction or incremental update.

### Decision 19: No production optimizer duplication

Convex clustering and any optimized tree-accelerated assignment use the general solver and neighbor-search abstractions from Spec 21. Clustering does not create a second optimizer or distance stack.

### Decision 20: Scope boundaries

Risk-sensitive or risk-standardized clustering structure, fairness constraints, and domain-specific risk objectives remain a later specification. This spec defines ordinary clustering quality, stability, missingness, and runtime. Any future risk standard integrates through clustering result and report fields without redefining the core algorithms.

## Testing Decisions

- **Good tests assert external behavior.** Distance tests assert mathematical and reference behavior. Clustering tests assert labels, metrics, objective properties, feasibility, determinism, and persistence. They do not assert private heap layout, mutation order, or implementation-specific tree shape.
- **Distance registry seam:** For every distance, verify non-negativity, symmetry, zero diagonal, reference values, and documented triangle-inequality status on deterministic cases.
- **Reference parity:** Compare distance values against SciPy, scikit-learn, or R implementations in benchmark-only tests. Core unit tests use fixed expected values and do not require external packages.
- **Mixed-data Gower seam:** Verify numeric range normalization, categorical mismatch, pairwise missing exclusion, minimum-overlap failure, and categorical weight behavior.
- **Degenerate distance seam:** Verify zero vectors, constant columns, empty overlap, one feature, extreme magnitudes, and non-finite input rejection.
- **Neighbor-search seam:** For exact brute force and exact indexed paths, assert identical k-neighbor and radius-neighbor sets on reference data. Assert evaluated-neighbor counts are reported.
- **Approximate tree-similarity seam:** Assert results are labeled similarity rather than distance, and that approximate mode appears in diagnostics.
- **HDBSCAN regression seam:** Existing HDBSCAN cluster and outlier behavior remains unchanged after acceleration. Exact indexed and brute-force paths produce equivalent reference output.
- **DBSCAN seam:** Verify known separated blobs, noise points, fixed-radius behavior, and consistency with an independent brute-force reference.
- **OPTICS seam:** Verify reachability ordering, noise handling, Xi extraction on known structure, and equivalence with brute-force neighborhood computation.
- **BIRCH seam:** Verify CF statistics, leaf split and merge behavior, deterministic incremental updates, bounded memory metadata, order-sensitivity reporting, and snapshot continuation.
- **Gower clustering seam:** Verify mixed numeric and categorical clustering, prototype contents, deterministic initialization, missing-value assignment, and objective computation.
- **Explainable clustering seam:** Verify each leaf rule reproduces its cluster membership on training data, rule depth and count are reported, and post-hoc extraction remains distinguishable.
- **Convex clustering seam:** Verify objective values against a small reference solution, cluster-path monotonicity under the solver contract, merge-event consistency, and explicit solver status.
- **e-means prototype seam:** Verify EMA update equations, cosine schedule, batch and step laws, finalization modes, dead-cluster behavior, and deterministic seed replay. Include the published rare-cluster counterexample.
- **Distance-aware revival seam:** Verify isolated small clusters survive while redundant close clusters may be revived, using fixed synthetic groups.
- **o-means prototype seam:** Verify cell construction, approximate assignment, cell rebuild, exact final assignment, miss-rate reporting, and distance-evaluation reporting.
- **Tree-accelerated prototype seam:** Verify pruned search matches exact search when bounds are valid and falls back when bounds are unavailable.
- **K-NN Tree prototype seam:** Verify incremental update, confidence-region behavior, bounded reference retention, pruning and calibration parameters, and comparison baselines.
- **Evaluation report seam:** Verify ARI, NMI, homogeneity, completeness, V-measure, silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn, stability, balance, noise rate, and objective reporting against independent reference values.
- **Noise handling seam:** Verify default exclusion of density-clustering noise and explicit sensitivity reporting when included.
- **Facade seam:** Verify supported engine names, unsupported combinations, metric validation, experimental labeling, and snapshot round-trip.
- **Persistence seam:** Verify that restored clustering predicts identically, retains distance scales, and continues deterministic BIRCH updates when incremental state is supported.
- **Benchmark smoke test:** Run every engine under a tiny budget on one deterministic dataset. Verify report schema and output artifacts without running the full benchmark.
- **Full benchmark:** Run separately. Compare methods under shared seeds, metrics, and preprocessing. Use known ground truth only for external validity and generated truth for structural checks.
- **Prior art:** Reuse patterns from existing solver and model benchmark programs, Rust core clustering tests, Python clustering tests, existing snapshot round-trip tests, and the current unified clustering benchmark.

## Out of Scope

- Deep embedding clustering, autoencoders, contrastive pretraining, and GPU clustering kernels.
- Distributed or GPU-native FAISS behavior.
- Risk-sensitive, fairness-constrained, or domain-specific risk-standard clustering objectives.
- Replacing stable defaults with e-means, o-means, tree-accelerated prototypes, or the historical K-NN Tree before benchmark evidence supports it.
- Full mixed-type BIRCH CF statistics before numeric BIRCH behavior is stable.
- Exact global optimality for k-means, k-medians, or explainable partitioning.
- Production-quality web-scale vector indexing comparable to FAISS IVF, HNSW, or CAGRA.
- Parallel or asynchronous clustering engines.
- A second independent distance implementation outside the shared registry.
- A second independent optimizer stack outside Spec 21.
- Duplicating the missing-value correction currently owned by another workstream.
- Committing or publishing tracker issues as part of this specification.

## Further Notes

- This specification extends the research note `references/research/2026-11-21-clustering-implementation-landscape.md` and the earlier tree-clustering survey `references/research/2026-11-01-tree-based-clustering-frontiers.md`.
- The historical K-NN Tree notebook is an early prototype. Its concepts overlap current TrieKNN and URF proximity, but its exact distance-confidence tree is distinct. The prototype exists to answer that question empirically.
- e-means primary sources:
  - Paper: Bessa et al., “e-means: outperforming k-means (in one-tenth the epochs),” [arXiv:2611.09501](https://arxiv.org/abs/2611.09501).
  - Project page and implementation notes: [e-means.github.io](https://e-means.github.io/).
  - FAISS implementation: `faiss/contrib/e_means` in the FAISS repository.
- o-means primary implementation sources:
  - Project implementation and design notes: [owlgebra-ai/faiss `contrib/o_means`](https://github.com/owlgebra-ai/faiss/tree/main/contrib/o_means).
  - FAISS contribution proposal: [facebookresearch/faiss pull request #5917](https://github.com/facebookresearch/faiss/pull/5917).
- The published e-means recipe keeps count and sum as exponential moving averages, derives centroid as sum divided by count, applies cosine-annealed step size, chooses batch size and peak step through scaling laws, and finalizes with last-epoch mean. The `mltreelib` prototype should reproduce these semantics rather than implement generic mini-batch k-means.
- The o-means implementation groups centroids into coarse cells, probes a limited number of cells per row, rebuilds cells while centroids move, and distinguishes approximate training assignments from exact final assignment. Its reported defaults and GPU assumptions are evidence from one reference implementation, not universal defaults for `mltreelib`.
- e-means has very recent primary evidence. Its mass-based dead-cluster revival has published counterexamples, and distance-aware revival has separate recent work. Default revival behavior in `mltreelib` must therefore be conservative.
- o-means is currently a FAISS pull request and contributor implementation rather than an established primary-paper algorithm. It remains research-grade.
- The KNN leaf module's ten distances are implemented but not adequately validated for clustering use cases. Consolidating them into a registry is a prerequisite for treating any of them as clustering metrics.
- The current HDBSCAN implementation stores numeric training data and supports only three metrics. Mixed-type HDBSCAN is not part of this spec.
- Current clustering benchmark scripts report ARI and fit time only. One shared report and one canonical benchmark are prerequisites for credible method comparison.
