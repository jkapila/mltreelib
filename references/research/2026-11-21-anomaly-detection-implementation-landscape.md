# Anomaly Detection Implementation Landscape for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Unsupervised and semi-supervised anomaly detection for tabular, mixed-type, grouped, hierarchical, streaming, and time-series data.  
**Decision informed:** Which methods belong in the Rust core, which belong behind optional interfaces, and which should remain external benchmarks.  
**Confidence:** High for classical algorithms and current package fit. Medium for recent learned methods because most require neural runtimes or incomplete primary-source verification.

## 1. Executive Summary

`mltreelib` already has seven tree-native anomaly engines: GritBot, Density Estimation Tree (DET), Extended Isolation Forest (EIF), Half-Space Tree (HS-Tree), Random Cut Tree (RCT), Active Anomaly Tree (AAD), and Context Anomaly Tree. Existing work is strongest on **contextual, tree-isolation, density, streaming, and sequential surprise**.

Main gap is not another isolation tree. It is a fast, deterministic, explainable **non-tree benchmark layer** covering marginal-tail, projection, and subsequence anomalies. Recommended order:

1. **ECOD / COPOD** — strongest immediate fit. Pure Rust, $O(nd)$, no tuning, per-feature explanations.
2. **LODA** — compact online projection-and-histogram detector. Fits streaming architecture.
3. **HBOS** — simple histogram baseline and component for detector ensembles.
4. **Matrix Profile** — strongest native time-series addition. Requires dedicated subsequence machinery, not row-wise anomaly scores.
5. **LookOut-style explanations** — detector-agnostic explanation layer for existing and future detectors.
6. **CHAODA-style hierarchical scoring** — leverage existing cluster trees without adding a deep detector.
7. **LOF and kernel detectors** — defer until core k-nearest-neighbor indexing supports them.

Deep time-series models such as USAD, TranAD, and Anomaly Transformer should remain **external benchmarks**, not Rust-core dependencies. They need autodiff, GPU-oriented training, and materially different operational assumptions.

---

## 2. Current Package Baseline

`crates/mltreelib-core/src/anomaly/mod.rs` exports:

| Method | Mechanism | Current role |
|---|---|---|
| `GritBot` | Contextual subgroup rules and robust within-subgroup deviations | Contextual/data-quality anomalies |
| `DensityEstimationTree` | Integrated squared error density leaves | Non-parametric density score |
| `ExtendedIsolationTree` | Random oblique isolation cuts | Axis-artifact-resistant isolation |
| `HalfSpaceTree` | Mass-based streaming tree | Streaming anomalies |
| `RandomCutTree` | Random cuts plus CoDisp | Robust stream anomaly detection |
| `ActiveAnomalyTree` | Human-in-the-loop weight updates | Active anomaly discovery |
| `ContextAnomalyTree` | Context Tree Weighting / probabilistic suffix tree | Sequential and categorical surprise |

This baseline already covers several families that many generic libraries treat as separate products. New work should create a common detector contract rather than another unrelated estimator.

Recommended common result:

```text
AnomalyScore
  score: f64
  contributions: Vec<FeatureContribution>
  detector: DetectorKind
  fitted: bool
```

Missing values must remain missing. Every score must be finite or explicitly missing. Unseen categorical levels must not silently become an ordinary known category.

---

## 3. Method Taxonomy

### 3.1 Marginal-tail detectors

**HBOS** estimates each feature's density from a histogram and sums negative log densities. It is fast but assumes features are approximately independent and is sensitive to binning.

**ECOD** estimates left and right empirical CDFs per feature. It aggregates tail probabilities in log space:

\[
O(x)=-\sum_j \log \big(\hat F_j(x_j)(1-\hat F_j(x_j))\big)
\]

ECOD is parameter-free, $O(nd)$ in its basic form, parallelizable across features, and exposes each feature's contribution. Its strong independence assumption misses multimodal and dependency-based anomalies. The authors report strong results on 30 benchmark datasets, but `mltreelib` should reproduce those claims through its own benchmark rather than inherit them.

**COPOD** also uses empirical tail probabilities, but selects left, right, or skewness-corrected tails per feature and takes the maximum score across variants. It remains parameter-free and explainable. ECOD and COPOD share enough machinery to share a univariate ECDF engine.

**Package fit:** excellent. Existing typed columns and categorical indexes support feature-wise passes. Continuous columns need sorted values or rank arrays. Categorical columns need a discrete-tail definition rather than numerical coercion.

**Recommended API:** one detector with `Ecod` and `Copod` tail policies, rather than duplicate modules.

### 3.2 Projection and streaming detectors

**LODA** projects observations with sparse random vectors and maintains one-dimensional histograms. Each stream has its own detector state. It is compact, online, and deterministic under an explicit seed.

**RS-Hash** and **xStream** use randomized hashing and sparse grids/counts. They target evolving streams and avoid storing raw history. Their shared machinery resembles LODA but differs in representation and update rule.

**Package fit:** excellent for LODA. Medium for xStream-style detectors because feature evolution and hash-table design need explicit memory bounds. `HalfSpaceTree` and `RandomCutTree` already cover a different streaming design, so new streaming methods should share a common stream-update interface.

### 3.3 Local-density and proximity detectors

**LOF** compares each point's local reachability density with neighbors. It detects local anomalies in heterogeneous densities, unlike global ECDF methods.

**LOF cost:** exact neighborhoods require $O(n^2)$ comparisons without an index. Core package has neighbor infrastructure, but no evidence supports claiming scalable exact LOF yet. Add LOF only after neighbor search can expose exact and approximate neighbor semantics cleanly.

**One-Class SVM** offers a principled boundary but needs kernel linear algebra and careful scaling. It should not be a first core addition.

**CBLOF** clusters data and scores points by cluster-level rarity. It may be useful after a common clustering interface exists, but it does not justify a separate implementation now.

### 3.4 Tree-native and distribution-aware detectors

Existing DET, EIF, HS-Tree, RCT, and GritBot already cover major tree-native families.

**Isolation Distributional Kernel (IDK)** uses isolation partitions to define a distribution kernel. It is a good research extension because it can reuse EIF partition logic. It does not displace ECOD/COPOD as baseline detectors.

**Mondrian Pólya Forest** combines random hierarchical partitions with non-parametric density estimation. It could fit a grouped or streaming API, but complexity is higher than LODA.

**CHAODA/CLAM** derives anomaly scores from hierarchical cluster geometry and local density. This matches existing URF, CLTree, and HDBSCAN-tree code better than a new neural detector.

**Package fit:** CHAODA-style scoring and IDK are Tier 2. Deep Isolation Forest adds learned representations and does not fit the lean Rust core unless represented by an optional crate.

### 3.5 Time-series detectors

**Matrix Profile** stores the distance from each subsequence to its nearest non-trivial neighbor. High profile values identify dissimilar subsequences.

For multivariate series, the multidimensional extension exposes a **K-of-N** problem: anomaly may occur in a subset of dimensions. Pre-sorting dimensions before nearest-neighbor search preserves correlation anomalies, while post-sorting is cheaper but can dilute subset anomalies.

**Package fit:** strong, but it needs a separate time-series index over sliding subsequences. Row-wise ECOD or EIF cannot substitute for it. `ContextAnomalyTree` covers symbolic sequence surprise, not continuous subsequence distance.

**Recommendation:** implement Matrix Profile after continuous-window representation and exclusion-zone logic are defined. Benchmark against STOMP, SCRIMP++, and simple rolling z-score. Require subsequence-length selection rather than exposing one undocumented default.

### 3.6 Explainability layer

**LookOut** searches for small, informative low-dimensional views that explain detector anomalies. It can wrap any detector.

**Shapley-style anomaly interpretation** assigns feature contributions to arbitrary scores but becomes expensive as feature count grows.

**Path attribution for isolation forests** fits the existing EIF/RCT structure well because anomalous paths already encode split evidence.

**Package fit:** excellent. Implement feature contributions first because ECOD, COPOD, HBOS, LODA, and per-feature local detectors expose them naturally. Add generic post-hoc explanations only when they provide evidence unavailable from the detector.

---

## 4. Data-Specific Assessment

| Data structure | Best current or proposed method | Reason |
|---|---|---|
| Numeric tabular | ECOD/COPOD; EIF; DET | Fast marginal evidence plus density/isolation diversity |
| Categorical tabular | ECOD/COPOD with discrete tails; AAD; ContextAnomalyTree | Avoid numerical coercion of codes |
| Mixed numeric/categorical | Same, with feature-type-specific tail models | One detector contract, typed feature behavior |
| Grouped rows | GritBot within groups; DET; group-aware streaming variants | Context must respect group boundaries |
| Hierarchical data | GritBot rules; hierarchical cluster anomaly scoring | Preserve parent-child context |
| Streaming | HS-Tree, RCT, LODA | Incremental state and bounded memory |
| Univariate time series | Matrix Profile; ContextAnomalyTree for symbols | Subsequence distance differs from row novelty |
| Multivariate time series | Multidimensional Matrix Profile | Detect subset and correlation anomalies |
| High-stakes audit | ECOD/COPOD, GritBot, path attribution | Per-feature and rule evidence |

---

## 5. Recommended Build Sequence

### Phase 1: Common contract and classical baselines

- Add `AnomalyDetector` fit/score interface.
- Add ECDF feature summary with continuous and categorical tail policies.
- Add ECOD and COPOD policies over one shared implementation.
- Add HBOS as a benchmark, not default winner.
- Add finite-score, missing-value, unseen-level, and deterministic-seed tests.

### Phase 2: Streaming and explanations

- Add LODA using explicit seeds and bounded histograms.
- Standardize per-feature contribution output.
- Add path attribution for EIF/RCT where path evidence can be mapped to features.
- Add detector ensemble only after component score normalization is defined.

### Phase 3: Time series

- Define continuous subsequence extraction and exclusion zones.
- Implement exact Matrix Profile baseline.
- Add multidimensional pre-sort and post-sort modes.
- Benchmark rolling z-score, STOMP, and SCRIMP++ under shared train/test protocols.

### Phase 4: Research extensions

- CHAODA-style score over existing cluster trees.
- Isolation distributional kernel.
- Approximate LOF after neighbor-index work.

---

## 6. Evaluation Protocol

Unsupervised anomaly ranking and contamination estimation are separate tasks. A correct evaluation needs:

- AUROC and average precision across labeled benchmarks.
- Precision and recall at a documented contamination rate.
- False-positive burden under a known small contamination assumption.
- Runtime, peak memory, fit time, and score time.
- Effect of missingness, unseen categories, correlated features, and feature scale.
- Rank stability across explicit seeds for randomized methods.

Recommended benchmark slices:

1. Existing generated datasets with controlled anomaly mechanisms.
2. Public tabular anomaly suites.
3. Public multivariate time-series suites.
4. Streaming data with drift and changing feature support.
5. Grouped data where anomalies are normal globally but contextual.

Do not compare across slices without reporting contamination, feature type, and missingness policy.

---

## 7. Rejected or Deferred Options

| Option | Decision | Reason |
|---|---|---|
| USAD / TranAD / Anomaly Transformer | External benchmark only | Autodiff, training pipeline, and GPU assumptions conflict with lean core |
| One-Class SVM | Defer | Kernel scaling and solver burden; limited mixed-type support |
| LOF | Defer | Exact neighbor cost until index semantics stabilize |
| OCSMM / hierarchical probabilistic group detectors | Defer | Kernel and model complexity exceed current payoff |
| Generic deep autoencoder detector | Reject for core | Heavy dependency, weak explainability, limited categorical support |
| Another axis-aligned isolation tree | Reject | Existing EIF/RCT already cover isolation family |

---

## 8. Key Risks

1. **False comparability.** Paper benchmark numbers may use different contamination, preprocessing, and scoring conventions.
2. **Categorical coercion.** Numeric code values are ordinal only if category order means something. ECDF tails must not assume that.
3. **Missing-data leakage.** Mean imputation can manufacture both normality and anomalies.
4. **Score direction drift.** Higher score must always mean more anomalous.
5. **Streaming contamination.** Detector state must define whether anomalies update the model.
6. **Explanation mismatch.** A contribution must identify evidence used by the detector, not a post-hoc correlation.

---

## 9. Primary Sources and Indexes

1. Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., and Chen, G. H. (2022). “ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.” [arXiv:2201.00382](https://arxiv.org/abs/2201.00382).
2. Li, Z., Zhao, Y., et al. (2020). “COPOD: Copula-Based Outlier Detection.” [arXiv:2009.09463](https://arxiv.org/abs/2009.09463).
3. Goldstein, M. and Dengel, A. (2012). “Histogram-Based Outlier Score (HBOS).” [KI 2012 poster](https://dfki.de/fileadmin/user_upload/import/6431_HBOS-poster.pdf).
4. Pevný, T. (2016). “LODA: Lightweight On-line Detector of Anomalies.” *Machine Learning* 102. [DOI](https://doi.org/10.1007/s10994-015-5521-0).
5. Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). “LOF: Identifying Density-Based Local Outliers.” *ACM SIGMOD Record*. [DOI](https://dl.acm.org/doi/10.1145/335191.335388).
6. Liu, F. T., Ting, K. M., and Zhou, Z. H. (2008). “Isolation Forest.” *ICDM*. [DOI](https://doi.org/10.1109/ICDM.2008.17).
7. Yeh, C.-C. M., Zhu, Y., Ulanova, L., et al. (2024/2026). “Matrix Profile for Anomaly Detection on Multidimensional Time Series.” [arXiv:2409.09298](https://arxiv.org/abs/2409.09298).
8. Yeh, C.-C. M., et al. (2016). “Matrix Profile I: All Pairs Similarity Joins for Time Series.” *ICDM*. [DOI](https://doi.org/10.1109/ICDM.2016.16).
9. Gupta, N., Eswaran, D., Shah, N., Akoglu, A., and Faloutsos, C. (2017/2018). “Beyond Outlier Detection: LookOut for Pictorial Explanation.” [arXiv:1710.05333](https://arxiv.org/abs/1710.05333).
10. Ishaq, N., Howard, T. J., and Daniels, N. M. (2021). “Clustered Hierarchical Anomaly and Outlier Detection Algorithms.” [arXiv:2103.11774](https://arxiv.org/abs/2103.11774). [Papers With Code](https://paperswithcode.com/paper/clustered-hierarchical-anomaly-and-outlier).
11. Hay, G. and Liberman, P. (2023). “SORTAD: Self-Supervised Optimized Random Transformations for Anomaly Detection in Tabular Data.” [Papers With Code](https://paperswithcode.com/paper/sortad-self-supervised-optimized-random).
12. Guggilam, S., Zaidi, S. M. A., Chandola, V., and Patra, A. K. (2019). “Integrated Clustering and Anomaly Detection for Streaming Data.” [Papers With Code](https://paperswithcode.com/paper/integrated-clustering-and-anomaly-detection).
13. KDD 2020. “Isolation Distributional Kernel.” [DOI](https://dl.acm.org/doi/10.1145/3394486.3403062).
14. Sathe, A. and Aggarwal, S. (2018). “xStream: Detecting Anomalies in High-Dimensional Streaming Data.” *KDD*. [DOI](https://dl.acm.org/doi/10.1145/3219819.3220107).
15. Akoglu, L., Tong, H., Vreeken, J., and Faloutsos, C. (2012). “Fast and Reliable Anomaly Detection in Categorical Data.” *CIKM*. [DOI](https://dl.acm.org/doi/10.1145/2382576.2382592).
16. Pang, G., Shen, C., Cao, L., and van den Hengel, A. (2021). “Deep Learning for Anomaly Detection: A Review.” *ACM Computing Surveys*. [DOI](https://doi.org/10.1145/3439950).
17. Schmidl, S., Wenig, P., and Papenbrock, T. (2022). “Anomaly Detection in Time Series: A Comprehensive Evaluation.” *PVLDB*. [DOI](https://doi.org/10.14778/3538598.3538602).

## 10. Source Notes

- alphaXiv discovery and paper reports supplied ECOD, COPOD, multidimensional Matrix Profile, and recent review candidates.
- Papers With Code surfaced INCAD, CLAM/CHAODA, SORTAD, and linked implementation status. Its index is historical metadata, so primary papers remain source of truth.
- treg catalog exposed Perplexity, Brave, and Scholar endpoints, but live treg calls failed because saved treg authentication expired. No claim in this note depends on a failed treg response.
