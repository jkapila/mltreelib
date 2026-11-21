# Component Analysis and Unified Model Reports for `mltreelib`

**Date:** 2026-11-21  
**Scope:** PCA, truncated SVD, whitening, ICA, NMF, related matrix factorization, their use in anomaly detection and clustering, and a common objective/learner/controller/diagnostics report contract.  
**Decision informed:** Which component-analysis capability belongs in `mltreelib`, how difficult an SVM implementation would be, and how model reports can work across estimators that do not support classical inference.  
**Confidence:** High for repository fit and numerical primitives. Medium for experimental method maturity and broad performance claims.

## 1. Executive Summary

`mltreelib` needs a consolidated component-analysis capability. PCA, truncated SVD, whitening, FastICA, NMF, CCA, PLS, and selected matrix factorizations currently have no unified native interface in this repository.

The repository already depends on `faer 0.24.4` and `nalgebra 0.33`. `faer` provides dense SVD, symmetric and general eigendecomposition, QR, Cholesky, and triangular solves. These primitives cover the first stable wave without new runtime dependencies:

1. Classical PCA and truncated SVD.
2. Whitening and inverse whitening.
3. FastICA using SVD whitening and symmetric decorrelation.
4. NMF using multiplicative updates.
5. PCA reconstruction anomaly detection.
6. NMF reconstruction anomaly detection and factor-based clustering.

Later component methods belong in an experimental namespace: robust PCA, sparse PCA, kernel PCA, probabilistic PCA, tensor PCA, nonlinear ICA, contrastive ICA, and autoencoder-based decomposition.

For reports, `mltreelib` should not force every estimator into a fake statsmodels table. Instead, every fitted estimator should expose one common report shell with four sections:

- **Objective:** what the estimator optimizes, objective value, units, and direction.
- **Learner:** mathematical family, fitted structure, capabilities, and complexity.
- **Controller:** selected configuration and tuned knobs.
- **Diagnostics:** convergence, fit quality, stability, missingness, fallback, runtime, and warnings.

Classical standard errors, confidence intervals, and p-values appear only when the estimator supports valid inference. PCA, KNN, clustering, ICA, NMF, and evolutionary optimizers usually provide descriptive diagnostics, bootstrap intervals where justified, or no coefficient-level inference.

## 2. Current Repository State

### Existing numerical foundation

`mltreelib-core` already depends on:

- `faer 0.24.4`
- `nalgebra 0.33`
- `rand`, `rand_chacha`, and `rand_distr`
- `statrs`
- `rayon`

`faer` exposes:

- `thin_svd` and full `svd`.
- `self_adjoint_eigen` and general `eigen`.
- QR, LU, Cholesky, and triangular solves.

These operations are enough for exact PCA, truncated SVD, whitening, fixed-point ICA with symmetric decorrelation, and basic NMF updates.

### Missing abstraction

There is no repository-level component-analysis module, no shared transform interface, and no common report for PCA-like fitted objects. Existing KNN metrics and encoders provide related preprocessing but are not component models.

### Rust ecosystem

Rust implementations exist, but no single crate supplies the desired consolidated package interface:

- `linfa-reduction`: PCA and dimensional reduction.
- `linfa-ica`: FastICA.
- `linfa-preprocessing`: whitening.
- `efficient_pca`: exact and randomized PCA.
- `aprender`: decomposition helpers including PCA and ICA.

Recommendation: implement the stable core on existing `faer` and `nalgebra`. Use external Rust crates as reference oracles in benchmark-only tests, not mandatory runtime dependencies.

## 3. Component-Analysis Portfolio

### 3.1 PCA

Classical PCA centers optionally standardized data and finds directions of maximum variance:

\[
X_c = U\Sigma V^\top,\qquad Z=X_cV.
\]

Useful outputs:

- Mean and scale.
- Component loadings.
- Singular values or eigenvalues.
- Explained and cumulative variance.
- Scores for training and new rows.
- Inverse transform.
- Reconstruction residual.

Stable options should include:

- Centered PCA.
- Standardized PCA.
- Truncated SVD without centering.
- Exact and randomized component computation.
- Explicit component count or variance threshold.

PCA weaknesses:

- Linear only.
- Sensitive to outliers, scale, and heavy tails.
- Anomalies inside retained subspace remain invisible.
- Missing values complicate covariance estimation.

### 3.2 Truncated SVD

Truncated SVD factors:

\[
X \approx U_k\Sigma_kV_k^\top.
\]

It does not require centering and suits sparse text or wide matrices. It should share one component model with PCA through common loading and transform semantics.

### 3.3 Whitening

Whitening transforms correlated data into approximately uncorrelated unit-variance coordinates:

\[
Z=X_c\Sigma^{-1/2}V.
\]

Whitening supports ICA, Mahalanobis-style scoring, and Gaussian mixture initialization. It must preserve the fitted whitening matrix so inverse transformation and serialization remain exact.

### 3.4 Independent Component Analysis

ICA assumes:

\[
X=AS,
\]

where sources are independent and non-Gaussian. It separates mixed signals rather than merely reducing dimension.

FastICA mechanics:

1. Center input.
2. Whiten with SVD.
3. Initialize an unmixing matrix.
4. Apply fixed-point updates.
5. Symmetrically decorrelate.
6. Stop on convergence.

Important limitations:

- Component order, sign, and scale are not identifiable.
- At most one Gaussian source is identifiable.
- Results depend on initialization and non-Gaussianity.
- Outliers can destabilize whitening.
- Component-level p-values are not generally meaningful.

Stable FastICA should report convergence, kurtosis or neg-entropy diagnostics, component order, signs, and reconstruction error. It is useful for source separation and explanatory decomposition, but not the default anomaly detector.

### 3.5 Nonnegative Matrix Factorization

NMF seeks:

\[
X \approx WH,\qquad W,H\geq 0.
\]

Multiplicative updates are simple and use only matrix operations already available through `faer` and `nalgebra`.

Strengths:

- Additive nonnegative parts.
- More interpretable factors than signed SVD.
- Useful for document-topic-like, count, and nonnegative tabular data.

Weaknesses:

- Joint non-convexity.
- Local optima and initialization sensitivity.
- Zero-update and numerical-underflow hazards.
- Different loss functions and regularization change meaning.
- Clustering interpretation depends on initialization and post-processing.

Stable NMF should support deterministic initialization, Frobenius reconstruction loss, maximum iterations, convergence tolerance, nonnegative clipping, reconstruction error, factor sparsity, and deterministic component ordering.

### 3.6 CCA and PLS

Canonical correlation analysis finds directions with maximum correlation between two views. Partial least squares finds directions with maximum covariance under a predictive weighting.

These are useful for:

- Paired tabular views.
- Multi-view data.
- Supervised latent components.
- Reduced-dimensional predictive models.

They should arrive after PCA, SVD, whitening, ICA, and NMF because they reuse fitted projection and cross-covariance machinery.

### 3.7 Experimental decomposition methods

Experimental methods include:

- Robust PCA and low-rank plus sparse decomposition.
- Sparse PCA.
- Kernel PCA.
- Probabilistic PCA.
- Factor analysis.
- Tensor and multilinear PCA.
- Nonlinear and contrastive ICA.
- Autoencoder-based components.
- Randomized subspace methods beyond basic randomized PCA.

These require explicit objectives, approximate-mode reporting, and independent benchmarks before promotion.

## 4. Component Models Inside Unsupervised Tasks

### 4.1 PCA anomaly detector

Fit PCA on reference data. Reconstruct a test row with rank-$k$ components and score:

\[
e(x)=\|x-\hat x_k\|^2.
\]

Useful diagnostics:

- Total and per-feature reconstruction error.
- Explained variance retained and discarded.
- Hotelling-style squared Mahalanobis reconstruction score.
- Per-feature reconstruction contributions.

PCA anomaly detection is strong when normality forms a low-dimensional linear subspace. It cannot detect anomalies within that subspace.

### 4.2 NMF anomaly detector

Fit NMF on nonnegative reference data and score reconstruction residual. Useful when normal observations have additive nonnegative structure.

Diagnostics must report loss, reconstruction error, factor sparsity, convergence, initialization, and data-domain compatibility.

### 4.3 Components as cluster features

PCA, ICA, NMF, or PLS scores can become local features for:

- K-means or k-prototypes.
- Explainable threshold trees.
- Density clustering.
- Discriminant analysis.
- Anomaly detection.

This reuse requires a common transformed-frame representation and train-only fitting. Component transformation used during prediction must never refit on test rows.

### 4.4 Components as leaf families

A component projection can become a leaf transform:

1. Fit components on leaf or node rows.
2. Fit the response model in component space.
3. Persist both transforms.

This risks leakage if components are fit on rows later used for response fitting. The implementation must define sample ownership and honest-sample behavior explicitly. A global pre-fit transformer followed by tree fitting is simpler and safer for first release.

## 5. Unified Task Facades

The primary user surface remains one facade per task:

```text
mltreelib.anomaly
mltreelib.cluster
mltreelib.rules
mltreelib.components
```

Each stable facade uses one method switch and shared controls. Users normally change a knob rather than import another algorithm.

Recommended stable names:

```python
from mltreelib.anomaly import AnomalyModel
from mltreelib.cluster import ClusterModel
from mltreelib.rules import RuleModel
from mltreelib.components import ComponentModel
```

Example:

```python
model = AnomalyModel(kind="pca", n_components=8, seed=42)
model.fit(reference)
result = model.score(data)
print(result.report)
```

Advanced algorithm-specific classes may remain available, but normal use does not require them.

Existing `AnomalyTree` and `ClusterTree` remain compatibility facades. Their kind switch grows with stable methods. Experimental methods live under:

```text
mltreelib.experimental.anomaly
mltreelib.experimental.cluster
mltreelib.experimental.rules
mltreelib.experimental.components
```

## 6. Common Report Contract

### 6.1 Four required sections

Every fitted model report contains:

1. **Objective**
   - Human-readable objective name.
   - Mathematical expression when useful.
   - Minimized, maximized, or reconstructed.
   - Final objective value.
   - Objective components and units.
   - Whether lower or higher is better.

2. **Learner**
   - Stable estimator family and method kind.
   - Mathematical representation.
   - Fitted structure such as depth, components, factors, centroids, clusters, rules, support vectors, or neighbors.
   - Effective and nominal complexity.
   - Supported capabilities: probability, density, interval, explanation, transform, incremental update.

3. **Controller**
   - Selected configuration.
   - Tuned regularization, component count, support, smoothing, or shrinkage.
   - Random seed.
   - Iteration and evaluation budgets.
   - Missing-data policy.
   - Fallback reason when a requested estimator could not fit.

4. **Diagnostics**
   - Convergence status and reason.
   - Fit and evaluation time.
   - Evaluation count.
   - Objective improvement and stability.
   - Quality metrics relevant to the task.
   - Missing, unseen-level, or finite-value warnings.
   - Calibration, coverage, or reconstruction diagnostics where supported.
   - Method-specific warnings and limitations.

### 6.2 Optional inference section

A common `inference` section is optional. It contains only mathematically valid quantities:

- Standard errors.
- Confidence intervals.
- Test statistics and p-values.
- Bootstrap intervals explicitly labeled as bootstrap.
- Likelihood, deviance, or information criteria when defined.

The report must not fabricate coefficient-level inference for PCA, KNN, clustering, ICA, NMF, or evolutionary optimizers.

### 6.3 Machine-readable and rendered forms

Every report exposes:

- A serializable structured form.
- A text renderer in the familiar R/statsmodels style.
- An optional Markdown renderer for documentation and agent consumption.
- JSON-compatible output for automation.

The structured form is authoritative. Text formatting never reparses numbers from strings.

## 7. Task-Specific Results

### 7.1 Anomaly result

```text
AnomalyResult
  scores
  predictions
  threshold
  feature_contributions
  report
```

### 7.2 Clustering result

```text
ClusterResult
  labels
  probabilities_or_memberships
  centers_or_prototypes
  distances_or_similarities
  rules
  stability
  report
```

### 7.3 Rules result

```text
RuleResult
  itemsets
  rules
  coverage
  redundancy
  classifier
  report
```

### 7.4 Component result

```text
ComponentResult
  components
  loadings
  scores
  explained_variance
  reconstruction
  report
```

## 8. SVM Difficulty

### 8.1 Linear SVM

A linear SVM is feasible without new dependencies if implemented as a convex primal problem.

Squared-hinge objective:

\[
\frac12\|w\|^2+\frac{C}{n}\sum_i\max(0,1-y_i(x_i^\top w+b))^2.
\]

Existing `faer` linear algebra and L-BFGS can fit this objective. This is simpler than a general QP solver and fits package architecture.

Limitations:

- Hinge loss itself is nonsmooth; squared hinge is smooth but changes the objective.
- Exact primal SVM with hinge loss requires a QP or specialized solver.
- Standardization, class weights, intercept handling, and regularization need explicit semantics.
- Probability estimates require an extra calibration model.

A linear squared-hinge SVM can become stable. Exact linear hinge-loss SVM remains harder.

### 8.2 Kernel SVM

Kernel SVM is not simple under the package's dependency policy.

Required machinery includes:

- Kernel evaluation and Gram matrix construction.
- QP or sequential minimal optimization.
- Support-vector selection.
- O(n^2) memory for explicit kernels.
- Precomputation for prediction.
- Numerical treatment of indefinite or ill-conditioned kernels.

Pure Rust SVM crates exist, including `linfa-svm`, `libsvm-rs`, and `anofox-ml-svm`, but adding one adds a separate numerical stack and maintenance burden.

Recommendation:

- Stable leaf: linear squared-hinge SVM implemented on existing primitives.
- Experimental: kernel SVM only through an optional adapter or separate optional crate.
- Do not promise LIBSVM parity in the core.

### 8.3 One-class and anomaly SVM

One-class SVM also uses QP or SMO and becomes costly for large row counts. It should remain experimental until the linear or kernel SVM foundation exists.

## 9. Stable Component Wave

1. Common component model and transform interface.
2. Classical PCA, standardized PCA, and truncated SVD.
3. Whitening and inverse whitening.
4. PCA reconstruction anomaly detector.
5. FastICA with deterministic initialization.
6. NMF with multiplicative updates and deterministic ordering.
7. NMF reconstruction anomaly detector.
8. Component scores as reusable train-only transforms.
9. Common report contract.
10. External reference and benchmark suite.

## 10. Experimental Component Wave

1. Randomized PCA and incremental PCA.
2. CCA and PLS.
3. Robust and sparse PCA.
4. Kernel PCA.
5. Probabilistic PCA and factor analysis.
6. Tensor PCA.
7. Nonlinear and contrastive ICA.
8. Autoencoder-based components.
9. Kernel and one-class SVM adapters.
10. Advanced NMF losses and regularization.

## 11. Evaluation

Component models require:

- Reconstruction or transform reference parity.
- Orthogonality and whitening checks.
- Explained-variance checks.
- Deterministic replay.
- Rotation and scale behavior.
- Missing-value and outlier behavior.
- Fit time, transform time, and memory.
- Use as anomaly detector and cluster-feature transform.

Anomaly and clustering facades require:

- Shared result schema.
- Common report schema.
- Correct threshold and label conventions.
- Stable and experimental method metadata.
- One benchmark program that keeps stable and experimental results separate.

## 12. Risks

1. **Synthetic diversity.** Hundreds of unrelated algorithms produce a confusing catalog.
2. **Inference fabrication.** Classical p-values do not apply to many estimators.
3. **Report inconsistency.** Independent text implementations drift.
4. **Dependency expansion.** `linfa`, `ndarray`, and BLAS stacks complicate the core.
5. **SVM misrepresentation.** Squared hinge is not exact hinge-loss SVM.
6. **NMF instability.** Local optima and zero factors require explicit handling.
7. **ICA ambiguity.** Component order and sign need deterministic canonicalization.
8. **PCA contamination.** Outliers distort fitted components.
9. **Refit leakage.** Components fitted during prediction invalidate results.
10. **Facade explosion.** Universal one-size dispatch would make agent decisions harder.

## 13. Primary Sources

1. Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd edition. Springer.
2. Halko, N., Martinsson, P.-G., and Tropp, J. A. (2010). “Finding Structure with Randomness.” *SIAM Review*. [DOI](https://doi.org/10.1137/090532886).
3. Hyvärinen, A. and Oja, E. (2000). “Independent Component Analysis: Algorithms and Applications.” *Neural Networks*. [DOI](https://doi.org/10.1016/S0893-6080(00)00053-2).
4. Hyvärinen, A., Karhunen, J., and Oja, E. (2001). *Independent Component Analysis*. Wiley.
5. Lee, D. D. and Seung, H. S. (1999). “Algorithms for Non-negative Matrix Factorization.” *NIPS*. [Paper](https://proceedings.neurips.cc/paper/1999/hash/fb80e41925b7a1be2eb9b57ee2a55e4c-Abstract.html).
6. Gillis, N. (2014). “The Nonnegative Matrix Factorization: A Survey.” *IEEE TPAMI*. [DOI](https://doi.org/10.1109/TPAMI.2013.2295274).
7. Chen, W. et al. (2024). “Nonnegative Matrix Factorization in Dimensionality Reduction: A Survey.” *ACM Computing Surveys*. [DOI](https://doi.org/10.1145/3767726).
8. Ahmed, I., Hu, B., Ben, X., Acharya, M. P., and Ding, Y. (2021). “Neighborhood Structure Assisted Non-negative Matrix Factorization and Its Application in Unsupervised Point-wise Anomaly Detection.” *JMLR* 22. [Paper](http://jmlr.org/papers/v22/19-924.html).
9. Hotelling, H. (1933). “Analysis of a Complex of Statistical Variables into Principal Components.” *Journal of Educational Psychology*. [DOI](https://doi.org/10.1037/h00756).
10. Cortes, C. and Vapnik, V. (1995). “Support-Vector Networks.” *Machine Learning*. [DOI](https://doi.org/10.1007/BF00994018).
11. Chang, C.-C. and Lin, C.-J. (2011). “LIBSVM: A Library for Support Vector Machines.” [Paper](https://www.csie.ntu.edu.tw/~cjlin/papers/tr.pdf).
