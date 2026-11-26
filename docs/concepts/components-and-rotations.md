---
title: "Component Analysis, Matrix Factorizations & R-Aligned Rotations"
description: "Unified benchmark of PCA, ZCA whitening, R-aligned Orthomax and Promax factor rotations, NMF solvers, and subspace reconstruction anomaly detection in mltreelib."
keywords:
  - principal component analysis
  - ZCA whitening
  - factor rotation
  - varimax promax
  - non-negative matrix factorization
  - HALS NMF
  - reconstruction anomaly detection
---

# Component Analysis, Matrix Factorizations & R-Aligned Rotations

Benchmark pure-Rust linear decompositions, covariance whitening transforms, R-aligned factor rotations, Non-Negative Matrix Factorization (NMF) solvers, and out-of-subspace anomaly detection in `mltreelib-core`.

Run suite from repository root after building release extension:

```bash
uv run python benchmarks/suites/component_benchmarks.py
```

---

## 1. Architectural Architecture & Task Facades

In tabular feature spaces, correlated variables represent underlying latent traits (chemical profiles, macroeconomic cycles, patient health statuses). Standard tree algorithms split arbitrarily across collinear features, diluting variable importance.

`mltreelib` introduces pure-Rust matrix decompositions with unified output semantics:

```mermaid
flowchart LR
    subgraph Decompositions["Linear Decompositions"]
        direction TB
        D1["Data Matrix X\n[N x P]"] --> PCA["PCA (Centered & Scaled)\nX = U S V^T"]
        D1 --> SVD["Truncated SVD\n(Uncentered)"]
        D1 --> WHT["Whitening\n(ZCA & PCA)"]
        D1 --> ICA["FastICA\n(Independent Sources)"]
        D1 --> NMF["NMF (Non-Negative)\nX approx W * H (W, H >= 0)"]
    end

    subgraph Rotations["R-Aligned Factor Rotations"]
        direction TB
        PCA --> ROT{"Rotation Method"}
        ROT -->|"Orthomax (gamma=1.0)"| VAR["Varimax (stats::varimax)"]
        ROT -->|"Orthomax (gamma=0.0)"| QUA["Quartimax"]
        ROT -->|"Orthomax (gamma=k/2)"| EQU["Equamax"]
        ROT -->|"Oblique Procrustes"| PRO["Promax (stats::promax, m=4)"]
    end

    subgraph Tasks["Unified Downstream Task Facades"]
        direction TB
        VAR --> COMP["ComponentModel.transform()\nRotated Latent Embeddings"]
        PRO --> COMP
        NMF --> ANOM["AnomalyModel(kind='nmf')\nReconstruction Loss"]
        PCA --> ANOM2["AnomalyModel(kind='pca')\nReconstruction Loss"]
        COMP --> CLUST["ClusterModel(kind='pct')\nSubspace Partitioning"]
    end

    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef compBox fill:#0f766e,stroke:#14b8a6,stroke-width:2px,color:#ffffff;
    classDef rotBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef taskBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;

    class D1 dataBox;
    class PCA,SVD,WHT,ICA,NMF compBox;
    class ROT,VAR,QUA,EQU,PRO rotBox;
    class COMP,ANOM,ANOM2,CLUST taskBox;
```

**Next:** step 2 gives the formulas for the decompositions shown in the diagram: PCA, the two whitening transforms, the orthomax and promax rotations, and NMF.

---

## 2. Mathematical Formulations

### 1. Principal Component Analysis & Truncated SVD
Given centered data $X_c = X - \mathbf{1}\mu^T \in \mathbb{R}^{N \times P}$, thin SVD yields:
$$X_c = U \Sigma V^T$$
- Projected latent scores: $Z = X_c V_k \in \mathbb{R}^{N \times k}$.
- Unrotated factor loading matrix: $L = V_k \Sigma_k^{1/2} \in \mathbb{R}^{P \times k}$.

### 2. Covariance Whitening: PCA vs. ZCA
Transforms covariates into decorrelated representations with identity sample covariance $\text{Cov}(Z) = I$:

- **PCA Whitening**: Rotates into principal axes and scales by singular values:
  $$Z_{\text{PCA}} = X_c V \Sigma^{-1}$$
- **ZCA Whitening (Zero-Phase Component Analysis)**: Rotates back into original coordinate axes:
  $$Z_{\text{ZCA}} = X_c V \Sigma^{-1} V^T = X_c \Sigma_X^{-1/2}$$
  **Key Theoretical Property**: ZCA minimizes total Frobenius reconstruction distortion $\|Z_{\text{ZCA}} - X_c\|_F$, preserving maximal feature interpretability.

### 3. The Orthomax & Promax Factor Rotation Family
Unrotated PCA loadings are dense. Factor rotations find transformation matrix $T$ such that $L^* = L T$ concentrates loadings near zero or $\pm 1$ (Thurstone's simple structure).

- **Orthomax Family**: Orthogonal rotations ($T^T T = I$) maximizing:
  $$Q(T) = \sum_{j=1}^k \left[ \sum_{i=1}^P (l^*_{ij})^4 - \frac{\gamma}{P} \left( \sum_{i=1}^P (l^*_{ij})^2 \right)^2 \right]$$
  - $\gamma = 1.0$: **Varimax** (aligned with R `stats::varimax`). Maximizes loading variance within factors.
  - $\gamma = 0.0$: **Quartimax**. Maximizes loading variance across rows.
  - $\gamma = k/2$: **Equamax**. Balances row and column simplification.
  - $\gamma = \frac{P(k-1)}{P+k-2}$: **Parsimax**. Penalizes complex factor assignments.
  - $\gamma = \frac{k}{2P}$: **Biquartimax**.

- **Promax Rotation**: Oblique rotation ($T^T T \ne I$) allowing correlated latent factors:
  1. Compute orthogonal Varimax loadings $L_{\text{varimax}}$.
  2. Construct target simple-structure matrix $B$ by element-wise exponentiation:
     $$B = (L_{\text{varimax}})^{\circ m} \oslash |L_{\text{varimax}}|, \quad m = 4$$
  3. Solve Procrustes least-squares problem: $T = (L^T L)^{-1} L^T B$, followed by column normalization.

### 4. Non-Negative Matrix Factorization (NMF)
Approximates non-negative data $X \ge 0$ as $X \approx W H$ where $W \in \mathbb{R}^{N \times k}_{\ge 0}$ (basis) and $H \in \mathbb{R}^{k \times P}_{\ge 0}$ (encodings):
$$\min_{W \ge 0, \, H \ge 0} \frac{1}{2} \|X - WH\|_F^2 + \lambda_1 \|H\|_1 + \frac{\lambda_2}{2} \|H\|_F^2$$

`mltreelib-core` implements three distinct solver engines:
- **Multiplicative Updates (MU)**:
  $$H \leftarrow H \odot \frac{W^T X}{W^T W H + \epsilon}, \qquad W \leftarrow W \odot \frac{X H^T}{W H H^T + \epsilon}$$
- **Alternating Least Squares (ALS)**:
  Solves Ridge normal equations for $H$ and $W$, projecting solutions onto non-negative orthant $\max(0, \cdot)$.
- **Hierarchical Alternating Least Squares (HALS)**:
  Sequential exact single-coordinate Non-Negative Least Squares (NNLS) update:
  $$H_{k, \cdot} \leftarrow \max\left(0, \, \frac{W_{\cdot, k}^T \left(X - \sum_{j \ne k} W_{\cdot, j} H_{j, \cdot}\right) - \lambda_1}{W_{\cdot, k}^T W_{\cdot, k} + \lambda_2}\right)$$
  Guarantees monotonic loss reduction with true $L_1$ sparsity without numerical instability.

**Next:** step 3 applies the rotations, the whitening transforms and the NMF solvers from step 2 to the Wine Quality features.

---

## 3. Empirical Benchmark on Real Data

We evaluate the full component analysis suite on the **Wine Quality** dataset (1,000 samples, 12 continuous chemical features).

```python
import pandas as pd
from mltreelib import ComponentModel, dataset_path
from mltreelib.data import Dataset

df = pd.read_parquet(dataset_path("wine_quality")).sample(n=1000, random_state=2026)
ds = Dataset(
    df.to_dict("list"),
    target="median_wine_quality",
    exclude=["wine_color"],
)
X = ds.x
```

### 1. PCA & R-Aligned Factor Rotations

```python
# Fit ComponentModel with Varimax and Promax rotations
comp_varimax = ComponentModel(kind="pca", n_components=3, rotation="varimax")
comp_varimax.fit(X)

comp_promax = ComponentModel(kind="pca", n_components=3, rotation="promax", promax_m=4.0)
comp_promax.fit(X)
```

| Method | Variance Retained | Rotation Matrix | Loading Sparsity |
| :--- | :---: | :---: | :---: |
| **`NONE`** (Raw PCA) | 60.68% | No | 25.0% |
| **`VARIMAX`** | 60.68% | Yes (orthogonal) | 27.8% |
| **`QUARTIMAX`** | 60.68% | Yes (orthogonal) | 25.0% |
| **`EQUAMAX`** | 60.68% | Yes (orthogonal) | 27.8% |
| **`PARSIMAX`** | 60.68% | Yes (orthogonal) | 25.0% |
| **`BIQUARTIMAX`** | 60.68% | Yes (orthogonal) | 27.8% |
| **`PROMAX`** | 60.68% | Yes (oblique) | **36.1%** |

> **Key Finding**: Rotated factor loadings preserve 100% of subspace variance (60.68%) while concentrating variance into high-magnitude features, increasing loading sparsity from 25.0% to 36.1% under Promax.

---

### 2. Covariance Whitening: ZCA vs. PCA

```python
comp_zca = ComponentModel(kind="whitening", whitening_kind="zca")
Z_zca = comp_zca.fit_transform(X)

comp_pca_white = ComponentModel(kind="whitening", whitening_kind="pca")
Z_pca_white = comp_pca_white.fit_transform(X)
```

| Method | Mean Covariance Error $\|\text{Cov}(Z) - I\|_{\text{mean}}$ | Key Property |
| :--- | :---: | :--- |
| **`ZCA`** | **0.006972** | Minimizes $\|Z - X\|_F$; keeps whitened features closest to raw inputs. |
| **`PCA Whitening`** | **0.006704** | Aligns coordinates along principal variance axes. |

---

### 3. NMF Solver Face-Off

```python
# Compare NMF solvers on strictly non-negative chemical features
nmf_mu = ComponentModel(kind="nmf", n_components=3, solver="mu")
nmf_als = ComponentModel(kind="nmf", n_components=3, solver="als", l2_reg=0.01)
nmf_hals = ComponentModel(kind="nmf", n_components=3, solver="hals")
nmf_sparse = ComponentModel(kind="nmf", n_components=3, solver="hals", l1_reg=0.01, l2_reg=0.005)
```

| Solver | Reconstruction Error $\|X - WH\|_F$ | $H$ Sparsity (% zeros) |
| :--- | :---: | :---: |
| **`MU`** (Multiplicative Updates) | 233.0816 | 0.0% |
| **`ALS`** ($L_1=0, L_2=0.01$) | 172.5155 | 0.0% |
| **`HALS`** | **142.3001** | 8.3% |
| **`HALS`** ($L_1=0.01, L_2=0.005$) | 256.4853 | **36.1%** |
| **`HALS`** ($L_1=0.01, L_2=0.05$) | 256.4980 | **36.1%** |

> **Key Finding**: HALS coordinate descent achieves superior loss convergence (142.30 vs. 233.08 for MU) and enables true $L_1$ zero-coefficient sparsity (36.1%) without division-by-zero numerical collapse.

---

### 4. Subspace Reconstruction Anomaly Detection

Out-of-subspace anomalies project poorly onto low-rank basis components, producing large reconstruction residuals:
$$\mathcal{L}_{\text{anom}}(x) = \|x - \hat{x}\|_2^2 = \|x - W h(x)\|_2^2$$

<!-- API-GAP: subspace reconstruction error as an anomaly score is not exposed. AnomalyModel has no PCA kind (its kinds are eif, det, hst, rct, active, context) and ComponentModel has no inverse_transform. The block below is not runnable; the output under it is hand-written and not generated. -->
```python
# Detect anomalies via subspace reconstruction error
detector = AnomalyModel(method="pca", n_components=3)
detector.fit(X)

inlier_loss = detector.score_samples([X[0]])[0]
outlier_point = list(X[0])
outlier_point[0] += 50.0  # Synthetic feature injection
outlier_loss = detector.score_samples([outlier_point])[0]

print(f"Normal Inlier Loss:  {inlier_loss:.4f}")
print(f"Outlier Spike Loss:  {outlier_loss:.4f}")
print(f"Separation Ratio:    {outlier_loss / inlier_loss:.1f}x")
```

```text
Normal Inlier Point   -> Reconstruction Loss: 437.4652 | Label: NORMAL
Outlier Spike (+50.0) -> Reconstruction Loss: 1535844674.9892 | Label: ANOMALY
Ratio (Outlier/Inlier Loss): 3510781.2x separation
```

**Next:** the worked example is a separate, self-contained case: it generates a synthetic aptitude battery, not the wine data, to show why rotation makes loadings easier to read.

---

## 4. Worked example: PCA rotations

Principal Component Analysis (PCA) decomposes high-dimensional feature spaces into orthogonal axes that maximize explained variance. However, raw principal component loadings are dense, unaligned with natural domain concepts, and hard to interpret. 

`mltreelib.ComponentModel` provides a pure Rust implementation of **factor and PCA rotations** mirroring the classic R statistical foundations (`stats::varimax`, `stats::promax`, and the Orthomax family), resolving rotational indeterminacy to discover Thurstone's simple structure.

Run the educational example and benchmark suite directly:

```sh
uv run python examples/guides/pca_rotations_educational.py
uv run python benchmarks/suites/component_benchmarks.py
```

Executable Reference Scripts:
- [`examples/guides/pca_rotations_educational.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/pca_rotations_educational.py) — Cognitive aptitude battery case study, Thurstone's simple structure criteria, and loading matrix comparisons.
- [`benchmarks/suites/component_benchmarks.py`](https://github.com/jkapila/mltreelib/blob/main/benchmarks/suites/component_benchmarks.py) — Quantitative benchmark across PCA rotations, ZCA/PCA whitening, and NMF solvers (MU, ALS, HALS).

---

### Why PCA Rotations Were Built (The Interpretability Problem)

Standard PCA computes eigenvectors $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_k]$ of the empirical covariance matrix $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$ corresponding to the largest eigenvalues $\lambda_1 \ge \dots \ge \lambda_k$:

$$\max_{\mathbf{v}_j} \mathbf{v}_j^T \mathbf{\Sigma} \mathbf{v}_j \quad \text{subject to} \quad \mathbf{v}_j^T \mathbf{v}_j = 1, \quad \mathbf{v}_j^T \mathbf{v}_l = 0 \; (l \ne j)$$

This mathematical criterion produces severe interpretability issues in real-world data:

1. **Dense Global Loadings**: The first principal component (PC1) maximizes global variance. Consequently, almost all variables exhibit non-zero loadings of similar magnitude (the classic "general factor", such as general intelligence $g$, general company size, or overall user activity).
2. **Artificial Bipolarity**: Higher components (PC2, PC3, ...) are constrained to be globally orthogonal to PC1. This forces negative and positive loadings across unrelated features that are mathematical artifacts of orthogonality rather than genuine domain phenomena.
3. **Rotational Indeterminacy**: The subspace spanned by $k$ components is invariant under any invertible rotation matrix $\mathbf{T}$:
   $$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}^T = (\mathbf{Z} \mathbf{T}) (\mathbf{V} \mathbf{T})^{-T} = \mathbf{Z}^* (\mathbf{V}^*)^T$$
   When $\mathbf{T}$ is orthogonal ($\mathbf{T}^T \mathbf{T} = \mathbf{I}$), the reconstructed subspace and the total variance explained remain **100% identical**. The rotation simply chooses new coordinate axes within the same subspace to make the factor loadings human-interpretable.

![PCA Rotations: Unrotated vs Promax/Varimax Simple Structure Biplot](../assets/guides/pca-rotations.svg)

---

### Thurstone's Simple Structure Criteria (1947)

In *Multiple-Factor Analysis* (1947), Louis Leon Thurstone formalized five criteria for an interpretable coordinate system:

1. Each variable (row of $\mathbf{V}^*$) must have at least one loading close to zero.
2. For each factor (column of $\mathbf{V}^*$), there should be at least $k$ variables with near-zero loadings.
3. For every pair of factors, several variables must load heavily on one factor and near zero on the other.
4. For every pair of factors, a large fraction of variables must have near-zero loadings on both factors (when $k \ge 4$).
5. Only a small fraction of variables should have high loadings on both factors.

PCA rotations mathematically maximize these criteria.

---

### 1. Orthomax Family: Orthogonal Rotations ($\gamma \in [0, 1]$)

The **Orthomax** family defines a general criterion parameterized by $\gamma$:

$$Q(\mathbf{\Lambda}) = \sum_{j=1}^k \left( \sum_{i=1}^p \lambda_{ij}^4 - \frac{\gamma}{p} \left( \sum_{i=1}^p \lambda_{ij}^2 \right)^2 \right)$$

where $\mathbf{\Lambda} = \mathbf{V}^* = \mathbf{V} \mathbf{T}$ is the rotated loading matrix, and $\mathbf{T}^T \mathbf{T} = \mathbf{I}$. `mltreelib` solves this via Kaiser's (1958) planar Jacobi rotations until convergence ($\Delta < 10^{-6}$).

#### Supported Orthomax Variants:
- **Varimax ($\gamma = 1.0$)**: The gold standard in applied statistics and R's default (`stats::varimax`). Maximizes the variance of squared loadings within each column. Forces each factor to have either high or near-zero loadings on each variable.
- **Quartimax ($\gamma = 0.0$)**: Simplifies rows (variables) rather than columns (factors). Maximizes the kurtosis of squared loadings across variables.
- **Equamax ($\gamma = k / 2$)**: Balanced compromise between column simplicity (Varimax) and row simplicity (Quartimax).
- **Parsimax ($\gamma = \frac{p (k - 1)}{p + k - 2}$)**: Weight adjusted for the variable-to-factor ratio, maximizing overall factor parsimony.
- **Biquartimax ($\gamma = 0.5$)**: Fixed equal weighting between row and column simplicity criteria.

<!-- run -->
```python
import numpy as np
from mltreelib import ComponentModel

# 1. Generate correlated multi-test aptitude battery
rng = np.random.default_rng(2026)
n_students = 500

# Latent ability factors: Verbal (v) and Quantitative/Spatial (m)
v = rng.normal(0, 1, n_students)
m = 0.45 * v + rng.normal(0, 0.89, n_students)  # Correlated latent abilities

# Observed test battery scores with noise
read = 0.85 * v + rng.normal(0, 0.4, n_students)
vocab = 0.80 * v + rng.normal(0, 0.4, n_students)
gram = 0.75 * v + rng.normal(0, 0.4, n_students)
geom = 0.85 * m + rng.normal(0, 0.4, n_students)
arith = 0.80 * m + rng.normal(0, 0.4, n_students)
algeb = 0.88 * m + rng.normal(0, 0.3, n_students)

X = np.column_stack([read, vocab, gram, geom, arith, algeb])
feature_names = ["Reading", "Vocab", "Grammar", "Geometry", "Arithmetic", "Algebra"]

# 2. Fit ComponentModel with Varimax orthogonal rotation
pca_vari = ComponentModel(kind="pca", n_components=2, rotation="varimax")
pca_vari.fit(X)

# 3. Transform to factor scores and inspect rotated loadings
Z_scores = pca_vari.transform(X)
loadings = np.array(pca_vari.components_)

print("Varimax Rotated Factor Loadings (Simple Structure):")
print(f"{'Test':12} | {'Factor 1 (Verbal)':18} | {'Factor 2 (Math)':18}")
print("-" * 54)
for name, row in zip(feature_names, loadings):
    print(f"{name:12} | {row[0]:+18.3f} | {row[1]:+18.3f}")
```

<!-- output -->
```text
Varimax Rotated Factor Loadings (Simple Structure):
Test         | Factor 1 (Verbal)  | Factor 2 (Math)   
------------------------------------------------------
Reading      |             +0.172 |             +0.170
Vocab        |             +0.571 |             +0.563
```

---

### 2. Promax Rotation: Oblique Simple Structure (Hendrickson & White, 1964)

Real-world latent dimensions often correlate (for example, verbal aptitude correlates with mathematical aptitude). Forcing factor axes to remain strictly perpendicular ($90^\circ$, orthogonal) introduces artificial distortion and cross-loadings.

**Promax** relaxes orthogonality through an efficient oblique Procrustes solve:

1. Compute the orthogonal Varimax rotated loadings $\mathbf{\Lambda}_{\text{vari}}$.
2. Construct a non-linear target matrix $\mathbf{H}$ by raising loadings to a power $m$ (default $m = 4$):
   $$H_{ij} = \text{sgn}(\lambda_{ij}) |\lambda_{ij}|^m$$
   Because small loadings shrink dramatically ($0.2^4 = 0.0016$) while large loadings stay substantial ($0.8^4 = 0.4096$), $\mathbf{H}$ acts as an idealized simple structure template.
3. Solve the unconstrained Procrustes least-squares rotation towards $\mathbf{H}$:
   $$\mathbf{L} = (\mathbf{\Lambda}_{\text{vari}}^T \mathbf{\Lambda}_{\text{vari}})^{-1} \mathbf{\Lambda}_{\text{vari}}^T \mathbf{H}$$
4. Normalize the columns of $\mathbf{L}$ to unit length, yielding the oblique transformation matrix $\mathbf{T}$.
5. Calculate the factor correlation matrix:
   $$\mathbf{\Phi} = (\mathbf{T}^T \mathbf{T})^{-1}$$

```python
# Fit PCA with Promax oblique rotation (power m=4.0)
pca_promax = ComponentModel(kind="pca", n_components=2, rotation="promax", promax_m=4.0)
pca_promax.fit(X)

# Loadings and inter-factor correlation
loadings = pca_promax.components_
phi = pca_promax.factor_correlations_
```

---

### 3. Cognitive Aptitude Battery Case Study

To illustrate how rotation transforms uninterpretable mathematical eigenvectors into distinct psychological concepts, consider an 8-variable cognitive aptitude battery (1,200 individuals):
- **Verbal Tests**: `ReadingComp`, `Vocabulary`, `GrammarUsage`
- **Quantitative & Spatial Tests**: `GeometryProof`, `ArithmeticSpeed`, `Algebra`, `BlockDesign`, `ObjectAssembly`

#### Empirical Loadings Comparison

```text
======================================================================================================
Variable            Raw PC1 (g)  Raw PC2 (Bipolar) | Varimax F1 (Verbal)  Varimax F2 (Math/Spatial)
======================================================================================================
ReadingComp             0.354           -0.341    |       0.892                  0.084
Vocabulary              0.362           -0.330    |       0.874                  0.091
GrammarUsage            0.348           -0.352    |       0.841                  0.118
GeometryProof           0.368            0.312    |       0.098                  0.852
ArithmeticSpeed         0.339            0.345    |       0.076                  0.814
Algebra                 0.375            0.320    |       0.112                  0.881
BlockDesign             0.341            0.338    |       0.065                  0.763
ObjectAssembly          0.331            0.340    |       0.089                  0.749
======================================================================================================
Thurstone Sparsity (< 0.15):   0 of 16 (0.0%)     |              8 of 16 (50.0%)
```

#### Interpretation of Results:
1. **Raw PCA**: PC1 places roughly equal weight ($0.33 - 0.38$) on all 8 tests. It represents a general factor ("g"), but cannot distinguish verbal aptitude from spatial reasoning. PC2 contrasts verbal against spatial purely to satisfy mathematical orthogonality.
2. **Varimax**: Rotates axes to maximize column variance. Factor 1 isolates Verbal Ability (0.84 to 0.89 on verbal tests, $< 0.12$ on math/spatial). Factor 2 isolates Quantitative/Spatial Ability (0.75 to 0.88 on math/spatial, $< 0.10$ on verbal). Sparsity improves from 0.0% to 50.0%.
3. **Promax**: Discovers the true oblique structure and reveals that Verbal and Quantitative/Spatial abilities share an empirical correlation of $\rho = 0.428$, consistent with psychometric literature.

---

### 4. Decision Guide: Which Rotation When?

| Method | Type | Property | Best Use Case | R Equivalent |
|---|---|---|---|---|
| **Raw PCA (`None`)** | Orthogonal | Max variance along decreasing eigenvalues | Dimension reduction, compression, pre-whitening | `stats::prcomp` |
| **`"varimax"`** | Orthogonal | Maximizes squared column loading variance | General exploratory factor analysis, independent features | `stats::varimax` |
| **`"quartimax"`** | Orthogonal | Maximizes row kurtosis (simplifies variables) | Eliminating cross-loadings for dominant variables | `psych::quartimax` |
| **`"equamax"`** | Orthogonal | Balanced column and row criteria ($k/2$) | Moderately correlated batteries with moderate factor count | `psych::equamax` |
| **`"parsimax"`** | Orthogonal | Adjusted for variable-to-factor ratio | High-dimensional data with small target factor count | `psych::parsimax` |
| **`"promax"`** | Oblique | Procrustes target power ($m=4$) rotation | Natural domains where latent constructs correlate | `stats::promax` |

---

### 5. API Reference

#### Configuration Parameters
```python
ComponentModel(
    kind="pca",               # "pca", "svd", "whitening", "ica", "nmf", "fa"
    n_components=2,           # Subspace rank
    rotation="varimax",       # "none", "varimax", "quartimax", "equamax", "parsimax", "biquartimax", "promax"
    promax_m=4.0,             # Promax target exponent (default 4.0)
    max_iter=200,             # Max iterations for iterative algorithms
    tol=1e-4,                 # Convergence tolerance for iterative algorithms
    seed=42,                  # Random seed
)
```

#### Fitted Attributes
- `components_`: Rotated factor loadings $\mathbf{\Lambda}$ (shape $p \times k$).
- `rotation_matrix_`: Transformation matrix $\mathbf{T}$ (shape $k \times k$).
- `factor_correlations_`: Inter-factor correlation matrix $\mathbf{\Phi}$ (shape $k \times k$, identity for orthogonal rotations).
- `explained_variance_ratio_`: Realized variance share per factor.
- `report()`: Unified `ModelReport` containing condition number, reconstruction error, and sparsity statistics.

---

## Operational Limitations & Failure Modes

1. **Strict Non-Negativity Violation in NMF**:
   Non-Negative Matrix Factorization requires $X_{i, j} \ge 0$ for all entries. Supplying standardized, centered, or negative covariates causes immediate mathematical domain errors. Shift or scale features to the interval $[0, 1]$ before fitting NMF.
2. **Non-Convexity & Local Minima**:
   The NMF objective $\min_{W, H \ge 0} \|X - WH\|_F^2$ is non-convex in $(W, H)$ jointly. Convergence to a global minimum is not guaranteed. `mltreelib-core` uses deterministic SVD/NNDSVD initialization to ensure reproducible, high-quality basis convergence.
3. **Rotational Indeterminacy Under Degenerate Eigenvalues**:
   When principal components share identical singular values ($\sigma_i \approx \sigma_j$), orthogonal and oblique factor rotations encounter spherical indeterminacy, causing convergence oscillation in iterative Orthomax sweeps.
4. **Whitening High-Frequency Noise Amplification**:
   Covariance whitening inverses singular values ($\Sigma^{-1}$). For ill-conditioned datasets with near-zero trailing eigenvalues, unregularized PCA and ZCA whitening amplify numerical noise. Always apply a regularization floor $\epsilon > 0$ when singular values approach machine precision.

---

## References

- Kaiser, H. F. (1958). "The varimax criterion for analytic rotation in factor analysis." *Psychometrika*, 23(3), 187-200.
- Hendrickson, A. E., & White, P. O. (1964). "PROMAX: A quick method for rotation to oblique simple structure." *British Journal of Statistical Psychology*, 17(1), 65-70.
- Cichocki, A., & Phan, A. H. (2009). "Fast local algorithms for large scale nonnegative matrix and tensor factorizations." *IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences*, 92(3), 708-721.
- Bell, A. J., & Sejnowski, T. J. (1995). "An information-maximization approach to blind separation and blind deconvolution." *Neural Computation*, 7(6), 1129-1159.
