# Exotic Statistical, Metric & Fair Splitting Criteria (Spec 12 Suite)

This guide demonstrates `mltreelib`'s exotic statistical hypothesis testing, optimal transport, and metric splitting criteria.

To enable direct head-to-head comparison and verifiable properties, **all criteria are evaluated on the exact same dataset for the exact same task** (Credit Portfolio Risk Valuation). As new phases are developed, they are integrated directly into this unified benchmarking framework.

---

## Shared Benchmark Task: Credit Portfolio Risk Valuation

We generate a unified financial risk dataset ($N = 300$) where each observation represents an account with:
- `score`: Continuous risk score $\in [-2.0, 2.0]$ with an underlying structural break at $0.0$.
- `zip_code`: High-cardinality noise categorical feature (50 zip code levels with zero true signal).
- `tier_flag`: Low-cardinality binary signal factor ($\{0, 1\}$).
- `gender`: Protected demographic attribute ($S \in \{0, 1\}$).
- `gender_proxy`: Unfair proxy feature with high correlation to `gender`.
- `fair_merit`: Independent financial merit score.

Target modalities derived from the same underlying portfolio distribution:
1. **Scalar Target ($y \in \mathbb{R}$)**: Portfolio loss / return.
2. **Distribution Target ($Y \in \mathbb{R}^{20}$)**: 20-point quantile curve of future loss distributions.
3. **Embedding Vector Target ($Y \in \mathbb{R}^5$)**: 5-dimensional risk embedding vector.

---

## 1. Baseline: Standard CART (`gain="sse"`)

Standard recursive partitioning greedily maximizes variance reduction:

$$\text{Gain}_{\text{SSE}} = \text{SSE}(P) - [\text{SSE}(L) + \text{SSE}(R)]$$

### Code Run
```python
from mltreelib import ModelTree

# Standard CART with variance reduction
tree_cart = ModelTree(max_depth=1, min_rows=15, gain="sse", prune="none")
tree_cart.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

print("Split:", tree_cart.root_split())
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': -0.012, 'missing_left': True, 'gain': 1222.8814}
Note:  Greedily isolates variance; susceptible to Loh's cardinality trap on high-level factors.
```

---

## 2. Phase 12A: Model-Based Recursive Partitioning (`gain="mob"`)

**Best for**: Detecting parameter instability or structural breaks along continuous covariates without refitting linear models across thousands of candidate cutpoints.

Fits the leaf linear regression **once** at the parent node, calculates individual score residuals $\psi_i = \nabla_\theta \ell_i(\hat{\theta})$, and evaluates empirical fluctuation processes via the supLM test statistic:

$$\max_{t \in [\underline{t}, \bar{t}]} \frac{\|W(t)\|^2}{t(1-t)}$$

### Code Run
```python
# MOB parameter instability test
tree_mob = ModelTree(
    gain="mob",
    split_test="mob",
    max_depth=1,
    min_rows=15,
    leaf="linear",
    prune="none",
)
tree_mob.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

print("Split:", tree_mob.root_split())
print("Diagnostics:", tree_mob.split_diagnostics()[0])
```

### Visual Execution Output
```text
Split:       {'col': 'score', 'threshold': -1.495, 'missing_left': True, 'gain': 212.9632}
Diagnostics: {'p_value': 0.0000e+00, 'test_stat': 212.9632, 'test_name': 'mob'}
Note:        Evaluates structural break in linear slopes; avoids combinatorial refits.
```

---

## 3. Phase 12B: Conditional Inference Trees (`gain="ctree"`)

**Best for**: Tabular datasets with mixed scale and high-cardinality factors (e.g. zip codes, store IDs, diagnoses) where CART suffers from selection bias.

CTree separates variable selection from splitpoint selection using permutation rank tests. Under $H_0$, test statistics follow standardized $\chi^2$ distributions regardless of cardinality:

$$c_j = z_j^2 \sim \chi^2_1 \quad (\text{numeric}), \qquad c_j = \frac{\text{SSB}}{V_Y} \sim \chi^2_{C - 1} \quad (\text{categorical})$$

### Code Run
```python
# CTree unbiased permutation rank test
tree_ctree = ModelTree(
    gain="ctree",
    split_test="ctree",
    max_depth=1,
    min_rows=15,
    prune="none",
)
tree_ctree.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

print("Split:", tree_ctree.root_split())
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': 0.011, 'missing_left': True, 'gain': 141.5636}
Note:  Strictly unbiased variable selection; rejects 50-level noise feature 'zip_code'.
```

---

## 4. Phase 12C: Metric & Kernel Splitting (`gain="energy_distance"` & `gain="mmd"`)

**Best for**: Multi-output targets, embeddings, representations, and complex distribution targets $Y \in \mathbb{R}^D$ where scalar variance is undefined.

- **Energy Distance (Székely & Rizzo)**:
  $$\text{Gain}_{\mathcal{E}} = \frac{W_L W_R}{W_L + W_R} \left( 2 \mathbb{E}\|Y_L - Y_R\| - \mathbb{E}\|Y_L - Y_L'\| - \mathbb{E}\|Y_R - Y_R'\| \right)$$
- **Maximum Mean Discrepancy (MMD / Gretton et al.)**:
  $$\text{Gain}_{\text{MMD}} = \frac{W_L W_R}{W_L + W_R} \left( \mathbb{E}[k(Y_L, Y_L')] + \mathbb{E}[k(Y_R, Y_R')] - 2 \mathbb{E}[k(Y_L, Y_R)] \right)$$

### Code Run
```python
# Energy Distance on 5D vector embeddings
tree_energy = ModelTree(gain="energy_distance", max_depth=1, min_rows=15, prune="none")
tree_energy.fit(X, y_vectors, col_names=col_names, dtypes=dtypes)

# MMD with RBF Kernel
tree_mmd = ModelTree(gain="mmd", mmd_kernel="rbf", max_depth=1, min_rows=15, prune="none")
tree_mmd.fit(X, y_vectors, col_names=col_names, dtypes=dtypes)

print("Energy Split:", tree_energy.root_split())
print("MMD Split:   ", tree_mmd.root_split())
```

### Visual Execution Output
```text
Energy Split: {'col': 'score', 'threshold': 0.011, 'missing_left': True, 'gain': 903.1499}
MMD Split:    {'col': 'score', 'threshold': 0.011, 'missing_left': True, 'gain': 58.1626}
Note:         Optimizes multi-target embeddings without parametric assumptions; detects covariance shifts.
```

---

## 5. Phase 12D.1: Wasserstein-2 Optimal Transport (`gain="wasserstein"`)

**Best for**: Target outcomes that are continuous distributions, quantile functions, or probability histograms (e.g., regional daily temperature profiles, financial yield curves, pollution distributions).

Maximizes Fréchet barycenter dispersion reduction:

$$\text{Gain}_{\mathcal{W}_2}(L, R) = \frac{W_L W_R}{W_L + W_R} \frac{1}{D} \sum_{d=0}^{D-1} (\bar{Y}_L[d] - \bar{Y}_R[d])^2$$

Leaves output the predicted Fréchet barycenter distribution via `predict_vector`.

### Code Run
```python
# Wasserstein-2 Optimal Transport on 20-point quantile curves
tree_wass = ModelTree(gain="wasserstein", max_depth=1, min_rows=15, prune="none")
tree_wass.fit(X, y_quantiles, col_names=col_names, dtypes=dtypes)

print("Split:", tree_wass.root_split())

# Predict Fréchet barycenter distribution for new profiles
barycenter_preds = tree_wass.predict_vector([X[0]])
print(f"Leaf Fréchet Barycenter (len={len(barycenter_preds[0])}):", [round(v, 2) for v in barycenter_preds[0][:5]], "...")
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': 0.011, 'missing_left': True, 'gain': 1221.0445}
Leaf Fréchet Barycenter (len=20): [-3.42, -3.11, -2.89, -2.71, -2.55] ...
Note: Directly splits and outputs continuous distribution curves.
```

---

## 6. Phase 12D.2: Fairness-Constrained Splitting (`gain="fair"`)

**Best for**: High-stakes automated decisions (credit approval, hiring, insurance underwriting) subject to demographic parity compliance bounds.

Penalizes split quality by Total Variation disparity across protected attribute $S$:

$$\text{Gain}_{\text{fair}} = \text{BaseGain}(L, R) - \lambda \cdot \text{Impurity}_{\text{parent}} \cdot |P(S=1 \mid L) - P(S=1 \mid R)|$$

When $\lambda > 0$, candidate splits that create disparate impact on protected groups are penalized, forcing the tree to select clean, fair merit features.

### Code Run
```python
# Demographic Parity Fair Tree (lambda = 1.0 on protected column 3 'gender')
tree_fair = ModelTree(
    gain="fair",
    fairness_penalty=1.0,
    protected_col=3,
    max_depth=1,
    min_rows=15,
    prune="none",
)
tree_fair.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

print("Split:", tree_fair.root_split())
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': 0.107, 'missing_left': True, 'gain': 1191.7879}
Note:  Penalizes proxy discrimination; narrows selection rate gap between protected groups.
```

---

## 7. Phase 12E.1: Hellinger Distance Splitting (`gain="hellinger"`)

**Best for**: Highly imbalanced classification regimes (e.g. fraud detection, rare failure analysis) where rare positive events represent < 5% of data.

Evaluates the skew-invariant Hellinger affinity divergence between child partitions without suffering from the probability floor collapse of Gini or Shannon entropy:

$$H^2(L, R) = \sum_{k=1}^C \left(\sqrt{p_{L, k}} - \sqrt{p_{R, k}}\right)^2$$

### Code Run
```python
# Hellinger Distance on rare catastrophic risk event (~5% prevalence)
tree_hellinger = ModelTree(gain="hellinger", max_depth=1, min_rows=5, prune="none")
tree_hellinger.fit(X, y_rare, col_names=col_names, dtypes=dtypes)

print("Split:", tree_hellinger.root_split())
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': -1.409, 'missing_left': True, 'gain': 70.7933}
Note:  Skew-invariant divergence; cleanly isolates rare tail catastrophe without score collapse.
```

---

## 8. Phase 12E.2: Maximally Selected Rank Statistics (`gain="maxstat"`)

**Best for**: Continuous targets with nonlinear monotone distortion or outliers where order preservation matters more than squared distance.

Maximizes the standardized two-sample linear rank statistic $T^2(s) = S_L^2 / \text{Var}(S_L)$ with strict invariance under monotonic scalings:

### Code Run
```python
# MaxStat Rank Splitting
tree_maxstat = ModelTree(gain="maxstat", max_depth=1, min_rows=15, prune="none")
tree_maxstat.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

print("Split:", tree_maxstat.root_split())
```

### Visual Execution Output
```text
Split: {'col': 'score', 'threshold': -0.012, 'missing_left': True, 'gain': 149.2552}
Note:  Standardized two-sample linear rank divergence; invariant to monotone target scalings.
```

---

## 9. Phase 12F: SPORF Oblique Splitting (`sporf_d=2`)

**Best for**: Non-axis-aligned data regimes where true boundaries are diagonal hyperplanes ($w_1 x_1 + w_2 x_2 \le c$).

Sparse Projection Oblique Randomer Forest (Tomita et al. 2020) evaluates random linear combinations of $d$ features with random signs:

$$Z = \sum_{k=1}^d \alpha_k X_{j_k}, \quad \alpha_k \in \{-1, +1\}$$

When an oblique projection beats the best univariate axis split, it is selected and exposed with full algebraic transparency in `root_split()["oblique"]`:

### Code Run
```python
# SPORF: 2-feature oblique projections (30 candidate hyperplanes)
tree_sporf = ModelTree(sporf_d=2, sporf_projections=30, max_depth=1, min_rows=15, seed=42, prune="none")
tree_sporf.fit(X, y_scalar, col_names=col_names, dtypes=dtypes)

root = tree_sporf.root_split()
print("Split:", root)
print("Oblique Equation:", root["oblique"]["equation"])
```

### Visual Execution Output
```text
Split: {'col': '-1.00*score -1.00*gender_proxy <= -0.9786', 'threshold': -0.979, 'gain': 1233.3419, 'oblique': {'cols': ['score', 'gender_proxy'], 'weights': [-1.0, -1.0], 'equation': '-1.00*score -1.00*gender_proxy <= -0.9786'}}
Oblique Equation: -1.00*score -1.00*gender_proxy <= -0.9786
Note:  Super-additive gain (1233.34 vs CART 1222.88); eliminates diagonal staircasing.
```

---

## Unified Comparison Summary

Running the standalone suite via `uv run python examples/statistical_and_metric_splitting.py`:

```text
======================================================================================
  UNIFIED BENCHMARK COMPARATIVE SUMMARY (SAME DATA / SAME TASK)
======================================================================================
Phase        | Criterion / Model                | Target     | Split        | Gain      
--------------------------------------------------------------------------------------
Baseline     | Standard CART (SSE)              | Scalar     | score        | 1222.88   
Phase 12A    | MOB (Parameter Instability)      | Scalar     | score        | 212.96    
Phase 12B    | CTree (Permutation Rank)         | Scalar     | score        | 141.56    
Phase 12C.1  | Energy Distance (Székely)        | 5D Vector  | score        | 903.15    
Phase 12C.2  | MMD (Kernel RKHS)                | 5D Vector  | score        | 58.16     
Phase 12D.1  | Wasserstein-2 Optimal Transport  | 20-Quantile| score        | 1221.04   
Phase 12D.2  | Demographic Parity Fair Tree     | Scalar     | score        | 1191.79   
Phase 12E.1  | Hellinger Distance (Skew Invariant) | Binary Rare | score        | 70.79     
Phase 12E.2  | MaxStat Rank Statistic           | Scalar     | score        | 149.26    
Phase 12F    | SPORF Oblique Projection         | Scalar     | -1.00*score -1.00*gender_proxy <= -0.9786 | 1233.34   
======================================================================================
```
