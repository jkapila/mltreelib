# Multiclass Tree Split Optimization: From Naive $O(N^2)$ to Streaming Histograms, Binned Gradients, and Softmax Quantization

**Date**: 2026-11-17  
**Author**: Jitin Kapila & Antigravity  
**Context**: `mltreelib-core` split engine, multiclass classification benchmarks (e.g., SDSS17 $N=62,442, C=3$)  
**Keywords**: Multiclass split criteria, Shannon entropy, Gini impurity, streaming histogram, histogram subtraction trick, LightGBM, scikit-learn, Softmax trees

---

## 1. Executive Summary & The Scalability Wall

In binary classification and continuous regression, tree split finding is naturally $O(N)$ per feature (after initial sorting or binning):
- Under Mean Squared Error (SSE), maintaining running moments $\sum w_i, \sum w_i y_i, \sum w_i y_i^2$ provides $O(1)$ child impurity evaluations at each candidate threshold.
- Under binary deviance, maintaining positive and negative sample weights similarly provides $O(1)$ updates.

However, for **multiclass classification** ($C \ge 3$) under information-theoretic criteria like Shannon Entropy ($-\sum p_c \log_2 p_c$) or Gini Impurity ($1 - \sum p_c^2$), naive implementations often introduce severe algorithmic regressions:
If child node impurities are evaluated by slicing the dataset at each candidate threshold and re-aggregating class counts over the slices, the split evaluation complexity scales as **$O(N^2 \cdot C)$** per feature.

### Concrete Impact on SDSS17
On the Sloan Digital Sky Survey benchmark (`SDSS17`, $N = 62,442$, $P = 11$, $C = 3$):
- **Naive slice evaluation**:
  $$62,442 \text{ candidate splits} \times 62,442 \text{ rows} \approx 3.9 \times 10^9 \text{ operations per column}$$
  Across 11 features: $\approx 4.3 \times 10^{10}$ operations per tree node split ($\sim 43\text{ billion operations}$).
  This causes catastrophic timeouts (> 180 seconds for a single node).
- **Streaming Class-Frequency Histogram**:
  $$62,442 \text{ candidate splits} \times 3 \text{ classes} \approx 1.87 \times 10^5 \text{ operations per column}$$
  Across 11 features: $\approx 2.0 \times 10^6$ operations ($\sim 2\text{ million operations}$).
  **Speedup**: **$\mathbf{20,800\times}$ faster**, reducing split evaluation from minutes to milliseconds.

---

## 2. Industry Architecture Comparison: How Modern Packages Solve Multiclass Splits

| Package / Algorithm | Multiclass Split Mechanism | Continuous Scan Complexity | Memory / Allocation Pattern | Handling of $C$ |
| :--- | :--- | :--- | :--- | :--- |
| **`scikit-learn`** (`_criterion.pyx`) | Streaming class histograms (`sum_left`, `sum_right`) | $O(N \log N) + O(N \cdot C)$ | In-place pre-allocated arrays of size $C$ | Direct $C$-way impurity update |
| **`LightGBM`** | 256-bin quantized histograms + Subtraction Trick | $O(N) + O(B \cdot C)$ where $B \le 256$ | Pre-allocated $B \times C$ flat buffers; zero per-node allocation | $K$ independent trees (one per class) OR vector splits |
| **`XGBoost`** | Quantile sketches + multi-threaded gradient histograms | $O(N) + O(B \cdot C)$ | Quantile sketch buffers + gradient histograms | $K$ trees per iteration via softmax gradient/hessian |
| **`CatBoost`** | Symmetric oblivious trees on quantized bins | $O(B \cdot C)$ with bitwise vectorization | Uniform trees across full depth; hardware SIMD | Multi-class cross-entropy or multi-RMSE |
| **`C4.5` / `J48` (Quinlan)** | Cumulative class frequency matrix during sorted sweep | $O(N \log N) + O(N \cdot C)$ | Static class count array $(2 \times C)$ | Multi-way entropy / Gain Ratio |
| **`Ranger` (R / C++)** | In-place pointer partitioning + fast Gini tracker | $O(N \log N) + O(N \cdot C)$ | Thread-local reusable buffers; zero copy | Twoing or Gini impurity |

---

## 3. Deep Dive into State-of-the-Art Solutions

### 3.1 Scikit-Learn: Incremental Streaming Updates (`_criterion.pyx`)

In scikit-learn's Cython engine, split criteria implement `init()`, `reset()`, and `update(new_pos)`:

```cython
# Simplified pseudo-Cython from sklearn/tree/_criterion.pyx
cdef class ClassificationCriterion:
    cdef double* sum_left
    cdef double* sum_right
    cdef double weight_left
    cdef double weight_right

    cdef void reset(self):
        # All samples initially in right node; left is empty
        for c in range(self.n_classes):
            self.sum_left[c] = 0.0
            self.sum_right[c] = self.sum_total[c]
        self.weight_left = 0.0
        self.weight_right = self.weight_total

    cdef void update(self, SIZE_t new_pos):
        # Move sample at new_pos from right child to left child in O(1)
        cdef SIZE_t sample_idx = self.samples[new_pos]
        cdef int c = <int>self.y[sample_idx]
        cdef double w = self.sample_weight[sample_idx]

        self.sum_left[c] += w
        self.sum_right[c] -= w
        self.weight_left += w
        self.weight_right -= w

    cdef double impurity_left(self):
        # Evaluated in O(C)
        cdef double entropy = 0.0
        cdef double p
        for c in range(self.n_classes):
            if self.sum_left[c] > 0.0:
                p = self.sum_left[c] / self.weight_left
                entropy -= p * log2(p)
        return entropy
```

**Key Advantage**: Zero memory allocation in the split search loop. Impurity computation is reduced to $C$ floating-point operations per threshold candidate.

### 3.2 LightGBM: 256-Bin Quantized Histograms & The Subtraction Trick

LightGBM avoids sorting continuous values during tree building by discretizing features into $B \le 256$ bins (fitting into `uint8` indices):
1. **Histogram Construction ($O(N)$)**:
   For node $P$, iterate over all samples $i \in P$:
   $$\text{Hist}[x_{i, \text{bin}}, y_i] \mathrel{+}= w_i$$
2. **Threshold Scan ($O(B \cdot C)$)**:
   Iterate over $b \in [0, B-1]$:
   $$\text{LeftHist}[c] \mathrel{+}= \text{Hist}[b, c], \quad \text{RightHist}[c] \mathrel{-}= \text{Hist}[b, c]$$
   Evaluate gain across the $B-1$ bin boundaries. With $B = 256$ and $C = 3$, scanning a feature requires only $256 \times 3 = 768$ operations, completely decoupled from $N$.
3. **Histogram Subtraction Trick ($O(B \cdot C)$)**:
   When splitting node $P \to (L, R)$, LightGBM only constructs the histogram for the smaller child from data (e.g. $L$ where $|L| \le |R|$). The larger child $R$'s histogram is computed via element-wise subtraction:
   $$\text{Hist}_R[b, c] = \text{Hist}_P[b, c] - \text{Hist}_L[b, c]$$
   This cuts tree construction cost across the entire forest by 50%.

### 3.3 Gradient-Boosted Multiclass Trees: Softmax Log-Loss Reformulation

Modern gradient-boosted trees rarely split on pure Shannon entropy or Gini impurity. Instead, they optimize multinomial log-loss directly using 2nd-order Taylor expansions:

$$\mathcal{L} = - \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log p_{i,c}, \quad \text{where } p_{i,c} = \frac{e^{f_{i,c}}}{\sum_{k=1}^C e^{f_{i,k}}}$$

1. **Gradients and Hessians**:
   $$g_{i,c} = p_{i,c} - y_{i,c}$$
   $$h_{i,c} = p_{i,c}(1 - p_{i,c})$$
2. **Architecture A: One-Tree-Per-Class ($K$-Trees / Iteration)**:
   At each boosting round, train $C$ separate trees. For class $c$, the tree splits on the scalar gradients $g_{i,c}$ and hessians $h_{i,c}$:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in L} g_{i,c})^2}{\sum_{i \in L} h_{i,c} + \lambda} + \frac{(\sum_{i \in R} g_{i,c})^2}{\sum_{i \in R} h_{i,c} + \lambda} - \frac{(\sum_{i \in P} g_{i,c})^2}{\sum_{i \in P} h_{i,c} + \lambda} \right]$$
   This completely reduces multiclass tree building to standard fast scalar tree building.
3. **Architecture B: Vector-Leaf Multiclass Trees**:
   A single tree is built where each leaf node predicts a $C$-dimensional vector of logits. The split criterion aggregates gradient gains across all $C$ dimensions:
   $$\text{Gain}_{\text{vector}} = \sum_{c=1}^C \text{Gain}_c$$
   This keeps tree depth and structural splits identical across all classes, maintaining joint representation while scaling linearly with $C$.

---

## 4. Analytical Comparison: Computational Cost per Continuous Feature

| Method | Candidate Thresholds | Cost per Threshold | Total Continuous Scan Cost |
| :--- | :--- | :--- | :--- |
| **Naive Slicing (Current Core)** | $N$ | $O(N)$ | $O(N^2)$ ($\approx 3.9 \times 10^9$ on SDSS17) |
| **Streaming Histogram (Proposed)** | $N$ | $O(C)$ | $O(N \cdot C)$ ($\approx 1.87 \times 10^5$ on SDSS17) |
| **Quantized Bins (LightGBM style)** | $B \le 256$ | $O(C)$ | $O(N) + O(B \cdot C)$ ($\approx 6.3 \times 10^4$ on SDSS17) |
| **Quantized Bins + Subtraction** | $B \le 256$ | $O(C)$ | $O(N/2) + O(B \cdot C)$ (amortized) |

---

## 5. Architectural Implementation Plan for `mltreelib-core`

### Phase 1: Streaming Histogram for `GainFn::Entropy` and `GainFn::Gini` (Immediate)
In `crates/mltreelib-core/src/split.rs`:
- Pre-allocate stack arrays `[f64; 32]` for `total_counts`, `left_counts`, and `right_counts` inside `SplitScratch`.
- For $C \le 32$ (covers virtually all tabular classification benchmarks):
  1. Accumulate total class counts:
     $$\text{total}[c] = \sum_{r} w_r \cdot \mathbb{I}(y_r = c)$$
  2. In the threshold scan loop, as candidate $k$ moves from right to left:
     $$\text{left}[y_k] \mathrel{+}= w_k, \quad \text{right}[y_k] \mathrel{-}= w_k$$
  3. Calculate entropy directly from `left` and `right` arrays in $O(C)$ via the identity:
     $$H(W, \{w_c\}) = \log_2(W) - \frac{1}{W} \sum_{c: w_c > 0} w_c \log_2(w_c)$$
- Eliminate the naive `gain_of(gfn, &yws_sorted[..k+1], &yws_sorted[k+1..])` fallback.

### Phase 2: Vector-Leaf Multiclass Softmax Splitting (Roadmap)
- When `leaf_family == LeafFamily::Softmax`, extend `SplitEval::Fast` to scan multi-class gradients directly rather than Shannon entropy.
- Eliminates class sorting and logarithmic evaluations during splitting, matching LightGBM / CatBoost speed.

---

## 6. References

1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC press.
2. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830. [Cython split engine: `sklearn/tree/_criterion.pyx`].
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS 2017)*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*.
6. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*.
7. Wright, M. N., & Ziegler, A. (2017). ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R. *Journal of Statistical Software*, 77(1).
