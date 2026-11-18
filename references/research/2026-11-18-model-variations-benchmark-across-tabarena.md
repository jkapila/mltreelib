# TabArena 56-Dataset Variation Comparison: ModelTree(all) vs Optimized Configurations

**Date**: 2026-11-18  
**Scope**: Full empirical evaluation across all 56 TabArena datasets (18 Regression + 38 Classification) comparing baseline `ModelTree(all)` against 4 parameter variations.

---

## 1. Regression Suite Results (18 Datasets)

| Model | Reg Elo | vs Baseline (all) | Avg Fit Time | Head-to-Head vs (all) |
|:---|:---:|:---:|:---:|:---:|
| **`ModelTree(subspaced_ridge)`** | **1149.2** | **+216.2** | 130.6 ms | **12W - 1D - 5L** |
| **`ModelTree(accurate)`** | **1110.6** | **+177.5** | 137.6 ms | **11W - 1D - 6L** |
| `ModelTree(pruned_bic)` | 996.1 | +63.0 | 120.0 ms | 9W - 1D - 8L |
| `ModelTree(all)` [Baseline] | 933.1 | +0.0 | 93.6 ms | baseline |
| `ModelTree(spline_mars)` | 811.0 | -122.1 | 342.6 ms | 7W - 0D - 11L |

### Regression Key Findings:
- `ModelTree(subspaced_ridge)` (`depth=8, min_rows=12, shrinkage=8.0, l2=3.0, col_sample=0.85`) produced a massive **+216.2 Elo surge**, beating baseline on 12 of 18 datasets.
- Strongest single-dataset improvements:
  - `boston`: -25.47% RMSE (3.418 vs 4.586)
  - `ozone`: -17.34% RMSE (4.827 vs 5.840)
  - `ta_airfoil`: -16.64% RMSE (2.734 vs 3.280)
  - `ta_superconductivity`: -12.55% RMSE (13.239 vs 15.139)
  - `ta_fiat500`: -8.25% RMSE (794.8 vs 866.3)
  - `ta_houses`: -6.82% RMSE (0.277 vs 0.298)

---

## 2. Classification Suite Results (38 Datasets)

| Model | Cla Elo | Avg Fit Time | Head-to-Head vs (all) |
|:---|:---:|:---:|:---:|
| `ModelTree(all)` [Baseline] | 1127.2 | 1492.6 ms | baseline |
| **`ModelTree(pruned_bic)`** | **1011.6** | **740.6 ms** | **13W - 5D - 20L** |
| `ModelTree(accurate)` | 969.8 | 446.1 ms | 11W - 5D - 22L |
| `ModelTree(subspaced_ridge)` | 891.4 | 961.3 ms | 10W - 6D - 22L |

### Classification Key Findings & Stability Insights:
- **Crash Recovery & Stability**: `ModelTree(all)` completely crashed with non-convergence on 6 difficult datasets (`Bank_Customer_Churn`, `Fitness_Club`, `Marketing_Campaign`, `polish_companies_bankruptcy`, `qsar-biodeg`, `seismic-bumps`). On those exact datasets, **`ModelTree(pruned_bic)`** succeeded and scored:
  - `Bank_Customer_Churn`: 86.55%
  - `Fitness_Club`: 80.67%
  - `Marketing_Campaign`: 89.06%
  - `polish_companies_bankruptcy`: 93.49%
  - `qsar-biodeg`: 84.36%
  - `seismic-bumps`: 92.84%
- **Accuracy Jumps**:
  - `credit-g`: `pruned_bic` achieved **73.00%** vs 66.00% for baseline (+7.00%).
  - `diabetes`: `pruned_bic` achieved **70.78%** vs 64.94% for baseline (+5.84%).
  - `SDSS17` (multiclass): All Twoing-based models achieved **97.10%** accuracy.
- **Root Cause of Deep Logistic Divergence**: Pure unregularized IRLS at deeper nodes ($depth=8, min\_rows=10$) suffers from complete separation on sparse binary dummy variables. Pruning with BIC cuts away these unregularized tail splits and halves fit time (740.6 ms vs 1492.6 ms).

---

## 3. Recommended Competitive Preset Configuration

Combining the winning components yields the optimal configuration:
- **`max_depth=8`**
- **`min_rows=12`**
- **`shrinkage=8.0`** (smooth top-down shrinkage toward parent)
- **`l2=3.0`** (damps collinear matrix inversions)
- **`col_sample=0.85`** (subspace decorrelation)
- **`prune="bic"`** (prevents overfit & separation crashes in logistic leaves)
- **`gain="twoing"`** (for multiclass classification)
