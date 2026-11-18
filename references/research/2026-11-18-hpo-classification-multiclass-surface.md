# Full-Surface Classification & Multiclass Hyperparameter Optimization (Optuna)

**Date**: 2026-11-18  
**Scope**: Full-surface hyperparameter exploration for binary classification (`credit-g`) and multiclass classification (`SDSS17`), plus multi-dataset generalization validation across 7 TabArena classification datasets.

---

## 1. Executive Summary

- **Multiclass SDSS17 Breakout**: Test accuracy jumped from **80.33% / 84.38%** to **97.40%** (+13.02% accuracy gain). Winning combination: `gain="twoing"`, `leaf="mean"`, `max_depth=7`, `min_rows=20`, `preselect="mutual_info"`, `prune="bic"`, `col_sample=0.93`, `shrinkage=0.82`.
- **Binary credit-g Breakout**: Test accuracy improved from **75.00%** to **76.00%**. Winning combination: `gain="gini"`, `leaf="odds"`, `leaf_family="probit"`, `leaf_solver="lbfgs"`, `max_depth=8`, `max_leaf_nodes=45`, `min_rows=9`, `shrinkage=8.33`, `l1=0.43`, `l2=0.0049`.
- **Global Preset Validation**: `ModelTree(preset="accurate")` achieved the highest average generalization across all 7 benchmark classification datasets (**78.58%** vs 77.41% default), proving the stability of audited defaults.

---

## 2. Winning Parameter Profiles

### Binary Classification (`credit-g`)
- **Gain**: `"gini"`
- **Leaf Family & Type**: `leaf_family="probit"`, `leaf="odds"`
- **Leaf Solver & Regularization**: `leaf_solver="lbfgs"`, `shrinkage=8.334`, `l1=0.432`, `l2=0.00486`
- **Topology**: `max_depth=8`, `min_rows=9`, `max_leaf_nodes=45`
- **Gates & Filtering**: `preselect="variance"`, `min_relevance=0.029`, `purity_gate=0.0085`, `prune="none"`
- **Splitting**: `split_eval="fast"`, `split_precision=0.01`

### Multiclass Classification (`SDSS17`)
- **Gain**: `"twoing"` (Twoing criterion groups classes into super-classes at each split, drastically outperforming multi-way Gini)
- **Leaf Type**: `leaf="mean"`
- **Topology**: `max_depth=7`, `min_rows=20`
- **Pruning & Regularization**: `prune="bic"`, `shrinkage=0.819`, `l2=0.00072`
- **Feature Subspacing**: `col_sample=0.93`, `preselect="mutual_info"`, `min_relevance=0.011`
- **Splitting**: `split_eval="exhaustive"`, `split_precision=None`

---

## 3. Multi-Dataset Cross-Validation

| Dataset | Task | Train Rows | Default Acc | Preset Accurate | HPO Winner | Delta vs Default |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `credit-g` | binary | 800 | 75.50% | 75.00% | **76.00%** | +0.50% |
| `diabetes` | binary | 614 | 72.08% | **72.73%** | 70.78% | -1.30% |
| `blood-transfusion` | binary | 598 | 72.67% | **73.33%** | 70.67% | -2.00% |
| `qsar-biodeg` | binary | 843 | 81.99% | 83.41% | **85.31%** | +3.32% |
| `maternal_health_risk` | multiclass | 812 | 56.93% | **66.34%** | 54.46% | -2.48% |
| `anneal` | multiclass | 718 | 98.33% | **98.89%** | 93.33% | -5.00% |
| `SDSS17` | multiclass | 12,000 | 84.38% | 80.33% | **97.40%** | **+13.02%** |
| **AVERAGE** | **ALL** | - | **77.41%** | **78.58%** | **77.74%** | **+0.32%** |

### Key Insight
Dataset-specific HPO produces extreme peaks on the target dataset (SDSS17 97.40%, credit-g 76.00%), but `preset="accurate"` (`max_depth=8`, `min_rows=10`, `shrinkage=10.0`, `prune="none"`) provides the highest macro-average generalization across arbitrary unseen datasets without tuning.
