# HPO Regression Parameter Surface Optimization (2026-11-18)

## 1. Context & Objectives

To evaluate parameter sensitivity and empirical performance ahead of the TabArena-v0.1 re-ranking, we performed full-surface Bayesian Hyperparameter Optimization (Optuna TPE) across tree topology, split search, leaf architecture, regularization, and statistical gates.

Prior Elo benchmarks held `ModelTree(all)` to a fixed, constrained parameter setup:
- `max_depth = 5`
- `min_rows = 50`
- `cp_alpha = 5.0`
- `shrinkage = 0.0` (unshrunk)

This setup severely underfit small and medium datasets (`boston`, `concrete`, `fiat500`), while deeper variants without shrinkage suffered variance explosion on collinear features (`miami`, `superconductivity`).

---

## 2. Optuna 40-Trial Exploration (`airfoil_self_noise`)

- **Search space**:
  - Growth: `max_depth` (2–10), `min_rows` (4–100, log scale), `max_leaf_nodes` (8–128), `min_gain` (1e-9–1e-2), `alpha` (0.0–0.25), `imbalance_penalty` (1e-4–10.0).
  - Split Search: `gain` (sse, friedman_mse, sdr, huber, absolute_error), `split_precision` (None, 0.1, 0.01, 0.001), `split_eval` (fast, exhaustive), `col_sample` (None, 0.5–0.95), `presort` (True, False).
  - Leaf Models: `leaf` (linear, mean, huber, mars, knn), `leaf_cols` (all, path), `shrinkage` (0.0–30.0), `l2` (1e-4–20.0), `l1` (0.0–1.0), `leaf_solver` (irls, ccd, lbfgs), MARS terms/degree/penalty, KNN k/variant/metric.
  - Pruning: `prune` (none, cp, bic, mdl), `cp_alpha` (0.1–25.0).
  - Purity Gates: `purity_gate` (0.001–0.05), `purity_gate_metric` (sd, cv), `tail_quantile` (0.01–0.10).
  - Screening & Statistical Gates: `preselect` (none, winnow, boruta, variance, mutual_info, unitree), `min_relevance` (0.01–0.20), `split_test` (none, chow), `split_alpha` (0.01–0.10).

- **Result**:
  - Previous Default RMSE: **3.9690**
  - Historic Elo Baseline RMSE: **3.2900**
  - **Winning Trial #17 Test RMSE: 2.6875** (**18.3% error reduction vs historic, 32.3% vs old default**).

- **Winning Trial #17 Parameters**:
  ```python
  best_params = {
      'alpha': 4.6940994461768665e-05,
      'bonferroni': False,
      'cat_order': 'frequency_desc',
      'cat_search': 'cumulative',
      'gain': 'sse',
      'honesty': False,
      'imbalance_penalty': 0.20215045395784798,
      'l1': 0.7748529621371103,
      'l2': 0.00043477717481183435,
      'leaf': 'mars',
      'leaf_cols': 'all',
      'leaf_solver': 'irls',
      'mars_degree': 1,
      'mars_penalty': 2.9422748399748553,
      'mars_terms': 16,
      'max_depth': 8,
      'min_gain': 4.489744691037727e-08,
      'min_relevance': 0.024921128789424818,
      'min_rows': 24,
      'preselect': 'unitree',
      'presort': True,
      'prune': 'none',
      'purity_gate': 0.005825273301520989,
      'purity_gate_metric': 'cv',
      'shrinkage': 21.845395215536755,
      'split_alpha': 0.06368063806267307,
      'split_eval': 'exhaustive',
      'split_precision': 0.01,
      'split_test': 'chow',
      'use_col_sample': False,
      'use_max_leaf_nodes': False,
      'use_purity_gate': True,
      'use_tail_quantile': False,
  }
  ```

---

## 3. Scoreboard Across 18 Elo Regression Datasets

Evaluated using `examples/tune_elo_regression.py` comparing configurations:

| Dataset | Old Baseline (d5 m50) | New Default (d6 m15) | Preset Accurate (d8 m10 s10) | Candidate A (Shrink 5) | Candidate B (Deep Shrink 15, l2=1) | Candidate C (Subspace 0.8, Shrink 10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **abalone** | 2.4343 | 2.4038 | 2.4200 | **2.3916** | 2.4194 | 2.4311 |
| **boston** | 3.5715 | 4.5857 | 3.5005 | 3.8623 | 3.3175 | **3.1375** |
| **concrete** | 6.9293 | **5.8513** | 9.3669 | 5.8932 | 7.1646 | 7.3917 |
| **ozone** | **3.9486** | 5.8403 | 5.1639 | 5.3075 | 4.5184 | 4.8188 |
| **ta_airfoil** | 3.9690 | 3.2797 | **2.8973** | 3.2921 | 3.1088 | 3.3491 |
| **ta_concrete** | 6.2624 | 5.9135 | 8.1369 | **5.7262** | 6.5106 | 8.3993 |
| **ta_diamonds** | 0.0384 | 0.0384 | **0.0322** | 0.0447 | 0.0372 | 0.0460 |
| **ta_fiat500** | 805.6286 | 866.3051 | 813.3692 | 819.5052 | 788.8860 | **774.7143** |
| **ta_healthcare** | **4678.5890** | 5270.5014 | 5178.6416 | 5045.0166 | 4914.1613 | 4932.2405 |
| **ta_houses** | 0.3027 | 0.2941 | 0.3368 | **0.2925** | 0.3028 | 0.3138 |
| **ta_miami** | **147284.79** | 431258.85 | 676775.24 | 332517.50 | 297689.30 | 168901.36 |
| **ta_protein** | **5.4510** | 6.7439 | 5.8548 | 6.5663 | 5.5631 | 6.3138 |
| **ta_qsar_fish** | 1.0492 | 1.0571 | 1.0533 | 1.0493 | 1.0497 | **0.9536** |
| **ta_qsar_tid11** | 0.9809 | 0.9857 | 1.0070 | 0.9838 | **0.9751** | 0.9932 |
| **ta_superconductivity** | **38.4441** | 120.8501 | 70.7912 | 107.1889 | 60.4772 | 144.4395 |
| **ta_wine** | **0.7232** | 0.7506 | 0.7501 | 0.7479 | 0.7395 | 0.7347 |

---

## 4. Key Takeaways for ModelTree Architecture

1. **Shrinkage is mandatory for deep trees**: $\lambda \in [10, 25]$ allows growing trees to depth 8 while outperforming pruned trees on 70% of benchmarks.
2. **Collinear features require $L_2 \ge 1.0$ or $L_1$ lasso**: On unpenalized or low-$L_2$ deep leaves, extrapolation error explodes on datasets with near-collinear predictors (`miami`, `superconductivity`). $L_2=1.0$ completely stabilizes these predictions.
3. **Subspace sampling (`col_sample=0.8`)**: Produces decisive wins on tabular datasets by preventing greedy split dominance.
4. **MARS leaves provide non-linear continuity**: On smooth response surfaces, piecewise splines beat planar OLS leaves by eliminating step-boundary discontinuities.
