# mltreelib Example Library & Benchmark Catalog

A structured directory of executable, self-contained examples, authoritative suites, and benchmark harnesses for `mltreelib`.

---

## Directory Overview

```text
examples/
├── quickstart.py                      <-- Start Here: 10-second end-to-end Concrete walkthrough
├── gen_api_doc.py                     <-- Tooling: Generates docs/reference/api/*.md
│
├── suites/                            <-- The 8 Consolidated Authoritative Feature Suites
│   ├── core_tree_architecture.py      (Suite 1: Two-Control decoupling, 0D stats, Chow gate)
│   ├── constraints_and_honesty.py     (Suite 2: Monotone splits/slopes, honesty, interaction isolation)
│   ├── missing_routing_benchmark.py   (Suite 3: Native missing data routing & surrogates on Scania APS)
│   ├── distributional_leaf_benchmarks.py (Suite 4: Gamma, Poisson, Tweedie, Tobit MLE leaves)
│   ├── stump_leaf_retrain.py          (Suite 5: Microsecond warm-start leaf retraining & Huber breakdown)
│   ├── statistical_and_metric_splitting.py (Suite 6: Chow, oblique ridge, Gain Ratio, Twoing, SDR)
│   ├── timeseries_additive_forecasting.py (Suite 7: Rust calendar features, day lags, trend extrapolation)
│   ├── unsupervised_trees.py          (Suite 8: Predictive Clustering Trees & Extended Isolation Forest)
│   └── solver_benchmarks.py           (Solver Pareto: Cholesky, L-BFGS, CCD, SHADE, CMA-ES, GA, sNES)
│
├── survival/                          <-- Time-to-Event & Competing Risks Suite
│   ├── competing_risks_pbc.py         (Multi-cause Gray's test, Aalen-Johansen CIF simplex identity)
│   ├── two_stage_survival.py          (Bank customer churn: Segregate-first 2-stage cohort discovery)
│   └── flchain_mortality.py           (FLCHAIN clinical trial follow-up with native missing creatinine)
│
└── benchmarks/                        <-- Large-Scale Arenas & Optimization Studies
    ├── elo/                           (TabArena 51-dataset Elo rating tournament runners)
    ├── hpo/                           (Optuna hyperparameter tuning routines & shapes)
    └── scale/                         (Large-scale speed, throughput, and memory scaling scans)
```

---

## 1. Quickstart

Run from repository root:

```bash
uv run python examples/quickstart.py
```

- **Dataset**: Yeh Concrete Compressive Strength (1,030 rows, 8 features).
- **Execution**: Fits standard CART vs `mltreelib` linear `ModelTree` in under 2 seconds.
- **Output**: Prints rpart-style ASCII tree, exports `docs/assets/quickstart_tree.dot`, and saves `docs/assets/quickstart_tree_vs_flat.png`.

---

## 2. The 8 Consolidated Suites (`examples/suites/`)

Each suite corresponds to an authoritative chapter in [docs/examples/](../docs/examples/index.md):

| Script | Documentation Guide | Key Mechanics Demonstrated |
| :--- | :--- | :--- |
| `core_tree_architecture.py` | [Core Architecture](../docs/examples/core-tree-architecture.md) | Decoupled `ModelTree` $\leftrightarrow$ `LeafControl`, 0D leaf stats, Chow hypothesis stopping. |
| `constraints_and_honesty.py` | [Constraints & Honesty](../docs/examples/constraints-and-honesty.md) | Dual monotone split and slope constraints, interaction isolation, sample-split honesty ($S^{\text{tr}}$ vs $S^{\text{est}}$). |
| `missing_routing_benchmark.py` | [Missing Data Routing](../docs/examples/missing-data-routing.md) | Learned `missing_left` direction, surrogate splits (`maxsurrogate=N`), zero-imputation triage on Scania APS. |
| `distributional_leaf_benchmarks.py` | [Distributional Leaves](../docs/examples/distributional-leaves.md) | Gamma, Poisson, Tweedie, and Tobit censored MLE leaves correcting OLS attenuation by 60%+. |
| `stump_leaf_retrain.py` | [Dynamic Leaf Retraining](../docs/examples/hierarchical-retrain.md) | Microsecond warm-start leaf retraining, Huber loss breakdown resistance (25.8 vs 54.3 MPa under sensor glitch). |
| `statistical_and_metric_splitting.py` | [Statistical & Metric Splitting](../docs/examples/statistical-and-metric-splitting.md) | Chow structural break gate, oblique ridge splits, Gain Ratio, Twoing, SDR, and Quinlan rule extraction. |
| `timeseries_additive_forecasting.py` | [Time Series Forecasting](../docs/examples/timeseries-forecasting.md) | Pure Rust calendar extraction, entity-grouped calendar day lags, additive decomposition, trend extrapolation. |
| `unsupervised_trees.py` | [Unsupervised Trees](../docs/examples/unsupervised-trees.md) | Predictive Clustering Trees (PCT) with rule extraction, Extended Isolation Forest (EIF) with oblique cuts. |
| `solver_benchmarks.py` | [Leaf Solver Benchmarks](../docs/examples/solver-benchmarks.md) | Pareto frontier of fit latency vs accuracy across Analytical, Quasi-Newton, Coordinate, and Evolutionary solvers. |

---

## 3. Survival & Competing Risks Suite (`examples/survival/`)

Dedicated time-to-event workflows implementing Spec 10:

- **`competing_risks_pbc.py`**:
  - Multi-cause setting: Cause 1 (Primary Relapse) vs Cause 2 (Competing Toxicity).
  - Evaluates Gray's subdistribution log-rank test (`gain="gray"`).
  - Demonstrates Aalen-Johansen multi-state Cumulative Incidence Functions (CIF) satisfying exact probability simplex invariance $\sum_{c=1}^J \hat{F}_c(t) + \hat{S}(t) = 1.0$.
  - Generates [docs/assets/competing_risks_cif_curves.png](../docs/assets/competing_risks_cif_curves.png).

- **`two_stage_survival.py`**:
  - Two-Stage Hierarchical Survival: Segregate First, Model Later.
  - Stage 1: Segment 10,000 bank customers into 4 transparent churn risk cohorts via `ModelTree`.
  - Stage 2: Fit non-parametric Kaplan-Meier step functions per leaf.
  - Generates [docs/assets/two_stage_survival_cohorts.png](../docs/assets/two_stage_survival_cohorts.png).

- **`flchain_mortality.py`**:
  - Real clinical follow-up on 7,874 patients from the Mayo Clinic FLCHAIN cohort.
  - Demonstrates native missing value routing on 1,350 records with missing creatinine without imputation.
  - Reports time-dependent IPCW Brier scores and C-indices at 1, 3, and 5 years.
  - Generates [docs/assets/guides/real-survival.svg](../docs/assets/guides/real-survival.svg).

---

## 4. Benchmarks & Arenas (`examples/benchmarks/`)

- **`elo/`**:
  - Full TabArena (51 datasets) automated Elo tournament runner and rating calculator.
  - `elo_round.py`: Runs multi-model rounds and records pairwise wins/losses.
  - `elo_report.py`: Aggregates pairwise game matrices and computes Bradley-Terry Elo ratings.
- **`hpo/`**:
  - Optuna hyperparameter optimization studies for regression and classification trees.
- **`scale/`**:
  - Scalability, throughput, and memory profiling across dataset sizes from 10k to 1M rows.
