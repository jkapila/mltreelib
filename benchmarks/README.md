# Elo Benchmarking & ModelTree Experimentation Guide

This guide describes how to run, manage, and extend the multi-language Elo benchmarking arena (comparing **Python, Rust, and R** models) and how to configure and benchmark experimental knobs for **`ModelTree`**.

---

## 1. Quick Start

All runs enforce a strict **300s process-group timeout** (hard kill via `SIGKILL` without thread leakage). External challengers are loaded ephemerally via `uv run --with ...`, keeping the core `mltreelib` virtualenv dependency-lean.

### Run Full Regression Suite (18 Datasets, 34 Models)
```bash
bash examples/elo_run_resume.sh
```
Or directly via Python orchestrator:
```bash
uv run \
  --with pandas --with pyarrow --with numpy \
  --with lightgbm --with catboost --with ctboost --with chimeraboost \
  --with interpret --with scikit-learn --with statsmodels \
  --with mars-earth --with trust-free --with forust --with perpetual \
  python examples/elo_score_all.py --task reg
```

### Run Full Classification Suite (24 Datasets, 28 Models)
```bash
bash examples/elo_run_cla.sh
```
Or directly:
```bash
uv run \
  --with pandas --with pyarrow --with numpy \
  --with lightgbm --with catboost --with ctboost --with chimeraboost \
  --with interpret --with scikit-learn --with statsmodels \
  --with mars-earth --with trust-free --with forust --with perpetual \
  python examples/elo_score_all.py --task cla
```
*(Note: ModelTree classification is currently undergoing internal development and will be attached to the classification board once complete.)*

### View Live Elo Leaderboards
```bash
uv run python examples/elo_report.py        # Regression Elo ratings
uv run python examples/elo_report_cla.py    # Classification Elo ratings
```

---

## 2. Managing Runs & Common Flags

| Environment Variable | Description | Example |
| :--- | :--- | :--- |
| `ELO_DATASETS` | Restrict run to specific comma-separated datasets | `ELO_DATASETS=ta_fiat500,boston` |
| `ELO_RETRY=1` | Force re-running pairs that previously scored or failed | `ELO_RETRY=1` |
| `ELO_EXCLUDE` | Exclude model name substrings (e.g. skip challengers) | `ELO_EXCLUDE=sklearn,statsmodels` |
| `--budget <sec>` | Adjust hard timeout per fit (default `300.0`s) | `--budget 60.0` |

Logs are streamed live to stdout and appended to `benchmarks/elo_run.log` (or `benchmarks/elo_cla_run.log`).

---

## 3. Data Ingestion & Categorical Taxonomy

All data in `benchmarks/elo_data/` and `benchmarks/elo_cla_data/` is kept **raw**:
- Categorical features are preserved as strings without one-hot encoding or arbitrary column capping.
- Each dataset includes `<dataset>.meta.json` defining column roles (`numeric` vs `categorical`).

In the output CSVs (`benchmarks/elo_scores.csv` and `benchmarks/elo_cla_scores.csv`), every entry tracks how categoricals were handled:
- **`internal`** (*native / as-is*): Consumes raw categories directly without driver encoding (`ModelTree`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `HistGradientBoosting`, `modeltrees`, R factors like `rpart`, `cubist`, `C5.0`, `ctree`, `glmtree`).
- **`external`** (*driver-encoded*): Requires driver-side `OrdinalEncoder` fit on train (`sklearn RF`, `sklearn GBM`, `ExtraTrees`, `sklearn tree`, `forust`, `perpetual`, `MARS`).
- **`none`** (*numeric-only*): Strictly requires numeric matrices; refuses categorical columns (`statsmodels`, `trust-free`, R `grf`).

---

## 4. `ModelTree` Knobs & Experimentation Guide

`ModelTree` (implemented in pure Rust with PyO3 bindings) exposes rich algorithmic knobs that can be systematically benchmarked.

### Available Knobs in `ModelTree(**kwargs)`

1. **Significance & Hypothesis Testing (MOB-style)**:
   - `split_test`: `"chow"` (structural break test) or `"none"`.
   - `split_alpha`: significance threshold (default `0.05`).
   - `bonferroni`: whether to apply Bonferroni candidate correction (default `false`).
   - *Result on `ta_fiat500`*: Drops RMSE from **789.07 down to 715.32**!

2. **Categorical Search Strategies**:
   - `cat_order`: `"frequency_desc"`, `"target_mean"`, or level list.
   - `cat_search`: `"cumulative"`, `"single_vs_rest"`, or `"both"`.
   - *Result on `credit-g`*: Drops logloss from **0.8776 down to 0.7156**!

3. **Honesty & Sample Splitting (grf-style)**:
   - `honesty`: `true` / `false` (structure / estimation split).
   - `honesty_fraction`: fraction of data for structure (default `0.5`).
   - *Result on `ta_fiat500`*: Drops RMSE from **789.07 down to 763.02**.

4. **Split Criteria (Gain)**:
   - `gain`: `"sse"`, `"friedman_mse"`, `"absolute_error"`, `"huber"`, `"poisson"`, `"gini"`, `"entropy"`.

5. **Leaf Models & Regularization**:
   - `leaf`: `"linear"`, `"mean"`, `"median"`, `"quantile"`, `"huber"`.
   - `l1`: Lasso penalty on leaf slopes.
   - `l2`: Ridge penalty on leaf slopes (default `0.5`).
   - `leaf_cols`: `"all"` vs `"path"` (split-path columns only).

6. **Screening & Preselection**:
   - `preselect`: `"none"`, `"winnow"`, `"boruta"`, `"shadowmax"`, `"variance"`, `"mutual_info"`, `"unitree"`.
   - `min_relevance`: threshold for screening.

### How to Add a New ModelTree Variant to the Elo Board

Edit [`examples/elo_models_reg.py`](file:///mnt/d/Programs/mltreelib/examples/elo_models_reg.py) or [`examples/elo_models_cla.py`](file:///mnt/d/Programs/mltreelib/examples/elo_models_cla.py):

```python
# In make_models():
ms["ModelTree(chow)"] = make_mt("linear", "all", split_test="chow", split_alpha=0.05, bonferroni=True)
ms["ModelTree(target_mean)"] = make_mt("linear", "all", cat_order="target_mean", cat_search="both")
ms["ModelTree(honest)"] = make_mt("linear", "all", honesty=True, honesty_fraction=0.5)
```

The orchestrator will automatically pick up the new variants on the next run.
