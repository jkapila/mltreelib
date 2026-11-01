# ELO Benchmark Arenas

One scoreboard, one protocol, every model scored on the **same prepared data**. The arena evaluates models across **Rust, Python, and R** under a strict 300s process-group timeout.

Detailed runbook &amp; ModelTree experimentation guide: [`benchmarks/README.md`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/README.md>).

---

## The Protocols

### 1\. Regression Arena

- **Data**: 18 regression benchmark datasets, one stratified 80/20 split (seed 7), prepared once by `examples/elo_prepare.py`.
- **Matchups**: Per dataset, every unordered pair of models "plays" once; lower test RMSE wins (tie = draw).
- **Scores**: Evaluates 36 models across Rust, Python, and R (`benchmarks/elo_scores.csv`).
- **Reproduce**: `uv run python examples/elo_report.py` (or `bash examples/elo_run_resume.sh`).

### 2\. Classification Arena

- **Data**: 24 TabArena datasets (20 binary, 4 multiclass) from OpenML suite 457, stratified 80/20 split within class (seed 7), prepared once by `examples/elo_prepare_cla.py` with raw strings for categoricals and `<ds>.meta.json`.
- **Matchups**: Per dataset, round-robin pairwise comparisons based on test **log loss** (cross-entropy), K=32, starting rating 1000.
- **Scores**: Evaluates 32 models across Python, Rust, and R (`benchmarks/elo_cla_scores.csv`), including RuleQuest `C5.0 (tree)` and `C5.0 (rules)` in R, `HistGradientBoosting`, `sklearn GBM`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `ranger`, `bart`, `glmtree`, and all 4 `ModelTree` variants (`all`, `path`, `mean`, `winnow`).
- **Reproduce**: `uv run python examples/elo_report_cla.py` (or `bash examples/elo_run_cla.sh`).

---

## Metadata Taxonomy

Every score tracked in the arena includes two core architectural dimensions: - **`lang`**: Model implementation language — **`Rust`**, **`Python`**, or **`R`**. - **`cat_handling`**: - **`internal`** (*native / as-is*): Direct native consumption of categoricals without driver encoding (`ModelTree`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `HistGradientBoosting`, `modeltrees`, R factors like `rpart`, `cubist`, `C5.0`, `ctree`, `glmtree`). - **`external`** (*driver-encoded*): Requires driver-side ordinal encoding fit on train (`sklearn RF`, `sklearn GBM`, `ExtraTrees`, `sklearn tree`, `forust`, `perpetual`, `MARS`). - **`none`** (*numeric-only*): Strictly requires numeric matrices; refuses categorical columns (`statsmodels`, `trust-free`, R `grf`).

---

## The Regression Board

```
== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==
+-----------+---------------------+------------------------+--------+------------+-----------+
|    ELO    |    FIT TIME (±sd)   |         MODEL          |  LANG  |    CAT     |    DATA   |
+-----------+---------------------+------------------------+--------+------------+-----------+
|   1690.0  |   12.01 s (±17.76)  |      ChimeraBoost      | Python |  internal  |   18/18   |
|   1622.3  |   18.24 s (±23.58)  |         ranger         |   R    |  internal  |   16/18   |
|   1618.1  |   19.20 s (±33.35)  |       sklearn RF       | Python |  external  |   18/18   |
|   1568.1  |   13.02 s (±33.28)  |       ExtraTrees       | Python |  external  |   18/18   |
|   1532.3  |   23.17 s (±31.44)  |        grf_llf         |   R    |    none    |   16/18   |
|   1470.6  |     0.35 s (±0.22)  |     LightGBM(200)      | Python |  internal  |   18/18   |
|   1439.3  |    7.20 s (±23.69)  |    perpetual (Rust)    |  Rust  |  external  |   17/18   |
|   1425.5  |     1.04 s (±1.10)  |     CatBoost(200)      | Python |  internal  |   18/18   |
|   1350.1  |     0.95 s (±1.91)  |  HistGradientBoosting  | Python |  internal  |   17/18   |
|   1319.3  |   65.08 s (±68.98)  |          EBM           | Python |  internal  |   15/18   |
|   1299.2  |     1.22 s (±0.86)  |     forust (Rust)      |  Rust  |  external  |   18/18   |
|   1210.8  |     0.76 s (±1.83)  |        CTBoost         | Python |  internal  |   17/18   |
|   1168.4  |   27.91 s (±39.58)  |          bart          |   R    |  internal  |   17/18   |
|   1166.7  |   18.06 s (±23.80)  |         grf_rf         |   R    |    none    |   18/18   |
|   1050.1  |   13.77 s (±20.73)  |     ModelTree(all)     |  Rust  |  internal  |   16/18   |
|   1030.9  |    7.00 s (±13.24)  |      sklearn GBM       | Python |  external  |   18/18   |
|   976.4   |   20.22 s (±47.48)  |         lmtree         |   R    |  internal  |   14/18   |
|   972.9   |   15.29 s (±51.54)  |   modeltrees lmtree    | Python |  internal  |   14/18   |
|   958.1   |   29.02 s (±57.11)  |        glmtree         |   R    |  internal  |   14/18   |
|   928.3   |     5.95 s (±8.90)  |   ModelTree(winnow)    |  Rust  |  internal  |   16/18   |
|   892.7   |    6.23 s (±15.19)  |         cubist         |   R    |  internal  |   17/18   |
|   873.1   |   20.49 s (±31.78)  |   MARS (mars-earth)    | Python |  external  |   10/18   |
|   761.5   |   20.60 s (±51.49)  |         mboost         |   R    |  internal  |   15/18   |
|   697.6   |     0.33 s (±0.68)  |    statsmodels OLS     | Python |    none    |    8/18   |
|   690.7   |   10.19 s (±27.50)  |         earth          |   R    |  internal  |   17/18   |
|   681.8   |     0.09 s (±0.10)  |          KNN           | Python |  external  |   18/18   |
|   655.7   |   21.17 s (±59.09)  |         ctree          |   R    |  internal  |   17/18   |
|   611.2   |    14.37 s (±7.33)  |       trust-free       | Python |    none    |    4/18   |
|   584.5   |     0.09 s (±0.23)  |     sklearn Linear     | Python |  external  |   18/18   |
|   547.3   |   10.80 s (±21.74)  |    ModelTree(path)     |  Rust  |  internal  |   18/18   |
|   539.8   |     0.34 s (±0.57)  |      sklearn tree      | Python |  external  |   18/18   |
|   527.1   |   17.48 s (±43.02)  |    statsmodels RLM     | Python |    none    |    8/18   |
|   482.0   |    5.34 s (±20.14)  |    ModelTree(mean)     |  Rust  |  internal  |   18/18   |
|   368.3   |   22.52 s (±59.15)  |  statsmodels QuantReg  | Python |    none    |    7/18   |
|   289.3   |     1.34 s (±3.71)  |         rpart          |   R    |  internal  |   17/18   |
+-----------+---------------------+------------------------+--------+------------+-----------+
```

### Interactive Regression Board (18 Datasets × 35 Models)

Click any column header to sort. The left key columns stay frozen while scrolling horizontally. **Green cells** indicate the best RMSE on that dataset; **blue** indicates the fastest model; **orange** indicates the Elo leader.

| ELO | FIT TIME (±sd) | MODEL | abalone | boston | concrete | fc2008 | ozone | airfoil | ta\_concrete | diamonds | fiat500 | food\_delivery | healthcare | houses | miami | protein | qsar\_fish | qsar\_tid11 | superconductivity | wine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1690.0** | 12.01±17.76s | **ChimeraBoost** 18/18 | 2.289 | 2.727 | 5.521 | 197.242 | 4.011 | 1.712 | 3.901 | 0.064 | 691.590 | **7.236** | 4425.975 | **0.208** | **85052.764** | 3.446 | 0.951 | **0.742** | 8.883 | 0.623 |
| **1622.3** | 18.24±23.58s | **ranger** 16/18 | 2.310 | 2.643 | 7.290 | 204.279 | 3.892 | · | 5.732 | 627.707 | **677.179** | 7.726 | 4634.886 | 0.238 | 101302.655 | 3.506 | · | 0.750 | 9.060 | 0.612 |
| **1618.1** | 19.20±33.35s | **sklearn RF** 18/18 | 2.298 | 3.216 | 6.060 | 201.606 | 3.881 | 1.838 | 4.847 | 0.171 | 684.020 | 7.929 | 4884.113 | 0.226 | 105171.270 | 3.538 | 0.942 | 0.743 | 9.153 | 0.613 |
| **1568.1** | 13.02±33.28s | **ExtraTrees** 18/18 | 2.345 | 2.804 | 6.062 | 202.052 | 4.099 | **1.430** | 4.600 | 0.066 | 693.526 | 8.640 | 5189.427 | 0.231 | 98024.414 | **3.392** | **0.930** | 1.027 | 8.771 | **0.602** |
| **1532.3** | 23.17±31.44s | **grf\_llf** 16/18 | **2.238** | 3.760 | 6.314 | · | 3.804 | 2.344 | 5.128 | 565.086 | 698.664 | 7.550 | 4542.951 | 0.231 | 102561.418 | 3.593 | 0.966 | · | **8.641** | 0.638 |
| **1470.6** | 0.35±0.22s | **LightGBM(200)** 18/18 | 2.433 | 3.473 | 5.848 | 202.615 | 4.093 | 1.626 | 4.288 | 0.070 | 716.880 | 7.673 | 5148.259 | 0.216 | 93946.869 | 3.919 | 0.972 | 0.745 | 9.425 | 0.630 |
| **1439.3** | 7.20±23.69s | **perpetual (Rust)** 17/18 | 2.316 | 3.440 | 5.781 | 222.651 | 3.882 | 2.245 | 4.746 | **0.059** | 699.336 | 7.485 | 4537.289 | 0.211 | 91666.478 | 3.781 | 0.955 | · | 9.358 | 0.657 |
| **1425.5** | 1.04±1.10s | **CatBoost(200)** 18/18 | 2.308 | **2.503** | 5.739 | 196.293 | 3.817 | 1.707 | 4.055 | 0.069 | 691.069 | 7.434 | 4511.499 | 0.219 | 87832.800 | 3.923 | 0.951 | 0.764 | 9.918 | 0.648 |
| **1350.1** | 0.95±1.91s | **HistGradientBoosting** 17/18 | 2.362 | 3.318 | 5.878 | 200.246 | 3.836 | 1.778 | 4.559 | 0.069 | 714.725 | · | 4976.682 | 0.225 | 99388.451 | 4.106 | 0.966 | 0.774 | 10.200 | 0.650 |
| **1319.3** | 65.08±68.98s | **EBM** 15/18 | 2.300 | 3.420 | **5.287** | **163.233** | **3.724** | 1.850 | **3.748** | 0.060 | 689.761 | 7.447 | 4489.402 | 0.218 | 91918.766 | · | 0.973 | · | · | 0.663 |
| **1299.2** | 1.22±0.86s | **forust (Rust)** 18/18 | 2.424 | 3.152 | 5.587 | 167.737 | 3.941 | 1.606 | 4.428 | 0.089 | 747.923 | 7.435 | 4832.590 | 0.224 | 93775.195 | 3.965 | 1.012 | 0.780 | 9.680 | 0.658 |
| **1210.8** | 0.76±1.83s | **CTBoost** 17/18 | 2.502 | 4.014 | 6.376 | 193.047 | 4.543 | 2.529 | 4.907 | 0.065 | 707.656 | · | 4509.975 | 0.226 | 107014.090 | 4.193 | 0.990 | 0.834 | 10.987 | 0.686 |
| **1168.4** | 27.91±39.58s | **bart** 17/18 | 2.341 | 3.384 | 5.507 | 195.823 | 3.778 | 1.874 | 3.971 | 539.746 | 701.686 | 7.801 | 4487.422 | 0.234 | 101243.023 | 4.328 | 0.969 | · | 11.834 | 0.690 |
| **1166.7** | 18.06±23.80s | **grf\_rf** 18/18 | 2.359 | 3.265 | 8.261 | 232.406 | 4.482 | 2.792 | 6.500 | 594.072 | 700.546 | 7.547 | 4608.092 | 0.246 | 124681.686 | 3.997 | 0.979 | 0.932 | 10.772 | 0.693 |
| **1050.1** | 13.77±20.73s | **ModelTree(all)** 16/18 | 2.409 | 4.537 | 6.421 | 265.250 | 4.311 | 3.290 | 6.485 | 0.062 | 789.071 | 7.420 | 4972.751 | 0.257 | 127066.860 | 4.516 | 0.998 | · | · | 0.705 |
| **1030.9** | 7.00±13.24s | **sklearn GBM** 18/18 | 2.291 | 3.209 | 6.377 | 190.926 | 3.881 | 2.622 | 5.380 | 0.068 | 700.354 | 7.640 | 4512.987 | 0.257 | 110163.529 | 4.702 | 0.962 | 0.961 | 12.615 | 0.694 |
| **976.4** | 20.22±47.48s | **lmtree** 14/18 | 2.272 | 4.052 | 7.644 | · | 4.268 | 2.580 | 6.635 | 728.364 | 718.839 | · | 4486.791 | 0.238 | 113425.118 | · | 1.036 | 0.960 | · | 0.709 |
| **972.9** | 15.29±51.54s | **modeltrees lmtree** 14/18 | 2.272 | 4.052 | 7.644 | · | 4.268 | 2.906 | 6.635 | · | 718.839 | · | 4486.791 | 0.238 | 104905.863 | 4.849 | 1.036 | 0.960 | · | 0.709 |
| **958.1** | 29.02±57.11s | **glmtree** 14/18 | 2.276 | 4.050 | 7.425 | · | 4.268 | 2.557 | 6.131 | 626.466 | 718.839 | 8.013 | 4486.791 | 0.235 | · | · | 1.029 | 0.960 | · | 0.713 |
| **928.3** | 5.95±8.90s | **ModelTree(winnow)** 16/18 | 2.409 | 4.537 | 6.421 | 266.405 | 4.242 | 3.550 | 6.485 | 0.062 | 769.640 | 7.864 | 4972.751 | 0.329 | 126937.727 | 4.960 | 0.998 | · | · | 0.719 |
| **892.7** | 6.23±15.19s | **cubist** 17/18 | 2.294 | 3.980 | 6.812 | · | 4.572 | 2.473 | 6.144 | 638.234 | 755.036 | 8.418 | 4531.016 | 0.230 | 114607.603 | 4.432 | 1.020 | 0.881 | 12.451 | 0.723 |
| **873.1** | 20.49±31.78s | **MARS (mars-earth)** 10/18 | 2.274 | 4.845 | 7.113 | · | 4.100 | 4.397 | 6.382 | · | 708.531 | · | 5809.859 | · | · | · | 0.988 | · | · | 0.727 |
| **761.5** | 20.60±51.49s | **mboost** 15/18 | 2.364 | 3.522 | 7.144 | 234.631 | 3.786 | · | 6.153 | 816.077 | 722.724 | · | **4372.440** | 0.298 | 134404.244 | 5.035 | · | 1.081 | 15.321 | 0.720 |
| **697.6** | 0.33±0.68s | **statsmodels OLS** 8/18 | · | · | 11.309 | · | 4.499 | · | 10.454 | · | · | · | · | 0.343 | · | 5.188 | 1.061 | 0.960 | 17.689 | · |
| **690.7** | 10.19±27.50s | **earth** 17/18 | 2.259 | 4.454 | 7.060 | · | 4.228 | 4.311 | 6.197 | 1078.948 | 714.051 | 7.994 | 5724.629 | 0.291 | 145755.854 | 5.026 | 0.999 | 1.018 | 17.701 | 0.730 |
| **681.8** | 0.09±0.10s | **KNN** 18/18 | 2.504 | 5.759 | 10.947 | 250.524 | 5.496 | 6.149 | 9.070 | 0.189 | 829.638 | 9.519 | 11145.185 | 0.519 | 147140.199 | 5.888 | 1.037 | 0.864 | 11.804 | 0.804 |
| **655.7** | 21.17±59.09s | **ctree** 17/18 | 2.417 | 4.019 | 9.757 | · | 4.260 | 3.788 | 7.098 | 603.732 | 780.636 | 7.875 | 4886.127 | 0.304 | 140802.325 | 4.594 | 1.078 | 1.059 | 12.731 | 0.746 |
| **611.2** | 14.37±7.33s | **trust-free** 4/18 | · | · | 7.511 | · | 4.471 | · | 6.646 | · | · | · | · | · | · | · | 1.087 | · | · | · |
| **584.5** | 0.09±0.23s | **sklearn Linear** 18/18 | 2.302 | 4.765 | 11.309 | 290.711 | 4.499 | 4.847 | 10.454 | 0.062 | 715.324 | 8.450 | 5854.983 | 0.343 | 189246.800 | 5.209 | 1.061 | 0.960 | 17.722 | 0.747 |
| **547.3** | 10.80±21.74s | **ModelTree(path)** 18/18 | 2.624 | 3.821 | 12.398 | 421.699 | 5.442 | 4.382 | 9.740 | 0.160 | 1130.844 | 7.598 | 4976.754 | 0.356 | 182240.202 | 5.332 | 1.140 | 1.331 | 15.398 | 0.740 |
| **539.8** | 0.34±0.57s | **sklearn tree** 18/18 | 3.237 | 4.769 | 7.931 | 210.311 | 5.456 | 2.801 | 5.665 | 0.110 | 854.893 | 10.593 | 6286.585 | 0.316 | 147951.144 | 5.068 | 1.273 | 1.057 | 11.611 | 0.846 |
| **527.1** | 17.48±43.02s | **statsmodels RLM** 8/18 | · | · | 11.438 | · | 4.600 | · | 10.518 | · | · | · | · | 0.345 | · | 5.202 | 1.069 | 0.970 | 17.745 | · |
| **482.0** | 5.34±20.14s | **ModelTree(mean)** 18/18 | 2.519 | 4.341 | 11.056 | 292.703 | 5.485 | 4.818 | 9.893 | 0.090 | 780.544 | 7.838 | 4791.068 | 0.358 | 177412.312 | 5.220 | 1.112 | 1.274 | 16.798 | 0.747 |
| **368.3** | 22.52±59.15s | **statsmodels QuantReg** 7/18 | · | · | 12.037 | · | 4.979 | · | 10.867 | · | · | · | · | 0.345 | · | 5.374 | 1.079 | · | 17.985 | · |
| **289.3** | 1.34±3.71s | **rpart** 17/18 | 2.588 | 4.310 | 9.069 | · | 4.707 | 4.443 | 9.787 | 1296.423 | 823.833 | 8.323 | 5088.406 | 0.378 | 192132.402 | 5.459 | 1.124 | 1.267 | 18.245 | 0.774 |

---

## The Classification Board

```
== Elo (round-robin on per-dataset LogLoss & Accuracy, K=32, start 1000) ==
+---------------+------------+----------------+--------------------------+---------+----------+
| ELO (LogLoss) | ELO (Acc)  | FIT TIME (±sd) | MODEL                    | LANG    | DATASETS |
+---------------+------------+----------------+--------------------------+---------+----------+
| 1473.4        | 1357.9     |   31.47 s      | EBM                      | Python  | 22/24    |
| 1468.1        | 1380.6     |    8.02 s      | perpetual (Rust)         | Rust    | 15/24    |
| 1440.8        | 1294.4     |    2.56 s      | CatBoost(200)            | Python  | 23/24    |
| 1435.0        | 1108.4     |    2.37 s      | ChimeraBoost             | Python  | 23/24    |
| 1429.7        | 1236.3     |    8.57 s      | bart                     | R       | 15/24    |
| 1360.8        | 1230.8     |    0.33 s      | CTBoost                  | Python  | 21/24    |
| 1343.8        | 1042.3     |    8.04 s      | sklearn GBM              | Python  | 23/24    |
| 1292.2        | 903.4      |   10.37 s      | grf_rf                   | R       |  7/24    |
| 1288.2        | 1244.2     |    7.60 s      | ranger                   | R       | 21/24    |
| 1226.1        | 1032.0     |   35.68 s      | MARS (mars-earth)        | Python  |  8/24    |
| 1170.4        | 996.1      |    2.10 s      | ctree                    | R       | 20/24    |
| 1138.0        | 894.8      |    8.07 s      | mboost                   | R       | 15/24    |
| 1084.6        | 884.1      |    0.41 s      | rpart                    | R       | 22/24    |
| 1065.6        | 1002.2     |    1.15 s      | forust (Rust)            | Rust    | 19/24    |
| 1049.8        | 1096.1     |    2.69 s      | sklearn RF               | Python  | 23/24    |
| 995.1         | 759.4      |    0.03 s      | statsmodels MNLogit      | Python  |  5/24    |
| 994.5         | 1317.2     |    0.71 s      | HistGradientBoosting     | Python  | 21/24    |
| 965.3         | 787.4      |    3.27 s      | c50_tree                 | R       | 24/24    |
| 954.9         | 934.2      |   16.04 s      | modeltrees multinomtree  | Python  |  9/24    |
| 950.7         | 989.9      |    1.21 s      | ExtraTrees               | Python  | 23/24    |
| 921.6         | 1280.8     |    0.60 s      | LightGBM(200)            | Python  | 23/24    |
| 891.5         | 737.2      |    2.44 s      | sklearn LogReg           | Python  | 23/24    |
| 853.5         | 1142.3     |    0.70 s      | ModelTree(path)          | Rust    | 18/24    |
| 787.4         | 1173.3     |    0.43 s      | ModelTree(mean)          | Rust    | 19/24    |
| 784.0         | 760.9      |   26.71 s      | modeltrees glmtree       | Python  | 14/24    |
| 763.2         | 674.4      |    1.74 s      | earth                    | R       | 16/24    |
| 762.9         | 700.1      |   22.37 s      | glmtree                  | R       | 14/24    |
| 745.1         | 997.1      |    6.07 s      | ModelTree(winnow)        | Rust    | 18/24    |
| 698.5         | 1056.1     |    7.42 s      | ModelTree(all)           | Rust    | 16/24    |
| 379.4         | 683.6      |    0.13 s      | KNN                      | Python  | 23/24    |
| 307.8         | 838.1      |    3.27 s      | c50_rules                | R       | 24/24    |
| -21.7         | 464.4      |    0.21 s      | sklearn tree             | Python  | 23/24    |
+---------------+------------+----------------+--------------------------+---------+----------+
```

### Interactive Classification Board (24 Datasets × 32 Models)

Click any column header to sort. The left key columns stay frozen while scrolling horizontally. **Green cells** indicate the best LogLoss on that dataset; **blue** indicates the fastest model; **orange** indicates the Elo leader.

| ELO | FIT TIME (±sd) | MODEL | Bank\_Customer\_Churn | E-CommereShippingData | Fitness\_Club | Is-this-a-good-customer | Marketing\_Campaign | NATICUSdroid | SDSS17 | bank-marketing | blood-transfusion-service-center | churn | coil2000\_insurance\_policies | credit-g | diabetes | hazelnut-spread-contaminant-detection | heloc | in\_vehicle\_coupon\_recommendation | jm1 | maternal\_health\_risk | online\_shoppers\_intention | qsar-biodeg | seismic-bumps | splice | students\_dropout\_and\_academic\_success | taiwanese\_bankruptcy\_prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1473.4** | 31.47±26.52s | **EBM** 22/24 | 0.326 | 0.509 | 0.436 | **0.313** | 0.270 | 0.177 | · | 0.301 | 0.501 | 0.172 | 0.208 | 0.512 | 0.563 | **0.123** | **0.540** | 0.493 | 0.424 | 0.564 | · | 0.305 | 0.244 | 0.198 | 0.571 | **0.071** |
| **1468.1** | 8.02±19.67s | **perpetual (Rust)** 15/24 | 0.335 | 0.512 | 0.446 | 0.329 | 0.298 | · | · | 0.304 | 0.502 | 0.174 | · | 0.538 | 0.632 | 0.280 | 0.547 | 0.497 | 0.424 | · | · | · | 0.237 | · | · | · |
| **1440.8** | 2.56±3.13s | **CatBoost(200)** 23/24 | 0.327 | 0.517 | 0.444 | 0.325 | **0.251** | **0.166** | 0.076 | 0.303 | 0.522 | **0.158** | 0.209 | 0.523 | 0.611 | 0.193 | 0.547 | 0.479 | 0.425 | 0.491 | · | 0.301 | 0.248 | 0.149 | 0.575 | 0.075 |
| **1435.0** | 2.37±2.56s | **ChimeraBoost** 23/24 | 0.328 | 0.509 | 0.438 | 0.316 | 0.260 | 0.183 | **0.071** | 0.303 | 0.498 | 0.164 | 0.203 | 0.529 | 0.576 | 0.136 | 0.545 | 0.539 | 0.424 | 0.527 | · | 0.327 | 0.241 | 0.146 | 0.573 | 0.079 |
| **1429.7** | 8.57±11.90s | **bart** 15/24 | **0.321** | **0.504** | · | 0.317 | · | 0.177 | · | 0.302 | 0.491 | 0.190 | **0.203** | **0.501** | 0.573 | 0.203 | 0.542 | 0.497 | · | · | · | 0.311 | 0.242 | · | · | · |
| **1360.8** | 0.33±0.45s | **CTBoost** 21/24 | 0.331 | 0.525 | 0.451 | 0.325 | · | 0.179 | · | **0.300** | 0.506 | 0.195 | 0.206 | 0.521 | 0.609 | 0.340 | 0.549 | 0.515 | 0.429 | 0.684 | · | 0.327 | 0.244 | **0.136** | 0.611 | 0.075 |
| **1343.8** | 8.04±24.70s | **sklearn GBM** 23/24 | 0.329 | 0.508 | 0.443 | 0.339 | 0.315 | 0.185 | 0.077 | 0.302 | 0.558 | 0.172 | 0.208 | 0.539 | 0.611 | 0.231 | 0.544 | 0.543 | 0.428 | 0.572 | · | 0.334 | 0.251 | 0.155 | **0.570** | 0.086 |
| **1292.2** | 10.37±9.76s | **grf\_rf** 7/24 | · | · | · | · | · | · | · | · | 0.500 | · | · | · | 0.562 | 0.332 | 0.553 | · | **0.423** | 0.638 | · | · | · | · | · | 0.081 |
| **1288.2** | 7.60±16.36s | **ranger** 21/24 | 0.337 | 0.518 | 0.468 | 0.341 | 0.291 | 0.186 | 0.083 | 0.304 | 0.550 | 0.181 | 0.229 | 0.506 | 0.589 | 0.273 | 0.547 | 0.529 | · | 0.515 | **0.223** | **0.297** | 0.255 | · | 0.587 | · |
| **1226.1** | 35.68±50.56s | **MARS (mars-earth)** 8/24 | · | 0.534 | · | 0.319 | · | · | · | · | 0.492 | · | · | 0.555 | 0.549 | · | · | 0.587 | · | 0.845 | · | · | 0.241 | · | · | · |
| **1170.4** | 2.10±0.75s | **ctree** 20/24 | 0.395 | 0.505 | 0.453 | 0.322 | · | 0.262 | · | 0.311 | 0.535 | 0.222 | 0.210 | 0.546 | 1.883 | 0.455 | 0.578 | 0.594 | 0.440 | 0.802 | · | 0.415 | **0.230** | 0.347 | · | 0.095 |
| **1138.0** | 8.07±9.35s | **mboost** 15/24 | 0.338 | 0.524 | 0.448 | 0.322 | · | 0.219 | · | 0.308 | 0.519 | · | · | 0.502 | 0.587 | 0.283 | 0.544 | · | · | 0.894 | · | 0.331 | 0.234 | · | 0.806 | · |
| **1084.6** | 0.41±0.64s | **rpart** 22/24 | 0.387 | 0.516 | 0.500 | 0.336 | 1.678 | 0.291 | · | 0.330 | 0.566 | 0.204 | 0.226 | 0.630 | 0.857 | 0.424 | 0.610 | 0.605 | 0.491 | 0.785 | 0.249 | 0.427 | 0.242 | 0.234 | · | 0.110 |
| **1065.6** | 1.15±0.79s | **forust (Rust)** 19/24 | 0.346 | 0.545 | 0.572 | 0.479 | 0.359 | 0.176 | · | 0.307 | 0.656 | 0.191 | 0.233 | 0.708 | 1.015 | 0.139 | 0.570 | 0.483 | 0.438 | · | · | 0.370 | 0.333 | · | · | 0.097 |
| **1049.8** | 2.69±6.37s | **sklearn RF** 23/24 | 0.343 | 0.528 | 0.800 | 0.527 | 0.304 | 0.238 | 0.121 | 0.484 | 0.827 | 0.213 | 0.561 | 0.512 | 0.606 | 0.331 | 0.551 | 0.522 | 0.617 | **0.422** | · | 0.306 | 0.664 | 0.315 | 0.695 | 0.097 |
| **995.1** | 0.03±0.01s | **statsmodels MNLogit** 5/24 | · | · | · | · | · | · | · | · | **0.486** | · | · | · | **0.520** | 0.212 | 0.563 | · | · | 0.830 | · | · | · | · | · | · |
| **994.5** | 0.71±0.33s | **HistGradientBoosting** 21/24 | 0.335 | 0.517 | 0.548 | 0.422 | · | 0.174 | · | 0.302 | 0.645 | 0.189 | 0.227 | 0.659 | 0.931 | 0.147 | 0.548 | 0.488 | 0.425 | 0.444 | · | 0.365 | 0.349 | 0.268 | 0.627 | 0.117 |
| **965.3** | 3.27±4.25s | **c50\_tree** 24/24 | 0.379 | 0.512 | 0.456 | 0.331 | 0.389 | 0.235 | 0.104 | 0.328 | 0.536 | 0.195 | 0.226 | 0.678 | 0.772 | 0.398 | 0.590 | 0.603 | 0.482 | 0.618 | 0.257 | 0.473 | 0.242 | 0.296 | 0.728 | 0.117 |
| **954.9** | 13.28±29.30s | **modeltrees multinomtree** 9/24 | · | 0.567 | · | 0.321 | · | · | · | · | · | · | · | 0.517 | 0.576 | 0.874 | 0.556 | · | · | 0.654 | · | 0.349 | 0.345 | · | · | · |
| **950.7** | 1.21±1.64s | **ExtraTrees** 23/24 | 0.413 | 0.545 | 0.723 | 0.451 | 0.713 | 0.899 | 0.131 | 0.847 | 3.449 | 0.243 | 0.888 | 0.522 | 0.574 | 0.263 | 0.553 | 0.688 | 0.667 | 0.970 | · | 0.439 | 0.735 | 0.396 | 0.596 | 0.100 |
| **921.6** | 0.60±1.01s | **LightGBM(200)** 23/24 | 0.349 | 0.539 | 0.628 | 0.587 | 0.367 | 0.186 | 0.125 | 0.307 | 0.684 | 0.238 | 0.260 | 0.818 | 1.321 | 0.149 | 0.564 | **0.468** | 0.434 | 0.466 | · | 0.543 | 0.450 | 0.145 | 0.655 | 0.168 |
| **891.5** | 2.44±4.80s | **sklearn LogReg** 23/24 | 0.442 | 0.556 | **0.435** | 0.320 | 0.384 | 0.193 | 0.641 | 0.334 | **0.486** | 0.333 | 0.212 | 0.559 | 0.520 | 0.693 | 0.575 | 0.639 | 0.541 | 0.831 | · | 0.336 | 0.265 | 0.442 | 0.597 | 0.185 |
| **853.5** | 0.70±0.72s | **ModelTree(path)** 18/24 | 0.387 | 0.507 | 0.455 | 0.449 | 0.326 | 0.289 | · | 0.311 | 0.524 | 0.267 | 0.225 | 0.868 | · | 0.544 | 0.581 | 0.565 | 0.441 | · | · | 0.511 | 0.439 | · | · | 0.118 |
| **787.4** | 0.43±0.50s | **ModelTree(mean)** 19/24 | 0.367 | 0.505 | 0.440 | 0.429 | 0.317 | 0.289 | · | 0.312 | 0.538 | 0.271 | 0.224 | 0.869 | 2.264 | 0.544 | 0.583 | 0.570 | 0.441 | · | · | 0.530 | 0.475 | · | · | 0.119 |
| **784.0** | 26.71±41.15s | **modeltrees glmtree** 14/24 | 0.325 | 0.576 | · | 0.321 | · | 0.211 | · | · | 0.500 | · | 0.215 | 0.517 | 0.576 | 0.870 | 0.555 | 0.564 | · | · | · | 0.349 | 0.338 | · | · | 1.215 |
| **763.2** | 1.74±3.83s | **earth** 16/24 | 0.343 | 0.522 | · | 0.345 | · | 0.201 | · | 0.306 | 0.493 | 0.286 | 0.207 | 0.536 | 0.585 | 0.222 | 0.542 | 0.589 | · | · | · | 0.388 | 0.272 | · | · | 1.367 |
| **762.9** | 22.37±35.04s | **glmtree** 14/24 | 0.325 | 0.576 | · | 0.321 | · | 0.211 | · | · | 0.500 | · | 0.215 | 0.517 | 0.576 | 0.870 | 0.555 | 0.564 | · | · | · | 0.349 | 0.265 | · | · | 1.519 |
| **745.1** | 6.07±11.28s | **ModelTree(winnow)** 18/24 | 0.378 | · | 0.453 | 0.444 | 0.351 | 0.199 | · | 0.312 | 0.503 | 0.235 | 0.259 | 0.938 | 0.980 | 0.544 | 0.672 | 0.562 | 0.511 | · | · | 0.540 | 0.406 | · | · | 0.191 |
| **698.5** | 7.42±13.26s | **ModelTree(all)** 16/24 | 0.430 | 0.510 | 0.493 | 0.482 | · | 0.200 | · | 0.311 | 0.529 | · | 0.305 | 1.018 | 1.043 | 0.544 | 0.689 | 0.559 | 0.511 | · | · | 0.598 | 0.449 | · | · | · |
| **379.4** | 0.13±0.20s | **KNN** 23/24 | 2.519 | 1.854 | 2.199 | 2.132 | 2.468 | 0.770 | 4.267 | 1.928 | 2.432 | 2.077 | 1.440 | 2.265 | 3.120 | 1.322 | 2.048 | 1.585 | 2.316 | 1.861 | · | 0.781 | 1.518 | 2.334 | 4.208 | 0.827 |
| **307.8** | 3.27±4.11s | **c50\_rules** 24/24 | 3.059 | 8.976 | 6.451 | 4.016 | 4.934 | 2.505 | 0.857 | 3.619 | 9.210 | 1.689 | 0.693 | 6.720 | 6.516 | 3.339 | 8.994 | 3.069 | 6.442 | 6.498 | 1.540 | 2.770 | 0.693 | 1.336 | 7.072 | 1.017 |
| **\-21.7** | 0.21±0.45s | **sklearn tree** 23/24 | 7.391 | 11.994 | 10.131 | 6.325 | 6.792 | 3.286 | 1.201 | 6.265 | 10.239 | 2.521 | 3.424 | 11.916 | 10.765 | 4.677 | 12.303 | 10.974 | 8.286 | 2.418 | · | 5.893 | 3.607 | 3.032 | 11.201 | 1.570 |

---

## Reading the Boards

- **Stock configs only**: All engines evaluated under their default configurations across all datasets.
- **Single Model Trees vs Ensembles**:
- `ModelTree(all)` scores **1050.1** in regression and `ModelTree(path)` scores **853.5** in classification, clearly outperforming single-tree model tree baselines like R `glmtree` (762.9, 22.37s) and Python `modeltrees glmtree` (784.0, 26.71s) while fitting in **0.70s** (38× faster).
- In pure accuracy, `ModelTree(mean)` achieves **Elo Acc 1173.3**, outperforming `sklearn RF` (1096.1), `sklearn GBM` (1042.3), and `c50_tree` (787.4).
- **Time Tiers**:
- `FIT TIME`: Average fit seconds across all scored datasets.
- `DATASETS`: Datasets where the model completed within the strict 180s budget (timeouts or numerical failures forfeit matchups).

## Development rounds (ours vs the frozen field)

The external field (R packages, LightGBM, CatBoost, CTBoost, forust, perpetual, MARS, sklearn, modeltrees) is scored **once** and frozen in `benchmarks/elo_baselines.csv` — *where we started*. Every development round re-scores only the ModelTree variants on the same prepared splits:

```
uv run python examples/elo_round.py                 # all datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # a subset
uv run python examples/elo_round.py --freeze        # deliberately re-anchor
```

Each round prints `started (frozen baseline) -> reached` Elo for every ModelTree flavor, with per-fit timings in the scores file — so progress across iterations is visible instead of silently overwritten. The full from-scratch rebuild (all models, groups of 5 datasets, shareable intermediate state after each group) is `bash examples/elo_run_groups.sh`.

The field's raw scores (per dataset RMSE/MAE/fit time) live in [`benchmarks/elo_scores.csv`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/elo_scores.csv>); the longer research log is in `docs/field-notes.md`.

## Running a round (repeatable, with provenance)

One command re-scores **our four flavors** against the frozen field:

```
uv run python examples/elo_round.py                    # all 18 datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # subset
```

Every round records what created its numbers:

- `benchmarks/rounds/<stamp>/settings.json` — the **exact settings dict** each flavor was fit with (single-sourced: the manifest records the same dict `variants()` builds from, so record and reality cannot drift), the git SHA, the dataset list, every result row, and Elo before → after.
- `benchmarks/elo_round.log` — stdlib `logging` output (file + console), one line per round: `timestamp, stamp, sha,   dataset count, per-model Elo before->after`. This is the first place to look for "what did we change and what did it move".
- `benchmarks/elo_scores.csv` — the live board (our rows upserted); `benchmarks/elo_baselines.csv` — the frozen anchor ("where we started").

Tuned settings from `examples/hpo_optuna.py` print in the same dict shape — paste them into `SETTINGS` in `elo_round.py` to score a tuned flavor.
