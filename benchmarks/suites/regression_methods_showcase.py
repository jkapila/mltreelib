"""Systematic Showcase and Benchmark of All Regression Methods in mltreelib.

Taxonomy of regression capabilities in mltreelib:
-----------------------------------------------------------------------------------------
1. Constant (0D) Leaf Trees:
   - CART Mean Leaf (Standard Anova SSE)
   - Median Leaf (L1 Least Absolute Deviation Location)
   - Huber Leaf (Outlier-resistant M-estimator Location)
2. Linear Hyperplane (1D) Model Trees:
   - OLS Linear Model Tree (Cholesky normal equations)
   - Huber Robust Linear Model Tree (Scale-adaptive psi-weighting)
   - NNLS Non-Negative Linear Leaf (Active-set positive coefficients)
   - Expectile Regression Leaf (Asymmetric squared loss tau)
3. Quantile Hyperplane Regression & Risk Diagnostics:
   - Quantile Linear Hyperplanes (Pinball loss tau = 0.10, 0.50, 0.90)
   - Quantile Crossing Diagnostics (detect_quantile_crossings)
4. Advanced Regularization & Structured Sparsity:
   - Hierarchical Leaf Shrinkage (Empirical Bayes leaf-to-parent shrinkage)
   - Group Lasso & Sparse-Group Lasso (Simon et al. 2013 proximal BCD)
   - Monotone Coefficient Constraints (leaf_monotone=[-1, 0, 1])
5. Generalized Linear Model (GLM) Trees:
   - Poisson Count Regression Tree (Log link)
   - Negative Binomial Dispersion Tree (Overdispersed counts)
   - Beta Regression Tree (Bounded rate (0, 1))
6. Smooth Non-Parametric & Adaptive Additive Methods:
   - MARS Regressor (Multivariate Adaptive Regression Splines with quantile knots)
   - Penalized B-Spline GAM (P-splines with second-difference penalty and GCV)
7. Fast Cardinality-Adaptive Splitting:
   - Fast Exact Split Scan (split_eval="fast")
   - Cardinality-Adaptive Binned Histogram (split_eval="histogram", max_bins=64)
-----------------------------------------------------------------------------------------

Run:
    uv run python benchmarks/suites/regression_methods_showcase.py
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from mltreelib import (
    GamRegressor,
    MarsRegressor,
    Metrics,
    ModelTree,
    detect_quantile_crossings,
)


def _eval_model(
    model: any,
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    metrics: Metrics,
) -> tuple[float, float, float, str, float]:
    """Fit model, measure runtime, predict test set, and compute error metrics."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0

    preds = model.predict(X_test)
    rep = metrics.regression(y_test, preds)
    medae = float(statistics.median(abs(yt - yp) for yt, yp in zip(y_test, preds)))

    if hasattr(model, "n_leaves"):
        complexity = f"{model.n_leaves()} leaves"
    elif hasattr(model, "n_terms"):
        n_terms = model.n_terms() if callable(model.n_terms) else model.n_terms
        complexity = f"{n_terms} terms"
    else:
        complexity = "-"

    return rep["rmse"], rep["mae"], medae, complexity, fit_time_ms


# ==============================================================================
# STUDY 1: Physical Continuous Response (Concrete Compressive Strength)
# ==============================================================================
def run_concrete_study(metrics: Metrics) -> None:
    print("\n" + "=" * 80)
    print("STUDY 1: Concrete Compressive Strength (Continuous Physical Process)")
    print(
        "Goal: Compare 0D vs 1D trees, test hierarchical shrinkage, evaluate smooth GAM."
    )
    print("=" * 80)

    path = Path("datasets/concrete_compressive_strength.parquet")
    df = pd.read_parquet(path)
    target = "ConcreteCompressiveStrength"
    features = [c for c in df.columns if c != target]

    X = [
        [float(v) for v in row]
        for row in df[features].itertuples(index=False, name=None)
    ]
    y = [float(v) for v in df[target]]

    indices = list(range(len(X)))
    Random(42).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
    print(
        f"Data: {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(features)} continuous inputs."
    )

    models = [
        # 0D Constant Leaves
        ("CART Mean Leaf (0D SSE)", ModelTree(leaf="mean", max_depth=4, min_rows=15)),
        (
            "Binned Histogram Split (0D SSE)",
            ModelTree(
                leaf="mean",
                max_depth=4,
                min_rows=15,
                split_eval="histogram",
                max_bins=64,
            ),
        ),
        (
            "Median Leaf (0D L1 Location)",
            ModelTree(leaf="median", max_depth=4, min_rows=15),
        ),
        # 1D Linear Model Trees & Shrinkage
        (
            "OLS Linear Tree (shrinkage=0.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=4,
                min_rows=20,
                l2=0.1,
                shrinkage=0.0,
            ),
        ),
        (
            "Hierarchical Shrunk Tree (shrinkage=10.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=4,
                min_rows=20,
                l2=0.1,
                shrinkage=10.0,
            ),
        ),
        (
            "Huber Robust Linear Tree",
            ModelTree(
                leaf="linear",
                leaf_family="huber",
                huber_delta=1.345,
                max_depth=4,
                min_rows=20,
                l2=0.1,
            ),
        ),
        (
            "NNLS Non-Negative Linear Tree",
            ModelTree(
                leaf="linear", non_negative=True, max_depth=4, min_rows=20, l2=0.1
            ),
        ),
        (
            "Expectile Linear Tree (tau=0.75)",
            ModelTree(
                leaf="linear",
                leaf_family="expectile",
                expectile_tau=0.75,
                max_depth=4,
                min_rows=20,
                l2=0.1,
            ),
        ),
        (
            "Quantile Linear Tree (tau=0.50)",
            ModelTree(
                leaf="linear",
                leaf_family="quantile",
                tau=0.50,
                max_depth=4,
                min_rows=20,
                l2=0.1,
            ),
        ),
        # Smooth Non-Parametric Estimators
        (
            "MARS Quantile Knots (max_terms=15)",
            MarsRegressor(
                max_terms=15, degree=1, quantile_knots=True, ridge_lambda=1e-3
            ),
        ),
        ("Penalized B-Spline GAM (n_knots=8)", GamRegressor(n_knots=8, degree=3)),
    ]

    print(
        f"\n{'Model Architecture':<42} | {'Test RMSE':<10} | {'Test MAE':<10} | {'Complexity':<10} | {'Fit Time':<9}"
    )
    print("-" * 90)
    for name, model in models:
        rmse, mae, _, comp, t_ms = _eval_model(
            model, X_train, y_train, X_test, y_test, metrics
        )
        print(
            f"{name:<42} | {rmse:<10.3f} | {mae:<10.3f} | {comp:<10} | {t_ms:<7.2f}ms"
        )


# ==============================================================================
# STUDY 2: Heavy-Tailed & Outlier Prone Response (Healthcare Insurance Expenses)
# ==============================================================================
def run_insurance_study(metrics: Metrics) -> None:
    print("\n" + "=" * 80)
    print("STUDY 2: Healthcare Insurance Expenses (Heavy-Tailed & Outliers)")
    print(
        "Goal: Test Huber robustness, Group Lasso on dummy blocks, and multi-tau quantiles."
    )
    print("=" * 80)

    path = Path("datasets/healthcare_insurance_expenses.parquet")
    df = pd.read_parquet(path)
    target = "charges"
    features = [c for c in df.columns if c != target]

    df_encoded = pd.get_dummies(df[features], drop_first=True)
    feat_names = list(df_encoded.columns)

    X = [
        [float(v) for v in row] for row in df_encoded.itertuples(index=False, name=None)
    ]
    y = [float(v) for v in df[target]]

    indices = list(range(len(X)))
    Random(101).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
    print(
        f"Data: {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(feat_names)} encoded features."
    )

    # Group definition: group region dummy indicators together
    region_cols = [i for i, c in enumerate(feat_names) if c.startswith("region_")]
    groups = {"region": region_cols} if region_cols else None

    models = [
        ("CART Mean Leaf", ModelTree(leaf="mean", max_depth=3, min_rows=25)),
        (
            "OLS Linear Tree (shrinkage=0.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=3,
                min_rows=30,
                l2=0.5,
                shrinkage=0.0,
            ),
        ),
        (
            "Hierarchical Shrunk Tree (shrinkage=15.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=3,
                min_rows=30,
                l2=0.5,
                shrinkage=15.0,
            ),
        ),
        (
            "Group Lasso Tree (region group_l2=5.0)",
            ModelTree(
                leaf="linear",
                alpha=1.0,
                l1=0.0,
                group_l2=5.0,
                groups=groups,
                max_depth=3,
                min_rows=30,
            ),
        ),
        (
            "Huber Robust Linear Tree",
            ModelTree(
                leaf="linear",
                leaf_family="huber",
                huber_delta=1.345,
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "NNLS Non-Negative Linear Tree",
            ModelTree(
                leaf="linear", non_negative=True, max_depth=3, min_rows=30, l2=0.5
            ),
        ),
        (
            "Expectile Linear Tree (tau=0.50)",
            ModelTree(
                leaf="linear",
                leaf_family="expectile",
                expectile_tau=0.50,
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "Median Leaf (0D L1 Location)",
            ModelTree(leaf="median", max_depth=3, min_rows=25),
        ),
        (
            "Huber Leaf (0D M-Location)",
            ModelTree(leaf="huber", max_depth=3, min_rows=25),
        ),
        (
            "Quantile Linear (tau=0.10 Lower Risk)",
            ModelTree(
                leaf="linear",
                leaf_family="quantile",
                tau=0.10,
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "Quantile Linear (tau=0.50 Median)",
            ModelTree(
                leaf="linear",
                leaf_family="quantile",
                tau=0.50,
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "Quantile Linear (tau=0.90 Upper Risk)",
            ModelTree(
                leaf="linear",
                leaf_family="quantile",
                tau=0.90,
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "MARS Quantile Knots (max_terms=15)",
            MarsRegressor(
                max_terms=15, degree=1, quantile_knots=True, ridge_lambda=1e-3
            ),
        ),
        ("Penalized B-Spline GAM", GamRegressor(n_knots=8, degree=3)),
    ]

    print(
        f"\n{'Model Architecture':<44} | {'Test RMSE':<10} | {'Test MAE':<10} | {'MedAE':<10} | {'Fit Time':<9}"
    )
    print("-" * 92)
    quantile_preds = []
    taus = [0.10, 0.50, 0.90]

    for name, model in models:
        rmse, mae, medae, _, t_ms = _eval_model(
            model, X_train, y_train, X_test, y_test, metrics
        )
        print(
            f"{name:<44} | {rmse:<10.1f} | {mae:<10.1f} | {medae:<10.1f} | {t_ms:<7.2f}ms"
        )
        if "tau=0.10" in name or "tau=0.50" in name or "tau=0.90" in name:
            preds = model.predict(X_test)
            quantile_preds.append(preds)

    # Diagnostic check for quantile monotonicity
    print("\n--- Diagnostic: Multi-Tau Quantile Crossing Analysis ---")
    diag = detect_quantile_crossings(quantile_preds, taus)
    print(f"Evaluated {len(X_test)} held-out cases across quantiles {taus}:")
    print(f"  - Crossing Inversion Pairs: {diag['n_crossings']}")
    print(f"  - Rows with Violations:     {diag['n_violating_rows']} / {len(X_test)}")
    print(f"  - Crossing Violation Rate:  {diag['crossing_rate']:.2%}")
    print(f"  - Maximum Dollar Violation: ${diag['max_violation']:.2f}")


# ==============================================================================
# STUDY 3: Bounded & Overdispersed Count Distributions (Wine Quality)
# ==============================================================================
def run_wine_quality_study(metrics: Metrics) -> None:
    print("\n" + "=" * 80)
    print("STUDY 3: Wine Quality (Overdispersed Counts & Bounded Score GLMs)")
    print(
        "Goal: Compare Gaussian OLS vs Poisson, Negative Binomial, and Beta GLM trees."
    )
    print("=" * 80)

    path = Path("datasets/wine_quality.parquet")
    df = pd.read_parquet(path)
    target = "median_wine_quality"
    features = [c for c in df.columns if c != target and c != "wine_color"]

    X = [
        [float(v) for v in row]
        for row in df[features].itertuples(index=False, name=None)
    ]
    y = [float(v) for v in df[target]]

    indices = list(range(len(X)))
    Random(777).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
    print(
        f"Data: {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(features)} physicochemical features."
    )

    models = [
        ("CART Mean Leaf", ModelTree(leaf="mean", max_depth=3, min_rows=30)),
        (
            "OLS Linear Tree (shrinkage=0.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=3,
                min_rows=30,
                l2=0.5,
                shrinkage=0.0,
            ),
        ),
        (
            "Hierarchical Shrunk Tree (shrinkage=10.0)",
            ModelTree(
                leaf="linear",
                leaf_family="gaussian",
                max_depth=3,
                min_rows=30,
                l2=0.5,
                shrinkage=10.0,
            ),
        ),
        (
            "Poisson GLM Tree (Log Link)",
            ModelTree(
                leaf="linear", leaf_family="poisson", max_depth=3, min_rows=30, l2=0.5
            ),
        ),
        (
            "Negative Binomial GLM Tree (Dispersion)",
            ModelTree(
                leaf="linear",
                leaf_family="negative_binomial",
                max_depth=3,
                min_rows=30,
                l2=0.5,
            ),
        ),
        (
            "Beta Regression Tree (Rate / 10)",
            ModelTree(
                leaf="linear", leaf_family="beta", max_depth=3, min_rows=30, l2=0.5
            ),
        ),
    ]

    print(
        f"\n{'Model Architecture':<44} | {'Test RMSE':<10} | {'Test MAE':<10} | {'Fit Time':<9}"
    )
    print("-" * 78)
    for name, model in models:
        t0 = time.perf_counter()
        if "Beta" in name:
            y_train_rate = [val / 10.0 for val in y_train]
            model.fit(X_train, y_train_rate)
            preds_rate = model.predict(X_test)
            preds = [p * 10.0 for p in preds_rate]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        t_ms = (time.perf_counter() - t0) * 1000.0

        rep = metrics.regression(y_test, preds)
        print(
            f"{name:<44} | {rep['rmse']:<10.3f} | {rep['mae']:<10.3f} | {t_ms:<7.2f}ms"
        )


# ==============================================================================
# STUDY 4: Controlled Ground Truth & Structured Group Sparsity
# ==============================================================================
def run_structured_sparsity_study(metrics: Metrics) -> None:
    print("\n" + "=" * 80)
    print("STUDY 4: Controlled Synthetic Ground Truth & Group Sparsity")
    print("Goal: Verify exact group zeroing and smooth sine wave reconstruction.")
    print("=" * 80)

    np.random.seed(42)
    n = 600
    # Group 1: features 0, 1 are strong signal; features 2, 3 are correlated noise
    # Group 2: features 4, 5, 6, 7 are pure independent noise
    # Feature 8: non-linear sine wave generator x8 in [0, 2pi]
    X_mat = np.random.uniform(-1.0, 1.0, (n, 8))
    x_sine = np.random.uniform(0.0, 2.0 * math.pi, n)

    # Ground truth DGP:
    # y = 3.0 + 2.5 * X0 - 1.8 * X1 + 2.0 * sin(x_sine) + noise
    y_vec = (
        3.0
        + 2.5 * X_mat[:, 0]
        - 1.8 * X_mat[:, 1]
        + 2.0 * np.sin(x_sine)
        + np.random.normal(0.0, 0.2, n)
    )
    X_full = np.column_stack([X_mat, x_sine])

    X_train, X_test = X_full[:450].tolist(), X_full[450:].tolist()
    y_train, y_test = y_vec[:450].tolist(), y_vec[450:].tolist()

    groups = {
        "signal_block": [0, 1, 2, 3],
        "noise_block": [4, 5, 6, 7],
    }

    models = [
        (
            "Standard OLS Linear Tree",
            ModelTree(
                leaf="linear", leaf_family="gaussian", max_depth=2, min_rows=30, l2=0.01
            ),
        ),
        (
            "Coordinate Lasso Tree (l1=2.0)",
            ModelTree(
                leaf="linear", alpha=1.0, l1=2.0, l2=0.01, max_depth=2, min_rows=30
            ),
        ),
        (
            "Group Lasso Tree (group_l2=5.0)",
            ModelTree(
                leaf="linear",
                alpha=1.0,
                l1=0.0,
                group_l2=5.0,
                groups=groups,
                max_depth=2,
                min_rows=30,
            ),
        ),
        (
            "Sparse-Group Lasso (l1=1.5, group_l2=4.0)",
            ModelTree(
                leaf="linear",
                alpha=1.0,
                l1=1.5,
                group_l2=4.0,
                groups=groups,
                max_depth=2,
                min_rows=30,
            ),
        ),
        ("Penalized B-Spline GAM on Sine Feature", GamRegressor(n_knots=12, degree=3)),
    ]

    print(
        f"\n{'Model Configuration':<44} | {'Test RMSE':<10} | {'Test MAE':<10} | {'Fit Time':<9}"
    )
    print("-" * 78)
    for name, model in models:
        t0 = time.perf_counter()
        if "GAM" in name:
            # Fit GAM strictly on the sine feature (column index 8)
            x_train_sin = [[row[8]] for row in X_train]
            x_test_sin = [[row[8]] for row in X_test]
            # Target for isolated sine recovery: residual y - linear_ground_truth
            y_train_sin = [
                y_train[i] - (3.0 + 2.5 * X_train[i][0] - 1.8 * X_train[i][1])
                for i in range(len(y_train))
            ]
            y_test_sin = [
                y_test[i] - (3.0 + 2.5 * X_test[i][0] - 1.8 * X_test[i][1])
                for i in range(len(y_test))
            ]
            model.fit(x_train_sin, y_train_sin)
            preds = model.predict(x_test_sin)
            rep = metrics.regression(y_test_sin, preds)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rep = metrics.regression(y_test, preds)
        t_ms = (time.perf_counter() - t0) * 1000.0

        print(
            f"{name:<44} | {rep['rmse']:<10.3f} | {rep['mae']:<10.3f} | {t_ms:<7.2f}ms"
        )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main() -> None:
    print("=" * 80)
    print("MLTREELIB REGRESSION METHODS BENCHMARK SUITE")
    print(
        "Comprehensive systematic evaluation of trees, splines, GLMs, and regularizers"
    )
    print("=" * 80)

    metrics = Metrics()
    run_concrete_study(metrics)
    run_insurance_study(metrics)
    run_wine_quality_study(metrics)
    run_structured_sparsity_study(metrics)

    print("\n" + "=" * 80)
    print("All regression benchmarks completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
