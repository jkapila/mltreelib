"""Showcase and benchmark regression methods across real-world datasets.

Demonstrates:
1. Standard CART mean leaf vs OLS linear model tree vs Quantile hyperplanes.
2. Robustness on heavy-tailed data (Insurance charges) vs symmetric data (Concrete strength) vs high-volume (Diamonds).
3. Quantile regression at multiple risk percentiles (tau = 0.10, 0.50, 0.90).
4. Multi-tau quantile crossing diagnostics (detect_quantile_crossings).
5. Cardinality-adaptive binned split evaluation (split_eval="histogram").

Run:
    uv run python examples/regression_methods_showcase.py
"""

from pathlib import Path
from random import Random
import statistics
import pandas as pd
from mltreelib import (
    GamRegressor,
    GrowthControl,
    LeafControl,
    MarsRegressor,
    Metrics,
    ModelTree,
    detect_quantile_crossings,
)


def pinball_loss_score(y_true, y_pred, tau: float) -> float:
    """Compute average empirical pinball (check) loss."""
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        r = yt - yp
        total += r * (tau if r >= 0.0 else tau - 1.0)
    return total / max(len(y_true), 1)


def run_concrete_benchmark():
    print("\n" + "=" * 70)
    print("DATASET 1: Concrete Compressive Strength (Physical / Continuous)")
    print("=" * 70)
    path = Path("datasets/concrete_compressive_strength.parquet")
    df = pd.read_parquet(path)
    target = "ConcreteCompressiveStrength"
    features = [c for c in df.columns if c != target]

    X = [[float(v) for v in row] for row in df[features].itertuples(index=False, name=None)]
    y = [float(v) for v in df[target]]

    # Train / Test split 80/20
    indices = list(range(len(X)))
    Random(42).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    metrics = Metrics()
    print(f"Loaded {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(features)} features.")

    configs = [
        ("CART Mean Leaf", ModelTree(leaf="mean", max_depth=4, min_rows=15)),
        ("Binned Histogram Split (Mean Leaf)", ModelTree(leaf="mean", max_depth=4, min_rows=15, split_eval="histogram", max_bins=64)),
        ("OLS Linear Model Tree (shrinkage=0.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=4, min_rows=20, l2=0.1, shrinkage=0.0)),
        ("Hierarchical Shrunk Tree (shrinkage=10.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=4, min_rows=20, l2=0.1, shrinkage=10.0)),
        ("Huber Linear Model Tree", ModelTree(leaf="linear", leaf_family="huber", huber_delta=1.345, max_depth=4, min_rows=20, l2=0.1)),
        ("NNLS Non-Negative Linear", ModelTree(leaf="linear", non_negative=True, max_depth=4, min_rows=20, l2=0.1)),
        ("Expectile Linear (tau=0.75)", ModelTree(leaf="linear", leaf_family="expectile", expectile_tau=0.75, max_depth=4, min_rows=20, l2=0.1)),
        ("Median Leaf (L1 Location)", ModelTree(leaf="median", max_depth=4, min_rows=15)),
        ("Quantile Linear (tau=0.50 Median)", ModelTree(leaf="linear", leaf_family="quantile", tau=0.50, max_depth=4, min_rows=20, l2=0.1)),
        ("MARS Quantile Knots (max_terms=15)", MarsRegressor(max_terms=15, degree=1, quantile_knots=True, ridge_lambda=1e-3)),
        ("Penalized B-Spline GAM (n_knots=8)", GamRegressor(n_knots=8, degree=3)),
    ]

    print(f"\n{'Model Configuration':<42} | {'Test RMSE':<10} | {'Test MAE':<10} | {'Complexity':<10}")
    print("-" * 78)
    for name, model in configs:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rep = metrics.regression(y_test, preds)
        terms = model.n_leaves() if hasattr(model, "n_leaves") else (model.n_terms() if callable(model.n_terms) else model.n_terms)
        complexity = f"{terms} {'leaves' if hasattr(model, 'n_leaves') else 'terms'}"
        print(f"{name:<42} | {rep['rmse']:<10.3f} | {rep['mae']:<10.3f} | {complexity:<10}")


def run_insurance_benchmark():
    print("\n" + "=" * 70)
    print("DATASET 2: Healthcare Insurance Expenses (Heavy-Tailed / Outlier Prone)")
    print("=" * 70)
    path = Path("datasets/healthcare_insurance_expenses.parquet")
    df = pd.read_parquet(path)
    target = "charges"
    features = [c for c in df.columns if c != target]

    # Convert categoricals
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    feat_names = list(df_encoded.columns)

    X = [[float(v) for v in row] for row in df_encoded.itertuples(index=False, name=None)]
    y = [float(v) for v in df[target]]

    indices = list(range(len(X)))
    Random(101).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    metrics = Metrics()
    print(f"Loaded {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(feat_names)} encoded features.")

    region_cols = [i for i, c in enumerate(feat_names) if c.startswith("region_")]
    groups = {"region": region_cols} if region_cols else None

    configs = [
        ("CART Mean Leaf", ModelTree(leaf="mean", max_depth=3, min_rows=25)),
        ("OLS Linear Model Tree (shrinkage=0.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=30, l2=0.5, shrinkage=0.0)),
        ("Hierarchical Shrunk Tree (shrinkage=15.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=30, l2=0.5, shrinkage=15.0)),
        ("Group Lasso Tree (region group_l2=5.0)", ModelTree(leaf="linear", alpha=1.0, l1=0.0, group_l2=5.0, groups=groups, max_depth=3, min_rows=30)),
        ("Huber Linear Model Tree", ModelTree(leaf="linear", leaf_family="huber", huber_delta=1.345, max_depth=3, min_rows=30, l2=0.5)),
        ("NNLS Non-Negative Linear", ModelTree(leaf="linear", non_negative=True, max_depth=3, min_rows=30, l2=0.5)),
        ("Expectile Linear (tau=0.50)", ModelTree(leaf="linear", leaf_family="expectile", expectile_tau=0.50, max_depth=3, min_rows=30, l2=0.5)),
        ("Median Leaf (0D L1)", ModelTree(leaf="median", max_depth=3, min_rows=25)),
        ("Huber Leaf (0D Location)", ModelTree(leaf="huber", max_depth=3, min_rows=25)),
        ("Quantile Linear (tau=0.10 Lower Risk)", ModelTree(leaf="linear", leaf_family="quantile", tau=0.10, max_depth=3, min_rows=30, l2=0.5)),
        ("Quantile Linear (tau=0.50 Median)", ModelTree(leaf="linear", leaf_family="quantile", tau=0.50, max_depth=3, min_rows=30, l2=0.5)),
        ("Quantile Linear (tau=0.90 Upper Risk)", ModelTree(leaf="linear", leaf_family="quantile", tau=0.90, max_depth=3, min_rows=30, l2=0.5)),
        ("MARS Quantile Knots (max_terms=15)", MarsRegressor(max_terms=15, degree=1, quantile_knots=True, ridge_lambda=1e-3)),
        ("Penalized B-Spline GAM", GamRegressor(n_knots=8, degree=3)),
    ]

    print(f"\n{'Model Configuration':<44} | {'Test RMSE':<10} | {'Test MAE':<10} | {'MedAE':<10}")
    print("-" * 80)
    quantile_preds = []
    taus = [0.10, 0.50, 0.90]

    for name, model in configs:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rep = metrics.regression(y_test, preds)
        medae = statistics.median(abs(yt - yp) for yt, yp in zip(y_test, preds))
        print(f"{name:<44} | {rep['rmse']:<10.1f} | {rep['mae']:<10.1f} | {medae:<10.1f}")
        if "tau=0.10" in name or "tau=0.50" in name or "tau=0.90" in name:
            quantile_preds.append(preds)

    # Multi-tau crossing detection
    print("\n--- Quantile Crossing Diagnostic (detect_quantile_crossings) ---")
    report = detect_quantile_crossings(quantile_preds, taus)
    print(f"Evaluated {len(X_test)} held-out cases across quantiles {taus}:")
    print(f"  - Total Inverted Crossing Pairs: {report['n_crossings']}")
    print(f"  - Violating Cases Count:         {report['n_violating_rows']} / {len(X_test)}")
    print(f"  - Crossing Violation Rate:       {report['crossing_rate']:.4%}")
    print(f"  - Maximum Inversion Violation:   ${report['max_violation']:.2f}")


def run_diamonds_benchmark():
    print("\n" + "=" * 70)
    print("DATASET 3: Diamonds (High-Volume Heterogeneous Tabular, N=53,940)")
    print("=" * 70)
    path = Path("datasets/diamonds.parquet")
    df = pd.read_parquet(path)
    target = "price"
    features = [c for c in df.columns if c != target]

    df_encoded = pd.get_dummies(df[features], drop_first=True)
    X = [[float(v) for v in row] for row in df_encoded.itertuples(index=False, name=None)]
    y = [float(v) for v in df[target]]

    # Train / Test split 10,000 / 5,000
    indices = list(range(len(X)))
    Random(2026).shuffle(indices)
    train_idx, test_idx = indices[:10000], indices[10000:15000]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    metrics = Metrics()
    print(f"Loaded {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(df_encoded.columns)} features.")

    configs = [
        ("Fast Exact Scan (Mean Leaf)", ModelTree(leaf="mean", max_depth=3, min_rows=50, split_eval="fast")),
        ("Adaptive Binned Split (max_bins=64)", ModelTree(leaf="mean", max_depth=3, min_rows=50, split_eval="histogram", max_bins=64)),
        ("OLS Linear Model Tree (shrinkage=0.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=80, l2=1.0, shrinkage=0.0)),
        ("Hierarchical Shrunk Tree (shrinkage=25.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=80, l2=1.0, shrinkage=25.0)),
        ("Huber Linear Model Tree", ModelTree(leaf="linear", leaf_family="huber", huber_delta=1.345, max_depth=3, min_rows=80, l2=1.0)),
        ("NNLS Non-Negative Linear", ModelTree(leaf="linear", non_negative=True, max_depth=3, min_rows=80, l2=1.0)),
        ("Expectile Linear (tau=0.50)", ModelTree(leaf="linear", leaf_family="expectile", expectile_tau=0.50, max_depth=3, min_rows=80, l2=1.0)),
        ("Quantile Linear (tau=0.50 Median)", ModelTree(leaf="linear", leaf_family="quantile", tau=0.50, max_depth=3, min_rows=80, l2=1.0)),
    ]

    import time
    print(f"\n{'Model Configuration':<44} | {'Fit Time':<10} | {'Test RMSE':<10} | {'Test MAE':<10}")
    print("-" * 80)
    for name, model in configs:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        preds = model.predict(X_test)
        rep = metrics.regression(y_test, preds)
        print(f"{name:<44} | {fit_time:<9.3f}s | {rep['rmse']:<10.1f} | {rep['mae']:<10.1f}")


def run_wine_quality_benchmark():
    print("\n" + "=" * 70)
    print("DATASET 4: Wine Quality (Overdispersed Counts & Bounded Score GLMs)")
    print("=" * 70)
    path = Path("datasets/wine_quality.parquet")
    df = pd.read_parquet(path)
    target = "median_wine_quality"
    features = [c for c in df.columns if c != target and c != "wine_color"]

    X = [[float(v) for v in row] for row in df[features].itertuples(index=False, name=None)]
    y = [float(v) for v in df[target]]

    # Train / Test split 80/20
    indices = list(range(len(X)))
    Random(777).shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    metrics = Metrics()
    print(f"Loaded {len(X)} rows ({len(train_idx)} train, {len(test_idx)} test), {len(features)} features.")

    configs = [
        ("CART Mean Leaf", ModelTree(leaf="mean", max_depth=3, min_rows=30)),
        ("OLS Linear Model Tree (shrinkage=0.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=30, l2=0.5, shrinkage=0.0)),
        ("Hierarchical Shrunk Tree (shrinkage=10.0)", ModelTree(leaf="linear", leaf_family="gaussian", max_depth=3, min_rows=30, l2=0.5, shrinkage=10.0)),
        ("Poisson GLM Tree", ModelTree(leaf="linear", leaf_family="poisson", max_depth=3, min_rows=30, l2=0.5)),
        ("Negative Binomial GLM Tree", ModelTree(leaf="linear", leaf_family="negative_binomial", max_depth=3, min_rows=30, l2=0.5)),
        ("Beta Regression Tree (Rate / 10)", ModelTree(leaf="linear", leaf_family="beta", max_depth=3, min_rows=30, l2=0.5)),
    ]

    print(f"\n{'Model Configuration':<44} | {'Test RMSE':<10} | {'Test MAE':<10}")
    print("-" * 70)
    for name, model in configs:
        if "Beta" in name:
            # Rescale count in 0..10 to (0, 1) rate
            y_train_rate = [val / 10.0 for val in y_train]
            model.fit(X_train, y_train_rate)
            preds_rate = model.predict(X_test)
            preds = [p * 10.0 for p in preds_rate]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rep = metrics.regression(y_test, preds)
        print(f"{name:<44} | {rep['rmse']:<10.3f} | {rep['mae']:<10.3f}")


if __name__ == "__main__":
    run_concrete_benchmark()
    run_insurance_benchmark()
    run_diamonds_benchmark()
    run_wine_quality_benchmark()
    print("\n" + "=" * 70)
    print("Regression methods benchmark showcase finished successfully.")
    print("=" * 70)
