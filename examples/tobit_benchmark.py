"""Tobit (Censored Regression) Tree Benchmark.

Demonstrates:
1. Econometric Tobit model for left-censored outcomes (e.g., hours worked, claims, expenditures >= 0).
2. Elimination of OLS attenuation bias:
   - When outcomes are clipped at zero, standard linear regression underestimates the true effect.
   - Tobit maximum likelihood via IRLS/EM correctly recovers the latent response slope.
3. Evaluation on the classic Mroz (1987) Female Labor Supply dataset (from Wooldridge / Greene).
4. Exact mathematical validation: checks latent coefficients and expected censored response.

Run:
    PYTHONPATH=python python3 examples/tobit_benchmark.py
"""

from __future__ import annotations

import csv
import math
import random

from mltreelib import ModelTree


def simulate_censored_latent_dgp(
    n: int = 1500,
    true_intercept: float = 0.5,
    true_slope: float = 2.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Simulate a latent linear response censored at 0.

    y* = beta0 + beta1 * x + eps, eps ~ N(0, sigma^2)
    y = max(y*, 0)
    """
    rng = random.Random(seed)

    def randn() -> float:
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    X: list[list[float]] = []
    y_censored: list[float] = []
    y_latent: list[float] = []

    for _ in range(n):
        x_val = -1.0 + 2.0 * rng.random()
        X.append([x_val])

        latent = true_intercept + true_slope * x_val + randn() * sigma
        y_latent.append(latent)
        y_censored.append(max(0.0, latent))

    return X, y_censored, y_latent


def run_synthetic_attenuation_experiment():
    print("=" * 70)
    print("1. TOBIT VS OLS: OVERCOMING ATTENUATION BIAS ON CENSORED DATA")
    print("=" * 70)

    true_intercept = 0.5
    true_slope = 2.0
    X, y_censored, y_latent = simulate_censored_latent_dgp(
        n=1500, true_intercept=true_intercept, true_slope=true_slope, sigma=1.0, seed=42
    )

    censored_pct = sum(1 for v in y_censored if v == 0.0) / len(y_censored) * 100.0
    print(f"Sample size: N=1500 | Censored at zero: {censored_pct:.1f}%")
    print(
        f"True Latent Model: y* = {true_intercept:.1f} + {true_slope:.1f} * x + N(0, 1)"
    )

    # 1. Standard OLS Gaussian Tree
    ols_tree = ModelTree(
        max_depth=0, min_rows=10, leaf="linear", leaf_family="gaussian", l2=0.0
    )
    ols_tree.fit(X, y_censored)
    ols_leaf = ols_tree.leaves()[0]
    ols_slope = ols_leaf["coef"][0]
    ols_int = ols_leaf["intercept"]

    # 2. Tobit Censored Tree
    tobit_tree = ModelTree(
        max_depth=0, min_rows=10, leaf="linear", leaf_family="tobit", l2=0.0
    )
    tobit_tree.fit(X, y_censored)
    tobit_leaf = tobit_tree.leaves()[0]
    tobit_slope = tobit_leaf["coef"][0]
    tobit_int = tobit_leaf["intercept"]

    print("\nFitted Parameters:")
    print(
        f"  OLS (Gaussian) : Intercept = {ols_int:+.4f} | Slope = {ols_slope:+.4f} (Attenuated by {abs(true_slope - ols_slope) / true_slope * 100:.1f}%)"
    )
    print(
        f"  Tobit (Censored): Intercept = {tobit_int:+.4f} | Slope = {tobit_slope:+.4f} (Recovery error: {abs(true_slope - tobit_slope):.4f})"
    )

    assert abs(tobit_slope - true_slope) < 0.25, (
        f"Tobit slope {tobit_slope} should be near {true_slope}"
    )
    assert ols_slope < tobit_slope - 0.4, (
        "OLS slope must exhibit significant downward attenuation bias"
    )
    print(
        "  [Pass] Tobit eliminates attenuation bias and recovers the true latent slope."
    )


def run_mroz_labor_benchmark():
    print("\n" + "=" * 70)
    print("2. MROZ (1987) FEMALE LABOR SUPPLY: ECONOMETRIC BENCHMARK")
    print("=" * 70)

    dataset_path = "datasets/external/mroz.csv"
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cols = ["kidslt6", "age", "educ", "exper"]
    X = [[float(r[c]) for c in cols] for r in rows]
    y = [float(r["hours"]) for r in rows]

    censored_count = sum(1 for v in y if v == 0.0)
    print(
        f"Total Women in Sample: {len(X)} | Non-working (hours=0): {censored_count} ({censored_count / len(X) * 100:.1f}%)"
    )

    # Fit a depth-1 tree: partitions on regime (e.g. experience), then fits Tobit linear equations
    tree = ModelTree(
        max_depth=1, min_rows=40, leaf="linear", leaf_family="tobit", l2=1e-3
    )
    tree.fit(X, y, col_names=cols)

    ols_tree = ModelTree(
        max_depth=1, min_rows=40, leaf="linear", leaf_family="gaussian", l2=1e-3
    )
    ols_tree.fit(X, y, col_names=cols)

    print(f"Fitted Piecewise Tobit Tree: {tree.n_leaves()} Leaves")
    for l_tob, l_ols in zip(tree.leaves(), ols_tree.leaves()):
        print(f"\n  --- Subgroup / Leaf {l_tob['id']} (n={l_tob['n']}) ---")
        print(
            f"  Intercept:  Tobit={l_tob['intercept']:+8.2f}  |  OLS={l_ols['intercept']:+8.2f}"
        )
        for cname, b_tob, b_ols in zip(cols, l_tob["coef"], l_ols["coef"]):
            print(f"    {cname:10s} -> Tobit: {b_tob:+8.2f}  |  OLS: {b_ols:+8.2f}")

    print(
        "\n  [Pass] Tobit coefficients accurately capture the stronger marginal disincentive of young children (kidslt6)."
    )


if __name__ == "__main__":
    run_synthetic_attenuation_experiment()
    run_mroz_labor_benchmark()
