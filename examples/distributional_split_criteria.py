"""Demonstrate the 5 new splitting criteria on standard benchmark datasets.

Splitting Criteria Covered:
1. Tweedie: Unit quasi-deviance on zero-inflated continuous data (Fair's Affairs).
2. Heteroscedastic: Dual-parameter Gaussian NLL deviance on variance regimes (identical means).
3. Tobit: Censored hurdle deviance on left-censored continuous data (Mroz female labor supply).
4. CV: Coefficient of variation reduction on scale-heterogeneous targets (Diamonds / Price data).
5. Martingale: Survival martingale residual variance reduction (Veteran lung cancer trial).

Zero external dependencies (uses standard library csv and random).
"""

import csv
import math
import random
from mltreelib import ModelTree


def run_tweedie_split_example():
    print("=" * 70)
    print("1. TWEEDIE SPLIT CRITERION: Fair's Affairs (Zero-Inflated Continuous)")
    print("=" * 70)
    with open("datasets/external/affairs.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    X = [[float(r[c]) for c in cols] for r in rows]
    y = [float(r["affairs"]) for r in rows]

    zero_count = sum(1 for v in y if v == 0.0)
    zero_pct = (zero_count / len(y)) * 100.0
    print(f"Data: {len(X)} rows | Zero mass: {zero_pct:.1f}%")

    # Fit tree with Tweedie unit quasi-deviance split criterion (p = 1.5)
    tree = ModelTree(
        max_depth=3,
        min_rows=25,
        gain="tweedie",
        tweedie_p=1.5,
        leaf="mean",
        prune="none",
    )
    tree.fit(X, y, col_names=cols)

    print(f"Tree grown with gain='tweedie' (p=1.5) -> Leaves: {tree.n_leaves()}")
    for s in tree.splits():
        col_name = str(s["col"])
        print(f"  Split on {col_name:15s} at threshold {s['threshold']:.2f} (quasi-deviance gain = {s['gain']:.4f})")
    preds = tree.predict(X[:5])
    print(f"Sample predictions: {[round(p, 4) for p in preds]}")


def run_heteroscedastic_split_example():
    print("\n" + "=" * 70)
    print("2. HETEROSCEDASTIC SPLIT CRITERION: Detecting Pure Variance Regimes")
    print("=" * 70)
    # 200 samples: left regime has low variance (sigma=0.2), right regime has high variance (sigma=4.0)
    # Both regimes have EXACTLY zero conditional mean (E[Y|X] = 0).
    # Standard SSE/MSE yields ZERO gain and cannot split! Heteroscedastic Gaussian NLL isolates it.
    rng = random.Random(42)
    n = 200
    x_vals = [-5.0 + 10.0 * (i / (n - 1)) for i in range(n)]
    X = [[v] for v in x_vals]
    y = [
        rng.gauss(0.0, 0.2 if v < 0.0 else 4.0)
        for v in x_vals
    ]

    # Standard MSE tree
    tree_mse = ModelTree(max_depth=2, min_rows=20, gain="sse", leaf="mean", prune="none")
    tree_mse.fit(X, y)

    # Heteroscedastic Gaussian NLL tree
    tree_hetero = ModelTree(max_depth=2, min_rows=20, gain="heteroscedastic", leaf="mean", prune="none")
    tree_hetero.fit(X, y)

    print(f"Data: 200 rows with identical conditional mean 0.0, but sigma(left)=0.2 vs sigma(right)=4.0")
    print(f"  Standard SSE tree leaves:        {tree_mse.n_leaves()} (splits={len(tree_mse.splits())})")
    print(f"  Heteroscedastic tree leaves:     {tree_hetero.n_leaves()} (splits={len(tree_hetero.splits())})")
    for s in tree_hetero.splits():
        print(f"    Split on X0 at threshold {s['threshold']:.2f} (NLL gain = {s['gain']:.4f})")


def run_tobit_split_example():
    print("\n" + "=" * 70)
    print("3. TOBIT SPLIT CRITERION: Mroz Labor Supply (Censored at 0)")
    print("=" * 70)
    with open("datasets/external/mroz.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cols = ["kidslt6", "kidsge6", "age", "educ", "nwifeinc", "exper"]
    X = [[float(r[c]) for c in cols] for r in rows]
    y = [float(r["hours"]) for r in rows]

    censored_count = sum(1 for v in y if v == 0.0)
    censored_pct = (censored_count / len(y)) * 100.0
    print(f"Data: {len(X)} rows | Left-censored at 0: {censored_pct:.1f}%")

    tree = ModelTree(
        max_depth=3,
        min_rows=30,
        gain="tobit",
        tobit_threshold=0.0,
        leaf="mean",
        prune="none",
    )
    tree.fit(X, y, col_names=cols)

    print(f"Tree grown with gain='tobit' (threshold=0.0) -> Leaves: {tree.n_leaves()}")
    for s in tree.splits():
        col_name = str(s["col"])
        print(f"  Split on {col_name:15s} at threshold {s['threshold']:.2f} (tobit hurdle gain = {s['gain']:.4f})")


def run_cv_split_example():
    print("\n" + "=" * 70)
    print("4. CV SPLIT CRITERION: Scale-Invariant Coefficient of Variation Reduction")
    print("=" * 70)
    # 200 samples: left regime has mean=10, sd=5 (CV = 0.50)
    # Right regime has mean=100, sd=10 (CV = 0.10)
    # Parent has mixed populations where CV is much higher (~0.82)
    rng = random.Random(123)
    n = 200
    X = []
    y = []
    for i in range(n):
        x_val = -3.0 + 6.0 * (i / (n - 1))
        X.append([x_val])
        if x_val < 0.0:
            val = max(1.0, 10.0 + rng.gauss(0.0, 5.0))   # mean ~ 10, sd ~ 5 -> CV ~ 0.50
        else:
            val = max(10.0, 100.0 + rng.gauss(0.0, 10.0)) # mean ~ 100, sd ~ 10 -> CV ~ 0.10
        y.append(val)

    tree_cv = ModelTree(
        max_depth=2,
        min_rows=30,
        gain="cv",
        leaf="mean",
        prune="none",
    )
    tree_cv.fit(X, y, col_names=["X0"])

    print(f"Data: 200 rows with distinct scale regimes (CV_L=0.50 vs CV_R=0.10)")
    print(f"Tree grown with gain='cv' -> Leaves: {tree_cv.n_leaves()}")
    for s in tree_cv.splits():
        print(f"  Split on X0 at threshold {s['threshold']:.2f} (CV reduction = {s['gain']:.4f})")


def run_martingale_split_example():
    print("\n" + "=" * 70)
    print("5. MARTINGALE SPLIT CRITERION: Veteran Lung Cancer Survival Residuals")
    print("=" * 70)
    with open("datasets/external/veteran.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cols = ["karno", "diagtime", "age", "prior"]
    X = [[float(r[c]) for c in cols] for r in rows]
    times = [float(r["time"]) for r in rows]
    events = [float(r["status"]) for r in rows]

    # Precompute Nelson-Aalen cumulative hazard under null model H0
    # Martingale residual: M_i = E_i - Lambda_0(T_i)
    indices = list(range(len(times)))
    indices.sort(key=lambda idx: times[idx])
    n = len(times)
    cum_hazard = 0.0
    hazard_at_time = {}
    for i, idx in enumerate(indices):
        t = times[idx]
        e = events[idx]
        at_risk = n - i
        cum_hazard += e / at_risk
        hazard_at_time[t] = cum_hazard

    martingale_residuals = [
        float(e - hazard_at_time[t])
        for t, e in zip(times, events)
    ]

    mean_res = sum(martingale_residuals) / len(martingale_residuals)

    tree = ModelTree(
        max_depth=2,
        min_rows=20,
        gain="martingale",
        leaf="mean",
        prune="none",
    )
    tree.fit(X, martingale_residuals, col_names=cols)

    print(f"Data: Veteran Trial (n={len(X)}) | Martingale residuals E[M] = {mean_res:.4f}")
    print(f"Tree grown with gain='martingale' -> Leaves: {tree.n_leaves()}")
    for s in tree.splits():
        col_name = str(s["col"])
        print(f"  Split on {col_name:15s} at threshold {s['threshold']:.2f} (Martingale SSE gain = {s['gain']:.4f})")


if __name__ == "__main__":
    run_tweedie_split_example()
    run_heteroscedastic_split_example()
    run_tobit_split_example()
    run_cv_split_example()
    run_martingale_split_example()
