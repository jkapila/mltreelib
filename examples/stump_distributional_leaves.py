"""Showcase of Distributional & Survival Stump Leaves in mltreelib.

Demonstrates the 3 new stump leaf families added in Spec 08:
1. Softmax (C >= 3): Multi-class probability vectors and class responses.
2. Survival (Nelson-Aalen & Kaplan-Meier): Right-censored survival curves,
   cumulative hazard, and median survival on the Veteran lung cancer study.
3. Heteroscedastic: Single-leaf location-scale estimation (mu, sigma) with
   conditional volatility bands.
4. Microsecond Frozen-Topology Retraining across entity slices.

Run:
    uv run python examples/stump_distributional_leaves.py
"""

import math
import random
import time
from mltreelib import ModelTree, LeafControl


def showcase_softmax():
    print("\n" + "=" * 78)
    print("1. Softmax Leaf: Multi-Class Probability Vectors (C=3)")
    print("=" * 78)

    # Synthetic 3-class data
    random.seed(42)
    X = []
    y = []
    classes = ["low_risk", "medium_risk", "high_risk"]
    for _ in range(300):
        c = random.choice([0, 1, 2])
        # Feature correlated with class
        f1 = c * 2.0 + random.gauss(0, 0.8)
        f2 = -c * 1.5 + random.gauss(0, 0.5)
        X.append([f1, f2])
        y.append(float(c))

    # Fit a Softmax stump (empirical distribution)
    tree = ModelTree(max_depth=0, leaf_family="softmax", min_rows=10)
    tree.fit(X, y)

    leaves = tree.leaves()
    leaf = leaves[0]
    print(f"Leaf samples: {leaf['n']}")
    print(f"Fitted Softmax Probs: {[round(p, 4) for p in leaf['softmax_probs']]}")

    # Unified predict queries
    test_X = [[0.0, 0.0], [2.0, -1.5], [4.0, -3.0]]
    pred_labels = tree.predict(test_X, kind="response")
    pred_probs = tree.predict(test_X, kind="proba")

    for i, (feat, lbl, probs) in enumerate(zip(test_X, pred_labels, pred_probs)):
        prob_str = ", ".join(f"{classes[c]}: {probs[c]:.3f}" for c in range(3))
        print(f"Sample {i+1} {feat} -> Predicted Class {int(lbl)} ({classes[int(lbl)]}) | Probabilities: [{prob_str}]")


def showcase_survival():
    print("\n" + "=" * 78)
    print("2. Survival Leaf: Veteran's Administration Lung Cancer Study")
    print("=" * 78)

    # Load veteran dataset
    veteran_path = "datasets/external/veteran.csv"
    times = []
    events = []
    X = []

    with open(veteran_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        time_idx = header.index("time")
        status_idx = header.index("status")
        karno_idx = header.index("karno")
        age_idx = header.index("age")

        for line in lines[1:]:
            parts = line.strip().split(",")
            t = float(parts[time_idx])
            e = float(parts[status_idx])
            karno = float(parts[karno_idx])
            age = float(parts[age_idx])
            times.append(t)
            events.append(e)
            X.append([karno, age])

    # Fit a Survival stump
    tree = ModelTree(max_depth=0, leaf_family="survival", min_rows=10)
    tree.fit(X, times, event=events)

    leaf = tree.leaves()[0]
    print(f"Cohort Size: {leaf['n']} patients")
    print(f"Median Survival Time: {leaf['median_time']:.1f} days")
    print(f"Distinct Event Times Logged: {len(leaf['survival_times'])} checkpoints")

    # Evaluate survival probability and cumulative hazard across key milestones
    eval_horizons = [30.0, 90.0, 180.0, 365.0]
    test_patient = [[50.0, 60.0]]  # Karnofsky score 50, Age 60
    
    surv_probs = tree.predict(test_patient, kind="survival", times=eval_horizons)[0]
    hazards = tree.predict(test_patient, kind="hazard", times=eval_horizons)[0]

    print("\nKaplan-Meier Survival & Nelson-Aalen Cumulative Hazard Milestones:")
    print(f"{'Days':<10} | {'S(t) Probability':<20} | {'H(t) Cum. Hazard':<20}")
    print("-" * 56)
    for day, s, h in zip(eval_horizons, surv_probs, hazards):
        print(f"{day:<10.0f} | {s:<20.4f} | {h:<20.4f}")


def showcase_heteroscedastic():
    print("\n" + "=" * 78)
    print("3. Heteroscedastic Leaf: Joint Location & Scale Estimation (mu, sigma)")
    print("=" * 78)

    random.seed(123)
    X = []
    y = []

    # True parameters: mu = 15.0, sigma = 3.5
    for _ in range(500):
        val = random.gauss(15.0, 3.5)
        X.append([1.0])
        y.append(val)

    tree = ModelTree(max_depth=0, leaf_family="heteroscedastic", min_rows=20)
    tree.fit(X, y)

    leaf = tree.leaves()[0]
    mu_hat = leaf["mean"]
    sigma_hat = leaf["scale"]
    print(f"True Process:   mu = 15.000, sigma = 3.500")
    print(f"Estimated Leaf: mu = {mu_hat:.3f}, sigma = {sigma_hat:.3f}")

    # Unified predict(kind="dist")
    dists = tree.predict([[1.0]], kind="dist")
    mu, sigma = dists[0]
    ci_low = mu - 1.96 * sigma
    ci_high = mu + 1.96 * sigma
    print(f"Predict Kind 'dist' Output: mu = {mu:.3f}, sigma = {sigma:.3f}")
    print(f"95% Volatility / Predictive Interval: [{ci_low:.3f}, {ci_high:.3f}]")


def showcase_retrain_benchmark():
    print("\n" + "=" * 78)
    print("4. High-Performance Frozen-Topology Retraining on Distributional Leaves")
    print("=" * 78)

    # Benchmark retraining speed on 10,000 observations
    random.seed(42)
    X = [[random.uniform(0, 10)] for _ in range(10000)]
    y_reg = [random.gauss(100, 15) for _ in range(10000)]

    tree = ModelTree(max_depth=0, leaf_family="heteroscedastic")
    t0 = time.perf_counter()
    tree.fit(X, y_reg)
    fit_time = (time.perf_counter() - t0) * 1000.0

    # Retrain on updated data
    y_updated = [random.gauss(120, 25) for _ in range(10000)]
    t1 = time.perf_counter()
    tree.retrain(X, y_updated)
    retrain_time = (time.perf_counter() - t1) * 1000.0

    leaf = tree.leaves()[0]
    print(f"Initial fit on 10,000 rows:   {fit_time:.3f} ms (mu={leaf['mean']:.2f}, sigma={leaf['scale']:.2f})")
    print(f"Retrain on 10,000 rows:       {retrain_time:.3f} ms (mu={leaf['mean']:.2f}, sigma={leaf['scale']:.2f})")
    print("Zero-allocation frozen topology refits leaves instantaneously for streaming & hierarchical routing.")
    print("=" * 78)


if __name__ == "__main__":
    showcase_softmax()
    showcase_survival()
    showcase_heteroscedastic()
    showcase_retrain_benchmark()
