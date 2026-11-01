"""Stump testbed: frozen-topology retraining across leaf model families.

Demonstrates using a single decision stump (max_depth=1) as an isolated
laboratory to fit and evaluate any leaf family without re-running split searches.

Run:
    uv run python examples/stump_leaf_retrain.py
"""

import time
from mltreelib import ModelTree, LeafControl


def main():
    print("=== Stump Retrain Laboratory ===")

    # 1. Generate 2-regime dataset partitioned strictly by x0 at 0.0
    # Left regime (x0 < 0):   y = 10.0 + 2.0 * x1
    # Right regime (x0 >= 0):  y = 50.0 + 5.0 * x1
    n = 200
    x = []
    y_reg = []
    for i in range(n):
        x0 = (i - 100) / 50.0  # Range -2.0 to +1.98
        x1 = (i % 10) * 0.5     # Range 0.0 to 4.5
        if x0 < 0.0:
            y_val = 10.0 + 2.0 * x1
        else:
            y_val = 50.0 + 5.0 * x1
        x.append([x0, x1])
        y_reg.append(y_val)

    # 2. Fit base stump (max_depth=1)
    t0 = time.perf_counter()
    stump = ModelTree(max_depth=1, min_rows=10, leaf="linear", l2=1e-3, prune="none")
    stump.fit(x, y_reg)
    fit_ms = (time.perf_counter() - t0) * 1000.0

    splits = stump.splits()
    print(f"Base fit time: {fit_ms:.2f} ms")
    print(f"Found {len(splits)} split: col={splits[0]['col']}, threshold={splits[0]['threshold']:.4f}")
    assert len(splits) == 1

    # 3. Retrain on frozen stump across leaf models
    test_points = [[-1.0, 2.0], [1.0, 2.0]]

    # A. Constant Mean leaves
    t0 = time.perf_counter()
    stump.retrain(x, y_reg, control=LeafControl(leaf="mean"))
    retrain_ms = (time.perf_counter() - t0) * 1000.0
    preds_mean = stump.predict(test_points)
    print(f"\n[Mean Leaf] Retrain time: {retrain_ms:.3f} ms")
    print(f"  Predictions at [-1, 2] and [+1, 2]: {[round(p, 2) for p in preds_mean]}")
    assert len(stump.splits()) == 1, "Split must remain unchanged"

    # B. Linear leaves with slight ridge (recovers slopes ~2.0 and ~5.0)
    t0 = time.perf_counter()
    stump.retrain(x, y_reg, control=LeafControl(leaf="linear", l2=1e-4))
    retrain_ms = (time.perf_counter() - t0) * 1000.0
    preds_linear = stump.predict(test_points)
    leaves = stump.leaves()
    print(f"\n[Linear Leaf (l2=1e-4)] Retrain time: {retrain_ms:.3f} ms")
    print(f"  Left leaf:  intercept={leaves[0]['intercept']:.2f}, coefs={leaves[0]['coef']}")
    print(f"  Right leaf: intercept={leaves[1]['intercept']:.2f}, coefs={leaves[1]['coef']}")
    print(f"  Predictions at [-1, 2] and [+1, 2]: {[round(p, 2) for p in preds_linear]}")

    # C. Linear leaves with heavy ridge penalty (l2=100.0) -> coefficients shrink
    stump.retrain(x, y_reg, leaf="linear", l2=100.0)
    leaves_ridge = stump.leaves()
    print(f"\n[Linear Leaf (l2=100.0)]")
    print(f"  Left leaf:  intercept={leaves_ridge[0]['intercept']:.2f}, coefs={leaves_ridge[0]['coef']}")
    print(f"  Right leaf: intercept={leaves_ridge[1]['intercept']:.2f}, coefs={leaves_ridge[1]['coef']}")

    # D. Robust Huber M-estimator under outlier pollution
    y_polluted = list(y_reg)
    y_polluted[0] = 10000.0  # Extreme outlier in left regime
    stump.retrain(x, y_polluted, leaf="huber")
    preds_huber = stump.predict(test_points)
    stump.retrain(x, y_polluted, leaf="mean")
    preds_polluted_mean = stump.predict(test_points)
    print(f"\n[Robust Huber Leaf vs Polluted Mean]")
    print(f"  Huber left prediction: {preds_huber[0]:.2f} (unaffected by 10,000.0 outlier)")
    print(f"  Mean left prediction:  {preds_polluted_mean[0]:.2f} (dragged upward)")

    # E. Empirical Quantile leaves (tau = 0.1, 0.5, 0.9)
    stump.retrain(x, y_reg, leaf_digest=True)
    q10 = stump.quantile(test_points, 0.10)
    q50 = stump.quantile(test_points, 0.50)
    q90 = stump.quantile(test_points, 0.90)
    print(f"\n[Quantile Bounds on Frozen Stump]")
    print(f"  Left  regime 10th-50th-90th: {q10[0]:.2f} <= {q50[0]:.2f} <= {q90[0]:.2f}")
    print(f"  Right regime 10th-50th-90th: {q10[1]:.2f} <= {q50[1]:.2f} <= {q90[1]:.2f}")

    # F. Retrain with Binomial leaf family on binary targets
    y_bin = [0.0 if i < 100 else 1.0 for i in range(n)]
    stump.retrain(x, y_bin, leaf_family="binomial", leaf="mean")
    probs = stump.predict_proba(test_points)
    labels = stump.predict_label(test_points, threshold=0.5)
    print(f"\n[Binomial Logit Family on Frozen Stump]")
    print(f"  Probabilities: {probs}")
    print(f"  Hard Labels:   {labels}")
    assert probs[0] == 0.0 and probs[1] == 1.0
    assert labels == [0, 1]

    # G. Retrain with Poisson count family
    y_counts = [2.0 if i < 100 else 20.0 for i in range(n)]
    stump.retrain(x, y_counts, leaf_family="poisson", leaf="mean")
    poisson_rates = stump.predict(test_points)
    print(f"\n[Poisson Log Family on Frozen Stump]")
    print(f"  Estimated rates: {poisson_rates}")
    assert abs(poisson_rates[0] - 2.0) < 1e-4
    assert abs(poisson_rates[1] - 20.0) < 1e-4

    print("\nAll stump retrain laboratory tests passed successfully!")


if __name__ == "__main__":
    main()
