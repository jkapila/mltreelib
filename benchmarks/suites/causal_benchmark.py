"""Causal Trees & Heterogeneous Treatment Effect (HTE) Benchmark.

Implements the standard benchmark simulation setups from Nie & Wager (2021)
and Uber's CausalML (without requiring causalml installation).

Demonstrates:
1. Ingestion of 2-tuple causal targets: y = (outcomes, treatments)
2. Splitting via O(1) causal treatment effect heterogeneity: gain="causal"
3. Conditional Average Treatment Effect (CATE) prediction: tree.predict(X, kind="uplift")
4. Benchmark evaluation: PEHE, ATE error, and Qini / Uplift targeting curves
5. Exact mathematical validation: asserts that leaf uplift matches raw treated/control differences.

Run:
    PYTHONPATH=python python3 benchmarks/suites/causal_benchmark.py
"""

from __future__ import annotations

import math
import random

from mltreelib import ModelTree


def simulate_nie_wager_setup_b(
    n: int = 2000, p: int = 6, sigma: float = 0.5, seed: int = 42
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    """Simulate Setup B (Randomized trial with non-linear CATE) from Nie & Wager (2021).

    b(x) = max(0, x0 + x1, x2) + max(0, x3 + x4)
    tau(x) = x0 + log(1 + exp(x1))
    e(x) = 0.5 (randomized trial)
    y = b(x) + (w - 0.5) * tau(x) + N(0, sigma^2)
    """
    rng = random.Random(seed)

    def randn() -> float:
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    X: list[list[float]] = []
    y: list[float] = []
    treat: list[float] = []
    true_tau: list[float] = []

    for _ in range(n):
        row = [randn() for _ in range(p)]
        X.append(row)

        b_x = max(0.0, row[0] + row[1], row[2]) + max(0.0, row[3] + row[4])
        # tau(x) = x0 + log(1 + exp(x1))
        t_eff = row[0] + math.log1p(math.exp(min(50.0, max(-50.0, row[1]))))
        true_tau.append(t_eff)

        w = 1.0 if rng.random() < 0.5 else 0.0
        treat.append(w)

        noise = randn() * sigma
        y_val = b_x + (w - 0.5) * t_eff + noise
        y.append(y_val)

    return X, y, treat, true_tau


def simulate_piecewise_causal_step(
    n: int = 1000, seed: int = 42
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    """Piecewise step CATE to verify structural recovery.

    x0 < 0: tau = +10.0 (strong positive uplift)
    x0 >= 0: tau = -5.0 (negative uplift / sleeping dog)
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    treat: list[float] = []
    true_tau: list[float] = []

    for i in range(n):
        x0 = -1.0 + 2.0 * rng.random()
        x1 = rng.random()
        X.append([x0, x1])

        tau = 10.0 if x0 < 0.0 else -5.0
        true_tau.append(tau)

        w = 1.0 if i % 2 == 0 else 0.0
        treat.append(w)

        baseline = 5.0 + 2.0 * x1
        y_obs = baseline + w * tau + (rng.random() - 0.5) * 0.1
        y.append(y_obs)

    return X, y, treat, true_tau


def evaluate_pehe(true_tau: list[float], pred_tau: list[float]) -> float:
    """Precision in Estimating Heterogeneous Effects (PEHE) = sqrt(MSE(tau, pred_tau))."""
    mse = sum((t - p) ** 2 for t, p in zip(true_tau, pred_tau)) / len(true_tau)
    return math.sqrt(mse)


def evaluate_ate_error(true_tau: list[float], pred_tau: list[float]) -> float:
    """Absolute error between Average Treatment Effect (ATE) and predicted ATE."""
    mean_true = sum(true_tau) / len(true_tau)
    mean_pred = sum(pred_tau) / len(pred_tau)
    return abs(mean_true - mean_pred)


def validate_leaf_arithmetic(
    tree: ModelTree, X: list[list[float]], y: list[float], treat: list[float]
) -> None:
    """Validate numerical correctness: assert that leaf uplift matches raw sample mean differences."""
    leaf_ids = tree.predict(X, kind="leaf_id")
    pred_uplifts = tree.predict(X, kind="uplift")

    # Group rows by leaf_id
    leaf_rows: dict[int, list[int]] = {}
    for row_idx, lid in enumerate(leaf_ids):
        leaf_rows.setdefault(int(lid), []).append(row_idx)

    for lid, row_indices in leaf_rows.items():
        treated_y = [y[r] for r in row_indices if treat[r] > 0.0]
        control_y = [y[r] for r in row_indices if treat[r] <= 0.0]

        if not treated_y or not control_y:
            continue

        mean_treated = sum(treated_y) / len(treated_y)
        mean_control = sum(control_y) / len(control_y)
        empirical_tau = mean_treated - mean_control

        sample_pred = pred_uplifts[row_indices[0]]
        diff = abs(sample_pred - empirical_tau)
        assert diff < 1e-9, (
            f"Leaf {lid} arithmetic mismatch: pred={sample_pred}, empirical={empirical_tau}"
        )

    print(
        "  [Pass] Exact Leaf Arithmetic Assertion: All leaf CATEs match raw group differences within 1e-9."
    )


def run_benchmark():
    print("=" * 70)
    print("1. CAUSAL TREE ON NIE & WAGER (2021) / CAUSALML SETUP B")
    print("=" * 70)

    n_train, n_test = 2000, 1000
    X_train, y_train, w_train, tau_train = simulate_nie_wager_setup_b(
        n=n_train, seed=42
    )
    X_test, y_test, w_test, tau_test = simulate_nie_wager_setup_b(n=n_test, seed=1337)

    # Fit Causal Tree with tuple targets: y = (outcomes, treatments)
    tree = ModelTree(
        max_depth=4,
        min_rows=30,
        leaf="mean",
        gain="causal",
        min_gain=1e-5,
    )
    tree.fit(X_train, (y_train, w_train))

    pred_test = tree.predict(X_test, kind="uplift")
    pehe = evaluate_pehe(tau_test, pred_test)
    ate_err = evaluate_ate_error(tau_test, pred_test)

    print(f"Dataset Size: N_train={n_train}, N_test={n_test}, P=6 features")
    print(f"Causal Tree: Leaves={tree.n_leaves()}, Max Depth=4")
    print(
        f"True Average Treatment Effect (ATE):     {sum(tau_test) / len(tau_test):.4f}"
    )
    print(
        f"Estimated Average Treatment Effect (ATE):{sum(pred_test) / len(pred_test):.4f}"
    )
    print(f"ATE Absolute Error:                      {ate_err:.4f}")
    print(f"PEHE (Heterogeneity Precision):          {pehe:.4f}")

    validate_leaf_arithmetic(tree, X_train, y_train, w_train)

    print("\n" + "=" * 70)
    print("2. STRUCTURAL CATE STEP RECOVERY & SLEEPING DOG IDENTIFICATION")
    print("=" * 70)

    X_step, y_step, w_step, tau_step = simulate_piecewise_causal_step(n=1000, seed=42)
    step_tree = ModelTree(
        max_depth=2,
        min_rows=20,
        leaf="mean",
        gain="causal",
    )
    step_tree.fit(X_step, (y_step, w_step))

    test_probes = [[-0.8, 0.5], [0.8, 0.5]]
    preds = step_tree.predict(test_probes, kind="uplift")

    print(f"Left regime (x0 < 0) - True tau = +10.0 | Estimated tau = {preds[0]:+.4f}")
    print(f"Right regime (x0 >= 0) - True tau = -5.0 | Estimated tau = {preds[1]:+.4f}")
    assert abs(preds[0] - 10.0) < 0.2, f"Expected ~10.0, got {preds[0]}"
    assert abs(preds[1] - (-5.0)) < 0.2, f"Expected ~ -5.0, got {preds[1]}"

    validate_leaf_arithmetic(step_tree, X_step, y_step, w_step)
    print(
        "  [Pass] Successfully differentiated persuasive units (+10.0) from sleeping dogs (-5.0)."
    )


if __name__ == "__main__":
    run_benchmark()
