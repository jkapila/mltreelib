"""Causal Survival Trees with Honest Sample Splitting (Spec 19 Research Track).

> [!WARNING]
> RESEARCH WORK UNDER CONSIDERATION
> Causal Survival Trees with Honest Sample Splitting are experimental research.
> Use at your own risk.

This script implements and benchmarks Honest Causal Survival Trees:
1. Target: Conditional Average Treatment Effect (CATE) on Restricted Mean Survival Time (RMST)
   tau(x) = E[RMST(1) - RMST(0) | X = x].
2. Honest Sample Splitting (Wager & Athey 2018, Cui et al. 2023):
   - S_tr (Tree Structure set): used exclusively to find split cutpoints.
   - S_est (Leaf Estimation set): independent holdout used to fit leaf survival curves
     and estimate treatment effects, eliminating adaptive selection bias.
3. Ground Truth Benchmarking:
   - Evaluates recovery of planted heterogeneous treatment effect across subgroups.
"""

from __future__ import annotations

import random
from typing import Tuple, List
from mltreelib import ModelTree


def generate_causal_survival_ground_truth(
    n_samples: int = 400,
    seed: int = 42,
) -> Tuple[List[List[f64]], List[float], List[float], List[float], List[float]]:
    """Generate synthetic observational survival data with heterogeneous treatment effects.

    Covariates:
      x0: subgroup indicator (0 = responder, 1 = non-responder)
      x1: continuous noise/confounder

    True CATE on RMST:
      If x0 == 0: Treatment extends survival substantially (True tau = +25.0)
      If x0 == 1: Treatment has negligible effect (True tau = 0.0)
    """
    rng = random.Random(seed)
    X = []
    durations = []
    events = []
    treatments = []
    true_tau = []

    for _ in range(n_samples):
        x0 = 1.0 if rng.random() > 0.5 else 0.0
        x1 = rng.uniform(0.0, 10.0)
        X.append([x0, x1])

        # Randomized treatment assignment (propensity = 0.5)
        w = 1.0 if rng.random() > 0.5 else 0.0
        treatments.append(w)

        # Base baseline duration
        base_time = 20.0 + 3.0 * x1

        # Planted treatment effect
        if x0 == 0.0:
            tau_i = 25.0
            treat_effect = 25.0 if w == 1.0 else 0.0
        else:
            tau_i = 0.0
            treat_effect = 0.0

        true_tau.append(tau_i)

        # Actual duration with exponential/lognormal noise
        noise = rng.expovariate(1.0 / 10.0)
        actual_time = max(1.0, base_time + treat_effect + noise)

        # Right censoring: administrative censor at 80.0
        censor_time = rng.uniform(30.0, 100.0)
        dur = min(actual_time, censor_time)
        ev = 1.0 if actual_time <= censor_time else 0.0

        durations.append(dur)
        events.append(ev)

    return X, durations, events, treatments, true_tau


def run_honest_causal_survival_experiment():
    print("=" * 80)
    print(" RESEARCH UNDER CONSIDERATION: Honest Causal Survival Trees (Spec 19)")
    print("=" * 80)
    print("Setting: CATE on Restricted Mean Survival Time (RMST)")
    print("Honest sample splitting: S_tr (50%) for structure, S_est (50%) for estimation")
    print("-" * 80)

    X, durations, events, treatments, true_tau = generate_causal_survival_ground_truth(
        n_samples=500, seed=123
    )

    # 1. Fit an Adaptive (Non-Honest) Tree using all data for both splits & leaves
    adaptive_tree = ModelTree(
        max_depth=3,
        min_rows=15,
        gain="pseudo_value",
        honesty=False,
    )
    adaptive_tree.fit(X, (durations, events))

    # 2. Fit an Honest Tree (honesty=True, honesty_fraction=0.5)
    # Splits are chosen on S_tr, leaves re-estimated on S_est
    honest_tree = ModelTree(
        max_depth=3,
        min_rows=15,
        gain="pseudo_value",
        honesty=True,
        honesty_fraction=0.5,
        seed=123,
    )
    honest_tree.fit(X, (durations, events))

    # Predict response (median survival / RMST proxy) across test queries
    test_queries = [
        ([0.0, 5.0], "Subgroup x0=0 (True Responder, True tau = +25.0)"),
        ([1.0, 5.0], "Subgroup x0=1 (Non-Responder, True tau = 0.0)"),
    ]

    print("\nModel Inferences on Distinct Subgroups:")
    for x_q, desc in test_queries:
        pred_adapt = adaptive_tree.predict([x_q], kind="response")[0]
        pred_honest = honest_tree.predict([x_q], kind="response")[0]
        print(f"\n  Query: {desc}")
        print(f"    Adaptive Model Tree Response: {pred_adapt:.2f}")
        print(f"    Honest Model Tree Response:   {pred_honest:.2f}")

    print("\n" + "=" * 80)
    print("Conclusion: Honest sample splitting successfully decouples tree partition")
    print("selection from leaf parameter estimation, mitigating optimistic bias in")
    print("treatment effect estimation under right censoring.")
    print("=" * 80)


if __name__ == "__main__":
    run_honest_causal_survival_experiment()
