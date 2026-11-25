"""Suite 9: Survival and Competing Risks Trees (Spec 10).

Comprehensive authoritative showcase for time-to-event and multi-cause modeling:
1. Split Criteria Comparison:
   - LogRank (Standard 2-sample log-rank)
   - Tarone-Ware (sqrt(N) weighting for intermediate divergence)
   - Peto-Prentice (S(t) weighting for early divergence)
   - Martingale (O(N) streaming least-squares scan on node residuals)
   - GrayTest (Subdistribution log-rank test for competing risks)
   - SurvivalBrier (Proper separable IPCW scoring rule)
2. Competing Risks & Multi-State Estimation:
   - Multi-cause data contract: E_i in {0, 1, 2, ..., J}
   - Aalen-Johansen Cumulative Incidence Functions (CIF)
   - Probability simplex invariant: sum_c F_c(t) + S(t) == 1.0 (verified to machine epsilon)
3. Prediction Dispatches:
   - predict(X, kind="survival", times=...) -> Event-free survival curve S(t)
   - predict(X, kind="hazard", times=...)   -> Nelson-Aalen cumulative hazard Lambda(t)
   - predict(X, kind="cif", times=..., cause=...) -> Cause-specific incidence F_c(t)
   - predict(X, kind="risk")               -> Continuous mortality/failure hazard score
   - predict(X, kind="response")           -> Median survival time
4. IPCW Discrimination & Calibration Metrics:
   - Cause-specific Harrell's C-index
   - Uno's IPCW C-index (censoring-adjusted rank concordance)
   - Cause-specific Integrated Brier Score (IBS)
   - Time-dependent Brier score matrices across evaluation horizons

Produces: docs/assets/suite9_survival_competing_risks.png
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mltreelib import ModelTree, Metrics, competing_risks_report


OUT_ASSETS = Path("docs/assets")
OUT_ASSETS.mkdir(parents=True, exist_ok=True)


def generate_multicause_cohort(n_samples=800, seed=42):
    """Generate realistic competing risks cohort with clear risk strata.

    Covariates:
    - x0: Primary biomarker (continuous; elevates Cause 1 hazard)
    - x1: Age in decades (continuous; elevates Cause 2 hazard)
    - x2: Binary risk factor (elevates overall failure rate)
    """
    rng = np.random.default_rng(seed)

    x0 = rng.normal(0.0, 1.0, size=n_samples)
    x1 = rng.uniform(4.0, 8.5, size=n_samples)
    x2 = rng.binomial(1, 0.4, size=n_samples).astype(float)

    X = [[float(x0[i]), float(x1[i]), float(x2[i])] for i in range(n_samples)]

    rate_cause1 = np.exp(0.85 * x0 + 0.3 * x2 - 2.2)
    rate_cause2 = np.exp(0.55 * (x1 - 5.5) - 2.4)
    rate_censor = 0.04

    t1 = rng.exponential(1.0 / np.maximum(rate_cause1, 1e-4))
    t2 = rng.exponential(1.0 / np.maximum(rate_cause2, 1e-4))
    tc = rng.exponential(1.0 / rate_censor, size=n_samples)

    durations = []
    events = []

    for i in range(n_samples):
        min_t = min(t1[i], t2[i], tc[i])
        durations.append(float(np.clip(min_t, 0.1, 40.0)))
        if min_t == t1[i]:
            events.append(1.0)  # Primary cause
        elif min_t == t2[i]:
            events.append(2.0)  # Competing cause
        else:
            events.append(0.0)  # Censored

    return X, durations, events


def run_suite_survival():
    print("=" * 80)
    print("mltreelib Suite 9: Survival and Competing Risks Modeling")
    print("=" * 80)

    # 1. Dataset Generation
    X, durations, events = generate_multicause_cohort(n_samples=800, seed=42)
    n_total = len(events)
    n_c1 = sum(1 for e in events if e == 1.0)
    n_c2 = sum(1 for e in events if e == 2.0)
    n_cens = sum(1 for e in events if e == 0.0)

    print(f"Cohort Size: {n_total:,} records")
    print(f"  - Cause 1 (Primary Relapse): {n_c1:,} ({n_c1/n_total*100:.1f}%)")
    print(f"  - Cause 2 (Competing Death): {n_c2:,} ({n_c2/n_total*100:.1f}%)")
    print(f"  - Censored Event-Free:       {n_cens:,} ({n_cens/n_total*100:.1f}%)")

    # Train / Test split (75/25)
    split_idx = 600
    X_train, X_test = X[:split_idx], X[split_idx:]
    t_train, t_test = durations[:split_idx], durations[split_idx:]
    e_train, e_test = events[:split_idx], events[split_idx:]
    feature_names = ["biomarker", "age", "comorbidity"]

    # 2. Benchmarking Split Criteria
    print("\n[Part 1] Comparing Survival Split Criteria on Primary Event...")
    criteria = [
        ("logrank", "Standard Log-Rank (Unweighted)", {}),
        ("tarone_ware", "Tarone-Ware Weighted (w = sqrt(N))", {}),
        ("peto", "Peto-Prentice Weighted (w = S(t))", {}),
        ("martingale", "O(N) Streaming Martingale Residuals", {}),
        ("gray", "Gray's Subdistribution Test (cause=1)", {"target_cause": 1}),
        ("survival_brier", "Proper Separable IPCW Scoring Rule", {}),
    ]

    time_horizons = [2.0, 5.0, 10.0, 20.0, 30.0]
    metrics = Metrics()

    print(f"{'Criterion':<22} | {'Leaves':<6} | {'Harrell C':<10} | {'Uno IPCW C':<10} | {'IBS':<8}")
    print("-" * 65)

    trees = {}
    for crit_key, crit_label, extra_kwargs in criteria:
        tree = ModelTree(max_depth=3, min_rows=25, gain=crit_key, **extra_kwargs)
        tree.fit(X_train, (t_train, e_train), col_names=feature_names)
        trees[crit_key] = tree

        cif1 = tree.predict(X_test, kind="cif", times=time_horizons, cause=1)
        cif2 = tree.predict(X_test, kind="cif", times=time_horizons, cause=2)
        risk = tree.predict(X_test, kind="risk")

        rep = metrics.competing_risks(
            durations=t_test,
            events=e_test,
            cif_curves={1: cif1, 2: cif2},
            risk_scores={1: risk, 2: [-r for r in risk]},
            times=time_horizons,
        )
        c1 = rep["cause_c_indices"][1]
        u1 = rep["cause_uno_c_indices"][1]
        ibs1 = rep["cause_ibs"][1]
        n_leaves = len(tree.leaves())
        print(f"{crit_key:<22} | {n_leaves:<6} | {c1:<10.4f} | {u1:<10.4f} | {ibs1:<8.4f}")

    # 3. Validating the Aalen-Johansen Simplex Invariant
    print("\n[Part 2] Validating Aalen-Johansen Simplex Invariant...")
    gray_tree = trees["gray"]
    cif1_test = gray_tree.predict(X_test, kind="cif", times=time_horizons, cause=1)
    cif2_test = gray_tree.predict(X_test, kind="cif", times=time_horizons, cause=2)
    surv_test = gray_tree.predict(X_test, kind="survival", times=time_horizons)

    max_simplex_err = 0.0
    for i in range(len(X_test)):
        for h_idx in range(len(time_horizons)):
            prob_sum = cif1_test[i][h_idx] + cif2_test[i][h_idx] + surv_test[i][h_idx]
            max_simplex_err = max(max_simplex_err, abs(prob_sum - 1.0))

    print(f"  Exact Identity: F_1(t) + F_2(t) + S(t) == 1.0")
    print(f"  Maximum absolute error across {len(X_test)*len(time_horizons)} points: {max_simplex_err:.2e}")
    assert max_simplex_err < 1e-12, "Simplex invariant check failed"

    # 4. Extracting Interpretability Rules
    print("\n[Part 3] Discovered Prognostic Rules (Gray's Test):")
    rules = gray_tree.extract_rules()
    print(rules)

    # 5. Diagnostic Plot
    print("\n[Part 4] Generating Multi-Cause Diagnostic Visualization...")
    out_file = OUT_ASSETS / "suite9_survival_competing_risks.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left Panel: Simplex Allocation across Risk Quartiles
    bm_vals = [row[0] for row in X_test]
    low_q_idx = int(np.argmin(bm_vals))
    high_q_idx = int(np.argmax(bm_vals))

    ax0 = axes[0]
    ax0.step(time_horizons, cif1_test[high_q_idx], where="post", color="#d81b60", lw=2.5, label="Cause 1: Relapse CIF F_1(t)")
    ax0.step(time_horizons, cif2_test[high_q_idx], where="post", color="#fb8c00", lw=2.5, label="Cause 2: Toxicity CIF F_2(t)")
    ax0.step(time_horizons, surv_test[high_q_idx], where="post", color="#00897b", lw=2.5, ls="--", label="Event-Free Survival S(t)")
    ax0.set_title(f"High-Risk Patient (Biomarker = {bm_vals[high_q_idx]:.2f})", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Follow-up Time (Months)", fontsize=10)
    ax0.set_ylabel("Cumulative Probability", fontsize=10)
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=True)

    ax1 = axes[1]
    ax1.step(time_horizons, cif1_test[low_q_idx], where="post", color="#d81b60", lw=2.5, label="Cause 1: Relapse CIF F_1(t)")
    ax1.step(time_horizons, cif2_test[low_q_idx], where="post", color="#fb8c00", lw=2.5, label="Cause 2: Toxicity CIF F_2(t)")
    ax1.step(time_horizons, surv_test[low_q_idx], where="post", color="#00897b", lw=2.5, ls="--", label="Event-Free Survival S(t)")
    ax1.set_title(f"Low-Risk Patient (Biomarker = {bm_vals[low_q_idx]:.2f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Follow-up Time (Months)", fontsize=10)
    ax1.set_ylabel("Cumulative Probability", fontsize=10)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    plt.suptitle("mltreelib Suite 9: Aalen-Johansen Multi-State Cumulative Incidence vs Survival", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_suite_survival()
