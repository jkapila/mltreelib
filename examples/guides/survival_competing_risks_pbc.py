"""Survival and Competing Risks Modeling: Aalen-Johansen CIF & Gray's Test.

Demonstrates Spec 10 (Survival and Competing Risks Trees):
1. Multi-cause competing risks setting:
   - Cause 1: Primary Event (e.g. Disease Relapse / Organ Failure)
   - Cause 2: Competing Risk (e.g. Treatment Toxicity / Other-Cause Mortality)
   - 0: Censored alive event-free
2. Fast O(N) streaming martingale residuals and Gray's subdistribution log-rank test.
3. Aalen-Johansen multi-state non-parametric Cumulative Incidence Function (CIF)
   satisfying the exact probability simplex invariant: F_1(t) + F_2(t) + S(t) == 1.0.
4. Proper scoring evaluation via Uno's IPCW C-index, Harrell's C-index, and cause-specific IBS.

Produces: docs/assets/competing_risks_cif_curves.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mltreelib import ModelTree, Metrics, competing_risks_report


def generate_competing_risks_benchmark(n_samples=600, seed=42):
    """Generate realistic clinical trial cohort with two mutually exclusive competing risks.

    Covariates:
    - x0: Biomarker level (continuous; drives Cause 1 hazard)
    - x1: Age in decades (continuous; drives Cause 2 hazard)
    - x2: Treatment arm (0=Standard Care, 1=Targeted Therapy)
    """
    rng = np.random.default_rng(seed)

    x0 = rng.normal(0.0, 1.0, size=n_samples)
    x1 = rng.uniform(4.0, 8.0, size=n_samples)
    x2 = rng.binomial(1, 0.5, size=n_samples).astype(float)

    X = [[float(x0[i]), float(x1[i]), float(x2[i])] for i in range(n_samples)]

    # Latent times to cause 1, cause 2, and right-censoring
    rate_cause1 = np.exp(0.8 * x0 - 0.5 * x2 - 2.0)
    rate_cause2 = np.exp(0.5 * (x1 - 5.0) - 2.5)
    rate_censor = 0.05

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


def run_competing_risks_showcase():
    print("=" * 80)
    print("mltreelib: Multi-Cause Survival and Competing Risks Trees (Spec 10)")
    print("=" * 80)

    # 1. Generate synthetic clinical cohort
    X, durations, events = generate_competing_risks_benchmark(n_samples=600, seed=42)
    n_total = len(events)
    n_cause1 = sum(1 for e in events if e == 1.0)
    n_cause2 = sum(1 for e in events if e == 2.0)
    n_censored = sum(1 for e in events if e == 0.0)

    print(f"Cohort Size: {n_total:,} patients")
    print(f"  - Cause 1 (Primary Relapse/Failure): {n_cause1:,} ({n_cause1/n_total*100:.1f}%)")
    print(f"  - Cause 2 (Competing Toxicity/Death): {n_cause2:,} ({n_cause2/n_total*100:.1f}%)")
    print(f"  - Censored Event-Free:               {n_censored:,} ({n_censored/n_total*100:.1f}%)")

    # Train / Test split (75/25)
    split_idx = 450
    X_train, X_test = X[:split_idx], X[split_idx:]
    t_train, t_test = durations[:split_idx], durations[split_idx:]
    e_train, e_test = events[:split_idx], events[split_idx:]

    # 2. Fit ModelTree with Gray's Test for Cause 1
    print("\n[1/3] Fitting ModelTree with Gray's Subdistribution Test (gain='gray', cause=1)...")
    tree_gray = ModelTree(
        max_depth=3,
        min_rows=25,
        gain="gray",
        target_cause=1,
    )
    tree_gray.fit(X_train, (t_train, e_train), col_names=["biomarker", "age", "treatment"])

    print(f"Tree structure: {len(tree_gray.leaves())} leaves")
    rules = tree_gray.extract_rules()
    print("Discovered Subdistribution Risk Strata:")
    print(rules)

    # 3. Predict Multi-Cause CIF and Survival Curves
    print("\n[2/3] Evaluating Aalen-Johansen Cumulative Incidence Functions (CIF)...")
    eval_horizons = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    cif1_pred = tree_gray.predict(X_test, kind="cif", times=eval_horizons, cause=1)
    cif2_pred = tree_gray.predict(X_test, kind="cif", times=eval_horizons, cause=2)
    surv_pred = tree_gray.predict(X_test, kind="survival", times=eval_horizons)
    risk_pred = tree_gray.predict(X_test, kind="risk")

    # Verify probability simplex invariant across all test rows and horizons
    max_simplex_err = 0.0
    for i in range(len(X_test)):
        for h_idx in range(len(eval_horizons)):
            total_prob = cif1_pred[i][h_idx] + cif2_pred[i][h_idx] + surv_pred[i][h_idx]
            err = abs(total_prob - 1.0)
            if err > max_simplex_err:
                max_simplex_err = err

    print(f"Verified Simplex Property: F_1(t) + F_2(t) + S(t) == 1.0 (max error: {max_simplex_err:.2e})")

    # 4. Multi-Cause Proper Scoring & Calibration Report
    print("\n[3/3] Computing Multi-Cause IPCW Performance Report...")
    metrics = Metrics()
    report = metrics.competing_risks(
        durations=t_test,
        events=e_test,
        cif_curves={1: cif1_pred, 2: cif2_pred},
        risk_scores={1: risk_pred, 2: [-r for r in risk_pred]},
        times=eval_horizons,
    )

    print("=" * 55)
    print("COMPETING RISKS DISCRIMINATION & CALIBRATION REPORT")
    print("=" * 55)
    for cause in report["causes"]:
        print(f"\n--- Cause {cause} ---")
        print(f"  Observed Test Events:      {report['cause_counts'][cause - 1]}")
        print(f"  Harrell's C-index:         {report['cause_c_indices'][cause]:.4f}")
        print(f"  Uno's IPCW C-index:        {report['cause_uno_c_indices'][cause]:.4f}")
        print(f"  Integrated Brier Score:    {report['cause_ibs'][cause]:.4f}")
        print("  Brier Scores by Horizon:")
        for t_str, bs_val in report["brier_at_horizons"][cause].items():
            print(f"    t = {float(t_str):5.1f}: BS = {bs_val:.4f}")

    # 5. Generate Multi-State Incidence Visualization
    os.makedirs("docs/assets", exist_ok=True)
    out_path = "docs/assets/competing_risks_cif_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Plot sample CIF trajectories for low vs high biomarker subjects
    test_bm = [X_test[i][0] for i in range(len(X_test))]
    low_bm_idx = int(np.argmin(test_bm))
    high_bm_idx = int(np.argmax(test_bm))

    # Left plot: Cumulative Incidence Curves for High Risk Patient
    ax0 = axes[0]
    ax0.step(eval_horizons, cif1_pred[high_bm_idx], label="Cause 1 (Relapse CIF)", color="#d81b60", lw=2.5, where="post")
    ax0.step(eval_horizons, cif2_pred[high_bm_idx], label="Cause 2 (Toxicity CIF)", color="#fb8c00", lw=2.5, where="post")
    ax0.step(eval_horizons, surv_pred[high_bm_idx], label="Event-Free S(t)", color="#00897b", lw=2.5, ls="--", where="post")
    ax0.set_title(f"High-Biomarker Cohort (x0={test_bm[high_bm_idx]:.2f})", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Follow-up Time (Months)", fontsize=10)
    ax0.set_ylabel("Probability", fontsize=10)
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=True)

    # Right plot: Cumulative Incidence Curves for Low Risk Patient
    ax1 = axes[1]
    ax1.step(eval_horizons, cif1_pred[low_bm_idx], label="Cause 1 (Relapse CIF)", color="#d81b60", lw=2.5, where="post")
    ax1.step(eval_horizons, cif2_pred[low_bm_idx], label="Cause 2 (Toxicity CIF)", color="#fb8c00", lw=2.5, where="post")
    ax1.step(eval_horizons, surv_pred[low_bm_idx], label="Event-Free S(t)", color="#00897b", lw=2.5, ls="--", where="post")
    ax1.set_title(f"Low-Biomarker Cohort (x0={test_bm[low_bm_idx]:.2f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Follow-up Time (Months)", fontsize=10)
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    plt.suptitle("mltreelib: Aalen-Johansen Cumulative Incidence vs Event-Free Survival (Simplex Invariant)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_competing_risks_showcase()
