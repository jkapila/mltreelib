"""Credit Scoring Case Study: Glass-Box Decision Trees under Asymmetric Misclassification Costs.

Evaluates 150,000 applicants from Give Me Some Credit (OpenML 46929).
Compares Empirical Prior Baseline, Linear Scorecard, Glass-Box Tree (depth=3),
and Monotone-Constrained Tree (depth=4) across a 20-seed benchmark.
Generates publication-quality calibration, cost-curve, and tree architecture diagrams.
"""

import os
import shutil
import subprocess
from pathlib import Path

# Ensure Matplotlib writes to /tmp to respect sandboxed environment
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mltreelib import Metrics, ModelTree


def load_credit_data(path: str = "datasets/GiveMeSomeCredit.parquet") -> pd.DataFrame:
    """Load Give Me Some Credit dataset."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GiveMeSomeCredit parquet file not found at {p}")
    return pd.read_parquet(p)


def evaluate_cost_and_metrics(y_true, y_prob, threshold, cost_fn=5.0, cost_fp=1.0):
    """Compute asymmetric misclassification loss, queue volume, and ranking metrics."""
    m = Metrics()
    n = len(y_true)
    fn = sum(1 for y, p in zip(y_true, y_prob) if y == 1 and p < threshold)
    fp = sum(1 for y, p in zip(y_true, y_prob) if y == 0 and p >= threshold)
    tp = sum(1 for y, p in zip(y_true, y_prob) if y == 1 and p >= threshold)
    tn = sum(1 for y, p in zip(y_true, y_prob) if y == 0 and p < threshold)

    expected_cost = (cost_fn * fn + cost_fp * fp) / n
    queue_rate = (tp + fp) / n * 100.0
    fnr = fn / max(1, (fn + tp)) * 100.0
    fpr = fp / max(1, (fp + tn)) * 100.0

    brier = m.brier_score(y_true, y_prob)
    roc_res = m.roc_curve(y_true, y_prob)
    pr_res = m.pr_curve(y_true, y_prob)

    return {
        "cost": expected_cost,
        "queue_rate": queue_rate,
        "fnr": fnr,
        "fpr": fpr,
        "brier": brier,
        "auc": roc_res["auc"],
        "ap": pr_res["average_precision"],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_credit_study(n_seeds: int = 20, output_dir: str = "docs/assets"):
    """Run full credit scoring study with single-seed diagnostics and 20-seed benchmark."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GLASS-BOX CREDIT SCORING UNDER ASYMMETRIC LOSS")
    print("=" * 80)

    df = load_credit_data()
    target = "FinancialDistressNextTwoYears"
    features = [c for c in df.columns if c != target]

    y_all = (df[target] == "Yes").astype(int).to_numpy()
    X_all = df[features].to_numpy()
    n_total = len(df)
    n_pos = sum(y_all)
    prevalence = n_pos / n_total * 100.0

    print(f"Total applicants: {n_total:,} | Distress events: {n_pos:,} ({prevalence:.2f}%)")
    missing_counts = df[features].isna().sum()
    print("Missing values in features:")
    for col, cnt in missing_counts[missing_counts > 0].items():
        print(f"  - {col}: {cnt:,} missing ({cnt / n_total * 100.0:.2f}%)")

    # Domain monotonicity constraints:
    # Utilization (+1), Age (-1), 30-59 DPD (+1), DebtRatio (+1), Income (-1),
    # OpenCreditLines (0), 90 DPD (+1), RealEstate (0), 60-89 DPD (+1), Dependents (0)
    monotone_vec = [1, -1, 1, 1, -1, 0, 1, 0, 1, 0]

    # Split sizes: 100,000 train, 25,000 validation, 25,000 test
    n_train = 100000
    n_val = 25000
    n_test = 25000

    # 1. Single-Seed In-Depth Diagnosis (Seed 42)
    print("\n--- Fitting Glass-Box Decision Tree (Seed 42) ---")
    rng0 = np.random.default_rng(42)
    perm0 = rng0.permutation(n_total)
    tr_idx, val_idx, te_idx = perm0[:n_train], perm0[n_train:n_train+n_val], perm0[n_train+n_val:]

    X_tr0, y_tr0 = X_all[tr_idx].tolist(), y_all[tr_idx].tolist()
    X_val0, y_val0 = X_all[val_idx].tolist(), y_all[val_idx].tolist()
    X_te0, y_te0 = X_all[te_idx].tolist(), y_all[te_idx].tolist()

    tree0 = ModelTree(
        leaf="mean",
        leaf_family="binomial",
        gain="deviance",
        max_depth=3,
        min_rows=100,
        prune="none",
    )
    tree0.fit(X_tr0, y_tr0, col_names=features)

    rules0 = tree0.extract_rules()
    print(f"\nExtracted Glass-Box Rules ({len(rules0.rules)} terminal leaves):")
    for r in rules0.rules:
        cond_str = " AND ".join(r.conditions)
        print(f"  Leaf {r.leaf_id} (n={r.cover:6d}): IF {cond_str} -> P(Distress) = {r.prediction:.4f}")

    # Generate tree diagram if dot is available
    dot_bin = shutil.which("dot")
    if dot_bin:
        dot_str = tree0.to_dot()
        svg_path = out / "credit_scoring_tree.svg"
        png_path = out / "credit_scoring_tree.png"
        subprocess.run([dot_bin, "-Tsvg", "-o", str(svg_path)], input=dot_str, text=True, check=True)
        subprocess.run([dot_bin, "-Tpng", "-Gdpi=300", "-o", str(png_path)], input=dot_str, text=True, check=True)
        print(f"Saved tree diagram: {svg_path} & {png_path}")

    # Find optimal threshold on validation set
    p_val0 = tree0.predict_proba(X_val0)
    p_te0 = tree0.predict_proba(X_te0)

    threshold_grid = np.linspace(0.01, 0.40, 40)
    costs_grid = []
    queues_grid = []
    for c in threshold_grid:
        m_c = evaluate_cost_and_metrics(y_val0, p_val0, c)
        costs_grid.append(m_c["cost"])
        queues_grid.append(m_c["queue_rate"])

    best_idx = int(np.argmin(costs_grid))
    opt_cutoff = float(threshold_grid[best_idx])
    opt_cost = costs_grid[best_idx]
    opt_queue = queues_grid[best_idx]
    print(f"\nOptimal Validation Cutoff under 5:1 loss: {opt_cutoff:.3f} | Cost: ${opt_cost:.4f} | Queue: {opt_queue:.1f}%")

    # Evaluate on held-out test set
    diag_te = evaluate_cost_and_metrics(y_te0, p_te0, opt_cutoff)
    print(f"Held-Out Test Results (Cutoff={opt_cutoff:.3f}):")
    print(f"  - Expected Cost: ${diag_te['cost']:.4f} / applicant")
    print(f"  - Review Queue: {diag_te['queue_rate']:.1f}% ({diag_te['tp'] + diag_te['fp']:,} applicants)")
    print(f"  - Missed Distress (FN): {diag_te['fn']:,} ({diag_te['fnr']:.1f}% of distress)")
    print(f"  - False Alerts (FP): {diag_te['fp']:,} ({diag_te['fpr']:.1f}% of solvent)")
    print(f"  - AUROC: {diag_te['auc']:.4f} | Average Precision: {diag_te['ap']:.4f} | Brier: {diag_te['brier']:.4f}")

    # Generate Publication Diagnostic Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Asymmetric Cost Curve & Queue Rate vs Threshold
    color_cost = "#991b1b"
    color_q = "#1e3a8a"

    ax1.plot(threshold_grid, costs_grid, color=color_cost, lw=2.5, label="Expected Cost ($ / applicant)")
    ax1.axvline(opt_cutoff, color="black", linestyle="--", lw=1.5, label=f"Optimal Cutoff ({opt_cutoff:.2f})")
    ax1.axvline(0.167, color="gray", linestyle=":", lw=1.2, label="Bayes Cutoff (1/6 ≈ 0.17)")
    ax1.set_xlabel("Review Cutoff Threshold (τ)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Expected Misclassification Cost ($)", fontsize=10, fontweight="bold", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.set_title("A: Misclassification Cost & Queue Volume vs Threshold", fontsize=11, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(threshold_grid, queues_grid, color=color_q, lw=2.0, linestyle="-.", label="Review Queue Volume (%)")
    ax1_twin.set_ylabel("Manual Review Queue Volume (%)", fontsize=10, fontweight="bold", color=color_q)
    ax1_twin.tick_params(axis="y", labelcolor=color_q)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", fontsize=8.5, framealpha=0.9)

    # Panel B: Reliability Curve (Calibration)
    bins = np.linspace(0.0, max(p_te0) + 0.05, 9)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    observed_rates = []
    bin_weights = []
    p_te0_np = np.array(p_te0)
    y_te0_np = np.array(y_te0)

    for i in range(len(bins) - 1):
        mask = (p_te0_np >= bins[i]) & (p_te0_np < bins[i+1])
        if np.sum(mask) > 0:
            observed_rates.append(np.mean(y_te0_np[mask]))
            bin_weights.append(np.sum(mask))
        else:
            observed_rates.append(np.nan)
            bin_weights.append(0)

    ax2.plot([0, 0.65], [0, 0.65], linestyle="--", color="gray", lw=1.5, label="Perfect Calibration")
    valid = ~np.isnan(observed_rates)
    ax2.plot(np.array(bin_centers)[valid], np.array(observed_rates)[valid], marker="s", lw=2.2, color="#0f766e", label="Glass-Box Tree (d=3)")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Observed Event Rate", fontsize=10, fontweight="bold")
    ax2.set_title("B: Held-Out Probability Calibration Diagnostic", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlim(0.0, 0.65)
    ax2.set_ylim(0.0, 0.65)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    calib_png = out / "credit_scoring_calibration_queue.png"
    calib_svg = out / "credit_scoring_calibration_queue.svg"
    fig.savefig(calib_png, dpi=300)
    fig.savefig(calib_svg)
    plt.close(fig)
    print(f"Saved diagnostic plots: {calib_png} & {calib_svg}")

    # 2. Multi-Seed Rigorous Benchmark (20 Seeds)
    print(f"\n--- Running {n_seeds}-Seed Rigorous Benchmark ---")
    cutoff_candidates = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.167, 0.20, 0.25, 0.30, 0.40]

    metrics_baseline = {"cost": [], "queue": [], "brier": [], "auc": [], "ap": []}
    metrics_scorecard = {"cost": [], "queue": [], "brier": [], "auc": [], "ap": []}
    metrics_tree_d3 = {"cost": [], "queue": [], "brier": [], "auc": [], "ap": []}
    metrics_tree_mono = {"cost": [], "queue": [], "brier": [], "auc": [], "ap": []}

    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        perm = rng.permutation(n_total)
        tr, val, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

        X_tr, y_tr = X_all[tr].tolist(), y_all[tr].tolist()
        X_val, y_val = X_all[val].tolist(), y_all[val].tolist()
        X_te, y_te = X_all[te].tolist(), y_all[te].tolist()

        # Architecture 1: Constant Rate Baseline
        prev_tr = float(np.mean(y_tr))
        m_base = evaluate_cost_and_metrics(y_te, [prev_tr] * len(y_te), 0.167)
        metrics_baseline["cost"].append(m_base["cost"])
        metrics_baseline["queue"].append(m_base["queue_rate"])
        metrics_baseline["brier"].append(m_base["brier"])
        metrics_baseline["auc"].append(0.5000)
        metrics_baseline["ap"].append(float(np.mean(y_te)))

        # Architecture 2: Linear Scorecard (Logistic Regression)
        t_sc = ModelTree(max_depth=0, leaf="linear", leaf_family="binomial", l2=1.0)
        t_sc.fit(X_tr, y_tr, col_names=features)
        p_val_sc = t_sc.predict_proba(X_val)
        p_te_sc = t_sc.predict_proba(X_te)
        # Select cutoff on val
        best_c_sc = min(cutoff_candidates, key=lambda c: evaluate_cost_and_metrics(y_val, p_val_sc, c)["cost"])
        m_sc = evaluate_cost_and_metrics(y_te, p_te_sc, best_c_sc)
        metrics_scorecard["cost"].append(m_sc["cost"])
        metrics_scorecard["queue"].append(m_sc["queue_rate"])
        metrics_scorecard["brier"].append(m_sc["brier"])
        metrics_scorecard["auc"].append(m_sc["auc"])
        metrics_scorecard["ap"].append(m_sc["ap"])

        # Architecture 3: Glass-Box Tree (Depth 3)
        t_d3 = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=3, min_rows=100, prune="none")
        t_d3.fit(X_tr, y_tr, col_names=features)
        p_val_d3 = t_d3.predict_proba(X_val)
        p_te_d3 = t_d3.predict_proba(X_te)
        best_c_d3 = min(cutoff_candidates, key=lambda c: evaluate_cost_and_metrics(y_val, p_val_d3, c)["cost"])
        m_d3 = evaluate_cost_and_metrics(y_te, p_te_d3, best_c_d3)
        metrics_tree_d3["cost"].append(m_d3["cost"])
        metrics_tree_d3["queue"].append(m_d3["queue_rate"])
        metrics_tree_d3["brier"].append(m_d3["brier"])
        metrics_tree_d3["auc"].append(m_d3["auc"])
        metrics_tree_d3["ap"].append(m_d3["ap"])

        # Architecture 4: Monotone Glass-Box Tree (Depth 4)
        t_mono = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=100, prune="none", monotone=monotone_vec)
        t_mono.fit(X_tr, y_tr, col_names=features)
        p_val_mono = t_mono.predict_proba(X_val)
        p_te_mono = t_mono.predict_proba(X_te)
        best_c_mono = min(cutoff_candidates, key=lambda c: evaluate_cost_and_metrics(y_val, p_val_mono, c)["cost"])
        m_mono = evaluate_cost_and_metrics(y_te, p_te_mono, best_c_mono)
        metrics_tree_mono["cost"].append(m_mono["cost"])
        metrics_tree_mono["queue"].append(m_mono["queue_rate"])
        metrics_tree_mono["brier"].append(m_mono["brier"])
        metrics_tree_mono["auc"].append(m_mono["auc"])
        metrics_tree_mono["ap"].append(m_mono["ap"])

        print(f"  Seed {seed+1:2d}/{n_seeds}: Cost d3=${m_d3['cost']:.4f} | Cost mono=${m_mono['cost']:.4f} | AUC d3={m_d3['auc']:.4f} mono={m_mono['auc']:.4f}")

    print("\n" + "=" * 80)
    print(f"FINAL 20-SEED BENCHMARK SUMMARY (N_TEST = {n_test:,})")
    print("=" * 80)
    print(f"{'Architecture':<32s} | {'Expected Cost ($)':<18s} | {'Review Queue':<14s} | {'Brier Score':<14s} | {'AUROC':<14s} | {'Avg Precision':<14s}")
    print("-" * 115)

    for name, met in [
        ("Constant Rate Baseline", metrics_baseline),
        ("Linear Scorecard (Logistic)", metrics_scorecard),
        ("Glass-Box Tree (Depth 3)", metrics_tree_d3),
        ("Monotone Glass-Box Tree (Depth 4)", metrics_tree_mono),
    ]:
        cost_m, cost_s = np.mean(met["cost"]), np.std(met["cost"])
        q_m, q_s = np.mean(met["queue"]), np.std(met["queue"])
        brier_m, brier_s = np.mean(met["brier"]), np.std(met["brier"])
        auc_m, auc_s = np.mean(met["auc"]), np.std(met["auc"])
        ap_m, ap_s = np.mean(met["ap"]), np.std(met["ap"])
        print(f"{name:<32s} | ${cost_m:.4f} ± ${cost_s:.4f} | {q_m:5.1f}% ± {q_s:4.1f}% | {brier_m:.4f} ± {brier_s:.4f} | {auc_m:.4f} ± {auc_s:.4f} | {ap_m:.4f} ± {ap_s:.4f}")

    # Win-rate comparison
    wins_mono_vs_d3 = sum(1 for m, d in zip(metrics_tree_mono["cost"], metrics_tree_d3["cost"]) if m < d)
    wins_d3_vs_sc = sum(1 for d, s in zip(metrics_tree_d3["cost"], metrics_scorecard["cost"]) if d < s)
    print(f"\nHead-to-head Cost Reductions:")
    print(f"  - Glass-Box Tree (d=3) beats Linear Scorecard: {wins_d3_vs_sc} / {n_seeds} seeds")
    print(f"  - Monotone Tree (d=4) beats Glass-Box Tree (d=3): {wins_mono_vs_d3} / {n_seeds} seeds")


if __name__ == "__main__":
    run_credit_study(n_seeds=20)
