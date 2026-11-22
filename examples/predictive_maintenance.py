"""Predictive Maintenance Case Study: Zero-Imputation Decision Trees on Scania Truck Telemetry.

Evaluates 76,000 Scania trucks with 170 anonymized sensor channels from APS Failure (OpenML 46908).
Audits over 1,000,000 missing sensor measurements.
Compares Blind Baseline, Median Imputation Tree, and Native Missing-Routing ModelTree
under the official Scania Industrial Cost metric (Cost = 10*FP + 500*FN) across a 20-seed benchmark.
Generates publication-quality cost curves, PR curves, and tree routing diagrams.
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


def load_aps_data(path: str = "datasets/APSFailure.parquet") -> pd.DataFrame:
    """Load APS Failure dataset."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"APSFailure parquet file not found at {p}")
    return pd.read_parquet(p)


def compute_scania_cost(y_true, y_prob, threshold, cost_fp=10.0, cost_fn=500.0):
    """Compute official Scania challenge cost: 10*FP + 500*FN."""
    fn = sum(1 for y, p in zip(y_true, y_prob) if y == 1 and p < threshold)
    fp = sum(1 for y, p in zip(y_true, y_prob) if y == 0 and p >= threshold)
    tp = sum(1 for y, p in zip(y_true, y_prob) if y == 1 and p >= threshold)
    tn = sum(1 for y, p in zip(y_true, y_prob) if y == 0 and p < threshold)

    total_cost = cost_fn * fn + cost_fp * fp
    queue_rate = (tp + fp) / len(y_true) * 100.0
    fnr = fn / max(1, fn + tp) * 100.0
    fpr = fp / max(1, fp + tn) * 100.0

    m = Metrics()
    brier = m.brier_score(y_true, y_prob)
    roc_res = m.roc_curve(y_true, y_prob)
    pr_res = m.pr_curve(y_true, y_prob)

    return {
        "cost": total_cost,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "tn": tn,
        "queue_rate": queue_rate,
        "fnr": fnr,
        "fpr": fpr,
        "auc": roc_res["auc"],
        "ap": pr_res["average_precision"],
        "brier": brier,
    }


def run_predictive_maintenance_study(n_seeds: int = 20, output_dir: str = "docs/assets"):
    """Run full predictive maintenance study with single-seed diagnostics and 20-seed benchmark."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ZERO-IMPUTATION PREDICTIVE MAINTENANCE: SCANIA TRUCK TELEMETRY")
    print("=" * 80)

    df = load_aps_data()
    target = "AirPressureSystemFailure"
    features = [c for c in df.columns if c != target]

    y_all = (df[target] == "pos").astype(int).to_numpy()
    X_raw = df[features]

    n_total = len(df)
    n_pos = int(sum(y_all))
    prevalence = n_pos / n_total * 100.0
    null_cells = int(X_raw.isna().sum().sum())
    total_cells = n_total * len(features)
    null_rate = null_cells / total_cells * 100.0

    print(f"Total fleet: {n_total:,} trucks | APS Failures: {n_pos:,} ({prevalence:.2f}%)")
    print(f"Sensor channels: {len(features)} | Missing sensor readings: {null_cells:,} / {total_cells:,} ({null_rate:.2f}%)")

    # Split sizes: 50,000 train, 13,000 validation, 13,000 test
    n_train = 50000
    n_val = 13000
    n_test = 13000

    # 1. Single-Seed In-Depth Diagnosis (Seed 42)
    print("\n--- Fitting Native Missing-Routing ModelTree (Seed 42) ---")
    rng0 = np.random.default_rng(42)
    perm0 = rng0.permutation(n_total)
    tr_idx, val_idx, te_idx = perm0[:n_train], perm0[n_train:n_train+n_val], perm0[n_train+n_val:]

    X_tr_native0 = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[tr_idx].itertuples(index=False, name=None)]
    X_val_native0 = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[val_idx].itertuples(index=False, name=None)]
    X_te_native0 = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[te_idx].itertuples(index=False, name=None)]

    y_tr0, y_val0, y_te0 = y_all[tr_idx].tolist(), y_all[val_idx].tolist(), y_all[te_idx].tolist()

    tree0 = ModelTree(
        leaf="mean",
        leaf_family="binomial",
        gain="deviance",
        max_depth=3,
        min_rows=100,
        prune="none",
    )
    tree0.fit(X_tr_native0, y_tr0, col_names=features)

    splits0 = tree0.splits()
    print(f"\nDiscovered Native Routing Splits ({len(splits0)} nodes):")
    for s in splits0[:5]:
        direction = "Left" if s.get("missing_left", True) else "Right"
        print(f"  - Split: {s['col']} <= {s['threshold']:.2f} | Missing routed: {direction} (Gain={s['gain']:.2f})")

    # Render tree diagram if dot is available
    dot_bin = shutil.which("dot")
    if dot_bin:
        dot_str = tree0.to_dot()
        svg_path = out / "predictive_maintenance_tree.svg"
        png_path = out / "predictive_maintenance_tree.png"
        subprocess.run([dot_bin, "-Tsvg", "-o", str(svg_path)], input=dot_str, text=True, check=True)
        subprocess.run([dot_bin, "-Tpng", "-Gdpi=300", "-o", str(png_path)], input=dot_str, text=True, check=True)
        print(f"Saved predictive maintenance tree: {svg_path} & {png_path}")

    # Threshold optimization on validation set
    p_val0 = tree0.predict_proba(X_val_native0)
    p_te0 = tree0.predict_proba(X_te_native0)

    threshold_grid = np.linspace(0.005, 0.15, 40)
    costs_grid = []
    queues_grid = []
    for c in threshold_grid:
        res_c = compute_scania_cost(y_val0, p_val0, c)
        costs_grid.append(res_c["cost"])
        queues_grid.append(res_c["queue_rate"])

    best_idx = int(np.argmin(costs_grid))
    opt_cutoff = float(threshold_grid[best_idx])
    opt_val_cost = costs_grid[best_idx]
    print(f"\nOptimal Validation Cutoff under 50:1 cost: {opt_cutoff:.3f} | Validation Cost: ${opt_val_cost:,.0f}")

    # Held-out evaluation on test set
    diag_te = compute_scania_cost(y_te0, p_te0, opt_cutoff)
    print(f"Held-Out Test Results (Cutoff={opt_cutoff:.3f}):")
    print(f"  - Total Scania Industrial Cost: ${diag_te['cost']:,.0f}")
    print(f"  - Missed Road Breakdowns (FN @ $500): {diag_te['fn']} ({diag_te['fnr']:.1f}% of failures)")
    print(f"  - False Workshop Checks (FP @ $10): {diag_te['fp']} ({diag_te['fpr']:.1f}% of healthy fleet)")
    print(f"  - Workshop Inspection Queue: {diag_te['queue_rate']:.1f}% ({diag_te['tp'] + diag_te['fp']:,} trucks)")
    print(f"  - AUROC: {diag_te['auc']:.4f} | Average Precision: {diag_te['ap']:.4f} | Brier: {diag_te['brier']:.4f}")

    # Generate Publication Diagnostic Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Industrial Cost Curve & Workshop Inspection Queue vs Threshold
    color_cost = "#991b1b"
    color_q = "#1e3a8a"

    ax1.plot(threshold_grid, costs_grid, color=color_cost, lw=2.5, label="Scania Industrial Cost ($)")
    ax1.axvline(opt_cutoff, color="black", linestyle="--", lw=1.5, label=f"Optimal Cutoff ({opt_cutoff:.3f})")
    bayes_cutoff = 10.0 / (500.0 + 10.0)
    ax1.axvline(bayes_cutoff, color="gray", linestyle=":", lw=1.2, label=f"Bayes Cutoff (10/510 ≈ {bayes_cutoff:.3f})")
    ax1.set_xlabel("Triage Inspection Threshold (τ)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total Fleet Cost (10*FP + 500*FN) [$]", fontsize=10, fontweight="bold", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.set_title("A: Scania Industrial Cost & Inspection Load vs Threshold", fontsize=11, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(threshold_grid, queues_grid, color=color_q, lw=2.0, linestyle="-.", label="Workshop Inspection Load (%)")
    ax1_twin.set_ylabel("Trucks Sent to Workshop (%)", fontsize=10, fontweight="bold", color=color_q)
    ax1_twin.tick_params(axis="y", labelcolor=color_q)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8.5, framealpha=0.9)

    # Panel B: Missingness Slice Diagnostic
    # Compare failure prediction across trucks with complete telemetry vs missing telemetry
    has_missing_te = X_raw.iloc[te_idx].isna().any(axis=1).to_numpy()
    p_te_np = np.array(p_te0)
    y_te_np = np.array(y_te0)

    complete_mask = ~has_missing_te
    missing_mask = has_missing_te

    n_complete = int(np.sum(complete_mask))
    n_missing = int(np.sum(missing_mask))

    obs_complete = np.mean(y_te_np[complete_mask]) * 100.0 if n_complete > 0 else 0.0
    pred_complete = np.mean(p_te_np[complete_mask]) * 100.0 if n_complete > 0 else 0.0

    obs_missing = np.mean(y_te_np[missing_mask]) * 100.0 if n_missing > 0 else 0.0
    pred_missing = np.mean(p_te_np[missing_mask]) * 100.0 if n_missing > 0 else 0.0

    x_pos = np.arange(2)
    width = 0.35

    ax2.bar(x_pos - width/2, [obs_complete, obs_missing], width, label="Observed Failure Rate (%)", color="#0284c7")
    ax2.bar(x_pos + width/2, [pred_complete, pred_missing], width, label="Model Predicted Risk (%)", color="#0d9488")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Complete Telemetry\n(n = {n_complete:,})", f"Missing Telemetry\n(n = {n_missing:,})"], fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Failure Prevalence (%)", fontsize=10, fontweight="bold")
    ax2.set_title("B: Native Missing Routing Across Telemetry Slices", fontsize=11, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    for i, (obs, pred) in enumerate(zip([obs_complete, obs_missing], [pred_complete, pred_missing])):
        ax2.text(i - width/2, obs + 0.05, f"{obs:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax2.text(i + width/2, pred + 0.05, f"{pred:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    cost_png = out / "predictive_maintenance_cost_curve.png"
    cost_svg = out / "predictive_maintenance_cost_curve.svg"
    fig.savefig(cost_png, dpi=300)
    fig.savefig(cost_svg)
    plt.close(fig)
    print(f"Saved diagnostic plots: {cost_png} & {cost_svg}")

    # 2. Multi-Seed Rigorous Benchmark (20 Seeds)
    print(f"\n--- Running {n_seeds}-Seed Rigorous Benchmark ---")
    cutoff_candidates = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15]

    metrics_baseline = {"cost": [], "fn": [], "fp": [], "queue": [], "auc": [], "brier": []}
    metrics_median = {"cost": [], "fn": [], "fp": [], "queue": [], "auc": [], "brier": []}
    metrics_native = {"cost": [], "fn": [], "fp": [], "queue": [], "auc": [], "brier": []}

    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        perm = rng.permutation(n_total)
        tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

        y_tr, y_va, y_te = y_all[tr].tolist(), y_all[va].tolist(), y_all[te].tolist()

        # Architecture 1: Blind Constant Baseline
        prev_tr = float(np.mean(y_tr))
        m_base = compute_scania_cost(y_te, [prev_tr] * len(y_te), bayes_cutoff)
        metrics_baseline["cost"].append(m_base["cost"])
        metrics_baseline["fn"].append(m_base["fn"])
        metrics_baseline["fp"].append(m_base["fp"])
        metrics_baseline["queue"].append(m_base["queue_rate"])
        metrics_baseline["auc"].append(0.5000)
        metrics_baseline["brier"].append(m_base["brier"])

        # Native data
        X_tr_nat = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[tr].itertuples(index=False, name=None)]
        X_va_nat = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[va].itertuples(index=False, name=None)]
        X_te_nat = [[None if pd.isna(v) else v for v in row] for row in X_raw.iloc[te].itertuples(index=False, name=None)]

        # Architecture 2: Median Imputed Tree
        meds = X_raw.iloc[tr].median()
        X_tr_med = X_raw.iloc[tr].fillna(meds).to_numpy().tolist()
        X_va_med = X_raw.iloc[va].fillna(meds).to_numpy().tolist()
        X_te_med = X_raw.iloc[te].fillna(meds).to_numpy().tolist()

        t_med = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=100)
        t_med.fit(X_tr_med, y_tr, col_names=features)
        p_va_med = t_med.predict_proba(X_va_med)
        p_te_med = t_med.predict_proba(X_te_med)
        best_c_med = min(cutoff_candidates, key=lambda c: compute_scania_cost(y_va, p_va_med, c)["cost"])
        m_med = compute_scania_cost(y_te, p_te_med, best_c_med)
        metrics_median["cost"].append(m_med["cost"])
        metrics_median["fn"].append(m_med["fn"])
        metrics_median["fp"].append(m_med["fp"])
        metrics_median["queue"].append(m_med["queue_rate"])
        metrics_median["auc"].append(m_med["auc"])
        metrics_median["brier"].append(m_med["brier"])

        # Architecture 3: Native Missing Routing Tree (Zero Imputation)
        t_nat = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=100)
        t_nat.fit(X_tr_nat, y_tr, col_names=features)
        p_va_nat = t_nat.predict_proba(X_va_nat)
        p_te_nat = t_nat.predict_proba(X_te_nat)
        best_c_nat = min(cutoff_candidates, key=lambda c: compute_scania_cost(y_va, p_va_nat, c)["cost"])
        m_nat = compute_scania_cost(y_te, p_te_nat, best_c_nat)
        metrics_native["cost"].append(m_nat["cost"])
        metrics_native["fn"].append(m_nat["fn"])
        metrics_native["fp"].append(m_nat["fp"])
        metrics_native["queue"].append(m_nat["queue_rate"])
        metrics_native["auc"].append(m_nat["auc"])
        metrics_native["brier"].append(m_nat["brier"])

        print(f"  Seed {seed+1:2d}/{n_seeds}: Native Cost=${m_nat['cost']:,.0f} | Median Cost=${m_med['cost']:,.0f} | Native AUC={m_nat['auc']:.4f}")

    print("\n" + "=" * 80)
    print(f"FINAL 20-SEED BENCHMARK SUMMARY (N_TEST = {n_test:,} TRUCKS)")
    print("=" * 80)
    print(f"{'Architecture':<34s} | {'Scania Total Cost ($)':<22s} | {'Missed (FN)':<14s} | {'False Checks (FP)':<18s} | {'AUROC':<14s} | {'Brier Score':<14s}")
    print("-" * 125)

    for name, met in [
        ("Blind Inspection Baseline", metrics_baseline),
        ("Median Imputation Decision Tree", metrics_median),
        ("Native Missing Routing (Zero Impute)", metrics_native),
    ]:
        cost_m, cost_s = np.mean(met["cost"]), np.std(met["cost"])
        fn_m, fn_s = np.mean(met["fn"]), np.std(met["fn"])
        fp_m, fp_s = np.mean(met["fp"]), np.std(met["fp"])
        auc_m, auc_s = np.mean(met["auc"]), np.std(met["auc"])
        brier_m, brier_s = np.mean(met["brier"]), np.std(met["brier"])
        print(f"{name:<34s} | ${cost_m:8,.0f} ± ${cost_s:6,.0f} | {fn_m:4.1f} ± {fn_s:3.1f} | {fp_m:5.1f} ± {fp_s:4.1f} | {auc_m:.4f} ± {auc_s:.4f} | {brier_m:.4f} ± {brier_s:.4f}")

    wins_native_vs_base = sum(1 for n, b in zip(metrics_native["cost"], metrics_baseline["cost"]) if n < b)
    cost_drop = (np.mean(metrics_baseline["cost"]) - np.mean(metrics_native["cost"])) / np.mean(metrics_baseline["cost"]) * 100.0
    print(f"\nCost Reduction Summary:")
    print(f"  - Native Missing Routing beats Blind Baseline: {wins_native_vs_base} / {n_seeds} seeds")
    print(f"  - Fleet Cost Reduced by {cost_drop:.1f}% without any imputation pipeline!")


if __name__ == "__main__":
    run_predictive_maintenance_study(n_seeds=20)
