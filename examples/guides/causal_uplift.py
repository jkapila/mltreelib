"""Causal Uplift Modeling Case Study: Honest Causal Trees vs Adaptive Trees and T-Learners.

Evaluates 42,693 customer records from the randomized Hillstrom email marketing trial.
Compares Uniform Treat-All, Adaptive Causal Trees, Two-Model T-Learners, and Honest Causal Trees
across a 20-seed benchmark.
Generates publication-quality Qini curves, decile uplift plots, and causal tree diagrams.
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


def load_hillstrom_data(path: str = "datasets/external/real_eval/hillstrom.csv") -> pd.DataFrame:
    """Load Hillstrom dataset and filter for Women's E-Mail vs Control."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Hillstrom dataset not found at {p}")
    df = pd.read_csv(p)
    df = df[df["segment"].isin(["Womens E-Mail", "No E-Mail"])].reset_index(drop=True)
    return df


def run_causal_uplift_study(n_seeds: int = 20, output_dir: str = "docs/assets"):
    """Run full causal uplift study with single-seed diagnostics and 20-seed benchmark."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HONEST CAUSAL UPLIFT MODELING ON RANDOMIZED TRIAL DATA")
    print("=" * 80)

    df = load_hillstrom_data()
    features = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
    types = ["numeric"] * 5 + ["categorical"] * 2

    X_all = [[None if pd.isna(v) else v for v in row] for row in df[features].itertuples(index=False, name=None)]
    y_all = df["visit"].astype(float).tolist()
    w_all = [float(s == "Womens E-Mail") for s in df["segment"]]

    n_total = len(df)
    n_treated = int(sum(w_all))
    n_control = n_total - n_treated
    overall_ate = (sum(y for y, w in zip(y_all, w_all) if w == 1.0) / n_treated -
                   sum(y for y, w in zip(y_all, w_all) if w == 0.0) / n_control) * 100.0

    print(f"Total customers: {n_total:,} | Treated (Womens E-Mail): {n_treated:,} | Control (No E-Mail): {n_control:,}")
    print(f"Overall Average Treatment Effect (ATE): +{overall_ate:.2f} percentage points incremental visits")

    # Split sizes: 60% train, 20% validation, 20% test
    n_train = int(0.60 * n_total)
    n_val = int(0.20 * n_total)
    n_test = n_total - n_train - n_val

    def pick(arr, idx):
        return [arr[i] for i in idx]

    # 1. Single-Seed In-Depth Diagnosis (Seed 42)
    print("\n--- Fitting Honest Causal Tree (Seed 42) ---")
    rng0 = np.random.default_rng(42)
    perm0 = rng0.permutation(n_total)
    tr_idx, val_idx, te_idx = perm0[:n_train], perm0[n_train:n_train+n_val], perm0[n_train+n_val:]

    xtr0, ytr0, wtr0 = pick(X_all, tr_idx), pick(y_all, tr_idx), pick(w_all, tr_idx)
    xte0, yte0, wte0 = pick(X_all, te_idx), pick(y_all, te_idx), pick(w_all, te_idx)

    honest_tree0 = ModelTree(
        gain="causal",
        leaf="mean",
        max_depth=2,
        min_rows=250,
        prune="none",
        honesty=True,
        seed=42,
    )
    honest_tree0.fit(xtr0, (ytr0, wtr0), col_names=features, dtypes=types)

    splits0 = honest_tree0.splits()
    leaves0 = honest_tree0.leaves()
    print(f"\nDiscovered Causal Splits ({len(splits0)} split nodes, {len(leaves0)} leaves):")
    for s in splits0:
        print(f"  - Split on {s['col']} (threshold={s['threshold']:.2f}, gain={s['gain']:.4f})")

    # Export tree diagram if dot is available
    dot_bin = shutil.which("dot")
    if dot_bin:
        dot_str = honest_tree0.to_dot()
        svg_path = out / "causal_uplift_tree.svg"
        png_path = out / "causal_uplift_tree.png"
        subprocess.run([dot_bin, "-Tsvg", "-o", str(svg_path)], input=dot_str, text=True, check=True)
        subprocess.run([dot_bin, "-Tpng", "-Gdpi=300", "-o", str(png_path)], input=dot_str, text=True, check=True)
        print(f"Saved causal tree diagram: {svg_path} & {png_path}")

    # Evaluate held-out uplift scores
    tau_honest0 = honest_tree0.predict(xte0, kind="uplift")

    # Fit Adaptive Tree (honesty=False)
    adapt_tree0 = ModelTree(
        gain="causal",
        leaf="mean",
        max_depth=2,
        min_rows=250,
        prune="none",
        honesty=False,
        seed=42,
    )
    adapt_tree0.fit(xtr0, (ytr0, wtr0), col_names=features, dtypes=types)
    tau_adapt0 = adapt_tree0.predict(xte0, kind="uplift")

    # Fit T-Learner (Two Separate Trees)
    tr_t_idx = [i for i, wt in enumerate(wtr0) if wt == 1.0]
    tr_c_idx = [i for i, wt in enumerate(wtr0) if wt == 0.0]

    m1_0 = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=150)
    m1_0.fit(pick(xtr0, tr_t_idx), pick(ytr0, tr_t_idx), col_names=features, dtypes=types)
    m0_0 = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=150)
    m0_0.fit(pick(xtr0, tr_c_idx), pick(ytr0, tr_c_idx), col_names=features, dtypes=types)

    p1_te0 = m1_0.predict_proba(xte0)
    p0_te0 = m0_0.predict_proba(xte0)
    tau_tlearner0 = [p1 - p0 for p1, p0 in zip(p1_te0, p0_te0)]

    m = Metrics()
    res_h0 = m.uplift(yte0, wte0, tau_honest0)
    res_a0 = m.uplift(yte0, wte0, tau_adapt0)
    res_t0 = m.uplift(yte0, wte0, tau_tlearner0)

    print("\nHeld-Out Single-Seed Performance (Seed 42):")
    print(f"  - Honest Causal Tree : Qini={res_h0['qini_coef']:.2f} | AUUC={res_h0['auuc']:.2f} | Top Quintile Uplift={res_h0['top_quintile_uplift']*100.0:.2f}%")
    print(f"  - Adaptive Tree      : Qini={res_a0['qini_coef']:.2f} | AUUC={res_a0['auuc']:.2f} | Top Quintile Uplift={res_a0['top_quintile_uplift']*100.0:.2f}%")
    print(f"  - Two-Model T-Learner: Qini={res_t0['qini_coef']:.2f} | AUUC={res_t0['auuc']:.2f} | Top Quintile Uplift={res_t0['top_quintile_uplift']*100.0:.2f}%")

    # Generate Publication Diagnostic Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Qini Curve Comparison
    # Extract Qini curve coordinates: (fraction_targeted, cumulative_gains)
    curve_h = res_h0["qini_curve"]
    curve_a = res_a0["qini_curve"]
    curve_t = res_t0["qini_curve"]

    # Random baseline line: (0, 0) to (1.0, total_incremental_gains)
    total_gains = curve_h[-1][1]
    ax1.plot([0.0, 1.0], [0.0, total_gains], linestyle="--", color="gray", lw=1.5, label=f"Random Targeting (ATE = +{overall_ate:.2f}%)")
    ax1.plot([p[0] for p in curve_h], [p[1] for p in curve_h], color="#0f766e", lw=2.5, label="Honest Causal Tree (Athey & Imbens)")
    ax1.plot([p[0] for p in curve_t], [p[1] for p in curve_t], color="#2563eb", lw=2.0, linestyle="-.", label="Two-Model T-Learner")
    ax1.plot([p[0] for p in curve_a], [p[1] for p in curve_a], color="#dc2626", lw=1.8, linestyle=":", label="Adaptive Causal Tree (No Honesty)")

    ax1.set_xlabel("Fraction of Customers Targeted", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cumulative Incremental Site Visits", fontsize=10, fontweight="bold")
    ax1.set_title("A: Qini Curves on Randomized Test Customers (N = 8,539)", fontsize=11, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # Panel B: Decile Uplift Comparison
    # Compute empirical decile uplifts for honest tree
    order_h = np.argsort(tau_honest0)[::-1]
    decile_size = len(order_h) // 10
    decile_uplifts = []
    decile_labels = [f"D{i+1}" for i in range(10)]

    for d in range(10):
        d_idx = order_h[d * decile_size : (d + 1) * decile_size]
        y_d = [yte0[i] for i in d_idx]
        w_d = [wte0[i] for i in d_idx]
        nt = sum(w_d)
        nc = len(w_d) - nt
        if nt > 0 and nc > 0:
            rate_t = sum(y for y, w in zip(y_d, w_d) if w == 1.0) / nt
            rate_c = sum(y for y, w in zip(y_d, w_d) if w == 0.0) / nc
            uplift_pct = (rate_t - rate_c) * 100.0
        else:
            uplift_pct = 0.0
        decile_uplifts.append(uplift_pct)

    colors = ["#047857" if u > overall_ate else "#0284c7" if u > 0 else "#e11d48" for u in decile_uplifts]
    bars = ax2.bar(decile_labels, decile_uplifts, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax2.axhline(overall_ate, color="#b91c1c", linestyle="--", lw=1.5, label=f"Average Uplift ({overall_ate:.2f} pp)")
    ax2.axhline(0.0, color="black", lw=0.8)

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.2 if h >= 0 else -0.5
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:+.1f}%", ha="center", va=va, fontsize=8.5, fontweight="bold")

    ax2.set_xlabel("Targeting Decile (D1 = Highest Predicted Uplift)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Empirical Incremental Visit Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title("B: Incremental Visit Rate by Honest Model Decile", fontsize=11, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    qini_png = out / "causal_uplift_qini_curve.png"
    qini_svg = out / "causal_uplift_qini_curve.svg"
    fig.savefig(qini_png, dpi=300)
    fig.savefig(qini_svg)
    plt.close(fig)
    print(f"Saved diagnostic plots: {qini_png} & {qini_svg}")

    # 2. Multi-Seed Rigorous Benchmark (20 Seeds)
    print(f"\n--- Running {n_seeds}-Seed Rigorous Benchmark ---")
    metrics_treat_all = {"qini": [], "auuc": [], "top_q": [], "top_d": []}
    metrics_adaptive = {"qini": [], "auuc": [], "top_q": [], "top_d": []}
    metrics_tlearner = {"qini": [], "auuc": [], "top_q": [], "top_d": []}
    metrics_honest = {"qini": [], "auuc": [], "top_q": [], "top_d": []}

    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        perm = rng.permutation(n_total)
        tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

        xtr, ytr, wtr = pick(X_all, tr), pick(y_all, tr), pick(w_all, tr)
        xte, yte, wte = pick(X_all, te), pick(y_all, te), pick(w_all, te)

        # Baseline: Uniform Treat All (Random Ordering / Constant Effect)
        res_all = m.uplift(yte, wte, [0.0] * len(yte))
        metrics_treat_all["qini"].append(0.0)
        metrics_treat_all["auuc"].append(res_all["auuc"])
        metrics_treat_all["top_q"].append(res_all["top_quintile_uplift"] * 100.0)
        metrics_treat_all["top_d"].append(res_all["top_decile_uplift"] * 100.0)

        # Model 1: Adaptive Causal Tree (No Honesty)
        m_a = ModelTree(gain="causal", leaf="mean", max_depth=2, min_rows=250, prune="none", honesty=False, seed=2000+seed)
        m_a.fit(xtr, (ytr, wtr), col_names=features, dtypes=types)
        tau_a = m_a.predict(xte, kind="uplift")
        res_a = m.uplift(yte, wte, tau_a)
        metrics_adaptive["qini"].append(res_a["qini_coef"])
        metrics_adaptive["auuc"].append(res_a["auuc"])
        metrics_adaptive["top_q"].append(res_a["top_quintile_uplift"] * 100.0)
        metrics_adaptive["top_d"].append(res_a["top_decile_uplift"] * 100.0)

        # Model 2: Two-Model T-Learner
        tr_t = [i for i, wt in enumerate(wtr) if wt == 1.0]
        tr_c = [i for i, wt in enumerate(wtr) if wt == 0.0]
        m1 = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=150)
        m1.fit(pick(xtr, tr_t), pick(ytr, tr_t), col_names=features, dtypes=types)
        m0 = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=150)
        m0.fit(pick(xtr, tr_c), pick(ytr, tr_c), col_names=features, dtypes=types)
        p1 = m1.predict_proba(xte)
        p0 = m0.predict_proba(xte)
        tau_t = [p1_i - p0_i for p1_i, p0_i in zip(p1, p0)]
        res_t = m.uplift(yte, wte, tau_t)
        metrics_tlearner["qini"].append(res_t["qini_coef"])
        metrics_tlearner["auuc"].append(res_t["auuc"])
        metrics_tlearner["top_q"].append(res_t["top_quintile_uplift"] * 100.0)
        metrics_tlearner["top_d"].append(res_t["top_decile_uplift"] * 100.0)

        # Model 3: Honest Causal Tree
        m_h = ModelTree(gain="causal", leaf="mean", max_depth=2, min_rows=250, prune="none", honesty=True, seed=2000+seed)
        m_h.fit(xtr, (ytr, wtr), col_names=features, dtypes=types)
        tau_h = m_h.predict(xte, kind="uplift")
        res_h = m.uplift(yte, wte, tau_h)
        metrics_honest["qini"].append(res_h["qini_coef"])
        metrics_honest["auuc"].append(res_h["auuc"])
        metrics_honest["top_q"].append(res_h["top_quintile_uplift"] * 100.0)
        metrics_honest["top_d"].append(res_h["top_decile_uplift"] * 100.0)

        print(f"  Seed {seed+1:2d}/{n_seeds}: Honest Qini={res_h['qini_coef']:5.2f} | T-Learner Qini={res_t['qini_coef']:5.2f} | Adaptive Qini={res_a['qini_coef']:5.2f}")

    print("\n" + "=" * 80)
    print(f"FINAL 20-SEED BENCHMARK SUMMARY (N_TEST = {n_test:,})")
    print("=" * 80)
    print(f"{'Architecture':<32s} | {'Qini Score':<16s} | {'AUUC':<16s} | {'Top Quintile Uplift':<20s} | {'Top Decile Uplift':<18s}")
    print("-" * 115)

    for name, met in [
        ("Uniform Treat All (Random)", metrics_treat_all),
        ("Adaptive Causal Tree (No Honesty)", metrics_adaptive),
        ("Two-Model T-Learner", metrics_tlearner),
        ("Honest Causal Tree (Athey & Imbens)", metrics_honest),
    ]:
        q_m, q_s = np.mean(met["qini"]), np.std(met["qini"])
        a_m, a_s = np.mean(met["auuc"]), np.std(met["auuc"])
        tq_m, tq_s = np.mean(met["top_q"]), np.std(met["top_q"])
        td_m, td_s = np.mean(met["top_d"]), np.std(met["top_d"])
        print(f"{name:<32s} | {q_m:6.2f} ± {q_s:5.2f} | {a_m:6.2f} ± {a_s:5.2f} | {tq_m:5.2f}% ± {tq_s:4.2f}% | {td_m:5.2f}% ± {td_s:4.2f}%")

    wins_honest_vs_adapt = sum(1 for h, a in zip(metrics_honest["qini"], metrics_adaptive["qini"]) if h > a)
    wins_honest_vs_tlearner = sum(1 for h, t in zip(metrics_honest["qini"], metrics_tlearner["qini"]) if h > t)
    print(f"\nHead-to-head Qini Superiority:")
    print(f"  - Honest Causal Tree beats Adaptive Tree: {wins_honest_vs_adapt} / {n_seeds} seeds")
    print(f"  - Honest Causal Tree beats T-Learner    : {wins_honest_vs_tlearner} / {n_seeds} seeds")


if __name__ == "__main__":
    run_causal_uplift_study(n_seeds=20)
