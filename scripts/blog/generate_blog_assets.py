"""Generate data, tree diagrams, and metrics diagrams for mltreelib blog articles.

Articles covered:
1. docs/blog/uplift-modeling-with-causal-trees.md
   - Data: Customer marketing discount campaign with Persuadables, Sleeping Dogs, Sure Things, Lost Causes.
   - Diagrams: blog_causal_tree.svg (tree) and blog_causal_metrics.png (Qini curve & decile uplift).
2. docs/blog/why-credit-scoring-needs-glass-box-trees.md
   - Data: Borrower credit underwriting scorecard (CIBIL score, income, delinquencies, DTI).
   - Diagrams: blog_credit_tree.svg (tree) and blog_credit_metrics.png (ROC, KS, Deciles, PSI).
3. docs/blog/zero-imputation-predictive-maintenance.md
   - Data: Industrial turbomachinery bearing sensor telemetry with missing sensor dropouts.
   - Diagrams: blog_maintenance_tree.svg (tree) and blog_maintenance_metrics.png (Residuals, Feature Importance).
"""

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from mltreelib import ModelTree

if HAS_MATPLOTLIB:
    from mltreelib import plots
else:
    plots = None

DOCS_ASSETS = Path("docs/assets")
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)


def render_dot_to_svg(dot_str: str, output_path: Path):
    """Render Graphviz DOT to SVG using the dot CLI."""
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_str.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    output_path.write_bytes(proc.stdout)
    print(f"  [SVG] Wrote {output_path} ({len(proc.stdout)} bytes)")


# ==============================================================================
# 1. Blog: Uplift Modeling with Causal Trees
# ==============================================================================
def run_blog_causal_uplift():
    print("\n--- 1. Generating Blog Uplift Modeling Assets ---")
    random.seed(42)
    n = 1000

    # Features: X0 = Loyalty / Tenure, X1 = Engagement / Frequency
    X: list[list[float]] = []
    y: list[float] = []
    w: list[float] = []
    true_tau: list[float] = []

    for _ in range(n):
        x0 = random.uniform(-1.0, 1.0)
        x1 = random.uniform(-1.0, 1.0)
        treat = 1.0 if random.random() < 0.5 else 0.0

        # Define 4 personas:
        if x0 > 0.0:
            # Persuadables: discount creates strong positive conversion
            p0 = 0.15
            p1 = 0.75
            tau = p1 - p0  # +0.60
        elif x1 > 0.0:
            # Sleeping Dogs: discount annoys customer, causes unsubscribe/churn
            p0 = 0.55
            p1 = 0.10
            tau = p1 - p0  # -0.45
        else:
            # Lost causes / Sure things: discount has zero effect
            p0 = 0.25
            p1 = 0.25
            tau = 0.0

        prob = p1 if treat == 1.0 else p0
        outcome = 1.0 if random.random() < prob else 0.0

        X.append([x0, x1])
        y.append(outcome)
        w.append(treat)
        true_tau.append(tau)

    # Split train / test
    split = 700
    X_train, y_train, w_train = X[:split], y[:split], w[:split]
    X_test, y_test, w_test = X[split:], y[split:], w[split:]

    # Fit Honest Causal ModelTree
    col_names = ["loyalty_tenure", "engagement_freq"]
    tree = ModelTree(
        gain="causal",
        max_depth=3,
        min_rows=30,
        prune="none",
        honesty=True,
        honesty_fraction=0.5,
        seed=42,
    )
    tree.fit(X_train, (y_train, w_train), col_names=col_names)

    # 1. Tree Diagram
    dot = tree.to_dot()
    tree_svg_path = DOCS_ASSETS / "blog_causal_tree.svg"
    render_dot_to_svg(dot, tree_svg_path)

    if not HAS_MATPLOTLIB:
        print("  [Note] matplotlib not installed; skipping PNG metrics plot")
        return

    # Predict test uplift (CATE)
    pred_tau = tree.predict(X_test, kind="uplift")

    # 2. Metrics Diagram: Qini / Cumulative Uplift Curve & Decile Uplift
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Cumulative Uplift (Qini Curve)
    # Sort test set by predicted uplift descending
    paired = sorted(zip(pred_tau, y_test, w_test), key=lambda t: t[0], reverse=True)
    n_test = len(paired)

    cum_treated = 0
    cum_control = 0
    cum_y_treated = 0
    cum_y_control = 0

    fractions = [0.0]
    qini_vals = [0.0]

    for i, (_, yi, wi) in enumerate(paired, 1):
        if wi == 1.0:
            cum_treated += 1
            cum_y_treated += yi
        else:
            cum_control += 1
            cum_y_control += yi

        if cum_treated > 0 and cum_control > 0:
            scale = cum_treated / cum_control if cum_control > 0 else 1.0
            qini = cum_y_treated - (cum_y_control * scale)
        else:
            qini = 0.0

        if i % 10 == 0 or i == n_test:
            fractions.append(i / n_test)
            qini_vals.append(qini)

    ax1 = axes[0]
    plots._style_axis(ax1)
    ax1.plot(
        fractions,
        qini_vals,
        color="#00897b",
        linewidth=2.5,
        label="Honest Causal Tree (Targeting Persuadables)",
    )
    # Baseline random targeting line
    final_qini = qini_vals[-1]
    ax1.plot(
        [0.0, 1.0],
        [0.0, final_qini],
        color="#78909c",
        linestyle="--",
        linewidth=1.5,
        label="Random Targeting (No Model)",
    )
    ax1.set_xlabel("Fraction of Population Targeted")
    ax1.set_ylabel("Cumulative Incremental Conversions (Qini)")
    ax1.set_title(
        "Uplift Curve: Targeting Persuadables Avoids Sleeping Dogs",
        fontweight="semibold",
    )
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Decile Uplift
    # Group test into 5 score buckets
    buckets = 5
    bucket_size = n_test // buckets
    decile_uplifts = []
    bucket_labels = []

    for b in range(buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < buckets - 1 else n_test
        chunk = paired[start:end]
        t_y = [yi for _, yi, wi in chunk if wi == 1.0]
        c_y = [yi for _, yi, wi in chunk if wi == 0.0]
        mean_t = sum(t_y) / len(t_y) if t_y else 0.0
        mean_c = sum(c_y) / len(c_y) if c_y else 0.0
        emp_uplift = (mean_t - mean_c) * 100.0  # percentage points
        decile_uplifts.append(emp_uplift)
        bucket_labels.append(f"Q{b + 1}\n(Top {int((b + 1) / buckets * 100)}%)")

    ax2 = axes[1]
    plots._style_axis(ax2)
    colors = ["#00897b" if u >= 0 else "#f4511e" for u in decile_uplifts]
    bars = ax2.bar(
        bucket_labels,
        decile_uplifts,
        color=colors,
        width=0.55,
        edgecolor="#263238",
        linewidth=0.5,
    )
    ax2.axhline(0, color="#263238", linestyle="-", linewidth=0.8)
    for bar, u in zip(bars, decile_uplifts):
        y_pos = bar.get_height() + (1.5 if u >= 0 else -4.0)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{u:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_xlabel("Predicted Uplift Quintile (Descending)")
    ax2.set_ylabel("Empirical Treatment Effect (Treated % - Control %)")
    ax2.set_title(
        "Quintile Uplift: Isolating Persuadables vs Sleeping Dogs",
        fontweight="semibold",
    )
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    metrics_png_path = DOCS_ASSETS / "blog_causal_metrics.png"
    fig.savefig(metrics_png_path, dpi=160)
    plt.close(fig)
    print(f"  [PNG] Wrote {metrics_png_path}")


# ==============================================================================
# 2. Blog: Credit Scoring with Glass-Box Monotone Trees
# ==============================================================================
def run_blog_credit_scoring():
    print("\n--- 2. Generating Blog Credit Scoring Assets ---")
    random.seed(123)
    n = 1200

    # Features:
    # 0: cibil_score (300 to 850) -> monotone decreasing (-1)
    # 1: monthly_income ($2000 to $20000)
    # 2: delinquencies_24m (0 to 6) -> monotone increasing (+1)
    # 3: debt_to_income (0.1 to 0.8) -> monotone increasing (+1)
    X: list[list[float]] = []
    y: list[float] = []

    for _ in range(n):
        cibil = random.gauss(680, 75)
        cibil = min(850.0, max(300.0, cibil))
        income = random.uniform(2000.0, 18000.0)
        delinq = float(
            random.choices([0, 1, 2, 3, 4], weights=[0.60, 0.20, 0.10, 0.06, 0.04])[0]
        )
        dti = random.uniform(0.15, 0.75)

        # Monotonic latent log-odds of default
        logit = (
            2.5
            - 0.012 * (cibil - 600.0)
            + 0.75 * delinq
            + 2.2 * (dti - 0.35)
            - 0.0001 * (income - 5000.0)
        )
        prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit))))
        default_flag = 1.0 if random.random() < prob else 0.0

        X.append([cibil, income, delinq, dti])
        y.append(default_flag)

    split = 800
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Glass-box tree with monotone constraints: cibil (-1), delinq (+1), dti (+1)
    col_names = ["cibil_score", "monthly_income", "delinquencies_24m", "debt_to_income"]
    tree = ModelTree(
        gain="gini",
        max_depth=3,
        min_rows=40,
        monotone=[-1, 0, 1, 1],
        prune="none",
        seed=123,
    )
    tree.fit(X_train, y_train, col_names=col_names)

    # 1. Tree Diagram
    dot = tree.to_dot()
    tree_svg_path = DOCS_ASSETS / "blog_credit_tree.svg"
    render_dot_to_svg(dot, tree_svg_path)

    if not HAS_MATPLOTLIB:
        print("  [Note] matplotlib not installed; skipping PNG metrics plot")
        return

    # Predictions (probabilities)
    scores_train = [min(1.0, max(0.0, p)) for p in tree.predict(X_train)]
    scores_test = [min(1.0, max(0.0, p)) for p in tree.predict(X_test)]

    # 2. Metrics Diagram: 4-Panel Scorecard Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    plots.plot_roc(y_test, scores_test, ax=axes[0, 0])
    axes[0, 0].set_title(
        "ROC Curve: Discrimination Power (Test Set)", fontweight="semibold"
    )

    plots.plot_ks(y_test, scores_test, ax=axes[0, 1])
    axes[0, 1].set_title(
        "Kolmogorov-Smirnov: Max Separation Distance", fontweight="semibold"
    )

    plots.plot_decile(y_test, scores_test, n_buckets=10, ax=axes[1, 0])
    axes[1, 0].set_title(
        "Scorecard Decile Profile: Monotonic Default Capture", fontweight="semibold"
    )

    plots.plot_psi(scores_train, scores_test, n_bins=10, ax=axes[1, 1])
    axes[1, 1].set_title(
        "Population Stability Index: Train vs Out-of-Time", fontweight="semibold"
    )

    plt.tight_layout()
    metrics_png_path = DOCS_ASSETS / "blog_credit_metrics.png"
    fig.savefig(metrics_png_path, dpi=160)
    plt.close(fig)
    print(f"  [PNG] Wrote {metrics_png_path}")


# ==============================================================================
# 3. Blog: Zero-Imputation Predictive Maintenance for IoT Sensors
# ==============================================================================
def run_blog_predictive_maintenance():
    print("\n--- 3. Generating Blog Predictive Maintenance Assets ---")
    random.seed(999)
    n = 1000

    # Features:
    # 0: bearing_temp (°C, 40 to 110)
    # 1: vibration_rms (mm/s, 0.5 to 7.0)
    # 2: oil_pressure (psi, with missingness ~20% when overheated)
    X: list[list[float | None]] = []
    y: list[float] = []

    for _ in range(n):
        temp = random.uniform(40.0, 105.0)
        vib = random.uniform(0.5, 6.5)

        # When temperature is critical (>85°C), sensor telemetry drops out (None)
        if temp > 85.0 and random.random() < 0.65:
            oil_pressure = None
        else:
            oil_pressure = random.uniform(25.0, 55.0)

        # Ground truth Remaining Useful Life (RUL) in hours:
        # High vibration + high temp or missing pressure drastically cuts RUL
        rul = 350.0 - 2.8 * temp - 25.0 * vib
        if oil_pressure is None:
            rul -= 40.0  # Sensor blackout penalty
        else:
            rul += 0.8 * (oil_pressure - 40.0)

        rul = max(5.0, rul + random.gauss(0, 12.0))
        X.append([temp, vib, oil_pressure])
        y.append(rul)

    split = 750
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Fit ModelTree with linear leaves and surrogate routing (maxsurrogate=1)
    col_names = ["bearing_temp", "vibration_rms", "oil_pressure"]
    tree = ModelTree(
        leaf="linear",
        max_depth=3,
        min_rows=30,
        maxsurrogate=1,
        prune="none",
        seed=999,
    )
    tree.fit(X_train, y_train, col_names=col_names)

    # 1. Tree Diagram
    dot = tree.to_dot()
    tree_svg_path = DOCS_ASSETS / "blog_maintenance_tree.svg"
    render_dot_to_svg(dot, tree_svg_path)

    if not HAS_MATPLOTLIB:
        print("  [Note] matplotlib not installed; skipping PNG metrics plot")
        return

    # Predictions
    pred_test = tree.predict(X_test)

    # 2. Metrics Diagram: Residuals & Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plots.plot_residuals(y_test, pred_test, ax=axes[0])
    axes[0].set_title(
        "Remaining Useful Life (RUL): Actual vs Predicted (Test)", fontweight="semibold"
    )

    imp = tree.feature_importance()
    feature_names = ["bearing_temp", "vibration_rms", "oil_pressure"]
    imp_pairs = [(name, float(val)) for name, (_, val) in zip(feature_names, imp)]
    plots.plot_importance(imp_pairs, ax=axes[1])
    axes[1].set_title(
        "Split Gain Importance Shares: Sensor Telemetry", fontweight="semibold"
    )

    plt.tight_layout()
    metrics_png_path = DOCS_ASSETS / "blog_maintenance_metrics.png"
    fig.savefig(metrics_png_path, dpi=160)
    plt.close(fig)
    print(f"  [PNG] Wrote {metrics_png_path}")


if __name__ == "__main__":
    run_blog_causal_uplift()
    run_blog_credit_scoring()
    run_blog_predictive_maintenance()
    print("\n[Done] All blog diagrams and metrics assets generated successfully!")
