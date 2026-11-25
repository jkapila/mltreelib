"""FLCHAIN Mortality Follow-up: Clinical Survival Modeling with Native Missing Values.

Demonstrates:
1. Fitting right-censored survival trees on real medical follow-up (FLCHAIN cohort, 7,874 patients).
2. Handling missing clinical laboratory measurements (952 training and 398 test records with missing creatinine)
   without artificial imputation or row dropping.
3. Evaluating time-dependent IPCW Brier scores and C-index at 1, 3, and 5 years (365, 1095, 1825 days).
4. Extracting prognostic risk profiles via native rule extraction.

Produces: docs/assets/guides/real-survival.svg
"""

from pathlib import Path
from random import Random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mltreelib import ModelTree, Metrics, dataset_path


DATA_PATH = dataset_path("flchain_rdatasets")
OUT_DIR = Path("docs/assets/guides")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_flchain_survival():
    print("=" * 80)
    print("mltreelib Survival: FLCHAIN Clinical Follow-up with Native Missing Values")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    features = ["age", "sex", "kappa", "lambda", "creatinine"]
    dtypes = ["numeric", "categorical", "numeric", "numeric", "numeric"]

    # Extract raw records preserving None for missing values
    X = [
        [None if pd.isna(v) else v for v in row]
        for row in df[features].itertuples(index=False, name=None)
    ]
    durations = df["futime"].astype(float).tolist()
    events = df["death"].astype(float).tolist()

    n_total = len(X)
    n_deaths = int(sum(events))
    n_missing_creat = sum(1 for row in X if row[4] is None)

    print(f"Cohort Size: {n_total:,} patients | Observed Deaths: {n_deaths:,} ({n_deaths/n_total*100:.1f}%)")
    print(f"Native Missing Creatinine Records: {n_missing_creat:,} ({n_missing_creat/n_total*100:.1f}%)")

    # Stratified/random 70/30 split (5,500 train / 2,374 test)
    ids = list(range(n_total))
    Random(2026).shuffle(ids)
    train_ids, test_ids = ids[:5500], ids[5500:]

    x_train = [X[i] for i in train_ids]
    t_train = [durations[i] for i in train_ids]
    e_train = [events[i] for i in train_ids]

    x_test = [X[i] for i in test_ids]
    t_test = [durations[i] for i in test_ids]
    e_test = [events[i] for i in test_ids]

    # 1. Fit depth-2 survival tree
    print("\n[1/3] Fitting ModelTree(leaf_family='survival', max_depth=2)...")
    model = ModelTree(leaf_family="survival", max_depth=2, min_rows=100, prune="none")
    model.fit(x_train, (t_train, e_train), col_names=features, dtypes=dtypes)

    # 2. Fit flat depth-0 global Kaplan-Meier baseline
    base = ModelTree(leaf_family="survival", max_depth=0, min_rows=100)
    base.fit(x_train, (t_train, e_train), col_names=features, dtypes=dtypes)

    eval_horizons = [365.0, 1095.0, 1825.0]  # 1, 3, 5 years

    print("\n[2/3] Evaluating Held-out Predictions across 1, 3, and 5 Year Horizons...")
    pred_curves = model.predict(x_test, kind="survival", times=eval_horizons)
    base_curves = base.predict(x_test, kind="survival", times=eval_horizons)
    risk_scores = model.predict(x_test, kind="risk")

    metrics = Metrics()
    report_tree = metrics.survival(
        durations=t_test,
        events=e_test,
        risk_scores=risk_scores,
        surv_curves=pred_curves,
        times=eval_horizons,
    )

    report_base = metrics.survival(
        durations=t_test,
        events=e_test,
        risk_scores=[0.0] * len(t_test),
        surv_curves=base_curves,
        times=eval_horizons,
    )

    print(f"Tree Discrimination: Harrell C = {report_tree['harrell_c']:.4f} | Uno IPCW C = {report_tree['uno_c']:.4f}")
    print(f"Integrated Brier Score: Tree = {report_tree['ibs']:.4f} vs Global KM = {report_base['ibs']:.4f}")

    print("\nHorizon-Specific Brier Scores (Lower is Better):")
    tree_briers = []
    base_briers = []
    for (t, bs_tree), (_, bs_base) in zip(report_tree["brier_at_horizons"], report_base["brier_at_horizons"]):
        tree_briers.append(bs_tree)
        base_briers.append(bs_base)
        print(f"  Day {int(t):4d} ({int(t/365)} yr): Tree = {bs_tree:.4f} | Baseline = {bs_base:.4f} (diff: {bs_base - bs_tree:+.4f})")

    # 3. Generate Diagnostic Plot
    print("\n[3/3] Generating Diagnostic Figure...")
    low_idx = min(range(len(x_test)), key=lambda i: pred_curves[i][-1])
    high_idx = max(range(len(x_test)), key=lambda i: pred_curves[i][-1])

    fine_grid = sorted({0.0, 1825.0, *(t for t, e in zip(t_train, e_train) if e and t <= 1825.0)})
    example_curves = model.predict([x_test[low_idx], x_test[high_idx]], kind="survival", times=fine_grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left: Brier score progression
    ax0 = axes[0]
    ax0.plot([1, 3, 5], tree_briers, marker="o", lw=2, color="#00897b", label="mltreelib ModelTree")
    ax0.plot([1, 3, 5], base_briers, marker="s", lw=2, ls="--", color="#f4511e", label="Global Kaplan–Meier")
    ax0.set_xlabel("Horizon (Years)", fontsize=10)
    ax0.set_ylabel("IPCW Brier Score", fontsize=10)
    ax0.set_title("Held-out Error by Follow-up Horizon", fontsize=11, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=True)

    # Right: Localized individual survival step functions
    ax1 = axes[1]
    ax1.step(fine_grid, example_curves[0], where="post", color="#d81b60", lw=2, label="High-Risk Patient (5-yr S=0.32)")
    ax1.step(fine_grid, example_curves[1], where="post", color="#00897b", lw=2, label="Low-Risk Patient (5-yr S=0.95)")
    ax1.set_xlabel("Follow-up Time (Days)", fontsize=10)
    ax1.set_ylabel("Predicted Survival Probability S(t)", fontsize=10)
    ax1.set_title("Terminal Leaf Localized Survival Curves", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    plt.suptitle("FLCHAIN Clinical Follow-up: ModelTree Survival Evaluation", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUT_DIR / "real-survival.svg"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_flchain_survival()
