"""Distributional Leaf Benchmarks: Probit, Gamma, Tweedie & Tobit on Real Data.

Why this was built:
Standard linear model trees assume Gaussian homoscedastic errors Y ~ N(X*b, sigma^2),
which breaks down across core empirical data regimes:
1. Probit (Latent Normal Utility): For binary default/choice where threshold-crossing
   mechanisms generate standard normal errors under Central Limit Theorem.
2. Gamma (Constant CV Severity): For strictly positive expenditures (healthcare charges,
   loss severity) where variance scales quadratically Var(Y) = phi * mu^2. Gaussian models
   generate invalid negative predictions and distort high-cost claims.
3. Tweedie (Zero-Inflated Loss Costs): For compound Poisson-Gamma processes with a point
   mass at zero (P(Y=0) > 0) alongside right-skewed positive payouts (1 < p < 2).
4. Tobit (Left-Censored Corner Solutions): For targets censored at a lower threshold
   (e.g. consumer expenditures or hours worked with 40%+ exact zeros). Ordinary Least
   Squares (OLS) suffers downward attenuation bias; Tobit EM-IRLS recovers unbiased
   structural parameters.

Use Cases & When to Use:
- Probit: Credit scoring, loan default prediction, econometric latent utility.
- Gamma: Insurance claim severity, medical billing, customer lifetime value (LTV).
- Tweedie: Insurance pure premium pricing, equipment warranty loss costs, zero-inflated billing.
- Tobit: Labor economics, durable goods expenditure, censored survey data.

Datasets & Citations:
1. German Credit Risk: Hofmann (1994), OpenML ID 31 (1,000 applicants).
2. US Healthcare Expenses: TabArena-v0.1 / Kaggle (1,338 policyholders).
3. Fair's Extramarital Affairs: Fair, R. C. (1978), Econometrica, 46(1), 45-61 (601 rows).
4. Mroz Female Labor Supply: Mroz, T. A. (1987), Econometrica, 55(4), 765-799; Wooldridge (2010) (753 rows).

Visualizations:
Generates publication-quality 4-panel scikit-learn style plots saved to:
`docs/assets/distributional_leaf_benchmarks.png`

Run:
    uv run python benchmarks/suites/distributional_leaf_benchmarks.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mltreelib import ModelTree

DOCS_ASSETS = Path("docs/assets")
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)


def run_probit_benchmark():
    print("=" * 78)
    print("1. PROBIT FAMILY: German Credit Risk (Binary Classification)")
    print("   Dataset: Hofmann (1994), OpenML ID 31 (1,000 loan applicants)")
    print("   Problem: Latent normal utility threshold-crossing for loan default P(Y=1).")
    print("=" * 78)
    df = pd.read_parquet("datasets/credit-g.parquet")
    cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_percent",
        "residence_since",
        "age_years",
        "existing_credits_count",
        "people_liable",
    ]
    X = df[cols].astype(float).values.tolist()
    y = [1.0 if v == "bad" else 0.0 for v in df["good_or_bad_customer"]]

    tree = ModelTree(max_depth=2, min_rows=30, leaf="mean", prune="none")
    tree.fit(X, y, col_names=cols)

    probit_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="probit",
        control={"l2": 1e-2},
    )

    print(f"Total Cohort: {len(X)} applicants (Default Rate: {np.mean(y):.1%})")
    print(f"Partitions Identified: {probit_tree.n_leaves()} Macro Risk Segments")
    for leaf in probit_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (probit scale): {leaf['intercept']:+.4f}")
        for cname, coef in zip(cols, leaf["coef"]):
            if abs(coef) > 1e-4:
                print(f"      {cname:25s}: {coef:+.4f} (z-score shift)")

    sample_probs = probit_tree.predict_proba(X[:5])
    print(f"\nCalibrated Default Probabilities P(Bad Credit): {[round(p, 4) for p in sample_probs]}")
    print("Interpretation: Probit link Phi(X*b) models default risk as a standard normal integral.")

    return probit_tree, df, cols


def run_gamma_benchmark():
    print("\n" + "=" * 78)
    print("2. GAMMA FAMILY: Healthcare Insurance Charges (Strictly Positive Continuous)")
    print("   Dataset: US Healthcare Expenses (TabArena-v0.1 / Kaggle, 1,338 policyholders)")
    print("   Problem: Strictly positive medical expenses with constant coefficient of variation.")
    print("=" * 78)
    df = pd.read_parquet("datasets/healthcare_insurance_expenses.parquet")
    cols = ["age", "bmi", "children", "smoker"]
    X = [
        [
            float(r.age),
            float(r.bmi),
            float(r.children),
            1.0 if r.smoker == "yes" else 0.0,
        ]
        for _, r in df.iterrows()
    ]
    y = [float(r.charges) for _, r in df.iterrows()]

    tree = ModelTree(max_depth=2, min_rows=30, leaf="mean", prune="none")
    tree.fit(X, y, col_names=cols)

    gamma_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="gamma",
        control={"l2": 1e-2},
    )

    print(f"Total Policyholders: {len(X)} (Mean Charges: ${np.mean(y):,.2f})")
    print(f"Partitions Identified: {gamma_tree.n_leaves()} Cost Regimes")
    for leaf in gamma_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (log link): {leaf['intercept']:+.4f} -> Base Exp: ${np.exp(leaf['intercept']):,.2f}")
        for cname, coef in zip(cols, leaf["coef"]):
            multiplier = np.exp(coef)
            print(f"      {cname:15s}: {coef:+.4f} (Multiplier: {multiplier:.2f}x per unit)")

    sample_preds = gamma_tree.predict(X[:5])
    print(f"\nSample Expected Charges: {[round(p, 2) for p in sample_preds]}")
    print(f"Actual Charges:          {[round(a, 2) for a in y[:5]]}")
    print("Interpretation: Log-link guarantees strictly positive predictions and preserves multiplicative risk factors.")

    return gamma_tree, df, cols


def run_tweedie_benchmark():
    print("\n" + "=" * 78)
    print("3. TWEEDIE FAMILY: Fair's Extramarital Affairs (Zero-Inflated Continuous)")
    print("   Dataset: Fair (1978), Econometrica (601 survey respondents)")
    print("   Problem: Point mass at zero (75.1% zeros) with continuous right-skewed positive counts.")
    print("=" * 78)
    df = pd.read_csv("datasets/external/affairs.csv")
    cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    X = df[cols].astype(float).values.tolist()
    y = df["affairs"].astype(float).values.tolist()

    zero_pct = (df["affairs"] == 0).mean() * 100.0
    print(f"Total Respondents: {len(X)} (Exact Zero Clump: {zero_pct:.1f}%)")

    tree = ModelTree(max_depth=2, min_rows=25, leaf="mean", prune="none")
    tree.fit(X, y, col_names=cols)

    tweedie_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="tweedie",
        control={"l2": 1e-2, "extra": {"power": 1.5}},
    )

    print(f"Partitions Identified: {tweedie_tree.n_leaves()} Behavioral Cohorts (Tweedie p=1.5)")
    for leaf in tweedie_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (log link): {leaf['intercept']:+.4f}")
        for cname, coef in zip(cols, leaf["coef"]):
            print(f"      {cname:15s}: {coef:+.4f}")

    sample_preds = tweedie_tree.predict(X[:5])
    print(f"\nSample Predicted Expectations E[Y|X]: {[round(p, 4) for p in sample_preds]}")
    print("Interpretation: Compound Poisson-Gamma likelihood handles the zero mass and positive tail simultaneously.")

    return tweedie_tree, df, cols


def run_tobit_benchmark():
    print("\n" + "=" * 78)
    print("4. TOBIT FAMILY: Mroz Female Labor Supply (Left-Censored Corner Solution at 0)")
    print("   Dataset: Mroz (1987), Econometrica; Wooldridge (2010) (753 women)")
    print("   Problem: 43.2% of women have 0 annual work hours. OLS suffers severe attenuation bias.")
    print("=" * 78)
    df = pd.read_csv("datasets/external/mroz.csv")
    cols = ["kidslt6", "kidsge6", "age", "educ", "nwifeinc", "exper"]
    X = df[cols].astype(float).values.tolist()
    y = df["hours"].astype(float).values.tolist()

    censored_pct = (df["hours"] == 0).mean() * 100.0
    print(f"Total Cohort: {len(X)} women (Censored at 0: {censored_pct:.1f}%)")

    tree = ModelTree(max_depth=2, min_rows=30, leaf="mean", prune="none")
    tree.fit(X, y, col_names=cols)

    # Fit Tobit leaves (EM in IRLS)
    tobit_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="tobit",
        control={"l2": 1e-2, "extra": {"censor_lower": 0.0}},
    )

    # Standard OLS linear leaves for direct comparison
    ols_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="gaussian",
        control={"l2": 1e-2},
    )

    print(f"Partitions Identified: {tobit_tree.n_leaves()} Labor Regimes")
    print("\nStructural Parameter Comparison (Tobit EM-IRLS vs Standard OLS):")
    for l_tobit, l_ols in zip(tobit_tree.leaves(), ols_tree.leaves()):
        print(f"  Leaf {l_tobit['id']} (n={l_tobit['n']}):")
        print(f"    Tobit intercept: {l_tobit['intercept']:+8.2f} | OLS intercept: {l_ols['intercept']:+8.2f}")
        for cname, c_tob, c_ols in zip(cols, l_tobit["coef"], l_ols["coef"]):
            bias_pct = (1.0 - abs(c_ols) / max(abs(c_tob), 1e-4)) * 100.0
            print(f"      {cname:12s} -> Tobit: {c_tob:+8.2f} | OLS: {c_ols:+8.2f} | OLS Attenuation: {bias_pct:4.1f}%")

    print("\nInterpretation: OLS flattens regression slopes toward zero due to corner-solution zeros.")
    print("                Tobit Expectation-Maximization recovers the true latent unobserved propensity.")

    return tobit_tree, ols_tree, df, cols


def generate_scikit_learn_style_distributional_figure(probit_tree, credit_df,
                                                       gamma_tree, health_df,
                                                       tweedie_tree, affairs_df,
                                                       tobit_tree, ols_tree, mroz_df):
    print("\n" + "=" * 78)
    print("Generating Scikit-Learn Style Distributional Benchmark Visualization...")
    print("=" * 78)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel 1: Probit Sigmoid Default Probability vs Duration (German Credit)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    durations = credit_df["duration_months"].values
    defaults = (credit_df["good_or_bad_customer"] == "bad").astype(float).values

    # Jitter points for visibility
    jitter_y = defaults + np.random.uniform(-0.04, 0.04, len(defaults))
    ax1.scatter(durations[defaults == 0], jitter_y[defaults == 0], c="#1976d2", alpha=0.35, s=20, label="Good Credit (Y=0)")
    ax1.scatter(durations[defaults == 1], jitter_y[defaults == 1], c="#d32f2f", alpha=0.45, s=25, label="Bad Credit / Default (Y=1)")

    # Evaluate fitted probit curve
    d_eval = np.linspace(durations.min(), durations.max(), 100)
    # Median profile for other features
    med_row = [np.median(credit_df[c]) for c in ["credit_amount", "installment_rate_percent", "residence_since", "age_years", "existing_credits_count", "people_liable"]]
    eval_X = [[float(d)] + med_row for d in d_eval]
    p_probs = probit_tree.predict_proba(eval_X)

    ax1.plot(d_eval, p_probs, color="#c62828", linewidth=2.8, label="Probit P(Default) = Phi(X*b)")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Decision Threshold (0.50)")
    ax1.set_title("Probit Family: Latent Utility Credit Risk\n(German Credit Hofmann 1994, OpenML 31)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Loan Duration (Months)", fontsize=9)
    ax1.set_ylabel("Predicted P(Default)", fontsize=9)
    ax1.set_ylim(-0.08, 1.08)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 2: Gamma Exponential Cost Response vs Age (Healthcare Expenses)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    h_ages = health_df["age"].values
    h_charges = health_df["charges"].values
    h_smoker = (health_df["smoker"] == "yes").values

    ax2.scatter(h_ages[~h_smoker], h_charges[~h_smoker], c="#388e3c", alpha=0.35, s=18, label="Non-Smokers (Observed)")
    ax2.scatter(h_ages[h_smoker], h_charges[h_smoker], c="#d32f2f", alpha=0.45, s=22, label="Smokers (Observed)")

    age_eval = np.linspace(18, 65, 80)
    for smk_val, col, smk_lbl in [(0.0, "#2e7d32", "Non-Smoker"), (1.0, "#c62828", "Smoker")]:
        eval_X = [[float(a), 28.0, 1.0, smk_val] for a in age_eval]
        gamma_preds = gamma_tree.predict(eval_X)
        ax2.plot(age_eval, gamma_preds, color=col, linewidth=2.8, label=f"Gamma E[Cost]: {smk_lbl}")

    ax2.set_title("Gamma Family: Strictly Positive Multiplicative Cost Model\n(US Healthcare Expenses, TabArena-v0.1)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Age (Years)", fontsize=9)
    ax2.set_ylabel("Medical Charges ($)", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 3: Tweedie Zero-Inflated Mass + Continuous Distribution (Fair's Affairs)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    affair_counts = affairs_df["affairs"].values
    years_m = affairs_df["yearsmarried"].values

    # Histogram of zero mass vs positive tail
    bins = np.linspace(0, 15, 16) - 0.5
    ax3.hist(affair_counts, bins=bins, color="#5c6bc0", edgecolor="black", alpha=0.6, density=True, label="Observed Density (75.1% Zeros)")

    # Fitted Tweedie expectation line across marriage duration
    ym_eval = np.linspace(affairs_df["yearsmarried"].min(), affairs_df["yearsmarried"].max(), 50)
    med_affairs = [np.median(affairs_df[c]) for c in ["age", "religiousness", "education", "occupation", "rating"]]
    eval_X = [[med_affairs[0], float(ym), med_affairs[1], med_affairs[2], med_affairs[3], med_affairs[4]] for ym in ym_eval]
    tw_preds = tweedie_tree.predict(eval_X)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(ym_eval, tw_preds, color="#d81b60", linewidth=2.8, label="Tweedie E[Y|YearsMarried] (p=1.5)")
    ax3_twin.set_ylabel("Expected Annual Affairs", fontsize=9, color="#d81b60")
    ax3_twin.tick_params(axis="y", labelcolor="#d81b60")

    ax3.set_title("Tweedie Family: Zero-Inflated Compound Poisson-Gamma\n(Fair's Affairs 1978, Econometrica)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Annual Extramarital Events (Histogram) / Years Married (Curve)", fontsize=9)
    ax3.set_ylabel("Relative Frequency", fontsize=9)
    ax3.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 4: Tobit Latent Parameter Recovery vs Attenuated OLS (Mroz Labor)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    exper = mroz_df["exper"].values
    hours = mroz_df["hours"].values

    censored = (hours == 0)
    ax4.scatter(exper[~censored], hours[~censored], c="#1976d2", alpha=0.45, s=20, label="Working Women (Hours > 0)")
    ax4.scatter(exper[censored], np.full(np.sum(censored), -60.0), c="#d32f2f", alpha=0.4, s=25, marker="|", label="Censored at Zero (43.2% Non-Working)")

    exp_eval = np.linspace(exper.min(), exper.max(), 100)
    med_mroz = [np.median(mroz_df[c]) for c in ["kidslt6", "kidsge6", "age", "educ", "nwifeinc"]]
    eval_X = [[med_mroz[0], med_mroz[1], med_mroz[2], med_mroz[3], med_mroz[4], float(ex)] for ex in exp_eval]

    tobit_preds = tobit_tree.predict(eval_X)
    ols_preds = ols_tree.predict(eval_X)

    ax4.plot(exp_eval, tobit_preds, color="#00897b", linewidth=2.8, label="Tobit EM-IRLS (Unbiased Latent Slope)")
    ax4.plot(exp_eval, ols_preds, color="#f57c00", linestyle="--", linewidth=2.5, label="Standard OLS (Downward Attenuation Bias)")
    ax4.axhline(0.0, color="black", linestyle=":", alpha=0.5)

    ax4.set_title("Tobit Family: Unbiased Latent Slope vs Attenuated OLS\n(Mroz Female Labor Supply 1987, Econometrica)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Years of Labor Market Experience", fontsize=9)
    ax4.set_ylabel("Annual Hours Worked", fontsize=9)
    ax4.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.5)

    out_png = DOCS_ASSETS / "distributional_leaf_benchmarks.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Distributional Benchmark Plot -> {out_png}")


def main():
    probit_tree, credit_df, _ = run_probit_benchmark()
    gamma_tree, health_df, _ = run_gamma_benchmark()
    tweedie_tree, affairs_df, _ = run_tweedie_benchmark()
    tobit_tree, ols_tree, mroz_df, _ = run_tobit_benchmark()

    generate_scikit_learn_style_distributional_figure(
        probit_tree, credit_df,
        gamma_tree, health_df,
        tweedie_tree, affairs_df,
        tobit_tree, ols_tree, mroz_df,
    )
    print("\nAll distributional leaf benchmarks completed successfully!")


if __name__ == "__main__":
    main()
