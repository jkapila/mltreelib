"""Demonstrate model trees across distributional leaf families on real benchmark datasets.

Datasets:
1. Probit: credit-g (OpenML 31, binary credit default)
2. Gamma: healthcare_insurance_expenses (TabArena, strictly positive continuous severity)
3. Tweedie: Fair's Affairs (zero-inflated continuous count / severity)
4. Tobit: Wooldridge Mroz (female labor supply hours censored at zero)
"""

import os
import pandas as pd
from mltreelib import ModelTree


def run_probit_benchmark():
    print("=" * 60)
    print("1. PROBIT FAMILY: German Credit Risk (Binary Classification)")
    print("=" * 60)
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

    # Macro tree partitioner
    tree = ModelTree(max_depth=2, min_rows=30, leaf="mean", prune="none")
    tree.fit(X, y, col_names=cols)

    # Retrain leaves with Probit GLM
    probit_tree = tree.retrain(
        X,
        y,
        mode="external",
        leaf="linear",
        leaf_family="probit",
        control={"l2": 1e-2},
    )

    print(f"Total rows: {len(X)}, Leaves: {probit_tree.n_leaves()}")
    for leaf in probit_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (probit scale): {leaf['intercept']:.4f}")
        for cname, coef in zip(cols, leaf["coef"]):
            if abs(coef) > 1e-4:
                print(f"      {cname:25s}: {coef:+.4f}")
    sample_probs = probit_tree.predict_proba(X[:5])
    print(f"Sample P(Bad Credit): {[round(p, 4) for p in sample_probs]}")


def run_gamma_benchmark():
    print("\n" + "=" * 60)
    print("2. GAMMA FAMILY: Healthcare Insurance Charges (Strictly Positive)")
    print("=" * 60)
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

    print(f"Total rows: {len(X)}, Leaves: {gamma_tree.n_leaves()}")
    for leaf in gamma_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (log link): {leaf['intercept']:.4f}")
        for cname, coef in zip(cols, leaf["coef"]):
            print(f"      {cname:15s}: {coef:+.4f}")
    sample_preds = gamma_tree.predict(X[:5])
    print(f"Sample Expected Charges: {[round(p, 2) for p in sample_preds]}")
    print(f"Actual Charges:          {[round(a, 2) for a in y[:5]]}")


def run_tweedie_benchmark():
    print("\n" + "=" * 60)
    print("3. TWEEDIE FAMILY: Fair Affairs (Zero-Inflated Continuous)")
    print("=" * 60)
    df = pd.read_csv("datasets/external/affairs.csv")
    cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    X = df[cols].astype(float).values.tolist()
    y = df["affairs"].astype(float).values.tolist()

    zero_pct = (df["affairs"] == 0).mean() * 100.0
    print(f"Total rows: {len(X)} (Zero mass: {zero_pct:.1f}%)")

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

    print(f"Leaves: {tweedie_tree.n_leaves()}")
    for leaf in tweedie_tree.leaves():
        print(f"  Leaf {leaf['id']} (n={leaf['n']}):")
        print(f"    Intercept (log link): {leaf['intercept']:.4f}")
        for cname, coef in zip(cols, leaf["coef"]):
            print(f"      {cname:15s}: {coef:+.4f}")
    sample_preds = tweedie_tree.predict(X[:5])
    print(f"Sample Predicted Expectations: {[round(p, 4) for p in sample_preds]}")


def run_tobit_benchmark():
    print("\n" + "=" * 60)
    print("4. TOBIT FAMILY: Mroz Female Labor Supply (Left-Censored at 0)")
    print("=" * 60)
    df = pd.read_csv("datasets/external/mroz.csv")
    cols = ["kidslt6", "kidsge6", "age", "educ", "nwifeinc", "exper"]
    X = df[cols].astype(float).values.tolist()
    y = df["hours"].astype(float).values.tolist()

    censored_pct = (df["hours"] == 0).mean() * 100.0
    print(f"Total rows: {len(X)} (Censored at 0: {censored_pct:.1f}%)")

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

    print(f"Leaves: {tobit_tree.n_leaves()}")
    for l_tobit, l_ols in zip(tobit_tree.leaves(), ols_tree.leaves()):
        print(f"  Leaf {l_tobit['id']} (n={l_tobit['n']}):")
        print(f"    Tobit intercept: {l_tobit['intercept']:.2f} | OLS intercept: {l_ols['intercept']:.2f}")
        for cname, c_tob, c_ols in zip(cols, l_tobit["coef"], l_ols["coef"]):
            print(f"      {cname:12s} -> Tobit: {c_tob:+8.2f} | OLS: {c_ols:+8.2f}")


if __name__ == "__main__":
    run_probit_benchmark()
    run_gamma_benchmark()
    run_tweedie_benchmark()
    run_tobit_benchmark()
