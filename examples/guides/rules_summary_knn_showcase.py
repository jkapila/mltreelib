"""Showcase of C5.0 Rule Extraction, R/statsmodels Leaf Summaries, and Cubist k-NN Adjustment.

Demonstrates:
1. R summary(lm) / Python statsmodels goodness-of-fit leaf summaries with covariance matrix inversion.
2. C5.0 rule-set extraction with path simplification and Quinlan's pessimistic binomial error pruning.
3. Cubist k-NN sample neighbor adjustment correcting leaf predictions via local training residuals.
"""

from __future__ import annotations

import math
import random

from mltreelib import ModelTree


def main():
    print("=" * 75)
    print("mltreelib Showcase: Rules, Leaf Summaries & k-NN Adjustments")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # PART 1: R summary(lm) / Python statsmodels Leaf Regression Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 1: R summary(lm) / Python statsmodels Leaf Regression Summary")
    print("=" * 75)
    print("Each linear leaf inverts the local covariance matrix (X^T X)^(-1)")
    print("to compute exact standard errors, t-values, and p-values.\n")

    random.seed(42)
    n = 200
    X_reg = []
    y_reg = []
    reg_cols = ["sqft", "beds", "age", "tax_rate"]

    for _ in range(n):
        sqft = random.uniform(800.0, 3500.0)
        beds = float(random.randint(1, 5))
        age = random.uniform(1.0, 50.0)
        tax = random.uniform(1.0, 3.5)

        # Distinct regimes
        if sqft > 2000.0:
            val = 140.0 + 0.16 * sqft + 22.0 * beds - 1.1 * age - 4.5 * tax
        else:
            val = 75.0 + 0.11 * sqft + 14.0 * beds - 0.7 * age - 2.8 * tax

        val += random.gauss(0.0, 8.0)
        X_reg.append([sqft, beds, age, tax])
        y_reg.append(val)

    tree_linear = ModelTree(
        leaf="linear",
        leaf_cols="all",
        max_depth=1,
        min_rows=30,
        seed=42,
    )
    tree_linear.fit(X_reg, y_reg, col_names=reg_cols)

    # Print exact R summary(lm) format for leaf 0
    print(tree_linear.leaf_summary(0))

    # -------------------------------------------------------------------------
    # PART 2: C5.0 Rule-Set Extraction with Pessimistic Pruning
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("PART 2: C5.0 Rule-Set Extraction with Quinlan's Pessimistic Pruning")
    print("=" * 75)
    print("Extracting decision paths into standalone IF-THEN rules...")
    print(
        "Preconditions are simplified and pruned using Quinlan's binomial bound (CF=0.25):\n"
    )

    random.seed(42)
    n_rule = 300
    X_rule = []
    y_rule = []
    rule_cols = ["credit_score", "debt_ratio", "income_k"]

    for _ in range(n_rule):
        cs = random.uniform(300.0, 850.0)
        dti = random.uniform(0.1, 0.8)
        inc = random.uniform(25.0, 150.0)

        # High credit & low DTI => approval score 85.0, else 35.0
        if cs > 680.0 and dti < 0.45:
            score = 85.0 + inc * 0.1
        elif cs <= 580.0 or dti >= 0.60:
            score = 25.0 + inc * 0.05
        else:
            score = 55.0 + inc * 0.08

        score += random.gauss(0.0, 3.0)
        X_rule.append([cs, dti, inc])
        y_rule.append(score)

    tree_rule = ModelTree(
        leaf="mean",
        max_depth=3,
        min_rows=15,
        seed=42,
    )
    tree_rule.fit(X_rule, y_rule, col_names=rule_cols)

    ruleset = tree_rule.extract_rules(X_rule, y_rule, prune=True, cf=0.25)
    print(ruleset)

    rule_preds = ruleset.predict(X_rule)
    rule_mae = sum(abs(p - t) for p, t in zip(rule_preds, y_rule)) / len(y_rule)
    print(f"Rule-Set Training MAE: {rule_mae:.2f}\n")

    # -------------------------------------------------------------------------
    # PART 3: Cubist k-NN Sample Neighbor Inference Adjustment
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("PART 3: Cubist k-NN Sample Neighbor Inference Adjustment")
    print("=" * 75)
    print("Cubist adjusts leaf predictions using normalized Manhattan distances")
    print("to nearest training instances: y_hat_adj(x) = y_hat(x) + sum(w_i * e_i)\n")

    random.seed(42)
    n_knn = 400
    X_knn = []
    y_knn = []
    knn_cols = ["sqft", "rooms", "distance_to_center"]

    for _ in range(n_knn):
        sqft = random.uniform(600.0, 3200.0)
        rooms = float(random.randint(1, 6))
        dist = random.uniform(1.0, 25.0)

        # Smooth non-linear neighborhood effect (e.g. localized hot-spots)
        local_hotspot = 40.0 * math.sin(sqft / 350.0) * math.cos(dist / 4.0)
        price = 80.0 + 0.14 * sqft + 18.0 * rooms - 3.2 * dist + local_hotspot
        price += random.gauss(0.0, 5.0)

        X_knn.append([sqft, rooms, dist])
        y_knn.append(price)

    split = 280
    X_tr, y_tr = X_knn[:split], y_knn[:split]
    X_te, y_te = X_knn[split:], y_knn[split:]

    tree_knn = ModelTree(
        leaf="mean",
        max_depth=3,
        min_rows=15,
        seed=42,
    )
    tree_knn.fit(X_tr, y_tr, col_names=knn_cols)

    base_preds = tree_knn.predict(X_te)
    base_mse = sum((p - t) ** 2 for p, t in zip(base_preds, y_te)) / len(y_te)
    base_mae = sum(abs(p - t) for p, t in zip(base_preds, y_te)) / len(y_te)

    adj_preds_3 = tree_knn.predict(X_te, neighbors=3)
    adj_mse_3 = sum((p - t) ** 2 for p, t in zip(adj_preds_3, y_te)) / len(y_te)
    adj_mae_3 = sum(abs(p - t) for p, t in zip(adj_preds_3, y_te)) / len(y_te)

    adj_preds_5 = tree_knn.predict(X_te, neighbors=5)
    adj_mse_5 = sum((p - t) ** 2 for p, t in zip(adj_preds_5, y_te)) / len(y_te)
    adj_mae_5 = sum(abs(p - t) for p, t in zip(adj_preds_5, y_te)) / len(y_te)

    adj_preds_9 = tree_knn.predict(X_te, neighbors=9)
    adj_mse_9 = sum((p - t) ** 2 for p, t in zip(adj_preds_9, y_te)) / len(y_te)
    adj_mae_9 = sum(abs(p - t) for p, t in zip(adj_preds_9, y_te)) / len(y_te)

    print(f"{'Inference Mode':<30} {'Test MSE':>12} {'Test MAE':>12}")
    print("-" * 56)
    print(f"{'Standard ModelTree':<30} {base_mse:>12.2f} {base_mae:>12.2f}")
    print(f"{'Cubist Adjustment (k=3)':<30} {adj_mse_3:>12.2f} {adj_mae_3:>12.2f}")
    print(f"{'Cubist Adjustment (k=5)':<30} {adj_mse_5:>12.2f} {adj_mae_5:>12.2f}")
    print(f"{'Cubist Adjustment (k=9)':<30} {adj_mse_9:>12.2f} {adj_mae_9:>12.2f}")
    print("-" * 56)

    reduction = ((base_mse - adj_mse_5) / base_mse) * 100.0
    print(f"\nResult: k-NN neighbor adjustment reduced test MSE by {reduction:.1f}%!")
    print("=" * 75)


if __name__ == "__main__":
    main()
