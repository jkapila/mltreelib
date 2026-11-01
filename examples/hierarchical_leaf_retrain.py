"""Hierarchical Macro-Tree to Micro-Leaf Retraining & Entity Clustering.

Demonstrates:
1. Fitting a global/macro ModelTree on cohort/segment attributes (city tier, car segment, model year).
2. Retraining leaves externally (mode="external") on micro-level vehicle telemetry
   (distance, service records, accident history, tire changes) with leaf_features subsetting.
3. Extracting entity primary keys per leaf via `leaf_samples(X, id_col="vin")` for downstream
   niche models (Tobit censored claims, Probit incident gates, survival hazard models).
"""

import random
from mltreelib import ModelTree


def generate_warranty_fleet(n: int = 600, seed: int = 42):
    """Simulate fleet warranty data with macro cohorts and micro telemetry."""
    rng = random.Random(seed)

    cities = [1.0, 2.0, 3.0]          # 1: Metro, 2: Suburban, 3: Rural
    segments = [1.0, 2.0, 3.0]        # 1: Economy, 2: SUV, 3: Commercial Fleet
    years = [2021.0, 2022.0, 2023.0, 2024.0]

    macro_rows = []
    full_rows = []
    targets = []
    vins = []

    for i in range(n):
        vin = f"1HGCR2F8{i:05d}"
        city = rng.choice(cities)
        segment = rng.choice(segments)
        year = rng.choice(years)

        # Micro vehicle parameters
        age_years = 2025.0 - year
        distance_km = rng.uniform(10_000, 120_000) * (0.8 + 0.4 * (segment == 3.0))
        service_count = max(0, int(rng.gauss(age_years * 2.0, 1.0)))
        accidents = rng.choice([0, 0, 0, 1, 2])
        tire_changes = max(0, int(distance_km / 40_000) + rng.choice([0, 1]))

        # Ground truth warranty claim cost:
        # Macro baseline sets the cohort floor
        macro_base = 200.0 * segment + 150.0 * city + 100.0 * age_years

        # Micro parameters drive individual car behavior
        # Notice: Commercial fleet (segment 3) has much higher sensitivity to distance & accidents
        if segment == 3.0:
            micro_effect = 0.015 * distance_km + 450.0 * accidents - 80.0 * service_count + 120.0 * tire_changes
        else:
            micro_effect = 0.005 * distance_km + 250.0 * accidents - 50.0 * service_count + 60.0 * tire_changes

        claim_cost = max(0.0, macro_base + micro_effect + rng.gauss(0.0, 50.0))

        macro_rows.append([city, segment, year])
        full_rows.append([city, segment, year, distance_km, float(service_count), float(accidents), float(tire_changes)])
        targets.append(claim_cost)
        vins.append(vin)

    return macro_rows, full_rows, targets, vins


def run_hierarchical_showcase():
    print("=" * 78)
    print("  HIERARCHICAL MACRO-TREE TO MICRO-LEAF RETRAINING & ENTITY ROUTING")
    print("=" * 78)

    macro_rows, full_rows, targets, vins = generate_warranty_fleet(n=800, seed=123)

    # -------------------------------------------------------------------------
    # Stage 1: Fit Global Macro Tree on Cohort Attributes
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Fitting Global Macro Tree (City Tier, Segment, Model Year)...")
    macro_tree = ModelTree(max_depth=2, min_rows=30, leaf="mean", prune="none")
    macro_cols = ["city", "segment", "year"]
    macro_tree.fit(macro_rows, targets, col_names=macro_cols)

    print(f"  Fitted macro tree with {macro_tree.n_leaves()} cohort leaf partitions.")
    for sp in macro_tree.splits():
        print(f"    Split on {sp['col']} at threshold {sp['threshold']:.2f} (gain: {sp['gain']:.2f})")

    # -------------------------------------------------------------------------
    # Stage 2: Mode='external' Retraining with Micro Features
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Retraining Micro Leaves (mode='external') using Vehicle Telemetry...")
    full_cols = ["city", "segment", "year", "distance_km", "service_count", "accidents", "tire_changes"]
    micro_cols = ["distance_km", "service_count", "accidents", "tire_changes"]

    # Retrain: routes via macro splits, but fits linear models using only micro features
    micro_model = macro_tree.retrain(
        full_rows,
        targets,
        mode="external",
        col_names=full_cols,
        leaf_features=micro_cols,
        leaf="linear",
        l2=1e-3,
    )

    print("  External retrain complete.")
    print("  Macro Tree leaves remain constant:", macro_tree.leaves()[0]["intercept"] is None)
    print("  Micro Model leaves are linear:", micro_model.leaves()[0]["intercept"] is not None)

    print("\n  Fitted Micro Leaf Models:")
    for lf in micro_model.leaves():
        coef_str = ", ".join(f"{c}: {v:+.4f}" for c, v in zip(lf["cols"], lf["coef"]))
        print(f"    Leaf #{lf['id']} (n={lf['n']:3d}): intercept={lf['intercept']:8.2f} | {coef_str}")

    # -------------------------------------------------------------------------
    # Stage 3: Entity Primary Key Routing (id_col="vin")
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Entity Primary Key Routing via leaf_samples(X, id_col='vin')...")
    leaf_clusters = micro_model.leaf_samples(full_rows, id_col=vins)

    for leaf_id, vin_list in sorted(leaf_clusters.items()):
        print(f"    Leaf #{leaf_id}: {len(vin_list)} vehicles clustered. Sample VINs: {vin_list[:3]}...")

    # -------------------------------------------------------------------------
    # Stage 4: Test Point Inference Comparison
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Prediction Comparison (Macro Cohort Baseline vs Micro-Tailored Model):")
    # Test car: Metro, Commercial Fleet, 2022, 90,000 km, 3 services, 1 accident, 2 tire changes
    test_car_macro = [[1.0, 3.0, 2022.0]]
    test_car_full = [[1.0, 3.0, 2022.0, 90_000.0, 3.0, 1.0, 2.0]]

    pred_macro = macro_tree.predict(test_car_macro)[0]
    pred_micro = micro_model.predict(test_car_full)[0]

    print(f"    Macro Cohort Mean Prediction : ${pred_macro:8.2f}")
    print(f"    Micro Telemetry-Aware Estimate: ${pred_micro:8.2f}")
    print(f"    Telemetry Adjustment         : ${pred_micro - pred_macro:+8.2f}")
    print("=" * 78)


if __name__ == "__main__":
    run_hierarchical_showcase()
