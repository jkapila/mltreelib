"""Two-control architecture, 0D leaf formulations, dual monotonicity, weights, and interaction constraints.

Run:
    python examples/two_controls_and_0d_showcase.py
"""

from mltreelib import LeafControl, ModelTree


def main():
    print("=" * 70)
    print("1. Two-Control Architecture: ModelTree + LeafControl")
    print("=" * 70)

    # LeafControl isolates local leaf model configuration:
    leaf_ctrl = LeafControl(
        leaf="linear",
        leaf_family="gaussian",
        l1=0.0,
        l2=1e-3,
        leaf_monotone=[1, -1],  # local feature slope constraints
    )
    print("Created LeafControl:", leaf_ctrl)

    # ModelTree constructor accepts control directly, keeping the tree API clean:
    model = ModelTree(
        max_depth=3,
        min_rows=10,
        tree_monotone=[1, 0],   # global split ordering
        col_weights=[2.0, 0.5], # feature split priorities
        control=leaf_ctrl,
    )
    print("ModelTree initialized with LeafControl.")

    print("\n" + "=" * 70)
    print("2. 0D Leaf Formulations & Stats Dictionary")
    print("=" * 70)

    # Synthetic binary classification data: 100 rows, 2 features
    X = [[float(i), float(i % 2)] for i in range(100)]
    y = [float(i % 2) for i in range(100)]
    weights = [1.0] * 100

    # Model tree with 0D probability leaves
    clf = ModelTree(leaf="prob", max_depth=2, min_rows=10)
    clf.fit(X, y, sample_weight=weights)

    print(f"Fitted tree with {clf.n_leaves()} leaves.")

    # Inspect the 0D statistics dictionary on leaves
    for i, leaf in enumerate(clf.leaves()):
        st = leaf["stats"]
        print(f"\nLeaf {i} (n={leaf['n']}):")
        print(f"  Classification stats: prob={st['prob']:.4f}, odds={st['odds']:.4f}, log_odds={st['log_odds']:.4f}, entropy={st['entropy']:.4f}, gini={st['gini']:.4f}")
        print(f"  Distribution stats  : mean={st['mean']:.4f}, std={st['std']:.4f}, var={st['var']:.4f}, median={st['median']:.4f}, mad={st['mad']:.4f}, iqr={st['iqr']:.4f}")
        print(f"  Shape stats         : skew={st['skew']:.4f}, kurtosis={st['kurtosis']:.4f}, zero_fraction={st['zero_fraction']:.4f}, dispersion={st['dispersion']:.4f}")

    # Inspect all leaf stats via m.leaf_stats()
    all_stats = clf.leaf_stats()
    print(f"\nm.leaf_stats() returned {len(all_stats)} leaf summary dictionaries.")

    print("\n" + "=" * 70)
    print("3. Flexible Prediction Kinds (0D Formulations)")
    print("=" * 70)

    test_samples = [[5.0, 0.0], [5.0, 1.0], [15.0, 0.0]]
    preds_prob = clf.predict(test_samples, kind="prob")
    preds_odds = clf.predict(test_samples, kind="odds")
    preds_lo = clf.predict(test_samples, kind="log_odds")
    preds_med = clf.predict(test_samples, kind="median")
    preds_stats = clf.predict(test_samples, kind="stats")

    for idx, sample in enumerate(test_samples):
        print(f"Sample {sample}:")
        print(f"  -> prob={preds_prob[idx]:.4f}, odds={preds_odds[idx]:.4e}, log_odds={preds_lo[idx]:.4f}, median={preds_med[idx]:.4f}")
        print(f"  -> row stats dict keys: {sorted(preds_stats[idx].keys())[:6]}...")

    print("\n" + "=" * 70)
    print("4. Interaction Constraints with Column Names")
    print("=" * 70)

    # 3 features: age, income, city_tier
    X_domain = [[float(i), float(i * 10), float(i % 3)] for i in range(100)]
    y_domain = [float(i % 3) for i in range(100)]

    # Allow age and income to interact, but isolate city_tier
    tree_interact = ModelTree(
        max_depth=3,
        min_rows=5,
        interaction_groups=[["age", "income"], ["city_tier"]],
    )
    tree_interact.fit(X_domain, y_domain, col_names=["age", "income", "city_tier"])

    paths = tree_interact.split_paths()
    print("Interaction groups successfully mapped from column names:")
    print("Resolved index groups:", tree_interact.get_params()["interaction_groups"])
    print("Split paths across tree:")
    for path in paths:
        print("  Root ->", " -> ".join(path) if path else "(stump)")

    print("\n" + "=" * 70)
    print("5. Row Weights & Column Split Priorities")
    print("=" * 70)

    # Priority weighting: column 0 has 3x priority over column 1
    col_weights = [3.0, 1.0, 0.5]
    tree_weighted = ModelTree(max_depth=2, col_weights=col_weights)
    tree_weighted.fit(
        X_domain,
        y_domain,
        sample_weight=[1.0 if i < 50 else 2.0 for i in range(100)],
        col_names=["age", "income", "city_tier"],
    )
    print("Tree fitted with row weights and column weights.")
    print("Reported params col_weights:", tree_weighted.get_params()["col_weights"])
    print("Finished showcase successfully!")


if __name__ == "__main__":
    main()
