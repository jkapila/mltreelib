from mltreelib import ModelTree


def test_unseen_categorical_level_maps_to_missing_without_explosion():
    # Train on categories 'A' and 'B'
    xtr = [
        ['A', 1.0],
        ['A', 1.2],
        ['B', 2.0],
        ['B', 2.2],
    ] * 20
    ytr = [10.0, 11.0, 20.0, 21.0] * 20
    dtypes = ['categorical', 'numeric']

    m = ModelTree(
        leaf="linear",
        leaf_cols="all",
        max_depth=4,
        min_rows=5,
        prune="none",
        leaf_na_policy="drop_fallback",
        seed=42,
    )
    m.fit(xtr, ytr, dtypes=dtypes)

    # Test on unseen category 'C'
    xte = [
        ['A', 1.0],
        ['B', 2.0],
        ['C', 1.5], # Unseen level 'C'
    ]
    preds = m.predict(xte)
    assert len(preds) == 3
    # Prediction on unseen level must be finite and within realistic range, not exploding to 1e300 or inf
    for p in preds:
        assert isinstance(p, float)
        assert 0.0 < p < 100.0
