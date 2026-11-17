from mltreelib import ModelTree


def test_causal_tree_tuple_targets_and_uplift_prediction():
    # 200 samples with treatment effect heterogeneity:
    # x < 0.0 -> tau = 10.0
    # x >= 0.0 -> tau = -5.0
    X = [[-1.0 + i * (2.0 / 199.0)] for i in range(200)]
    treat = [1.0 if i % 2 == 0 else 0.0 for i in range(200)]
    outcomes = [2.0 + (10.0 if X[i][0] < 0.0 else -5.0) * treat[i] for i in range(200)]

    tree = ModelTree(gain="causal", max_depth=2, min_rows=10, leaf="mean")
    tree.fit(X, (outcomes, treat))

    test_X = [[-0.5], [0.5]]
    uplifts = tree.predict(test_X, kind="uplift")
    assert len(uplifts) == 2
    assert abs(uplifts[0] - 10.0) < 0.01
    assert abs(uplifts[1] - (-5.0)) < 0.01


def test_train_groups_and_train_ids_filtering():
    X = [[i, 100.0, -1.0 + i * (2.0 / 199.0)] for i in range(200)]
    treat = [1.0 if i % 2 == 0 else 0.0 for i in range(200)]
    outcomes = [2.0 + (10.0 if X[i][2] < 0.0 else -5.0) * treat[i] for i in range(200)]

    # By default, train_groups=True and train_ids=False:
    # id_col is dropped from feature matrix, while group_col remains
    tree = ModelTree(
        gain="causal",
        max_depth=2,
        min_rows=10,
        leaf="mean",
        train_groups=True,
        train_ids=False,
    )
    tree.fit(
        X, (outcomes, treat), id_col=0, group_col=1, col_names=["id", "grp", "feature"]
    )

    # Predict with only the remaining features [grp, feature]
    test_X = [[100.0, -0.5], [100.0, 0.5]]
    uplifts = tree.predict(test_X, kind="uplift")
    assert abs(uplifts[0] - 10.0) < 0.01
    assert abs(uplifts[1] - (-5.0)) < 0.01


def test_train_groups_false_drops_group_column():
    X = [[i, 100.0, -1.0 + i * (2.0 / 199.0)] for i in range(200)]
    treat = [1.0 if i % 2 == 0 else 0.0 for i in range(200)]
    outcomes = [2.0 + (10.0 if X[i][2] < 0.0 else -5.0) * treat[i] for i in range(200)]

    # If train_groups=False and train_ids=False: both are dropped
    tree = ModelTree(
        gain="causal",
        max_depth=2,
        min_rows=10,
        leaf="mean",
        train_groups=False,
        train_ids=False,
    )
    tree.fit(
        X, (outcomes, treat), id_col=0, group_col=1, col_names=["id", "grp", "feature"]
    )

    # Predict with only [feature]
    test_X = [[-0.5], [0.5]]
    uplifts = tree.predict(test_X, kind="uplift")
    assert abs(uplifts[0] - 10.0) < 0.01
    assert abs(uplifts[1] - (-5.0)) < 0.01
