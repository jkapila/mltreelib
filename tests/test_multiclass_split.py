import pytest
from mltreelib import ModelTree


@pytest.mark.parametrize("split_eval", ["fast", "histogram"])
def test_gini_split_with_class_observed_only_when_feature_is_missing(split_eval):
    x = [[0.0], [1.0], [2.0], [None]]
    tree = ModelTree(
        max_depth=1,
        min_rows=1,
        gain="gini",
        leaf="mean",
        prune="none",
        split_precision=None,
        split_eval=split_eval,
    )
    tree.fit(x, [0.0, 1.0, 0.0, 2.0], sample_weight=[1.0, 1.0, 2.0, 4.0])
    assert tree.predict(x) == pytest.approx([8 / 5, 1 / 3, 1 / 3, 8 / 5])
