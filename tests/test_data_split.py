"""Tests for mltreelib.data.split: deterministic index splitting."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from collections import Counter

import pytest
from mltreelib.data.split import (
    kfold,
    repeated_kfold,
    take,
    train_test_split,
    train_val_test_split,
)

NAN = float("nan")


def _same(a, b) -> bool:
    """Deep equality that treats NaN as equal to NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    return a == b


def _assert_partition(train, test, n):
    assert sorted(train + test) == list(range(n))
    assert set(train).isdisjoint(test)
    assert train == sorted(train) and test == sorted(test)


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------


def test_train_test_split_plain_pinned_seed0():
    # Pinned from the first run. Test count = round(10 * 0.2) = 2.
    assert train_test_split(10, test_size=0.2, seed=0) == (
        [0, 1, 2, 3, 5, 6, 7, 8],
        [4, 9],
    )


def test_train_test_split_size_rules():
    # Float: round half up. 0.25 * 10 = 2.5 -> 3. Int: absolute count.
    assert len(train_test_split(10, 0.25)[1]) == 3
    assert len(train_test_split(10, 4)[1]) == 4
    # Float clamped to [1, n-1]: 0.2 * 2 = 0.4 -> 0 -> clamped to 1.
    assert train_test_split(2, 0.2) == ([1], [0])
    with pytest.raises(ValueError):
        train_test_split(10, 0.0)
    with pytest.raises(ValueError):
        train_test_split(10, 10)  # int must leave one train row
    with pytest.raises(ValueError):
        train_test_split(1, 0.5)  # need at least 2 rows
    with pytest.raises(TypeError):
        train_test_split(10, 0.2, seed=1.5)


def test_train_test_split_partition_and_plain_empty_edge():
    for n in (2, 3, 17, 100):
        for size in (0.2, 0.5, 0.9, 1):
            train, test = train_test_split(n, size, seed=3)
            _assert_partition(train, test, n)
            assert len(test) >= 1 and len(train) >= 1
    # n=0 and n=1 cannot be split: a clear error, not a crash.
    with pytest.raises(ValueError):
        train_test_split(0)


def test_train_test_split_stratify_proportions_hand_checked():
    # counts a=7, b=5, c=3, n=15, test=4. Exact shares 1.87, 1.33, 0.80.
    # Largest remainder gives a=2, b=1, c=1.
    labels = ["a"] * 7 + ["b"] * 5 + ["c"] * 3
    _, test = train_test_split(15, 4, seed=0, stratify=labels)
    assert Counter(labels[i] for i in test) == {"a": 2, "b": 1, "c": 1}
    _assert_partition(*train_test_split(15, 4, seed=0, stratify=labels), 15)


def test_train_test_split_stratify_within_one_row():
    labels = ["p"] * 61 + ["q"] * 23 + ["r"] * 9 + ["s"] * 7
    n = len(labels)
    for test_count in (5, 17, 40):
        _, test = train_test_split(n, test_count, seed=7, stratify=labels)
        got = Counter(labels[i] for i in test)
        for cls, size in Counter(labels).items():
            exact = size * test_count / n
            assert math.floor(exact) <= got[cls] <= math.ceil(exact)


def test_train_test_split_singleton_class_goes_to_train():
    labels = ["x", "y", "y", "y", "y", "y"]
    train, test = train_test_split(6, 0.5, seed=0, stratify=labels)
    assert 0 in train
    assert 0 not in test


def test_train_test_split_groups_no_straddle_and_target():
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g4", "g5"]
    for seed in range(5):
        train, test = train_test_split(10, 3, seed=seed, groups=groups)
        _assert_partition(train, test, 10)
        test_groups = {groups[i] for i in test}
        train_groups = {groups[i] for i in train}
        assert test_groups.isdisjoint(train_groups)
        assert len(test) >= 3


def test_train_test_split_groups_with_stratify_is_group_integral():
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5", "g6", "g6"]
    labels = ["a", "a", "a", "a", "b", "b", "b", "b", "a", "a", "b", "b"]
    train, test = train_test_split(12, 4, seed=1, groups=groups, stratify=labels)
    _assert_partition(train, test, 12)
    assert {groups[i] for i in test}.isdisjoint({groups[i] for i in train})


def test_train_test_split_time_order_last_rows_no_shuffle():
    # Keys sort to 0(idx4) 1(idx1) 2(idx2) 3(idx0) 5(idx5) then missing(idx3).
    # Last 2 by (key, index): idx5 and idx3 (missing sorts last).
    keys = [3, 1, 2, None, 0, 5]
    train, test = train_test_split(6, 2, time_order=keys)
    assert test == [3, 5]
    assert train == [0, 1, 2, 4]
    # Same answer for any seed: no shuffling in time mode.
    assert train_test_split(6, 2, seed=99, time_order=keys) == (train, test)


def test_train_test_split_time_order_ties_by_index():
    keys = [1, 1, 1, 1]
    assert train_test_split(4, 2, time_order=keys) == ([0, 1], [2, 3])


def test_train_test_split_mode_validation():
    with pytest.raises(ValueError):
        train_test_split(6, 2, stratify=list("aabbcc"), time_order=list(range(6)))
    with pytest.raises(ValueError):
        train_test_split(6, 2, groups=list("aabbcc"), time_order=list(range(6)))
    with pytest.raises(ValueError):
        train_test_split(6, 2, stratify=list("aabbc"))  # length mismatch


def test_train_test_split_one_group_holding_all_rows_raises():
    with pytest.raises(ValueError):
        train_test_split(4, 2, groups=["only"] * 4)


def test_train_test_split_none_and_nan_labels_are_own_level():
    stratify = ["a", None, "a", NAN, "b", None, "b", "a", NAN, "b"]
    snapshot = copy.deepcopy(stratify)
    train, test = train_test_split(10, 0.5, seed=0, stratify=stratify)
    _assert_partition(train, test, 10)
    assert _same(stratify, snapshot)
    # None and NaN form one level of size 4: 2 of them go to test (0.5 of 4).
    missing_positions = {1, 3, 5, 8}
    assert len(missing_positions & set(test)) == 2

    groups = [None, NAN, None, "g", "g", NAN, "h", "h"]
    gsnap = copy.deepcopy(groups)
    train, test = train_test_split(8, 0.5, seed=0, groups=groups)
    _assert_partition(train, test, 8)
    assert _same(groups, gsnap)
    # Missing is one group: rows 0,1,2,5 must sit on the same side.
    sides = {i in test for i in (0, 1, 2, 5)}
    assert len(sides) == 1


def test_train_test_split_time_order_missing_keys_kept():
    keys = [NAN, 2, 1, None]
    snapshot = copy.deepcopy(keys)
    train, test = train_test_split(4, 1, time_order=keys)
    _assert_partition(train, test, 4)
    # Missing keys (idx 0 and 3) sort after 1 and 2; the last one by index is idx 3.
    assert test == [3]
    assert _same(keys, snapshot)


def test_train_test_split_empty_and_all_missing_stratify_is_sensible():
    # All-missing labels: one level, so it behaves like a plain split.
    train, test = train_test_split(10, 0.2, seed=0, stratify=[None] * 10)
    assert len(test) == 2
    _assert_partition(train, test, 10)


def test_train_test_split_determinism_same_seed_same_output():
    a = train_test_split(50, 0.3, seed=11, stratify=["u"] * 25 + ["v"] * 25)
    b = train_test_split(50, 0.3, seed=11, stratify=["u"] * 25 + ["v"] * 25)
    assert a == b
    assert train_test_split(50, 0.3, seed=12) != train_test_split(50, 0.3, seed=11)


# ---------------------------------------------------------------------------
# train_val_test_split
# ---------------------------------------------------------------------------


def test_train_val_test_split_partition_and_val_fraction_of_original():
    train, val, test = train_val_test_split(100, val_size=0.2, test_size=0.2, seed=0)
    assert len(test) == 20
    assert len(val) == 20  # fraction of the original n, not of the remainder
    assert len(train) == 60
    assert sorted(train + val + test) == list(range(100))
    assert set(train).isdisjoint(val) and set(val).isdisjoint(test)


def test_train_val_test_split_stratified_and_groups_and_errors():
    labels = ["a"] * 12 + ["b"] * 8
    train, val, test = train_val_test_split(20, 0.25, 0.25, seed=2, stratify=labels)
    assert sorted(train + val + test) == list(range(20))
    assert Counter(labels[i] for i in test) == {"a": 3, "b": 2}
    assert Counter(labels[i] for i in val) == {"a": 3, "b": 2}

    groups = [f"g{i // 2}" for i in range(10)]
    train, val, test = train_val_test_split(10, 0.2, 0.2, seed=0, groups=groups)
    for part in (train, val, test):
        assert len({groups[i] for i in part}) >= 1
    assert {groups[i] for i in val}.isdisjoint({groups[i] for i in train})
    assert {groups[i] for i in test}.isdisjoint({groups[i] for i in train + val})

    with pytest.raises(ValueError):
        train_val_test_split(4, 3, 1, seed=0)  # no training rows left


def test_train_val_test_split_time_order_is_chronological():
    keys = list(range(10))
    train, val, test = train_val_test_split(10, 2, 2, seed=0, time_order=keys)
    assert test == [8, 9]
    assert val == [6, 7]
    assert train == list(range(6))


def test_train_val_test_split_none_keys_and_determinism():
    keys = [None, 4, 3, NAN, 1, 0]
    snapshot = copy.deepcopy(keys)
    a = train_val_test_split(6, 1, 1, seed=0, time_order=keys)
    assert a == train_val_test_split(6, 1, 1, seed=0, time_order=keys)
    assert _same(keys, snapshot)
    # Test = last by sorted (key, index): missing keys idx 0 and 3 sort last, so test is [3].
    assert a == ([1, 2, 4, 5], [0], [3])


# ---------------------------------------------------------------------------
# kfold
# ---------------------------------------------------------------------------


def test_kfold_plain_pinned_and_hand_checked_sizes():
    folds = kfold(10, k=3, seed=0)
    assert folds == [
        ([1, 2, 3, 6, 7, 8], [0, 4, 5, 9]),
        ([0, 3, 4, 5, 6, 8, 9], [1, 2, 7]),
        ([0, 1, 2, 4, 5, 7, 9], [3, 6, 8]),
    ]
    sizes = [len(test) for _, test in folds]
    assert sorted(sizes) == [3, 3, 4]  # 10 = 4 + 3 + 3


def test_kfold_disjoint_and_covering():
    for n, k in ((10, 3), (17, 5), (5, 5), (30, 2)):
        folds = kfold(n, k, seed=4)
        assert len(folds) == k
        tests = [i for _, test in folds for i in test]
        assert sorted(tests) == list(range(n))  # each row in exactly one test fold
        for train, test in folds:
            _assert_partition(train, test, n)
        sizes = [len(test) for _, test in folds]
        assert max(sizes) - min(sizes) <= 1


def test_kfold_stratified_round_robin_per_class():
    labels = ["a"] * 5 + ["b"] * 4 + ["c"] * 2
    folds = kfold(11, k=3, seed=0, stratify=labels)
    tests = [i for _, test in folds for i in test]
    assert sorted(tests) == list(range(11))
    for cls, size in Counter(labels).items():
        per_fold = [sum(labels[i] == cls for i in test) for _, test in folds]
        assert max(per_fold) - min(per_fold) <= 1
        assert sum(per_fold) == size
    sizes = [len(test) for _, test in folds]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_stratified_small_class_does_not_raise():
    labels = ["rare"] + ["common"] * 9
    folds = kfold(10, k=4, seed=0, stratify=labels)
    assert sorted(i for _, t in folds for i in t) == list(range(10))


def test_kfold_groups_integrity_and_smallest_fold():
    groups = ["g1"] * 4 + ["g2"] * 1 + ["g3"] * 2 + ["g4"] * 3
    folds = kfold(10, k=3, seed=0, groups=groups)
    tests = [i for _, t in folds for i in t]
    assert sorted(tests) == list(range(10))
    for train, test in folds:
        _assert_partition(train, test, 10)
        assert {groups[i] for i in test}.isdisjoint({groups[i] for i in train})
    # Each group sits in exactly one test fold.
    for g in set(groups):
        homes = [j for j, (_, t) in enumerate(folds) if any(groups[i] == g for i in t)]
        assert len(homes) == 1


def test_kfold_groups_hand_checked():
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g5", "g5", "g5"]
    folds = kfold(10, k=3, seed=0, groups=groups)
    assert [test for _, test in folds] == [[4, 5, 6], [0, 1, 7, 8, 9], [2, 3]]


def test_kfold_groups_with_none_is_own_group():
    groups = [None, None, "a", "a", NAN, "b"]
    snapshot = copy.deepcopy(groups)
    folds = kfold(6, k=3, seed=0, groups=groups)
    assert _same(groups, snapshot)
    tests = [i for _, t in folds for i in t]
    assert sorted(tests) == list(range(6))
    # None and NaN rows form one group, so they share a single test fold.
    homes = {j for j, (_, t) in enumerate(folds) for i in t if i in (0, 1, 4)}
    assert len(homes) == 1


def test_kfold_time_order_expanding_window_hand_checked():
    # 6 rows, k=3 -> 4 blocks of sizes 2,2,1,1: [0,1] [2,3] [4] [5].
    folds = kfold(6, k=3, time_order=[0, 1, 2, 3, 4, 5])
    assert folds == [
        ([0, 1], [2, 3]),
        ([0, 1, 2, 3], [4]),
        ([0, 1, 2, 3, 4], [5]),
    ]


def test_kfold_time_order_sorts_and_missing_keys_last():
    keys = [5, NAN, 0, 4, 1, 3]
    snapshot = copy.deepcopy(keys)
    folds = kfold(6, k=2, time_order=keys)
    assert _same(keys, snapshot)
    # Sorted (key, index): 0(idx2) 1(idx4) 3(idx5) 4(idx3) 5(idx0) NaN(idx1).
    # k=2 gives 3 blocks of size 2: [2,4] [3,5] [0,1]. The NaN-key row (idx1) lands in the last block.
    assert folds == [([2, 4], [3, 5]), ([2, 3, 4, 5], [0, 1])]


def test_kfold_validation_errors():
    with pytest.raises(ValueError, match="k must be at least 2"):
        kfold(10, k=1)
    with pytest.raises(ValueError, match="exceeds n"):
        kfold(4, k=5)
    with pytest.raises(ValueError, match="at least k"):
        kfold(10, k=4, groups=["a"] * 9 + ["b"])  # 2 groups for 4 folds
    with pytest.raises(ValueError):
        kfold(4, k=4, time_order=[0, 1, 2, 3])  # time mode needs n >= k+1
    with pytest.raises(ValueError):
        kfold(6, k=2, groups=["a"] * 6, stratify=list("ababab"))


def test_kfold_empty_and_all_missing_edges():
    # n=k=2 plain: each fold tests one row (pinned).
    assert kfold(2, k=2, seed=0) == [([1], [0]), ([0], [1])]
    # All-missing stratify is one level; still a full partition.
    folds = kfold(4, k=2, stratify=[None, None, None, None], seed=0)
    assert folds == [([0, 3], [1, 2]), ([1, 2], [0, 3])]


def test_kfold_determinism():
    labels = list("abcabcabc") + ["d"]
    assert kfold(10, 3, seed=5, stratify=labels) == kfold(
        10, 3, seed=5, stratify=labels
    )
    assert kfold(10, 3, seed=5) != kfold(10, 3, seed=6)


# ---------------------------------------------------------------------------
# repeated_kfold
# ---------------------------------------------------------------------------


def test_repeated_kfold_uses_seed_plus_r():
    runs = repeated_kfold(12, k=3, repeats=3, seed=10)
    assert len(runs) == 3
    for r, run in enumerate(runs):
        assert run == kfold(12, 3, seed=10 + r)
    assert runs[0] != runs[1]


def test_repeated_kfold_stratify_and_errors():
    labels = ["a"] * 6 + ["b"] * 6
    runs = repeated_kfold(12, k=3, repeats=2, seed=0, stratify=labels)
    assert len(runs) == 2 and all(len(run) == 3 for run in runs)
    with pytest.raises(ValueError):
        repeated_kfold(12, k=3, repeats=0)


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------


def test_take_dict_order_and_missing_kept():
    columns = {"a": [1, None, 3, NAN], "b": ["x", "y", "z", "w"]}
    snapshot = copy.deepcopy(columns)
    out = take(columns, [3, 1, 0])
    assert list(out) == ["a", "b"]
    assert out["b"] == ["w", "y", "x"]
    assert out["a"][1] is None and math.isnan(out["a"][0])
    assert _same(columns, snapshot)


def test_take_list_returns_list_and_tuple_returns_tuple():
    assert take([10, 20, 30], [2, 0]) == [30, 10]
    assert take((10, 20, 30), [1]) == (20,)


def test_take_repeats_and_empty():
    assert take({"a": [5, 6]}, [1, 1, 0]) == {"a": [6, 6, 5]}
    assert take({"a": [5, 6], "b": [None, None]}, []) == {"a": [], "b": []}
    assert take([], []) == []


def test_take_out_of_range_raises_with_index_in_message():
    with pytest.raises(IndexError, match="index 7"):
        take({"a": [1, 2, 3]}, [0, 7])
    with pytest.raises(IndexError, match="index -1"):
        take([1, 2], [-1])
    with pytest.raises(TypeError):
        take([1, 2], [1.0])
    with pytest.raises(TypeError):
        take("abc", [0])


def test_take_does_not_modify_input_indices():
    indices = [2, 0]
    take({"a": [1, 2, 3]}, indices)
    assert indices == [2, 0]


# ---------------------------------------------------------------------------
# Determinism across processes (hash seed independence)
# ---------------------------------------------------------------------------

_PROBE = r"""
import json
from mltreelib.data.split import kfold, train_test_split, train_val_test_split
labels = ["alpha", "beta", "gamma", "alpha", "beta", "delta", None, "gamma", "alpha", "beta"]
groups = ["g-x", "g-y", "g-x", "g-z", "g-y", "g-w", None, "g-z", "g-w", "g-x"]
out = {
    "strat": train_test_split(10, 0.4, seed=3, stratify=labels),
    "groups": train_test_split(10, 0.4, seed=3, groups=groups),
    "sgroups": train_test_split(10, 0.4, seed=3, groups=groups, stratify=labels),
    "kf_strat": kfold(10, 3, seed=8, stratify=labels),
    "kf_groups": kfold(10, 3, seed=8, groups=groups),
    "tvt": train_val_test_split(10, 0.2, 0.2, seed=4, stratify=labels),
}
print(json.dumps(out))
"""


def _probe(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return proc.stdout


def test_cross_process_determinism_with_different_hash_seeds():
    out1 = _probe("1")
    out2 = _probe("2")
    assert out1 == out2
    # Compare against the in-process result too.
    labels = [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "delta",
        None,
        "gamma",
        "alpha",
        "beta",
    ]
    groups = ["g-x", "g-y", "g-x", "g-z", "g-y", "g-w", None, "g-z", "g-w", "g-x"]
    expected = json.loads(out1)
    assert (
        json.loads(json.dumps(train_test_split(10, 0.4, seed=3, stratify=labels)))
        == expected["strat"]
    )
    assert (
        json.loads(json.dumps(kfold(10, 3, seed=8, groups=groups)))
        == expected["kf_groups"]
    )
