"""Tests for mltreelib.data.binning (hand-checkable values, stdlib only)."""

from __future__ import annotations

import copy
import itertools
import math
import random
import time
from bisect import bisect_left
from fractions import Fraction

import pytest
from mltreelib.data._common import is_missing
from mltreelib.data.binning import (
    _dp_prebin_edges,
    bin_edges,
    binner,
    information_value_table,
    iv_bins,
    iv_levels,
)

NAN = float("nan")


def _same(a, b) -> bool:
    """NaN-safe deep equality."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------------------
# bin_edges
# ---------------------------------------------------------------------------


def test_bin_edges_quantile_hand_values():
    # Type-7 quantiles of 1..10 at 0.2, 0.4, 0.6, 0.8:
    # pos = 9q -> 2.8, 4.6, 6.4, 8.2 (interpolated between neighbours).
    values = list(range(1, 11))
    edges = bin_edges(values, n_bins=5)
    assert len(edges) == 4
    for got, want in zip(edges, [2.8, 4.6, 6.4, 8.2]):
        assert math.isclose(got, want, rel_tol=1e-12)


def test_bin_edges_width_hand_values():
    # [0, 10] split into 4 equal widths of 2.5.
    edges = bin_edges(list(range(11)), n_bins=4, method="width")
    assert edges == [2.5, 5.0, 7.5]


def test_bin_edges_ties_give_fewer_bins_and_drop_top_edge():
    # Type-7 quantiles at 0.25, 0.5, 0.75 are 1, 1, 5. The duplicate 1 is
    # removed; the cut at the observed max 5 is dropped (it would leave an
    # empty top bin). The cut at the minimum 1 is kept: (-inf, 1] holds the 1s.
    values = [1, 1, 1, 1, 1, 1, 5, 5, 5, 5]
    assert bin_edges(values, n_bins=4) == [1.0]


def test_bin_edges_keeps_min_cut_for_rare_flag():
    # 95% zeros: every decile is 0. Without the minimum cut the rare flag is
    # one bin and IV is 0. The cut at 0 gives (-inf, 0] and (0, inf].
    y = [0] * 95 + [1] * 5
    assert bin_edges(y, n_bins=10) == [0.0]
    r = iv_bins(y, y, min_bin_frac=0.05)
    assert [b["n"] for b in r["bins"]] == [95, 5]
    assert r["iv"] > 0.5
    rows = information_value_table({"flag": y}, y)
    assert rows[0]["strength"] == "suspicious"


def test_bin_edges_min_obs_count_merges_small_bins():
    # Quantile cuts for 1..10 at n_bins=5 give 5 bins of 2 values each.
    # Requiring 3 observed values per bin merges them until each bin has >= 3.
    values = list(range(1, 11))
    edges = bin_edges(values, n_bins=5, min_obs=3)
    counts = binner(values, edges=edges)["counts"]
    assert all(c >= 3 for c in counts)
    assert len(edges) < 4


def test_bin_edges_min_obs_fraction_matches_count():
    values = list(range(1, 11))
    assert bin_edges(values, n_bins=5, min_obs=0.3) == bin_edges(
        values, n_bins=5, min_obs=3
    )


def test_bin_edges_missing_counted_not_dropped_and_input_unchanged():
    values = [1.0, None, 2.0, NAN, 3.0, 4.0, 5.0, 6.0]
    original = copy.deepcopy(values)
    edges = bin_edges(values, n_bins=2)
    assert values == original or _same(values, original)
    # Observed finite values 1..6 (missing ignored): median 3.5.
    assert edges == [3.5]


def test_bin_edges_empty_and_all_missing_return_no_edges():
    assert bin_edges([]) == []
    assert bin_edges([None, NAN]) == []
    assert bin_edges([7.0, 7.0, 7.0]) == []


def test_bin_edges_invalid_arguments_raise():
    with pytest.raises(ValueError):
        bin_edges([1, 2], n_bins=0)
    with pytest.raises(ValueError):
        bin_edges([1, 2], method="kmeans")


def test_bin_edges_deterministic():
    values = [3.3, 1.1, None, 2.2, 9.9, 4.4, 4.4, 0.5]
    assert bin_edges(values, n_bins=4) == bin_edges(values, n_bins=4)


# ---------------------------------------------------------------------------
# binner
# ---------------------------------------------------------------------------


def test_binner_exact_labels_counts_and_edge_convention():
    values = list(range(1, 11))
    out = binner(values, n_bins=5)
    assert out["labels"] == [
        "(-inf, 2.8]",
        "(2.8, 4.6]",
        "(4.6, 6.4]",
        "(6.4, 8.2]",
        "(8.2, inf]",
    ]
    assert out["counts"] == [2, 2, 2, 2, 2]
    # Value 2 is below the edge 2.8, so left bin; a value exactly on an edge goes left.
    assert out["index"][1] == 0
    e = binner([1, 2.5, 3], edges=[2.5])
    assert e["index"] == [0, 0, 1]


def test_binner_right_false_and_labels():
    out = binner([1, 2, 3], edges=[2.0], right=False)
    # [2, 3) is the upper bin: value 2 is on the edge and goes right.
    assert out["index"] == [0, 1, 1]
    assert out["labels"] == ["[-inf, 2)", "[2, inf)"]


def test_binner_missing_kept_and_infinities_counted():
    values = [1.0, float("inf"), float("-inf"), 2.0, None, NAN]
    original = copy.deepcopy(values)
    out = binner(values, edges=[1.5])
    assert out["index"] == [0, 1, 0, 1, None, None]
    assert out["n_missing"] == 2
    assert out["n_inf"] == 2
    assert sum(out["counts"]) + out["n_missing"] == len(values)
    assert _same(values, original)


def test_binner_labels_digits_and_finite_edge_format():
    out = binner([0.0, 1.0], edges=[0.123456, 7.0], digits=2)
    assert out["labels"] == ["(-inf, 0.12]", "(0.12, 7]", "(7, inf]"]


def test_binner_empty_and_all_missing():
    empty = binner([])
    assert empty["labels"] == ["(-inf, inf]"]
    assert empty["counts"] == [0]
    allmiss = binner([None, NAN])
    assert allmiss["index"] == [None, None]
    assert allmiss["n_missing"] == 2


def test_binner_rejects_non_numeric_and_unsorted_edges():
    with pytest.raises(ValueError):
        binner([1, "x"], edges=[1.5])
    with pytest.raises(ValueError):
        binner([1, 2], edges=[2.0, 1.0])


def test_binner_deterministic():
    values = [5, 1, None, 3, 3.0, 9]
    assert binner(values, n_bins=3) == binner(values, n_bins=3)


# ---------------------------------------------------------------------------
# iv_bins
# ---------------------------------------------------------------------------


def test_iv_bins_hand_computed_two_bins():
    # x = [1,2,3,4], y = [0,0,1,1], edges [2.5] -> bin A: n=2, events 0;
    # bin B: n=2, events 2. E = N = 2, k = 2, s = 0.5.
    # pe_A = 0.5/3 = 1/6, pn_A = 2.5/3 = 5/6, WOE_A = ln(0.2),
    # IV_A = (1/6 - 5/6) * ln(0.2) = (2/3) * ln 5. Bin B is symmetric.
    # Total IV = (4/3) * ln 5 = 2.1459172...
    out = iv_bins([1, 2, 3, 4], [0, 0, 1, 1], edges=[2.5], min_bin_frac=0)
    assert math.isclose(out["iv"], (4 / 3) * math.log(5), rel_tol=1e-12)
    a, b = out["bins"]
    assert (a["lo"], a["hi"], a["n"], a["events"]) == (-math.inf, 2.5, 2, 0)
    assert (b["lo"], b["hi"], b["n"], b["events"]) == (2.5, math.inf, 2, 2)
    assert math.isclose(a["woe"], math.log(0.2), rel_tol=1e-12)
    assert math.isclose(b["woe"], math.log(5), rel_tol=1e-12)
    assert a["event_rate"] == 0.0 and b["event_rate"] == 1.0
    assert out["trend"] == "increasing"
    assert out["missing_bin"]["n"] == 0 and out["missing_bin"]["iv"] == 0.0


def test_iv_bins_missing_bin_never_dropped():
    values = [1, 2, 3, 4, None, NAN]
    y = [0, 0, 1, 1, 1, 0]
    original_v, original_y = copy.deepcopy(values), copy.deepcopy(y)
    out = iv_bins(values, y, edges=[2.5], min_bin_frac=0)
    assert out["n_rows"] == 6
    assert out["n_used"] == 6
    assert out["missing_bin"]["n"] == 2
    assert out["missing_bin"]["events"] == 1
    assert out["missing_bin"]["non_events"] == 1
    assert math.isfinite(out["missing_bin"]["woe"])
    total = sum(b["n"] for b in out["bins"]) + out["missing_bin"]["n"]
    assert total == out["n_used"] == len(values)
    assert _same(values, original_v) and _same(y, original_y)


def test_iv_bins_y_missing_excluded_and_counted():
    values = [1, 2, 3, 4, 5]
    y = [0, None, 1, NAN, 1]
    out = iv_bins(values, y, edges=[2.5], min_bin_frac=0)
    assert out["n_rows"] == 5
    assert out["n_y_missing"] == 2
    assert out["n_used"] == 3
    assert sum(b["n"] for b in out["bins"]) + out["missing_bin"]["n"] == 3


def test_iv_bins_zero_cells_are_finite_with_smoothing():
    # Bin (-inf,1.5] has 0 events and bin (1.5,2.5] has 0 events: both zero cells.
    values = [1, 2, 3, 4]
    y = [0, 0, 1, 1]
    out = iv_bins(values, y, edges=[1.5, 2.5], min_bin_frac=0)
    assert len(out["bins"]) == 3
    first = out["bins"][0]
    # E = 2, N = 2, k = 3: pe = 0.5/3.5, pn = 1.5/3.5, WOE = ln(1/3).
    assert first["events"] == 0
    assert math.isclose(first["woe"], math.log(1 / 3), rel_tol=1e-12)
    assert all(math.isfinite(b["woe"]) and math.isfinite(b["iv"]) for b in out["bins"])
    assert out["iv"] >= 0


def test_iv_bins_pav_makes_non_monotone_input_increasing():
    # Event rates by bin: 1, 0, 1, 0, 1, 1 (one row per bin).
    values = [1, 2, 3, 4, 5, 6]
    y = [1, 0, 1, 0, 1, 1]
    edges = [1.5, 2.5, 3.5, 4.5, 5.5]
    free = iv_bins(values, y, edges=edges, min_bin_frac=0)
    assert free["trend"] == "none"
    assert len(free["bins"]) == 6

    out = iv_bins(values, y, edges=edges, monotone="increasing", min_bin_frac=0)
    rates = [b["event_rate"] for b in out["bins"]]
    assert rates == sorted(rates)
    # PAV pools (1,0) -> 0.5 and (1,0) -> 0.5, then 1 and 1: bins merge.
    assert len(out["bins"]) == 4
    assert [(b["n"], b["events"]) for b in out["bins"]] == [
        (2, 1),
        (2, 1),
        (1, 1),
        (1, 1),
    ]
    assert out["trend"] == "increasing"


def test_iv_bins_decreasing_and_auto_pick_monotone_direction():
    values = [1, 2, 3, 4, 5, 6]
    y = [0, 1, 0, 1, 0, 0]  # rates 0,1,0,1,0,0
    dec = iv_bins(
        values,
        y,
        edges=[1.5, 2.5, 3.5, 4.5, 5.5],
        monotone="decreasing",
        min_bin_frac=0,
    )
    rates = [b["event_rate"] for b in dec["bins"]]
    assert rates == sorted(rates, reverse=True)
    assert dec["trend"] in ("decreasing", "none")
    edges = [1.5, 2.5, 3.5, 4.5, 5.5]
    inc = iv_bins(values, y, edges=edges, monotone="increasing", min_bin_frac=0)
    auto = iv_bins(values, y, edges=edges, monotone="auto", min_bin_frac=0)
    # auto returns the direction with the larger total IV.
    assert auto["iv"] == max(dec["iv"], inc["iv"])
    assert auto["trend"] in ("increasing", "decreasing")


def test_iv_bins_max_bins_greedy_merge_recovers_separation():
    # 4 single-value bins with rates 0,0,1,1. Greedy IV-loss merging ends at
    # the perfect split {1,2} | {3,4}, whose IV is (4/3) ln 5 (see the hand case).
    out = iv_bins(
        [1, 2, 3, 4], [0, 0, 1, 1], edges=[1.5, 2.5, 3.5], max_bins=2, min_bin_frac=0
    )
    assert [(b["n"], b["events"]) for b in out["bins"]] == [(2, 0), (2, 2)]
    assert math.isclose(out["iv"], (4 / 3) * math.log(5), rel_tol=1e-12)


def test_iv_bins_min_bin_frac_merges_small_bins():
    values = list(range(1, 11))
    y = [0] * 5 + [1] * 5
    # Bins: {1} (n=1), {2..9} (n=8), {10} (n=1). 0.1 * 10 = 1: nothing is below 1.
    kept = iv_bins(values, y, edges=[1.5, 9.5], min_bin_frac=0.1)
    assert len(kept["bins"]) == 3
    # 0.2 * 10 = 2: the two single-value end bins merge until all bins have >= 2.
    merged = iv_bins(values, y, edges=[1.5, 9.5], min_bin_frac=0.2)
    assert len(merged["bins"]) == 1
    assert merged["bins"][0]["n"] == 10
    assert merged["trend"] is None


def test_iv_bins_empty_and_all_missing():
    empty = iv_bins([], [])
    assert empty["iv"] == 0.0 and empty["bins"] == [] and empty["trend"] is None
    assert empty["n_rows"] == 0 and empty["n_used"] == 0
    all_missing = iv_bins([None, NAN], [0, 1])
    assert all_missing["bins"] == []
    assert all_missing["missing_bin"]["n"] == 2
    assert all_missing["missing_bin"]["events"] == 1
    assert all_missing["iv"] == 0.0


def test_iv_bins_invalid_arguments_raise():
    with pytest.raises(ValueError):
        iv_bins([1, 2, 3], [0, 1, 2])  # three target values
    with pytest.raises(ValueError):
        iv_bins([1, 2], [0, 1], smoothing=0)
    with pytest.raises(ValueError):
        iv_bins([1, 2], [0, 1], monotone="peak")
    with pytest.raises(ValueError):
        iv_bins([1, 2], [0])


def test_iv_bins_deterministic():
    values = [0.3, 2.2, None, 5.1, 4.4, 1.0, 9.0, 3.3]
    y = [0, 1, 1, 0, 1, 0, 1, 0]
    a = iv_bins(values, y, n_bins=3, monotone="auto", max_bins=2)
    b = iv_bins(values, y, n_bins=3, monotone="auto", max_bins=2)
    assert _same(a, b)


# ---------------------------------------------------------------------------
# iv_levels
# ---------------------------------------------------------------------------


def test_iv_levels_hand_values_and_sorting():
    # a: n=2, events 1; b: n=2, events 0. Sorted by event rate ascending: b, a.
    # Totals E = 1, N = 3, k = 2 levels (no missing), s = 0.5:
    # b: pe = (0 + 0.5)/(1 + 1) = 0.25, pn = (2 + 0.5)/(3 + 1) = 0.625.
    out = iv_levels(["a", "b", "a", "b"], [1, 0, 0, 0])
    assert [b["level"] for b in out["bins"]] == ["b", "a"]
    b, a = out["bins"]
    assert (b["n"], b["events"], a["n"], a["events"]) == (2, 0, 2, 1)
    pe, pn = 0.25, 0.625
    assert math.isclose(b["woe"], math.log(pe / pn), rel_tol=1e-12)
    assert math.isclose(b["iv"], (pe - pn) * math.log(pe / pn), rel_tol=1e-12)
    assert out["trend"] is None


def test_iv_levels_pools_rare_levels_and_missing_kept():
    values = ["a", "b", "a", "c", "a", "b", None]
    y = [1, 0, 0, 1, 0, 0, 1]
    original = copy.deepcopy(values)
    out = iv_levels(values, y, min_level_frac=0.2)
    levels = [b["level"] for b in out["bins"]]
    # n_used = 7, threshold 1.4: c (n=1) pools into __other__ (n=1, events=1).
    assert "__other__" in levels and "c" not in levels
    assert out["missing_bin"]["n"] == 1 and out["missing_bin"]["events"] == 1
    total = sum(b["n"] for b in out["bins"]) + out["missing_bin"]["n"]
    assert total == out["n_used"] == 7
    assert _same(values, original)


def test_iv_levels_nan_level_goes_to_missing_bin():
    values = ["a", NAN, None, "a", NAN]
    y = [1, 1, 0, 0, 1]
    original = copy.deepcopy(values)
    out = iv_levels(values, y)
    assert [b["level"] for b in out["bins"]] == ["a"]
    assert out["missing_bin"]["n"] == 3
    assert out["missing_bin"]["events"] == 2
    assert not any(isinstance(b["level"], float) for b in out["bins"])
    assert _same(values, original)


def test_iv_levels_empty_and_all_missing_and_y_missing():
    empty = iv_levels([], [])
    assert empty["bins"] == [] and empty["iv"] == 0.0
    out = iv_levels(["a", None], [None, 1])
    assert out["n_y_missing"] == 1
    assert out["n_used"] == 1
    # Row 0 has a missing target and is excluded; row 1 (missing level, y=1) is kept.
    assert out["missing_bin"]["n"] == 1
    assert out["missing_bin"]["events"] == 1
    assert out["bins"] == []


def test_iv_levels_deterministic():
    values = ["x", "y", "z", "x", None, "y"]
    y = [0, 1, 1, 0, 1, 0]
    assert _same(
        iv_levels(values, y, min_level_frac=0.3),
        iv_levels(values, y, min_level_frac=0.3),
    )


# ---------------------------------------------------------------------------
# information_value_table
# ---------------------------------------------------------------------------


def test_information_value_table_sorted_and_labelled():
    cols = {
        "x": [1, 2, 3, 4],
        "c": ["a", "b", "a", "b"],
        "empty": [None, None, None, None],
        "partial": [NAN, 1, None, 2],
    }
    y = [0, 0, 1, 1]
    original = copy.deepcopy(cols), copy.deepcopy(y)
    rows = information_value_table(cols, y, n_bins=2)
    assert _same((cols, y), original)
    # x: edges [2.5] give (2.5 / 4.5 hand case): IV = (4/3) ln 5.
    assert math.isclose(
        {r["name"]: r for r in rows}["x"]["iv"], (4 / 3) * math.log(5), rel_tol=1e-12
    )
    names = [r["name"] for r in rows]
    assert names[0] == "x"
    ivs = [r["iv"] for r in rows]
    assert ivs == sorted(ivs, reverse=True)
    by_name = {r["name"]: r for r in rows}
    assert by_name["x"]["kind"] == "numeric"
    assert by_name["x"]["n_bins"] == 2
    assert by_name["x"]["strength"] == "suspicious"  # IV = 2.146 > 0.5
    assert by_name["c"]["kind"] == "categorical"
    assert by_name["c"]["iv"] == 0.0  # both levels have the same event rate
    assert by_name["c"]["strength"] == "useless"


def test_information_value_table_strength_thresholds():
    # Strength labels follow the rule of thumb. IV is always >= 0 here.
    from mltreelib.data import binning as mod

    assert mod._strength(0.01) == "useless"
    assert mod._strength(0.05) == "weak"
    assert mod._strength(0.2) == "medium"
    assert mod._strength(0.4) == "strong"
    assert mod._strength(0.6) == "suspicious"


def test_information_value_table_deterministic_and_empty():
    cols = {"x": [0.5, None, 3.0, 2.0], "c": ["p", "q", None, "p"]}
    y = [0, 1, 1, 0]
    assert information_value_table(cols, y) == information_value_table(cols, y)
    assert information_value_table({}, []) == []


# ---------------------------------------------------------------------------
# iv_bins(optimize="dp"): exact optimal binning
# ---------------------------------------------------------------------------


def _seeded_data(seed: int, n: int) -> tuple[list, list]:
    """Feature with a logistic signal, some missing features and some missing targets."""
    rng = random.Random(seed)
    x: list = []
    y: list = []
    for _ in range(n):
        v = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-1.2 * v))
        x.append(round(v, 3))
        y.append(1 if rng.random() < p else 0)
    for i in range(n):
        r = rng.random()
        if r < 0.05:
            x[i] = None
        elif r < 0.07:
            x[i] = NAN
        if rng.random() < 0.03:
            y[i] = None
    return x, y


def _shape_ok(rates: list[Fraction], kind: str | None) -> bool:
    """Independent check of a trend constraint on exact event rates."""
    if kind is None:
        return True
    pairs = list(itertools.pairwise(rates))
    if kind == "increasing":
        return all(a <= b for a, b in pairs)
    if kind == "decreasing":
        return all(a >= b for a, b in pairs)
    if kind == "auto":
        return _shape_ok(rates, "increasing") or _shape_ok(rates, "decreasing")
    for p in range(len(rates)):
        if (
            kind == "peak"
            and all(a <= b for a, b in pairs[:p])
            and all(a >= b for a, b in pairs[p:])
        ):
            return True
        if (
            kind == "valley"
            and all(a >= b for a, b in pairs[:p])
            and all(a <= b for a, b in pairs[p:])
        ):
            return True
    return False


def _exhaustive(values, y, *, n_prebins, max_bins, min_bin_frac, trend, s=0.5):
    """Best (objective, bin boundaries) over every contiguous partition of the pre-bins.

    Written from the definition in the docstring; it shares only the pre-bin
    edges with the implementation. Returns None if no partition is feasible.
    """
    if trend == "auto":
        cands = [
            _exhaustive(
                values,
                y,
                n_prebins=n_prebins,
                max_bins=max_bins,
                min_bin_frac=min_bin_frac,
                trend=t,
                s=s,
            )
            for t in ("increasing", "decreasing")
        ]
        cands = [c for c in cands if c is not None]
        if not cands:
            return None
        return max(cands, key=lambda c: c[0])  # ties: increasing first
    rows = [
        (float(v), int(t))
        for v, t in zip(values, y)
        if not is_missing(v) and not is_missing(t)
    ]
    edges = _dp_prebin_edges([v for v, _ in rows], None, n_prebins, "quantile")
    P = len(edges) + 1
    pn = [0] * P
    pe = [0] * P
    for v, t in rows:
        b = bisect_left(edges, v)  # right-closed bins: edge e belongs to the left bin
        pn[b] += 1
        pe[b] += t
    E = sum(pe)
    N = sum(pn) - E
    min_n = max(math.ceil(round(min_bin_frac * len(rows), 9)), 1)
    bounds = [-math.inf, *edges, math.inf]
    best = None
    for mask in range(2 ** (P - 1)):
        starts = [0] + [c + 1 for c in range(P - 1) if mask >> c & 1]
        ends = starts[1:] + [P]
        k = len(starts)
        if not (min(2, P) <= k <= min(max_bins, P)):
            continue
        ns = [sum(pn[a:b]) for a, b in zip(starts, ends)]
        es = [sum(pe[a:b]) for a, b in zip(starts, ends)]
        if any(n < min_n or e == 0 or e == n for n, e in zip(ns, es)):
            continue
        if not _shape_ok([Fraction(e, n) for n, e in zip(ns, es)], trend):
            continue
        obj = 0.0
        for n, e in zip(ns, es):
            ne = n - e
            obj += (e / E - ne / N) * math.log(
                ((e + s) / (E + s)) / ((ne + s) / (N + s))
            )
        cand = (obj, starts, ends)
        if (
            best is None
            or obj > best[0] + 1e-12
            or (abs(obj - best[0]) <= 1e-12 and (k, starts) < (len(best[1]), best[1]))
        ):
            best = cand
    if best is None:
        return None
    obj, starts, ends = best
    return obj, [(bounds[a], bounds[b]) for a, b in zip(starts, ends)]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "trend", [None, "increasing", "decreasing", "peak", "valley", "auto"]
)
def test_dp_matches_exhaustive_enumeration(seed, trend):
    rng = random.Random(100 + seed)
    n = rng.randint(60, 200)
    n_prebins = rng.randint(4, 10)
    max_bins = rng.randint(2, 4)
    frac = rng.choice([0.0, 0.05, 0.1])
    x, y = _seeded_data(seed, n)
    if trend == "decreasing":
        # The signal rises with x; flip it so a decreasing partition can exist.
        x = [v if is_missing(v) else -v for v in x]
    dp = iv_bins(
        x,
        y,
        optimize="dp",
        n_prebins=n_prebins,
        max_bins=max_bins,
        min_bin_frac=frac,
        trend=trend,
    )
    want = _exhaustive(
        x, y, n_prebins=n_prebins, max_bins=max_bins, min_bin_frac=frac, trend=trend
    )
    ordered = [b for b in dp["bins"] if not b["special"]]
    if want is None:
        # No partition meets the level-0 rules: the DP must say so and fall back.
        assert dp["constraint_relaxed"] is True
        assert len(ordered) == 1
        return
    assert dp["constraint_relaxed"] is False
    assert math.isclose(dp["objective"], want[0], rel_tol=0, abs_tol=1e-12)
    assert [(b["lo"], b["hi"]) for b in ordered] == want[1]


@pytest.mark.parametrize("seed", range(5))
def test_greedy_never_beats_dp_on_the_same_prebins(seed):
    x, y = _seeded_data(10 + seed, 150)
    feat = [v for v, t in zip(x, y) if not is_missing(v) and not is_missing(t)]
    edges = _dp_prebin_edges(feat, None, 8, "quantile")
    n_feat = len(feat)
    mc = math.ceil(round(0.05 * n_feat, 9))
    compared = 0
    for trend in (None, "increasing", "decreasing", "auto"):
        dp = iv_bins(
            x, y, optimize="dp", n_prebins=8, max_bins=3, min_bin_frac=0.05, trend=trend
        )
        greedy = iv_bins(
            x,
            y,
            edges=edges,
            optimize="greedy",
            max_bins=3,
            min_bin_frac=0.05,
            trend=trend,
        )
        g_bins = [b for b in greedy["bins"] if not b["special"]]
        feasible = (
            2 <= len(g_bins) <= 3
            and all(b["n"] >= mc and 0 < b["events"] < b["n"] for b in g_bins)
            and _shape_ok([Fraction(b["events"], b["n"]) for b in g_bins], trend)
        )
        if feasible:
            compared += 1
            assert greedy["objective"] <= dp["objective"] + 1e-12
    assert compared >= 3


@pytest.mark.parametrize(
    "trend", ["increasing", "decreasing", "peak", "valley", "auto"]
)
def test_dp_event_rate_sequence_has_requested_shape(trend):
    x, y = _seeded_data(42, 300)
    if trend == "decreasing":
        x = [v if is_missing(v) else -v for v in x]
    r = iv_bins(
        x, y, optimize="dp", n_prebins=12, max_bins=5, min_bin_frac=0.02, trend=trend
    )
    rates = [Fraction(b["events"], b["n"]) for b in r["bins"] if not b["special"]]
    assert len(rates) >= 2
    assert _shape_ok(rates, trend)
    assert r["trend"] in ("increasing", "decreasing", "peak", "valley", "none")


def test_dp_peak_and_valley_are_found_when_they_are_the_best_shape():
    # Three blocks of 10 rows (x = 1..30). With n_prebins=3 the quantile
    # pre-bins are exactly the blocks. Event rates: low, high, low (peak)
    # and high, low, high (valley).
    x = list(range(1, 31))
    low = [1, 1] + [0] * 8
    high = [1] * 8 + [0, 0]
    peak = iv_bins(
        x,
        low + high + low,
        optimize="dp",
        n_prebins=3,
        max_bins=3,
        min_bin_frac=0,
        trend="peak",
    )
    assert peak["constraint_relaxed"] is False
    assert [b["events"] for b in peak["bins"]] == [2, 8, 2]
    assert peak["trend"] == "peak"
    valley = iv_bins(
        x,
        high + low + high,
        optimize="dp",
        n_prebins=3,
        max_bins=3,
        min_bin_frac=0,
        trend="valley",
    )
    assert valley["constraint_relaxed"] is False
    assert [b["events"] for b in valley["bins"]] == [8, 2, 8]
    assert valley["trend"] == "valley"
    # Increasing forbids the peak: the only feasible 2+ bin split is low | high+low.
    inc = iv_bins(
        x,
        low + high + low,
        optimize="dp",
        n_prebins=3,
        max_bins=3,
        min_bin_frac=0,
        trend="increasing",
    )
    assert inc["constraint_relaxed"] is False
    assert [b["events"] for b in inc["bins"]] == [2, 10]
    assert inc["trend"] == "increasing"


def test_dp_respects_min_bin_frac_and_max_bins():
    x, y = _seeded_data(7, 180)
    r = iv_bins(
        x, y, optimize="dp", n_prebins=15, max_bins=4, min_bin_frac=0.1, trend="auto"
    )
    n_feat = sum(1 for v, t in zip(x, y) if not is_missing(v) and not is_missing(t))
    ordered = [b for b in r["bins"] if not b["special"]]
    assert r["constraint_relaxed"] is False
    assert 2 <= len(ordered) <= 4
    assert all(b["n"] >= math.ceil(round(0.1 * n_feat, 9)) for b in ordered)
    assert all(b["events"] >= 1 and b["non_events"] >= 1 for b in ordered)


def test_dp_missing_feature_is_its_own_bin_and_y_missing_is_excluded():
    values = [1.0, None, 2.0, NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [0, 1, None, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    original_v, original_y = copy.deepcopy(values), copy.deepcopy(y)
    r = iv_bins(
        values, y, optimize="dp", n_prebins=4, max_bins=3, min_bin_frac=0.1, trend=None
    )
    assert r["n_rows"] == 12
    assert r["n_y_missing"] == 1
    assert r["n_used"] == 11
    assert r["missing_bin"]["n"] == 2
    assert r["missing_bin"]["events"] == 2
    ordered = sum(b["n"] for b in r["bins"] if not b["special"])
    assert ordered == 9
    assert ordered + r["missing_bin"]["n"] == r["n_used"]
    assert math.isfinite(r["missing_bin"]["woe"])
    assert _same(values, original_v) and _same(y, original_y)


def test_dp_special_values_get_own_bins_outside_the_ordered_bins():
    x = [-999, -999, -999, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, 999]
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    original = copy.deepcopy(x), copy.deepcopy(y)
    r = iv_bins(
        x,
        y,
        optimize="dp",
        n_prebins=5,
        max_bins=3,
        min_bin_frac=0.05,
        trend="increasing",
        special_values=[-999, 0, 12345],
    )
    specials = [b for b in r["bins"] if b["special"]]
    ordered = [b for b in r["bins"] if not b["special"]]
    # 12345 never occurs, so it gets no bin.
    assert [(b["lo"], b["hi"], b["n"]) for b in specials] == [
        (-999.0, -999.0, 3),
        (0.0, 0.0, 2),
    ]
    # Ordered values: 1..9 and 999 (10 rows). Specials: 3 and 2. Missing: 1.
    assert sum(b["n"] for b in ordered) == 10
    assert sum(b["n"] for b in r["bins"]) + r["missing_bin"]["n"] == r["n_used"] == 16
    rates = [Fraction(b["events"], b["n"]) for b in ordered]
    assert _shape_ok(rates, "increasing")
    assert _same((x, y), original)
    greedy = iv_bins(x, y, n_bins=3, special_values=[-999, 0])
    assert [b["special"] for b in greedy["bins"]].count(True) == 2
    assert sum(b["n"] for b in greedy["bins"]) + greedy["missing_bin"]["n"] == 16


def test_dp_relaxation_is_reported_when_no_partition_is_feasible():
    # Events below the median, non-events above: a 'increasing' trend cannot hold
    # for any split into two or more bins, so the bin count is relaxed to one.
    x = list(range(1, 13))
    y = [1] * 6 + [0] * 6
    inc = iv_bins(
        x,
        y,
        optimize="dp",
        n_prebins=12,
        max_bins=4,
        min_bin_frac=0,
        trend="increasing",
    )
    assert inc["constraint_relaxed"] is True
    assert len(inc["bins"]) == 1
    # Without a trend, every split of the step data breaks the presence rule
    # (only one bin can hold both classes), so the DP relaxes presence and keeps
    # a split into two or more bins.
    free = iv_bins(
        x, y, optimize="dp", n_prebins=12, max_bins=4, min_bin_frac=0, trend=None
    )
    assert free["constraint_relaxed"] is True
    assert len(free["bins"]) >= 2


def test_dp_single_class_and_empty_inputs_do_not_raise():
    single = iv_bins(
        list(range(10)), [0] * 10, optimize="dp", n_prebins=4, max_bins=3, trend="auto"
    )
    assert single["objective"] == 0.0 and math.isfinite(single["iv"])
    assert single["constraint_relaxed"] is True
    empty = iv_bins([], [], optimize="dp")
    assert empty["bins"] == [] and empty["objective"] == 0.0
    assert empty["constraint_relaxed"] is True
    all_missing = iv_bins([None, NAN], [0, 1], optimize="dp")
    assert all_missing["bins"] == [] and all_missing["missing_bin"]["n"] == 2


def test_dp_is_deterministic():
    x, y = _seeded_data(3, 400)
    a = iv_bins(x, y, optimize="dp", n_prebins=20, max_bins=5, trend="auto")
    b = iv_bins(x, y, optimize="dp", n_prebins=20, max_bins=5, trend="auto")
    assert _same(a, b)


def test_dp_runtime_sanity_for_20000_rows():
    x, y = _seeded_data(99, 20000)
    original = copy.deepcopy(x), copy.deepcopy(y)
    start = time.perf_counter()
    r = iv_bins(x, y, optimize="dp", n_prebins=50, max_bins=8, trend="auto")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert len([b for b in r["bins"] if not b["special"]]) <= 8
    assert math.isfinite(r["iv"]) and math.isfinite(r["objective"])
    assert _same((x, y), original)


def test_trend_and_monotone_are_aliases_and_conflicts_raise():
    x, y = _seeded_data(5, 80)
    a = iv_bins(x, y, optimize="dp", max_bins=3, monotone="increasing")
    b = iv_bins(x, y, optimize="dp", max_bins=3, trend="increasing")
    c = iv_bins(
        x, y, optimize="dp", max_bins=3, monotone="increasing", trend="increasing"
    )
    assert _same(a, b) and _same(b, c)
    with pytest.raises(ValueError):
        iv_bins(x, y, optimize="dp", monotone="increasing", trend="decreasing")
    with pytest.raises(ValueError):
        iv_bins(x, y, trend="peak")  # greedy has no peak constraint
    with pytest.raises(ValueError):
        iv_bins(x, y, optimize="dp", trend="sigmoid")
    with pytest.raises(ValueError):
        iv_bins(x, y, optimize="simplex")
    with pytest.raises(ValueError):
        iv_bins(x, y, optimize="dp", max_bins=1)
    with pytest.raises(ValueError):
        iv_bins(x, y, special_values=["a"])


def test_greedy_default_output_reports_optimize_and_objective():
    r = iv_bins([1, 2, 3, 4], [0, 0, 1, 1], edges=[2.5], min_bin_frac=0)
    assert r["optimize"] == "greedy"
    assert r["constraint_relaxed"] is False
    assert isinstance(r["objective"], float) and math.isfinite(r["objective"])
    assert all(b["special"] is False for b in r["bins"])
