"""Binning and information value (IV) for binary targets (standard library only).

Numeric columns are cut into bins (quantile or equal width). For a binary
target, each bin gets events, non-events, weight of evidence (WOE) and its IV
contribution. Missing feature values form their own bin and are never dropped.
Rows with a missing target are excluded from the analysis and counted.

Smoothing: with prior ``s`` and ``k`` cells (ordered bins, plus the missing bin
when it is non-empty), the smoothed shares are
``pe_i = (events_i + s) / (E + k*s)`` and ``pn_i = (non_events_i + s) / (N + k*s)``.
They sum to one over cells, so zero cells stay finite. WOE is
``ln(pe_i / pn_i)`` (event share over non-event share, the sign convention of
this package) and IV is ``sum((pe_i - pn_i) * WOE_i)``, which is never negative.

Two optimisers share the same output. ``optimize="greedy"`` (default) runs PAV
on event rates, then size merges, then greedy IV-loss merges. ``optimize="dp"``
is an exact dynamic program over fine pre-bins that maximises a separable
objective under the same kind of constraints (trend, bin count, bin size,
event and non-event presence). See ``iv_bins`` for the objective.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Sequence
from typing import Any

from mltreelib.data._common import (
    as_columns,
    is_infinite,
    is_missing,
    is_number,
    numeric_observed,
    quantile,
)

__all__ = ["bin_edges", "binner", "information_value_table", "iv_bins", "iv_levels"]

_METHODS = ("quantile", "width")
_MONOTONE = (None, "increasing", "decreasing", "auto")
_TRENDS = (None, "increasing", "decreasing", "peak", "valley", "auto")
_OPTIMIZE = ("greedy", "dp")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# validation and small helpers
# ---------------------------------------------------------------------------


def _check_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_fraction(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number in [0, 1]")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _check_smoothing(smoothing: Any) -> None:
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError("smoothing must be a positive number")
    if not smoothing > 0 or not math.isfinite(smoothing):
        raise ValueError("smoothing must be a positive finite number")


def _checked_edges(edges: Sequence[Any]) -> list[float]:
    cuts = [float(e) for e in edges]
    if any(not math.isfinite(e) for e in cuts):
        raise ValueError("edges must be finite")
    if any(b <= a for a, b in itertools.pairwise(cuts)):
        raise ValueError("edges must be strictly increasing")
    return cuts


def _segment_counts(sorted_obs: list[float], edges: list[float]) -> list[int]:
    """Counts per bin (right-closed bins, as in ``binner`` with right=True)."""
    cuts = [bisect.bisect_right(sorted_obs, e) for e in edges]
    bounds = [0, *cuts, len(sorted_obs)]
    return [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]


def _bin_of(value: float, edges: list[float], right: bool) -> int:
    if right:
        return bisect.bisect_left(edges, value)
    return bisect.bisect_right(edges, value)


def _fmt(x: float, digits: int) -> str:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = repr(round(x, digits))
    return text.removesuffix(".0")


def _labels(edges: list[float], right: bool, digits: int) -> list[str]:
    bounds = [-math.inf, *edges, math.inf]
    labels = []
    for lo, hi in itertools.pairwise(bounds):
        lo_txt, hi_txt = _fmt(lo, digits), _fmt(hi, digits)
        labels.append(f"({lo_txt}, {hi_txt}]" if right else f"[{lo_txt}, {hi_txt})")
    return labels


def _min_count(min_obs: Any, n_obs: int) -> int:
    if isinstance(min_obs, bool):
        raise TypeError("min_obs must be an int count or a float fraction")
    if isinstance(min_obs, float):
        _check_fraction(min_obs, "min_obs")
        return math.ceil(round(min_obs * n_obs, 9))
    if isinstance(min_obs, int):
        if min_obs < 0:
            raise ValueError("min_obs must be non-negative")
        return min_obs
    raise TypeError("min_obs must be an int count or a float fraction")


def _merge_small_edges(
    obs: list[float], edges: list[float], threshold: int
) -> list[float]:
    """Remove edges until every bin has at least ``threshold`` observed values.

    The first short bin is merged with its smaller neighbour (ties go left).
    """
    edges = list(edges)
    while edges:
        counts = _segment_counts(obs, edges)
        j = next((i for i, c in enumerate(counts) if c < threshold), None)
        if j is None:
            break
        if j == 0:
            drop = 0
        elif j == len(counts) - 1:
            drop = j - 1
        else:
            drop = j - 1 if counts[j - 1] <= counts[j + 1] else j
        del edges[drop]
    return edges


def _is_numeric_column(values: Sequence[Any]) -> bool:
    return all(
        is_missing(v) or (isinstance(v, (int, float)) and not isinstance(v, bool))
        for v in values
    )


def _event_flags(ys: Sequence[Any]) -> list[int]:
    """1 for events, 0 for non-events. The larger of two distinct values is the event."""
    distinct = list(dict.fromkeys(ys))
    if len(distinct) > 2:
        raise ValueError("y must be binary: found more than two distinct values")
    if len(distinct) == 2:
        event = max(distinct)
        return [1 if v == event else 0 for v in ys]
    if len(distinct) == 1:
        return [1 if distinct[0] == 1 else 0] * len(ys)
    return []


def _usable(
    values: Sequence[Any], y: Sequence[Any]
) -> tuple[list[Any], list[int], int]:
    """Rows with a present target, their flags, and the count of rows dropped."""
    keep = [i for i in range(len(y)) if not is_missing(y[i])]
    flags = _event_flags([y[i] for i in keep])
    return [values[i] for i in keep], flags, len(y) - len(keep)


# ---------------------------------------------------------------------------
# scoring and bin surgery
# ---------------------------------------------------------------------------


def _score(
    groups: list[tuple[int, int]], miss_n: int, miss_ev: int, smoothing: float
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Stats for ordered ``(n, events)`` groups and the missing cell.

    Returns (per-group stats, missing stats, total IV).
    """
    total = sum(n for n, _ in groups) + miss_n
    ev_total = sum(e for _, e in groups) + miss_ev
    ne_total = total - ev_total
    k = len(groups) + (1 if miss_n > 0 else 0)
    s = smoothing

    def cell(n: int, e: int) -> dict[str, Any]:
        pe = (e + s) / (ev_total + k * s)
        pn = (n - e + s) / (ne_total + k * s)
        woe = math.log(pe / pn)
        return {
            "n": n,
            "events": e,
            "non_events": n - e,
            "event_rate": e / n,
            "woe": woe,
            "iv": (pe - pn) * woe,
        }

    stats = [cell(n, e) for n, e in groups]
    if miss_n > 0:
        miss = cell(miss_n, miss_ev)
    else:
        miss = {
            "n": 0,
            "events": 0,
            "non_events": 0,
            "event_rate": None,
            "woe": None,
            "iv": 0.0,
        }
    total_iv = sum(c["iv"] for c in stats) + miss["iv"]
    return stats, miss, total_iv


def _pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "lo": a["lo"],
        "hi": b["hi"],
        "n": a["n"] + b["n"],
        "events": a["events"] + b["events"],
    }


def _merge_at(bins: list[dict[str, Any]], d: int) -> list[dict[str, Any]]:
    """Merge bins ``d`` and ``d + 1``."""
    return bins[:d] + [_pair(bins[d], bins[d + 1])] + bins[d + 2 :]


def _enforce_min(bins: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Merge the smallest bin below ``threshold`` into its smaller neighbour, repeatedly."""
    bins = list(bins)
    while len(bins) > 1:
        short = [i for i, b in enumerate(bins) if b["n"] < threshold]
        if not short:
            break
        j = min(short, key=lambda i: (bins[i]["n"], i))
        if j == 0:
            d = 0
        elif j == len(bins) - 1:
            d = j - 1
        else:
            d = j - 1 if bins[j - 1]["n"] <= bins[j + 1]["n"] else j
        bins = _merge_at(bins, d)
    return bins


def _pav(bins: list[dict[str, Any]], decreasing: bool) -> list[dict[str, Any]]:
    """Pool adjacent violators on event rates, weighted by counts (exact cross-multiplication)."""
    stack: list[dict[str, Any]] = []
    for b in bins:
        stack.append(b)
        while len(stack) >= 2:
            a, c = stack[-2], stack[-1]
            lhs = a["events"] * c["n"]
            rhs = c["events"] * a["n"]
            violates = lhs < rhs if decreasing else lhs > rhs
            if not violates:
                break
            stack[-2:] = [_pair(a, c)]
    return stack


def _cap_bins(
    bins: list[dict[str, Any]],
    max_bins: int,
    miss_n: int,
    miss_ev: int,
    smoothing: float,
    extra: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Greedy: merge the adjacent pair whose merge keeps the most total IV.

    ``extra`` holds fixed (n, events) cells, such as special-value bins, that
    take part in the IV totals but are never merged.
    """
    while len(bins) > max_bins and len(bins) > 1:
        best_d, best_iv = 0, None
        for d in range(len(bins) - 1):
            cand = _merge_at(bins, d)
            iv = _score(
                [(b["n"], b["events"]) for b in cand] + list(extra),
                miss_n,
                miss_ev,
                smoothing,
            )[2]
            if best_iv is None or iv > best_iv:
                best_d, best_iv = d, iv
        bins = _merge_at(bins, best_d)
    return bins


def _trend(rates: list[float]) -> str | None:
    if len(rates) < 2:
        return None
    if all(a == rates[0] for a in rates):
        return "none"
    if all(a <= b for a, b in itertools.pairwise(rates)):
        return "increasing"
    if all(a >= b for a, b in itertools.pairwise(rates)):
        return "decreasing"
    return "none"


# ---------------------------------------------------------------------------
# exact optimal binning (dynamic programming)
# ---------------------------------------------------------------------------


def _contribution(n: int, e: int, big_e: int, big_n: int, s: float) -> float:
    """DP objective term for one ordered bin (``n`` rows, ``e`` events).

    ``big_e`` and ``big_n`` are the event and non-event totals over the
    usable rows, and ``s`` is the prior. With fixed totals the term is additive
    over bins: ``(e/E - ne/N) * ln(((e+s)/(E+s)) / ((ne+s)/(N+s)))``.
    """
    if big_e == 0 or big_n == 0:
        return 0.0
    ne = n - e
    p = (e + s) / (big_e + s)
    q = (ne + s) / (big_n + s)
    return (e / big_e - ne / big_n) * math.log(p / q)


def _rate_sign(n1: int, e1: int, n2: int, e2: int) -> int:
    """Sign of (e2/n2 - e1/n1) with exact integer arithmetic (n1, n2 > 0)."""
    lhs, rhs = e2 * n1, e1 * n2
    return (lhs > rhs) - (lhs < rhs)


def _shape(groups: Sequence[tuple[int, int]]) -> str | None:
    """Trend of ordered (n, events) bins: increasing, decreasing, peak, valley, none."""
    if len(groups) < 2:
        return None
    cmps = [
        _rate_sign(n1, e1, n2, e2) for (n1, e1), (n2, e2) in itertools.pairwise(groups)
    ]
    if all(c == 0 for c in cmps):
        return "none"
    if all(c >= 0 for c in cmps):
        return "increasing"
    if all(c <= 0 for c in cmps):
        return "decreasing"
    first_fall = next(i for i, c in enumerate(cmps) if c < 0)
    if all(c >= 0 for c in cmps[:first_fall]) and all(
        c <= 0 for c in cmps[first_fall:]
    ):
        return "peak"
    first_rise = next(i for i, c in enumerate(cmps) if c > 0)
    if all(c <= 0 for c in cmps[:first_rise]) and all(
        c >= 0 for c in cmps[first_rise:]
    ):
        return "valley"
    return "none"


def _phase_moves(trend: str, c: int, ph: int) -> tuple[int, ...]:
    """Phases reachable after a rate step of sign ``c`` from phase ``ph``.

    Phase 0 is the first leg (rising for peak, falling for valley); phase 1 is
    the second leg. Equal rates (c == 0) may start the second leg early.
    """
    if trend == "increasing":
        return (0,) if c >= 0 else ()
    if trend == "decreasing":
        return (0,) if c <= 0 else ()
    if trend == "peak":
        if ph == 0:
            return tuple(p for p, ok in ((0, c >= 0), (1, c <= 0)) if ok)
        return (1,) if c <= 0 else ()
    if ph == 0:
        return tuple(p for p, ok in ((0, c <= 0), (1, c >= 0)) if ok)
    return (1,) if c >= 0 else ()


def _dp_prebin_edges(
    ord_vals: Sequence[Any], edges: Sequence[Any] | None, n_prebins: int, method: str
) -> list[float]:
    """Pre-bin cut points for the DP: cuts from ``bin_edges`` (or ``edges``), with no empty pre-bin."""
    cuts = (
        _checked_edges(edges)
        if edges is not None
        else bin_edges(ord_vals, n_bins=n_prebins, method=method)
    )
    everything = sorted(float(v) for v in ord_vals)  # infinities included
    return _merge_small_edges(everything, cuts, 1)


def _dp_solve(
    cn: list[int],
    ce: list[int],
    big_e: int,
    big_n: int,
    s: float,
    kmin: int,
    kmax: int,
    min_count: int,
    presence: bool,
    trend: str | None,
) -> tuple[float, list[tuple[int, int]]] | None:
    """Best contiguous partition of the pre-bins into ``kmin..kmax`` bins.

    ``cn``/``ce`` are prefix counts (length P+1) of the pre-bins. A bin is
    feasible when it has at least ``min_count`` rows (and at least one) and,
    with ``presence``, at least one event and one non-event. ``trend`` is None
    or one of increasing, decreasing, peak, valley (non-strict on ties).

    Returns ``(objective, ranges)`` with ``ranges`` as ``[start, end)`` pre-bin
    pairs, or None when no partition is feasible. The state for a trend is
    (bins used, last bin start and end, phase); without a trend the last bin
    start is not needed.
    """
    P = len(cn) - 1
    need = max(min_count, 1)
    contrib: list[list[float | None]] = [[None] * (P + 1) for _ in range(P + 1)]
    for i in range(P):
        for j in range(i + 1, P + 1):
            n, e = cn[j] - cn[i], ce[j] - ce[i]
            if n < need or (presence and (e == 0 or e == n)):
                continue
            contrib[i][j] = _contribution(n, e, big_e, big_n, s)

    chosen: tuple[int, float] | None = None  # (bins, objective); fewer bins win ties
    if trend is None:
        best = [[None] * (P + 1) for _ in range(kmax + 1)]
        back = [[0] * (P + 1) for _ in range(kmax + 1)]
        for j in range(1, P + 1):
            best[1][j] = contrib[0][j]
        for k in range(2, kmax + 1):
            for j in range(k, P + 1):
                for i in range(k - 1, j):
                    g, prev = contrib[i][j], best[k - 1][i]
                    if g is None or prev is None:
                        continue
                    if best[k][j] is None or prev + g > best[k][j]:
                        best[k][j] = prev + g
                        back[k][j] = i
        for k in range(kmin, kmax + 1):
            v = best[k][P]
            if v is not None and (chosen is None or v > chosen[1] + _EPS):
                chosen = (k, v)
        if chosen is None:
            return None
        k, value = chosen
        ranges: list[tuple[int, int]] = []
        j = P
        for kk in range(k, 1, -1):
            i = back[kk][j]
            ranges.append((i, j))
            j = i
        ranges.append((0, j))
        ranges.reverse()
        return value, ranges

    moves = {(c, ph): _phase_moves(trend, c, ph) for c in (-1, 0, 1) for ph in (0, 1)}
    # H[k][i][j][ph]: best value with k bins whose last bin is [i, j), in phase ph.
    H = [
        [[[None, None] for _ in range(P + 1)] for _ in range(P + 1)]
        for _ in range(kmax + 1)
    ]
    B: list[list[list[list[tuple[int, int] | None]]]] = [
        [[[None, None] for _ in range(P + 1)] for _ in range(P + 1)]
        for _ in range(kmax + 1)
    ]
    for j in range(1, P + 1):
        if contrib[0][j] is not None:
            H[1][0][j][0] = contrib[0][j]
    for k in range(2, kmax + 1):
        for j in range(1, P):
            for i in range(j):
                n1, e1 = cn[j] - cn[i], ce[j] - ce[i]
                for ph in (0, 1):
                    prev = H[k - 1][i][j][ph]
                    if prev is None:
                        continue
                    for loc in range(j + 1, P + 1):
                        g = contrib[j][loc]
                        if g is None:
                            continue
                        c = _rate_sign(n1, e1, cn[loc] - cn[j], ce[loc] - ce[j])
                        for nph in moves[(c, ph)]:
                            cur = H[k][j][loc][nph]
                            if cur is None or prev + g > cur:
                                H[k][j][loc][nph] = prev + g
                                B[k][j][loc][nph] = (i, ph)
    for k in range(kmin, kmax + 1):
        kbest: tuple[float, int, int] | None = None
        for i in range(P):
            for ph in (0, 1):
                v = H[k][i][P][ph]
                if v is not None and (kbest is None or v > kbest[0]):
                    kbest = (v, i, ph)
        if kbest is not None and (chosen is None or kbest[0] > chosen[1] + _EPS):
            chosen = (k, kbest[0])
            last = (kbest[1], kbest[2])
    if chosen is None:
        return None
    k, value = chosen
    a, p = last
    b = P
    ranges = []
    kk = k
    while True:
        ranges.append((a, b))
        if kk == 1:
            break
        pa, pp = B[kk][a][b][p]  # type: ignore[misc]
        b, a, p = a, pa, pp
        kk -= 1
    ranges.reverse()
    return value, ranges


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def bin_edges(
    values: Sequence[Any],
    n_bins: int = 10,
    method: str = "quantile",
    min_obs: float | None = None,
) -> list[float]:
    """Interior cut points for numeric binning (missing and infinite values ignored).

    Quantile cuts are R type-7 quantiles of the finite observed values. Equal
    width cuts split ``[min, max]``. Cuts at or above the observed maximum
    are dropped, since they would leave an empty top bin with ``right=True``.
    A cut equal to the observed minimum is kept, so a rare flag at the
    minimum still gets its own bin. With ``right=False`` (labels
    ``[a, b)``) that cut leaves an empty bottom bin, which ``binner`` keeps
    as is. Duplicates are removed, so ties give fewer bins.

    Args:
        values: Numeric values; None and NaN are missing and ignored here.
        n_bins: Target number of bins (at most).
        method: ``"quantile"`` or ``"width"``.
        min_obs: If given, adjacent bins are merged until each has at least
            this many observed values. An int is a count; a float is a
            fraction of the observed values (rounded up).

    Returns:
        Strictly increasing cut points. Empty when there is nothing to cut.

    Raises:
        ValueError: For an invalid ``n_bins`` or ``method``.

    Example:
        >>> [round(e, 6) for e in bin_edges([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_bins=5)]
        [2.8, 4.6, 6.4, 8.2]
    """
    _check_positive_int(n_bins, "n_bins")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    obs = sorted(numeric_observed(values))
    if not obs or obs[0] == obs[-1]:
        return []
    lo, hi = obs[0], obs[-1]
    if method == "quantile":
        raw = [quantile(obs, i / n_bins) for i in range(1, n_bins)]
    else:
        step = (hi - lo) / n_bins
        raw = [lo + i * step for i in range(1, n_bins)]
    edges: list[float] = []
    for e in raw:
        if lo <= e < hi and (not edges or e > edges[-1]):
            edges.append(e)
    if min_obs is not None:
        edges = _merge_small_edges(obs, edges, _min_count(min_obs, len(obs)))
    return edges


def binner(
    values: Sequence[Any],
    n_bins: int = 10,
    method: str = "quantile",
    min_obs: float | None = None,
    edges: Sequence[float] | None = None,
    right: bool = True,
    digits: int = 4,
) -> dict[str, Any]:
    """Assign numeric values to bins and count them.

    Infinite values go to the first (-inf) or last (+inf) bin and are counted
    in ``n_inf``. Missing values get ``None`` in ``index`` and are counted in
    ``n_missing``. ``counts`` include the infinite rows, so
    ``sum(counts) + n_missing == len(values)``.

    Args:
        values: Numeric values; missing values are kept, not dropped.
        n_bins: Passed to ``bin_edges`` when ``edges`` is None.
        method: Passed to ``bin_edges`` when ``edges`` is None.
        min_obs: Passed to ``bin_edges`` when ``edges`` is None.
        edges: Strictly increasing cut points, used as given.
        right: If True, bins are ``(a, b]`` and a value equal to an edge goes
            to the left bin; if False, bins are ``[a, b)``.
        digits: Decimal places in labels.

    Returns:
        Dict with keys:
            ``edges`` (list[float]): cut points used.
            ``labels`` (list[str]): e.g. ``"(-inf, 1.5]"``, ``"(7.2, inf]"``;
                infinite ends are written ``inf``/``-inf``.
            ``counts`` (list[int]): rows per bin, including infinite rows.
            ``n_missing`` (int): missing rows.
            ``n_inf`` (int): infinite rows (also in ``counts``).
            ``index`` (list[int | None]): 0-based bin per input row, None if missing.

    Raises:
        ValueError: For non-numeric, non-missing values, or invalid edges.

    Example:
        >>> r = binner([1, 2, 3, 4, None], edges=[2.5])
        >>> r["labels"], r["counts"], r["index"]
        (['(-inf, 2.5]', '(2.5, inf]'], [2, 2], [0, 0, 1, 1, None])
    """
    cuts = (
        _checked_edges(edges)
        if edges is not None
        else bin_edges(values, n_bins=n_bins, method=method, min_obs=min_obs)
    )
    k = len(cuts) + 1
    counts = [0] * k
    index: list[int | None] = []
    n_missing = 0
    n_inf = 0
    for v in values:
        if is_missing(v):
            n_missing += 1
            index.append(None)
            continue
        if is_infinite(v):
            n_inf += 1
            b = 0 if v < 0 else k - 1
        elif is_number(v):
            b = _bin_of(float(v), cuts, right)
        else:
            raise ValueError(f"binner expects numeric values, got {v!r}")
        counts[b] += 1
        index.append(b)
    return {
        "edges": cuts,
        "labels": _labels(cuts, right, digits),
        "counts": counts,
        "n_missing": n_missing,
        "n_inf": n_inf,
        "index": index,
    }


def _resolve_trend(
    monotone: str | None, trend: str | None, optimize: str
) -> str | None:
    """Merge the ``monotone`` alias into ``trend``; reject conflicts and unsupported shapes."""
    if monotone is not None and trend is not None and monotone != trend:
        raise ValueError(
            f"monotone={monotone!r} and trend={trend!r} conflict; "
            "monotone is an alias of trend, pass one of them"
        )
    t = trend if trend is not None else monotone
    if t not in _TRENDS:
        raise ValueError(f"trend must be one of {_TRENDS}")
    if optimize == "greedy" and t in ("peak", "valley"):
        raise ValueError("trend 'peak' and 'valley' need optimize='dp'")
    return t


def _checked_specials(special_values: Any) -> list[float]:
    if special_values is None:
        return []
    out: list[float] = []
    for v in special_values:
        if not is_number(v):
            raise ValueError(f"special_values must be finite numbers, got {v!r}")
        out.append(float(v))
    return list(dict.fromkeys(out))


def iv_bins(
    values: Sequence[Any],
    y: Sequence[Any],
    n_bins: int = 10,
    method: str = "quantile",
    monotone: str | None = None,
    max_bins: int | None = None,
    min_bin_frac: float = 0.05,
    smoothing: float = 0.5,
    edges: Sequence[float] | None = None,
    optimize: str = "greedy",
    n_prebins: int = 50,
    trend: str | None = None,
    special_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Information value of a numeric feature against a binary target.

    Usable rows are those with a present target. Rows with a missing target
    are excluded and counted in ``n_y_missing``. Missing feature values form
    ``missing_bin``, which is never merged or dropped. Infinite feature values
    fall into the end bins. The events are the larger of the two target values
    (``True`` or ``1`` for a 0/1 target).

    **optimize="greedy"** (default, unchanged): bin (``edges``, or ``bin_edges``
    on the usable rows); merge empty bins; enforce a monotone event rate by PAV
    if requested; merge bins smaller than ``min_bin_frac`` of the usable rows,
    smallest first; then merge the adjacent pair losing the least IV until at
    most ``max_bins`` ordered bins remain.

    **optimize="dp"**: exact optimal binning. Fine pre-bins are built first
    (``n_prebins`` quantile, or width, cuts on the observed values, or the given
    ``edges``; empty pre-bins are merged away, so ties may give fewer). Then the
    partition of the ordered pre-bins into contiguous bins is chosen to maximise
    ``objective`` subject to:

    - between 2 and ``max_bins`` ordered bins (``max_bins`` defaults to ``n_bins``);
    - each ordered bin holds at least ``min_bin_frac`` of the feature-present
      usable rows, and at least one event and one non-event;
    - the event rate across ordered bins follows ``trend``: None (free),
      ``"increasing"``, ``"decreasing"``, ``"peak"`` (non-decreasing, then
      non-increasing), ``"valley"`` (non-increasing, then non-decreasing), or
      ``"auto"`` (the better of increasing and decreasing). Equal rates are
      allowed everywhere.

    If no partition meets the presence and size rules, they are relaxed in that
    order, and finally the bin count may drop to one (always feasible).
    ``constraint_relaxed`` is True whenever that happens, or when fewer than two
    pre-bins exist. The trend itself is relaxed only by that last step. Ties in
    ``objective`` (within 1e-12) prefer fewer bins, then the earliest cuts.

    **DP objective** (additive over ordered bins, with the fixed prior ``s`` and
    the totals ``E``, ``N`` of events and non-events over the feature-present
    usable rows, special-value rows included, missing feature values excluded)::

        contribution(bin) = (e/E - ne/N) * ln(((e+s)/(E+s)) / ((ne+s)/(N+s)))

    ``objective`` is the sum over ordered bins of this term, for both optimisers,
    so the two can be compared directly. ``iv`` is the standard IV with the
    per-cell smoothing of this module (missing and special cells included).

    **Complexity** (dp): contributions for all pre-bin ranges come from prefix
    sums in O(P^2). The DP has state (bins used, last bin start, last bin end,
    trend phase) and runs in O(max_bins * P^3) time for P = ``n_prebins``, with
    the phase factor 2 for peak and valley. A 20 000-row feature with P = 50,
    max_bins = 8 and trend="auto" takes well under a second.

    **Special values.** Each value in ``special_values`` that occurs in the
    feature forms its own bin, reported with ``"special": True`` and
    ``lo == hi == value``. Special rows are counted, never dropped, and sit
    outside the ordered bins: they do not take part in the trend, the PAV or the
    greedy merges, and they are not subject to ``min_bin_frac``.

    **Relation to optbinning.** This solves the same problem class as the
    optbinning package (ordered numeric bins for a binary target, with a trend
    constraint, bounds on bin count and bin size, and special codes kept in their
    own bins). Here the optimum is exact for the chosen pre-bins by dynamic
    programming, with no solver library. Not supported: continuous or
    multiclass targets, categorical optimal grouping, and concave or convex
    trends (only the four shapes above).

    Args:
        values: Numeric feature values.
        y: Binary target; exactly two distinct values, or one (no contrast).
        n_bins: Target number of quantile or width bins (greedy); also the
            default ``max_bins`` for dp.
        method: ``"quantile"`` or ``"width"`` for the initial bins.
        monotone: Alias of ``trend`` for ``None``, ``"increasing"``,
            ``"decreasing"``, ``"auto"``. Conflicts with a different ``trend``
            raise ValueError.
        max_bins: Upper bound on ordered (non-missing, non-special) bins. None
            means no cap for greedy, and ``n_bins`` for dp. dp needs at least 2.
        min_bin_frac: Minimum share of feature-present usable rows per ordered bin,
            in [0, 1]. For greedy, the share of all usable rows.
        smoothing: Positive prior count added to each cell.
        edges: Greedy: cut points used as given. Dp: the pre-bins, used as given.
        optimize: ``"greedy"`` (default) or ``"dp"``.
        n_prebins: dp only: number of quantile pre-bins to start from (at most).
        trend: None, ``"increasing"``, ``"decreasing"``, ``"peak"``, ``"valley"``
            or ``"auto"``. greedy supports the first three and ``"auto"``.
        special_values: Finite numbers that get their own bins (e.g. ``[-999, 0]``).

    Returns:
        Dict with keys:
            ``iv`` (float): total IV, including the missing and special bins.
            ``bins`` (list[dict]): ordered bins first, then special bins. Each has
                ``lo``, ``hi`` (floats, +-inf at the ends; equal for specials),
                ``special`` (bool), ``n``, ``events``, ``non_events``,
                ``event_rate``, ``woe``, ``iv``.
            ``missing_bin`` (dict): ``n``, ``events``, ``non_events``,
                ``event_rate``, ``woe``, ``iv``. When empty, ``n`` is 0,
                ``woe`` and ``event_rate`` are None and ``iv`` is 0.0.
            ``n_rows`` (int): ``len(values)``.
            ``n_used`` (int): rows with a present target.
            ``n_y_missing`` (int): rows with a missing target.
            ``trend`` (str | None): ``"increasing"``, ``"decreasing"``, ``"peak"``
                and ``"valley"`` (dp only), ``"none"`` (constant or other shape),
                or None with fewer than two ordered bins.
            ``optimize`` (str): the optimiser used.
            ``objective`` (float): sum of the DP contribution over ordered bins.
            ``constraint_relaxed`` (bool): dp: True when a rule was relaxed (see
                above). greedy: always False, since its rules are not the DP ones.

    Raises:
        ValueError: For a non-binary target, invalid options, conflicting
            ``monotone``/``trend``, ``peak``/``valley`` with greedy, dp with
            ``max_bins`` below 2, non-numeric values, or mismatched lengths.

    Example:
        >>> r = iv_bins([1, 2, 3, 4], [0, 0, 1, 1], edges=[2.5], min_bin_frac=0)
        >>> round(r["iv"], 6), r["trend"]
        (2.145917, 'increasing')
        >>> d = iv_bins([1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 1, 1], optimize="dp",
        ...             n_prebins=6, max_bins=2, min_bin_frac=0)
        >>> [(b["lo"], b["hi"], b["n"], b["events"]) for b in d["bins"]]
        [(-inf, 3.5, 3, 1), (3.5, inf, 3, 2)]
    """
    _check_smoothing(smoothing)
    _check_fraction(min_bin_frac, "min_bin_frac")
    if monotone not in _MONOTONE:
        raise ValueError(f"monotone must be one of {_MONOTONE}")
    if optimize not in _OPTIMIZE:
        raise ValueError(f"optimize must be one of {_OPTIMIZE}")
    t = _resolve_trend(monotone, trend, optimize)
    _check_positive_int(n_prebins, "n_prebins")
    if max_bins is not None:
        _check_positive_int(max_bins, "max_bins")
    if len(values) != len(y):
        raise ValueError("values and y must have the same length")
    cap = max_bins if max_bins is not None else n_bins
    if optimize == "dp":
        _check_positive_int(n_bins, "n_bins")
        if cap < 2:
            raise ValueError("optimize='dp' needs max_bins >= 2")
    s = smoothing
    requested = _checked_specials(special_values)

    vals, flags, n_y_missing = _usable(values, y)
    n_used = len(vals)
    pos = {v: p for p, v in enumerate(requested)}
    sp_counts = [[0, 0] for _ in requested]
    ord_vals: list[Any] = []
    ord_flags: list[int] = []
    miss_n = 0
    miss_ev = 0
    for v, f in zip(vals, flags):
        if is_missing(v):
            miss_n += 1
            miss_ev += f
            continue
        if is_number(v) and float(v) in pos:
            cell = sp_counts[pos[float(v)]]
            cell[0] += 1
            cell[1] += f
            continue
        if not (is_number(v) or is_infinite(v)):
            raise ValueError(f"iv_bins expects numeric values, got {v!r}")
        ord_vals.append(v)
        ord_flags.append(f)
    specials = [(requested[p], c[0], c[1]) for p, c in enumerate(sp_counts) if c[0] > 0]
    extra = [(n, e) for _, n, e in specials]
    n_feat = len(ord_vals) + sum(n for _, n, _ in specials)
    big_e = sum(ord_flags) + sum(e for _, _, e in specials)
    big_n = n_feat - big_e

    if optimize == "greedy":
        cuts = (
            _checked_edges(edges)
            if edges is not None
            else bin_edges(ord_vals, n_bins=n_bins, method=method)
        )
        index = binner(ord_vals, edges=cuts)["index"]
        k = len(cuts) + 1
        counts = [0] * k
        events = [0] * k
        for idx, f in zip(index, ord_flags):
            counts[idx] += 1
            events[idx] += f
        lows = [-math.inf, *cuts]
        highs = [*cuts, math.inf]
        base = [
            {"lo": lows[i], "hi": highs[i], "n": counts[i], "events": events[i]}
            for i in range(k)
        ]

        def build(decreasing: bool | None) -> list[dict[str, Any]]:
            if sum(b["n"] for b in base) == 0:
                return []
            bins = _enforce_min(base, 1)
            if decreasing is not None:
                bins = _pav(bins, decreasing)
            bins = _enforce_min(bins, min_bin_frac * n_used)
            if max_bins is not None:
                bins = _cap_bins(bins, max_bins, miss_n, miss_ev, smoothing, extra)
            return bins

        def total_iv(bins: list[dict[str, Any]]) -> float:
            cells = [(b["n"], b["events"]) for b in bins] + extra
            return _score(cells, miss_n, miss_ev, smoothing)[2]

        if t == "auto":
            bins = max([build(False), build(True)], key=total_iv)
        else:
            bins = build({None: None, "increasing": False, "decreasing": True}[t])
        groups = [(b["n"], b["events"]) for b in bins]
        bounds = [(b["lo"], b["hi"]) for b in bins]
        trend_out = _trend([b["events"] / b["n"] for b in bins])
        relaxed = False
    else:
        cuts = _dp_prebin_edges(ord_vals, edges, n_prebins, method)
        if ord_vals:
            pre_index = binner(ord_vals, edges=cuts)["index"]
        else:
            pre_index = []
        P = len(cuts) + 1 if ord_vals else 0
        pn = [0] * P
        pe = [0] * P
        for idx, f in zip(pre_index, ord_flags):
            pn[idx] += 1
            pe[idx] += f
        cn = [0]
        ce = [0]
        for a, b in zip(pn, pe):
            cn.append(cn[-1] + a)
            ce.append(ce[-1] + b)
        min_count = math.ceil(round(min_bin_frac * n_feat, 9))

        if P == 0:
            groups, bounds, trend_out, relaxed = [], [], None, True
        else:
            bounds_all = [-math.inf, *cuts, math.inf]
            kmax = min(cap, P)
            options = ["increasing", "decreasing"] if t == "auto" else [t]
            k2 = min(2, P)
            ladder = (
                (k2, min_count, True),
                (k2, min_count, False),
                (k2, 1, False),
                (1, 1, False),
            )
            found = None
            for level, (kmin_, mc, presence) in enumerate(ladder):
                cands = []
                for opt in options:
                    res = _dp_solve(
                        cn, ce, big_e, big_n, s, kmin_, kmax, mc, presence, opt
                    )
                    if res is not None:
                        cands.append(res)
                if cands:
                    best = cands[0]
                    for c in cands[1:]:
                        if c[0] > best[0] + _EPS:
                            best = c
                    found = (level, best[1])
                    break
            assert found is not None  # a single bin is always feasible for P >= 1
            level, ranges = found
            relaxed = level > 0 or P < 2
            groups = [(cn[j] - cn[i], ce[j] - ce[i]) for i, j in ranges]
            bounds = [(bounds_all[i], bounds_all[j]) for i, j in ranges]
            trend_out = _shape(groups)

    stats, miss_stats, iv = _score([*groups, *extra], miss_n, miss_ev, s)
    m = len(groups)
    ordered = [
        {"lo": lo, "hi": hi, "special": False, **st}
        for (lo, hi), st in zip(bounds, stats[:m])
    ]
    special_out = [
        {"lo": v, "hi": v, "special": True, **st}
        for (v, _, _), st in zip(specials, stats[m:])
    ]
    objective = sum(_contribution(n, e, big_e, big_n, s) for n, e in groups)
    return {
        "iv": iv,
        "bins": ordered + special_out,
        "missing_bin": miss_stats,
        "n_rows": len(values),
        "n_used": n_used,
        "n_y_missing": n_y_missing,
        "trend": trend_out,
        "optimize": optimize,
        "objective": objective,
        "constraint_relaxed": relaxed,
    }


def iv_levels(
    values: Sequence[Any],
    y: Sequence[Any],
    min_level_frac: float = 0.0,
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Information value of a categorical feature against a binary target.

    One bin per level, sorted by event rate ascending (ties by level name).
    Levels with fewer than ``min_level_frac`` of the usable rows are pooled into
    a level named ``"__other__"``. Missing values form ``missing_bin``. Target
    handling, smoothing and the returned keys follow ``iv_bins``, with ``level``
    in place of ``lo``/``hi`` and ``trend`` always None.

    Args:
        values: Categorical values (any hashable); None and NaN are missing.
        y: Binary target.
        min_level_frac: Pool levels rarer than this share of usable rows, in [0, 1].
        smoothing: Positive prior count added to each cell.

    Returns:
        Dict with keys ``iv``, ``bins`` (dicts with ``level``, ``n``,
        ``events``, ``non_events``, ``event_rate``, ``woe``, ``iv``),
        ``missing_bin``, ``n_rows``, ``n_used``, ``n_y_missing``, ``trend``.

    Example:
        >>> r = iv_levels(["a", "b", "a", "b"], [0, 0, 1, 1])
        >>> [b["level"] for b in r["bins"]], round(r["iv"], 6)
        (['a', 'b'], 0.0)
    """
    _check_smoothing(smoothing)
    _check_fraction(min_level_frac, "min_level_frac")
    if len(values) != len(y):
        raise ValueError("values and y must have the same length")
    _, flags, n_y_missing = _usable(values, y)
    keep = [i for i in range(len(y)) if not is_missing(y[i])]
    n_used = len(keep)

    counts: dict[Any, list[int]] = {}
    miss_n = 0
    miss_ev = 0
    for pos, i in enumerate(keep):
        v = values[i]
        f = flags[pos]
        if is_missing(v):
            miss_n += 1
            miss_ev += f
            continue
        cell = counts.setdefault(v, [0, 0])
        cell[0] += 1
        cell[1] += f

    threshold = min_level_frac * n_used
    kept: list[tuple[Any, int, int]] = []
    other = [0, 0]
    for level, (n, e) in counts.items():
        if n < threshold:
            other[0] += n
            other[1] += e
        else:
            kept.append((level, n, e))
    if other[0] > 0:
        kept.append(("__other__", other[0], other[1]))
    kept.sort(key=lambda t: (t[2] / t[1], str(t[0])))

    stats, miss_stats, iv = _score(
        [(n, e) for _, n, e in kept], miss_n, miss_ev, smoothing
    )
    return {
        "iv": iv,
        "bins": [{"level": lvl, **st} for (lvl, _, _), st in zip(kept, stats)],
        "missing_bin": miss_stats,
        "n_rows": len(values),
        "n_used": n_used,
        "n_y_missing": n_y_missing,
        "trend": None,
    }


def _strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


def information_value_table(
    columns: Any,
    y: Sequence[Any],
    n_bins: int = 10,
    method: str = "quantile",
    monotone: str | None = None,
    max_bins: int | None = None,
    min_bin_frac: float = 0.05,
    smoothing: float = 0.5,
    min_level_frac: float = 0.0,
) -> list[dict[str, Any]]:
    """IV for every column: ``iv_bins`` for numeric, ``iv_levels`` for categorical.

    A column is numeric when every non-missing value is an int or float (bool
    excluded). Strength follows the usual rule of thumb: below 0.02 useless,
    below 0.1 weak, below 0.3 medium, below 0.5 strong, otherwise suspicious.

    Args:
        columns: Dict of columns, or a table ``as_columns`` accepts.
        y: Binary target, same length as each column.
        n_bins, method, monotone, max_bins, min_bin_frac, smoothing: Passed to
            ``iv_bins`` for numeric columns.
        min_level_frac: Passed to ``iv_levels`` for categorical columns.

    Returns:
        List of dicts with ``name``, ``iv``, ``kind`` ("numeric" or
        "categorical"), ``n_bins`` (ordered or level bins, missing excluded),
        and ``strength``. Sorted by IV descending, then name.

    Example:
        >>> t = information_value_table({"x": [1, 2, 3, 4]}, [0, 0, 1, 1], n_bins=2)
        >>> t[0]["name"], t[0]["kind"], t[0]["n_bins"]
        ('x', 'numeric', 2)
    """
    rows = []
    for name, col in as_columns(columns).items():
        if _is_numeric_column(col):
            r = iv_bins(
                col,
                y,
                n_bins=n_bins,
                method=method,
                monotone=monotone,
                max_bins=max_bins,
                min_bin_frac=min_bin_frac,
                smoothing=smoothing,
            )
            kind = "numeric"
            n_bins_out = len(r["bins"])
        else:
            r = iv_levels(col, y, min_level_frac=min_level_frac, smoothing=smoothing)
            kind = "categorical"
            n_bins_out = len(r["bins"])
        rows.append(
            {
                "name": str(name),
                "iv": r["iv"],
                "kind": kind,
                "n_bins": n_bins_out,
                "strength": _strength(r["iv"]),
            }
        )
    rows.sort(key=lambda row: (-row["iv"], row["name"]))
    return rows
