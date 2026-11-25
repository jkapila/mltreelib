"""Profiling, missingness, duplicate and schema helpers (standard library only).

Every function takes ``columns`` as a dict of column name to list, or any table
that :func:`mltreelib.data._common.as_columns` accepts. Missing means ``None``
or NaN: it is counted, never imputed, dropped or reordered. Infinite values are
counted as ``n_inf`` and ignored by statistics; ``inf_as_missing=True`` also
counts them as missing in the counts. Nothing here modifies its input.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mltreelib.data._common import (
    as_columns,
    is_infinite,
    is_missing,
    is_number,
    numeric_observed,
    quantile,
)

__all__ = [
    "class_balance",
    "constant_columns",
    "describe",
    "duplicate_columns",
    "duplicate_rows",
    "infer_types",
    "missing_profile",
    "missing_vs_target",
    "near_zero_variance",
    "outlier_flags",
    "schema_capture",
    "schema_validate",
]

_DEFAULT_N_CATEGORY = 20
_DEFAULT_PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_DEFAULT_TRIM = 0.1
_SCHEMA_MAX_LEVELS = 200
_KIND_OVERRIDES = ("numeric", "categorical", "ordinal")
_HINT_OF_OVERRIDE = {"ordinal": "ordinal_candidate"}
_BASE_NUMERIC_KEYS = (
    "mean",
    "std",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "skew",
    "kurtosis",
)
_ROBUST_NUMERIC_KEYS = (
    "iqr",
    "mad",
    "cv",
    "trimmed_mean",
    "trimmed_std",
    "n_outliers_iqr",
    "pct_outliers_iqr",
    "n_zero",
    "n_negative",
    "n_positive",
)
_MISSING_KEY = object()  # comparison marker: all missing values compare equal


def _n_rows(cols: Mapping[str, Sequence[Any]]) -> int:
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        raise ValueError("all columns must have the same number of rows")
    return lengths.pop() if lengths else 0


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _mean(xs: Sequence[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def _is_plain_number(v: Any) -> bool:
    """int or float, bool excluded; infinities included."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_integer_valued(v: Any) -> bool:
    """For a plain number: an int, or a finite float with no fractional part."""
    return isinstance(v, int) or (math.isfinite(v) and v.is_integer())


def _kind_hint(observed: Sequence[Any], n_category: int) -> str:
    """Data-based kind hint shared by describe, infer_types and the schema functions.

    - Any value that is not an int or float (bool included) gives ``categorical``.
    - All numbers, integer-valued, with fewer than 6 distinct values give
      ``ordinal_candidate`` (often a category code or an ordered rating).
    - All numbers that are not all integer-valued, or with more than
      ``n_category`` distinct values, give ``numeric``.
    - Otherwise (integer-valued, 6 to ``n_category`` distinct) ``categorical``.

    A column with no observed values is ``numeric``.
    """
    if not observed:
        return "numeric"
    if not all(_is_plain_number(v) for v in observed):
        return "categorical"
    integer_valued = all(_is_integer_valued(v) for v in observed)
    n_distinct = len(set(observed))
    if integer_valued and n_distinct < 6:
        return "ordinal_candidate"
    if not integer_valued or n_distinct > n_category:
        return "numeric"
    return "categorical"


def _check_kinds(
    kinds: Mapping[str, str] | None, names: Sequence[str]
) -> dict[str, str]:
    """Validate a user ``kinds`` override dict against the column names."""
    if not kinds:
        return {}
    unknown = [name for name in kinds if name not in names]
    if unknown:
        raise ValueError(f"kinds names unknown column(s): {unknown}")
    bad = {name: kind for name, kind in kinds.items() if kind not in _KIND_OVERRIDES}
    if bad:
        raise ValueError(
            f"kinds values must be 'numeric', 'categorical' or 'ordinal'; got {bad}"
        )
    return dict(kinds)


def _check_trim(trim: float) -> None:
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5); got {trim}")


def _percentile_keys(
    percentiles: Sequence[float],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Quantiles and their output key names ``p05``, ``p25``, ... (whole percents only)."""
    qs = tuple(float(q) for q in percentiles)
    keys: list[str] = []
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"percentiles must be in [0, 1]; got {q}")
        pct = q * 100
        if abs(pct - round(pct)) > 1e-9:
            raise ValueError(
                f"percentiles must be whole percents, such as 0.05; got {q}"
            )
        keys.append(f"p{round(pct):02d}")
    if len(set(qs)) != len(qs):
        raise ValueError(f"percentiles must not repeat; got {list(qs)}")
    return qs, tuple(keys)


def _sample_std(xs: Sequence[float], mean: float) -> float | None:
    if len(xs) < 2:
        return None
    return math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def _numeric_stats(
    observed: Sequence[Any],
    qs: Sequence[float],
    pkeys: Sequence[str],
    trim: float,
) -> dict[str, Any]:
    """Statistics over the finite observed values of a numeric-valued column.

    Skew is the adjusted Fisher-Pearson G1 and kurtosis the adjusted excess G2.
    The Tukey outlier count uses fences ``q1 - 1.5*iqr`` and ``q3 + 1.5*iqr``.
    Sign counts are always returned (0 when there are no finite values).
    """
    out: dict[str, Any] = dict.fromkeys(
        (*_BASE_NUMERIC_KEYS, *pkeys, *_ROBUST_NUMERIC_KEYS)
    )
    finite = sorted(numeric_observed(observed))
    n = len(finite)
    out.update(
        n_zero=sum(1 for x in finite if x == 0),
        n_negative=sum(1 for x in finite if x < 0),
        n_positive=sum(1 for x in finite if x > 0),
    )
    if n == 0:
        return out
    mean = math.fsum(finite) / n
    std = _sample_std(finite, mean)
    q1, med, q3 = quantile(finite, 0.25), quantile(finite, 0.5), quantile(finite, 0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = sum(1 for x in finite if x < lo or x > hi)
    # trim: floor(n * trim) values removed from each end; the epsilon guards float error
    k = math.floor(n * trim + 1e-9)
    core = finite[k : n - k]
    out.update(
        mean=mean,
        std=std,
        min=finite[0],
        max=finite[-1],
        q1=q1,
        median=med,
        q3=q3,
        iqr=iqr,
        mad=quantile(sorted(abs(x - med) for x in finite), 0.5),
        cv=std / mean if std is not None and mean != 0 else None,
        n_outliers_iqr=n_outliers,
        pct_outliers_iqr=100.0 * n_outliers / n,
    )
    for key, q in zip(pkeys, qs):
        out[key] = quantile(finite, q)
    if core:
        trimmed_mean = math.fsum(core) / len(core)
        out["trimmed_mean"] = trimmed_mean
        out["trimmed_std"] = _sample_std(core, trimmed_mean)
    if len(set(finite)) >= 2:  # zero variance: shape statistics undefined
        m2 = math.fsum((x - mean) ** 2 for x in finite)
        s2 = m2 / n
        if n >= 3:
            g1 = (math.fsum((x - mean) ** 3 for x in finite) / n) / s2**1.5
            out["skew"] = g1 * math.sqrt(n * (n - 1)) / (n - 2)
        if n >= 4:
            g2 = (math.fsum((x - mean) ** 4 for x in finite) / n) / s2**2 - 3.0
            out["kurtosis"] = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return out


def _categorical_stats(observed: Sequence[Any]) -> dict[str, Any]:
    """Mode and top levels. Infinities are excluded (they are ignored by statistics)."""
    levels = [v for v in observed if not is_infinite(v)]
    ranked = sorted(Counter(levels).items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {
        "mode": ranked[0][0] if ranked else None,
        "top": [(level, count) for level, count in ranked[:10]],
    }


def _level_key(v: Any) -> tuple[str, str]:
    return (type(v).__name__, str(v))


def _missing_test(inf_as_missing: bool) -> Callable[[Any], bool]:
    """Missing test used for counts: infinities too when ``inf_as_missing`` is True."""
    if inf_as_missing:
        return lambda v: is_missing(v) or is_infinite(v)
    return is_missing


def _resolve_kind(
    name: str, hint: str, overrides: Mapping[str, str]
) -> tuple[str, str]:
    if name in overrides:
        return overrides[name], "user"
    return hint, "inferred"


def describe(
    columns: Any,
    n_category: int = _DEFAULT_N_CATEGORY,
    kinds: Mapping[str, str] | None = None,
    trim: float = _DEFAULT_TRIM,
    percentiles: Sequence[float] = _DEFAULT_PERCENTILES,
    inf_as_missing: bool = False,
) -> list[dict[str, Any]]:
    """Describe each column: kind, missingness, distinct count, and statistics.

    Statistics are computed on the finite observed values. Numeric statistics
    are computed for every column whose observed values are all ints or floats
    (bools excluded), whatever its kind, so a 0/1 flag or a 1 to 5 rating gets a
    mean. Categorical statistics (``mode`` and ``top``, up to 10 ``(level,
    count)`` pairs ordered by count descending then ``str(level)``) are computed
    for every column; infinities are excluded from them. Numeric keys are
    ``None`` when the column is not numeric-valued or the value is not
    computable.

    The ``kind`` is a hint, never enforced:

    - ``'categorical'``: any value that is not an int or float (bool included).
    - ``'ordinal_candidate'``: all values integer-valued with fewer than 6
      distinct values (likely a category code or an ordered rating such as 1 to
      5; the user decides, and ordered levels suit monotone constraints).
    - ``'numeric'``: all numbers that are not all integer-valued, or with more
      than ``n_category`` distinct values.
    - otherwise ``'categorical'`` (integer-valued with 6 to ``n_category``
      distinct values).

    ``kinds`` overrides the hint per column with ``'numeric'``, ``'categorical'``
    or ``'ordinal'`` (``'ordinal'`` is accepted only here). The override is
    echoed as ``kind`` without checking it against the data; ``kind_source`` is
    then ``'user'`` instead of ``'inferred'``.

    Args:
        columns: Dict of column name to list, or a table.
        n_category: Integer-valued columns with more distinct values than this
            are numeric; see the kind rule above.
        kinds: Optional ``{column: 'numeric' | 'categorical' | 'ordinal'}``.
        trim: Fraction removed from each end before the trimmed mean and
            standard deviation, in ``[0, 0.5)``. ``floor(n * trim)`` values are
            removed from each end; ``trim=0`` gives the plain mean and std.
        percentiles: Quantiles in ``[0, 1]``, whole percents only (e.g. 0.05).
            Each gives a key ``p05``, ``p25``, ... (``p`` plus two digits).
        inf_as_missing: If True, infinities are also counted in ``n_missing``
            and ``pct_missing``. Statistics ignore infinities either way.

    Returns:
        One dict per column, in input order, with keys:

        - ``name``, ``kind``, ``kind_source`` (``'inferred'`` or ``'user'``).
        - ``n`` (rows), ``n_missing``, ``pct_missing`` (0-100), ``n_inf``,
          ``n_distinct`` (distinct observed values, infinities included).
        - ``mean``, ``std`` (sample), ``min``, ``q1``, ``median``, ``q3``,
          ``max``, ``skew`` and ``kurtosis`` (excess).
        - One ``pNN`` key per requested percentile (default ``p05``, ``p25``,
          ``p50``, ``p75``, ``p95``).
        - ``iqr`` (q3 - q1), ``mad`` (median absolute deviation from the median,
          unscaled), ``cv`` (std / mean; ``None`` when the mean is 0).
        - ``trimmed_mean`` and ``trimmed_std`` after trimming (see ``trim``).
        - ``n_outliers_iqr`` and ``pct_outliers_iqr`` (percent of finite values)
          outside the Tukey fences ``q1 - 1.5*iqr`` and ``q3 + 1.5*iqr``.
        - ``n_zero``, ``n_negative``, ``n_positive`` (finite values).
        - ``mode`` and ``top``.

        Every column has the same keys; keys that do not apply are ``None``.

    Raises:
        ValueError: For an invalid ``trim``, a bad percentile, or a ``kinds``
            entry naming an unknown column or kind.

    Example:
        >>> rows = describe({"x": [1.5, None, 2.5, 3.5]})
        >>> rows[0]["kind"], rows[0]["n_missing"], rows[0]["mean"]
        ('numeric', 1, 2.5)
        >>> describe({"flag": [0, 1, 1, None]})[0]["kind"]
        'ordinal_candidate'
    """
    cols = as_columns(columns)
    overrides = _check_kinds(kinds, list(cols))
    _check_trim(trim)
    qs, pkeys = _percentile_keys(percentiles)
    missing = _missing_test(inf_as_missing)
    out: list[dict[str, Any]] = []
    for name, values in cols.items():
        n = len(values)
        observed = [v for v in values if not is_missing(v)]
        n_missing = sum(1 for v in values if missing(v))
        hint = _kind_hint(observed, n_category)
        kind, source = _resolve_kind(name, hint, overrides)
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "kind_source": source,
            "n": n,
            "n_missing": n_missing,
            "pct_missing": _pct(n_missing, n),
            "n_inf": sum(1 for v in observed if is_infinite(v)),
            "n_distinct": len(set(observed)),
        }
        if all(_is_plain_number(v) for v in observed):
            entry.update(_numeric_stats(observed, qs, pkeys, trim))
        else:
            entry.update(
                dict.fromkeys((*_BASE_NUMERIC_KEYS, *pkeys, *_ROBUST_NUMERIC_KEYS))
            )
        entry.update(_categorical_stats(observed))
        out.append(entry)
    return out


def missing_profile(columns: Any, inf_as_missing: bool = False) -> dict[str, Any]:
    """Missingness per column, per row, by pattern, and as a co-missing matrix.

    Args:
        columns: Dict of column name to list, or a table.
        inf_as_missing: If True, infinities count as missing in every count,
            pattern and per-row total below.

    Returns:
        Dict with keys ``n_rows``, ``n_cols``, ``per_column`` (``{name:
        {'n_missing', 'pct_missing'}}``), ``per_row`` (missing count per row),
        ``n_complete_rows``, ``patterns`` (list of ``{'columns': tuple of names
        missing together, 'count': int}``, sorted by count descending then
        columns; the all-observed pattern is ``columns=()``) and ``co_missing``
        (``{'names': [...], 'matrix': [[int]]}``, the count of rows where both
        columns are missing; the diagonal is each column's missing count).

    Example:
        >>> p = missing_profile({"a": [1, None], "b": [None, None]})
        >>> p["n_complete_rows"], p["co_missing"]["matrix"]
        (0, [[1, 1], [1, 2]])
    """
    cols = as_columns(columns)
    names = list(cols)
    n_rows = _n_rows(cols)
    missing = _missing_test(inf_as_missing)
    per_row = [0] * n_rows
    row_missing: list[list[int]] = [[] for _ in range(n_rows)]
    per_column: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        n_miss = 0
        for i, v in enumerate(cols[name]):
            if missing(v):
                n_miss += 1
                per_row[i] += 1
                row_missing[i].append(j)
        per_column[name] = {"n_missing": n_miss, "pct_missing": _pct(n_miss, n_rows)}

    pattern_counts = Counter(tuple(names[j] for j in idx) for idx in row_missing)
    patterns = [
        {"columns": cols_, "count": count}
        for cols_, count in sorted(
            pattern_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]

    m = len(names)
    matrix = [[0] * m for _ in range(m)]
    for idx in row_missing:
        for a in idx:
            row = matrix[a]
            for b in idx:
                row[b] += 1

    return {
        "n_rows": n_rows,
        "n_cols": m,
        "per_column": per_column,
        "per_row": per_row,
        "n_complete_rows": sum(1 for c in per_row if c == 0),
        "patterns": patterns,
        "co_missing": {"names": names, "matrix": matrix},
    }


def missing_vs_target(
    columns: Any, y: Sequence[Any], inf_as_missing: bool = False
) -> dict[str, dict[str, Any]]:
    """Target mean when each column is missing versus observed.

    Rows whose target is missing or infinite are left out of every count and
    mean; their number is reported as ``n_y_missing_excluded``. The counts
    ``n_missing`` and ``n_observed`` cover only the rows used for the means.

    Args:
        columns: Dict of column name to list, or a table.
        y: Numeric target (0/1 or continuous), one value per row.
        inf_as_missing: If True, a column's infinite values count as missing
            (they are not missing by default).

    Returns:
        ``{name: {'n_missing', 'n_observed', 'y_mean_when_missing',
        'y_mean_when_observed', 'n_y_missing_excluded'}}``. Means are ``None``
        when there are no rows in that group.

    Raises:
        ValueError: If ``y`` and the columns differ in length.
        TypeError: If ``y`` holds a non-numeric value.

    Example:
        >>> r = missing_vs_target({"x": [1, None, 3, 4]}, [1, 0, 0, None])
        >>> r["x"]["n_missing"], r["x"]["y_mean_when_missing"], r["x"]["n_y_missing_excluded"]
        (1, 0.0, 1)
    """
    cols = as_columns(columns)
    n = _n_rows(cols)
    y_vals = list(y)
    if len(y_vals) != n:
        raise ValueError("y must have one value per row")
    missing = _missing_test(inf_as_missing)
    usable = [False] * n
    y_num = [0.0] * n
    for i, v in enumerate(y_vals):
        if is_missing(v) or is_infinite(v):
            continue
        if not _is_plain_number(v):
            raise TypeError(f"y must be numeric; got {type(v).__name__} at row {i}")
        usable[i] = True
        y_num[i] = float(v)
    n_excluded = n - sum(usable)

    out: dict[str, dict[str, Any]] = {}
    for name, values in cols.items():
        when_missing: list[float] = []
        when_observed: list[float] = []
        for i in range(n):
            if usable[i]:
                (when_missing if missing(values[i]) else when_observed).append(y_num[i])
        out[name] = {
            "n_missing": len(when_missing),
            "n_observed": len(when_observed),
            "y_mean_when_missing": _mean(when_missing),
            "y_mean_when_observed": _mean(when_observed),
            "n_y_missing_excluded": n_excluded,
        }
    return out


def infer_types(
    columns: Any,
    n_category: int = _DEFAULT_N_CATEGORY,
    kinds: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Kind and number of distinct observed levels per column.

    Uses the same kind rule as :func:`describe`, including the ``kinds``
    override (see there for its values).

    Args:
        columns: Dict of column name to list, or a table.
        n_category: Threshold for integer-valued columns (see :func:`describe`).
        kinds: Optional ``{column: 'numeric' | 'categorical' | 'ordinal'}``.

    Returns:
        ``{name: {'kind', 'n_levels', 'ordinal_candidate'}}``. ``kind`` is the
        override if given, else the hint (``'numeric'``, ``'categorical'`` or
        ``'ordinal_candidate'``). ``n_levels`` is the number of distinct observed
        non-missing values. ``ordinal_candidate`` is True when the data-based hint
        is ``'ordinal_candidate'``, whatever the override.

    Raises:
        ValueError: For a ``kinds`` entry naming an unknown column or kind.

    Example:
        >>> infer_types({"c": ["a", "b", None], "n": [1.5, 2.0, 3.0]})["c"]
        {'kind': 'categorical', 'n_levels': 2, 'ordinal_candidate': False}
    """
    cols = as_columns(columns)
    overrides = _check_kinds(kinds, list(cols))
    out: dict[str, dict[str, Any]] = {}
    for name, values in cols.items():
        observed = [v for v in values if not is_missing(v)]
        hint = _kind_hint(observed, n_category)
        kind, _ = _resolve_kind(name, hint, overrides)
        out[name] = {
            "kind": kind,
            "n_levels": len(set(observed)),
            "ordinal_candidate": hint == "ordinal_candidate",
        }
    return out


def near_zero_variance(
    columns: Any, freq_cut: float = 19.0, unique_cut: float = 10.0
) -> list[dict[str, Any]]:
    """Columns flagged by the caret near-zero-variance rule.

    Computed on observed values. ``freq_ratio`` is the count of the most common
    value divided by the count of the second most common; it is ``inf`` when
    there are fewer than two distinct values. ``pct_unique`` is 100 times the
    distinct count over the observed count (0.0 when nothing is observed). A
    column is flagged when it has at most one distinct value, or when
    ``freq_ratio > freq_cut`` and ``pct_unique < unique_cut``.

    Args:
        columns: Dict of column name to list, or a table.
        freq_cut: Frequency ratio cutoff.
        unique_cut: Percent-unique cutoff.

    Returns:
        Flagged columns only, in input order, as dicts with keys ``name``,
        ``freq_ratio``, ``pct_unique`` and ``zero_var``.

    Example:
        >>> near_zero_variance({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
        [{'name': 'a', 'freq_ratio': inf, 'pct_unique': 25.0, 'zero_var': True}]
    """
    flagged: list[dict[str, Any]] = []
    for name, values in as_columns(columns).items():
        observed = [v for v in values if not is_missing(v)]
        counts = sorted(Counter(observed).values(), reverse=True)
        distinct = len(counts)
        zero_var = distinct <= 1
        freq_ratio = math.inf if zero_var else counts[0] / counts[1]
        pct_unique = _pct(distinct, len(observed))
        if zero_var or (freq_ratio > freq_cut and pct_unique < unique_cut):
            flagged.append(
                {
                    "name": name,
                    "freq_ratio": freq_ratio,
                    "pct_unique": pct_unique,
                    "zero_var": zero_var,
                }
            )
    return flagged


def constant_columns(columns: Any) -> list[str]:
    """Names of columns with at most one distinct observed value.

    All-missing columns are included.

    Args:
        columns: Dict of column name to list, or a table.

    Returns:
        Column names in input order.

    Example:
        >>> constant_columns({"a": [1, 1, None], "b": [1, 2, 3], "c": [None, None, None]})
        ['a', 'c']
    """
    return [
        name
        for name, values in as_columns(columns).items()
        if len({v for v in values if not is_missing(v)}) <= 1
    ]


def _comparable(v: Any) -> Any:
    return _MISSING_KEY if is_missing(v) else v


def duplicate_rows(columns: Any) -> list[list[int]]:
    """Groups of identical rows, as ascending row indices.

    Missing equals missing (``None`` and NaN are the same marker), and ``1``
    equals ``1.0``. Only groups of two or more rows are returned.

    Args:
        columns: Dict of column name to list, or a table.

    Returns:
        List of index lists, ordered by each group's first index.

    Example:
        >>> duplicate_rows({"a": [1, 2, 1, None], "b": ["x", "y", "x", None]})
        [[0, 2]]
    """
    cols = as_columns(columns)
    if not cols:
        return []
    keyed = [[_comparable(v) for v in values] for values in cols.values()]
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(zip(*keyed)):
        groups.setdefault(row, []).append(i)
    return [idx for idx in groups.values() if len(idx) >= 2]


def duplicate_columns(columns: Any) -> list[list[str]]:
    """Groups of columns with identical values row by row.

    Compared under the same rule as :func:`duplicate_rows`.

    Args:
        columns: Dict of column name to list, or a table.

    Returns:
        List of name lists (input order within a group), ordered by each
        group's first column. Only groups of two or more columns are returned.

    Example:
        >>> duplicate_columns({"a": [1, 2], "b": [1.0, 2.0], "c": [2, 1]})
        [['a', 'b']]
    """
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name, values in as_columns(columns).items():
        key = tuple(_comparable(v) for v in values)
        groups.setdefault(key, []).append(name)
    return [names for names in groups.values() if len(names) >= 2]


def class_balance(y: Sequence[Any]) -> dict[str, Any]:
    """Class counts and balance for a target.

    Proportions and the imbalance ratio use observed (non-missing) values only.

    Args:
        y: Target values, one per row.

    Returns:
        Dict with keys ``counts`` (``{level: n}``, count descending then
        ``str(level)``), ``proportions`` (``{level: share of observed rows}``),
        ``imbalance_ratio`` (largest over smallest count, ``None`` under two
        levels), ``n_missing`` and ``n`` (all rows).

    Example:
        >>> b = class_balance([0, 0, 0, 1, None])
        >>> b["counts"], b["imbalance_ratio"], b["n_missing"]
        ({0: 3, 1: 1}, 3.0, 1)
    """
    values = list(y)
    observed = [v for v in values if not is_missing(v)]
    ranked = sorted(Counter(observed).items(), key=lambda kv: (-kv[1], str(kv[0])))
    total = len(observed)
    ratio = ranked[0][1] / ranked[-1][1] if len(ranked) >= 2 else None
    return {
        "counts": dict(ranked),
        "proportions": {level: count / total for level, count in ranked}
        if total
        else {},
        "imbalance_ratio": ratio,
        "n_missing": len(values) - total,
        "n": len(values),
    }


def outlier_flags(
    values: Sequence[Any], method: str = "iqr", k: float | None = None
) -> list[bool | None]:
    """Flag outliers in one numeric column. Flags only; nothing is removed.

    Methods:
        ``"iqr"`` (default ``k=1.5``): flag values outside
        ``[q1 - k*iqr, q3 + k*iqr]``.
        ``"mad"`` (default ``k=3.5``): flag when the modified z-score
        ``0.6745 * |x - median| / MAD`` exceeds ``k``. When MAD is zero nothing
        is flagged.

    Quartiles, median and MAD use the finite observed values.

    Args:
        values: One numeric column.
        method: ``"iqr"`` or ``"mad"``.
        k: Threshold multiplier, or ``None`` for the method default.

    Returns:
        One entry per value: ``None`` for missing, ``True`` for infinite,
        otherwise ``True`` or ``False``.

    Raises:
        ValueError: If ``method`` is unknown.
        TypeError: If a value is neither missing, infinite nor a number.

    Example:
        >>> outlier_flags([1, 2, 3, 4, 100, None])
        [False, False, False, False, True, None]
    """
    if method not in ("iqr", "mad"):
        raise ValueError(f"method must be 'iqr' or 'mad', got {method!r}")
    vals = list(values)
    for v in vals:
        if not (is_missing(v) or is_infinite(v) or is_number(v)):
            raise TypeError(
                f"outlier_flags needs numeric values; got {type(v).__name__}"
            )
    finite = sorted(numeric_observed(vals))

    def _flagger() -> Any:
        if not finite:
            return lambda x: False
        if method == "iqr":
            kk = 1.5 if k is None else k
            q1, q3 = quantile(finite, 0.25), quantile(finite, 0.75)
            lo, hi = q1 - kk * (q3 - q1), q3 + kk * (q3 - q1)
            return lambda x: x < lo or x > hi
        kk = 3.5 if k is None else k
        med = quantile(finite, 0.5)
        mad = quantile(sorted(abs(x - med) for x in finite), 0.5)
        if mad == 0:
            return lambda x: False
        return lambda x: 0.6745 * abs(x - med) / mad > kk

    flag = _flagger()
    out: list[bool | None] = []
    for v in vals:
        if is_missing(v):
            out.append(None)
        elif is_infinite(v):
            out.append(True)
        else:
            out.append(bool(flag(float(v))))
    return out


def schema_capture(columns: Any) -> dict[str, Any]:
    """Capture a schema (kind, missing count, range or levels) for later validation.

    Each column records its inferred kind (``kind_source`` is always
    ``'inferred'`` here, as this function takes no ``kinds`` override). A
    ``'numeric'`` column records the finite min and max, with ``levels`` None.
    Any other kind (``'categorical'`` or ``'ordinal_candidate'``) records the
    sorted observed levels, at most 200; beyond that the entry gets
    ``'levels_truncated': True``, and min and max are None.

    Args:
        columns: Dict of column name to list, or a table.

    Returns:
        ``{'columns': {name: {'kind', 'kind_source', 'n_missing', 'min', 'max',
        'levels'}}, 'n_rows': int}``.

    Example:
        >>> s = schema_capture({"x": [1.5, 3.5, None], "c": ["b", "a", "b"]})
        >>> s["columns"]["x"]["max"], s["columns"]["c"]["levels"]
        (3.5, ['a', 'b'])
    """
    cols = as_columns(columns)
    out: dict[str, dict[str, Any]] = {}
    for name, values in cols.items():
        observed = [v for v in values if not is_missing(v)]
        n_missing = len(values) - len(observed)
        kind = _kind_hint(observed, _DEFAULT_N_CATEGORY)
        if kind == "numeric":
            finite = numeric_observed(observed)
            out[name] = {
                "kind": kind,
                "kind_source": "inferred",
                "n_missing": n_missing,
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
                "levels": None,
            }
        else:
            levels = sorted(set(observed), key=_level_key)
            entry: dict[str, Any] = {
                "kind": kind,
                "kind_source": "inferred",
                "n_missing": n_missing,
                "min": None,
                "max": None,
                "levels": levels[:_SCHEMA_MAX_LEVELS],
            }
            if len(levels) > _SCHEMA_MAX_LEVELS:
                entry["levels_truncated"] = True
            out[name] = entry
    return {"columns": out, "n_rows": _n_rows(cols)}


def _kind_disagrees(spec: Mapping[str, Any], kind: str, numeric_valued: bool) -> bool:
    """Whether the data breaks the stored kind.

    Inferred kinds are compared by value type only: numeric-valued (every value
    an int or float) against non-numeric. The stored value type is read from the
    schema: ``levels`` None means numeric; otherwise numeric when every level is
    an int or float. So ``numeric``, ``ordinal_candidate`` and ``categorical``
    differences are never violations. A user-declared kind (``kind_source``
    ``'user'``) must equal the data hint, with ``'ordinal'`` matching
    ``'ordinal_candidate'``.
    """
    if spec.get("kind_source", "inferred") == "user":
        return kind != _HINT_OF_OVERRIDE.get(spec["kind"], spec["kind"])
    levels = spec.get("levels")
    stored_numeric = levels is None or all(_is_plain_number(v) for v in levels)
    return numeric_valued != stored_numeric


def schema_validate(
    columns: Any, schema: Mapping[str, Any], allow_missing: bool = True
) -> list[dict[str, str]]:
    """Check columns against a schema from :func:`schema_capture`.

    Problems reported:

    - ``missing_column``: in the schema, absent from the data.
    - ``extra_column``: in the data, absent from the schema.
    - ``kind_mismatch``: the data breaks the stored kind. For an inferred kind
      this means only a change of value type (numbers against non-numbers); a
      change between ``numeric``, ``ordinal_candidate`` and ``categorical`` is
      not a violation. A user-declared kind must match the data hint exactly.
      All-missing columns fit any kind.
    - ``out_of_range``: finite numeric values outside the captured min and max
      (numeric columns only). Infinite values are not range-checked, matching
      how statistics ignore them.
    - ``unknown_level``: observed values not among the captured levels (columns
      with levels). Skipped when the captured levels were truncated.
    - ``unexpected_missing``: missing values when ``allow_missing`` is False.

    Args:
        columns: Dict of column name to list, or a table.
        schema: The dict returned by :func:`schema_capture`.
        allow_missing: Whether missing values are acceptable.

    Returns:
        List of ``{'column', 'problem', 'detail'}`` dicts. An empty list means
        the data is valid. Schema columns come first, in schema order, then
        extra columns in data order.

    Example:
        >>> s = schema_capture({"x": [1.5, 2.5]})
        >>> schema_validate({"x": [5.5, None]}, s)[0]["problem"]
        'out_of_range'
    """
    cols = as_columns(columns)
    expected = schema["columns"]
    violations: list[dict[str, str]] = []

    def add(column: str, problem: str, detail: str) -> None:
        violations.append({"column": column, "problem": problem, "detail": detail})

    for name, spec in expected.items():
        if name not in cols:
            add(name, "missing_column", "column is in the schema but not in the data")
            continue
        values = cols[name]
        observed = [v for v in values if not is_missing(v)]
        n_missing = len(values) - len(observed)
        kind = _kind_hint(observed, _DEFAULT_N_CATEGORY)
        numeric_valued = all(_is_plain_number(v) for v in observed)
        if observed and _kind_disagrees(spec, kind, numeric_valued):
            add(
                name,
                "kind_mismatch",
                f"stored kind {spec['kind']}, data kind {kind}",
            )
        elif spec.get("levels") is None:  # stored numeric: range check
            lo, hi = spec.get("min"), spec.get("max")
            if numeric_valued and (lo is not None or hi is not None):
                bad = [
                    x
                    for x in numeric_observed(observed)
                    if (lo is not None and x < lo) or (hi is not None and x > hi)
                ]
                if bad:
                    add(
                        name,
                        "out_of_range",
                        f"{len(bad)} value(s) outside [{lo}, {hi}], e.g. {bad[0]}",
                    )
        elif not spec.get("levels_truncated"):
            allowed = set(spec["levels"])
            unknown = sorted({v for v in observed if v not in allowed}, key=_level_key)
            if unknown:
                add(
                    name,
                    "unknown_level",
                    f"{len(unknown)} unseen level(s), e.g. {unknown[:5]}",
                )
        if not allow_missing and n_missing:
            add(name, "unexpected_missing", f"{n_missing} missing value(s)")

    for name in cols:
        if name not in expected:
            add(name, "extra_column", "column is in the data but not in the schema")
    return violations
