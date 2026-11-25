"""Association and screening helpers: correlation, Cramer's V, VIF and leakage hints.

Pure Python, standard library only. Missing values (None or NaN) are never
imputed, dropped or recoded. Pairwise statistics report how many rows each
pair used (``n_pairs``); complete-case statistics report ``n_used``. Infinite
values are ignored by every statistic (they are neither missing nor usable).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._common import as_columns, is_infinite, is_missing, is_number

__all__ = [
    "correlation_filter",
    "correlation_matrix",
    "cramers_v_matrix",
    "leakage_hints",
    "vif",
]

_N_CATEGORY = 20
_TOL = 1e-10


def _observed(value: Any) -> bool:
    return not is_missing(value) and not is_infinite(value)


def _n_rows(cols: Mapping[str, Sequence[Any]]) -> int:
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        raise ValueError("all columns must have the same length")
    return lengths.pop() if lengths else 0


def _kind(values: Sequence[Any], n_category: int = _N_CATEGORY) -> str:
    """'categorical' for strings/bools or integer codes with <= n_category levels, else 'numeric'.

    A column with no observed values is 'numeric' (nothing to classify).
    """
    obs = [v for v in values if _observed(v)]
    if not obs:
        return "numeric"
    if not all(is_number(v) for v in obs):
        return "categorical"
    if all(float(v).is_integer() for v in obs) and len(set(obs)) <= n_category:
        return "categorical"
    return "numeric"


def _is_numeric_column(values: Sequence[Any]) -> bool:
    return all(is_number(v) for v in values if _observed(v))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Two-pass Pearson correlation; None if n < 3 or a variance is zero."""
    n = len(x)
    if n < 3 or min(x) == max(x) or min(y) == max(y):
        return None
    mx = math.fsum(x) / n
    my = math.fsum(y) / n
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    r = sxy / (math.sqrt(sxx) * math.sqrt(syy))
    return max(-1.0, min(1.0, r))


def correlation_matrix(columns: Any, method: str = "pearson") -> dict[str, Any]:
    """Pairwise correlation between numeric columns on pairwise-complete rows.

    A row counts for a pair only when both values are finite numbers. Spearman
    is Pearson on average ranks, with the ranks taken over the pair's rows.

    Args:
        columns: Dict of column name to values (or a table accepted by the data helpers).
        method: ``"pearson"`` or ``"spearman"``.

    Returns:
        dict with keys:
            names: numeric column names, in input order.
            matrix: symmetric list of lists of float or None. Correlation is None when
                the pair has fewer than 3 rows or a zero variance. The diagonal is 1.0
                when the column has at least 2 observed values and non-zero variance.
            n_pairs: symmetric list of lists of int, rows used per pair (diagonal:
                observed count of the column).
            skipped: non-numeric column names (strings or booleans), in input order.
            method: the method used.

    Raises:
        ValueError: if ``method`` is not supported or columns differ in length.

    Example:
        >>> r = correlation_matrix({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        >>> round(r["matrix"][0][1], 6), r["n_pairs"][0][1]
        (1.0, 4)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    cols = as_columns(columns)
    _n_rows(cols)
    names: list[str] = []
    skipped: list[str] = []
    for name, values in cols.items():
        (names if _is_numeric_column(values) else skipped).append(name)
    data = {nm: [float(v) if is_number(v) else None for v in cols[nm]] for nm in names}
    p = len(names)
    matrix: list[list[float | None]] = [[None] * p for _ in range(p)]
    n_pairs = [[0] * p for _ in range(p)]
    for i in range(p):
        a = data[names[i]]
        obs = [v for v in a if v is not None]
        n_pairs[i][i] = len(obs)
        if len(obs) >= 2 and min(obs) != max(obs):
            matrix[i][i] = 1.0
        for j in range(i + 1, p):
            b = data[names[j]]
            rows = [r for r in range(len(a)) if a[r] is not None and b[r] is not None]
            x = [a[r] for r in rows]
            y = [b[r] for r in rows]
            if method == "spearman":
                x = _average_ranks(x)
                y = _average_ranks(y)
            r = _pearson(x, y)
            matrix[i][j] = matrix[j][i] = r
            n_pairs[i][j] = n_pairs[j][i] = len(rows)
    return {
        "names": names,
        "matrix": matrix,
        "n_pairs": n_pairs,
        "skipped": skipped,
        "method": method,
    }


def correlation_filter(
    result: Mapping[str, Any], cutoff: float = 0.9
) -> dict[str, Any]:
    """Greedy caret ``findCorrelation`` rule on a :func:`correlation_matrix` result.

    Repeatedly takes the pair with the largest absolute correlation above
    ``cutoff`` and drops the column with the larger mean absolute correlation to
    the remaining columns (ties drop the later column in input order). Stops when
    no remaining pair exceeds the cutoff. None entries are ignored.

    Args:
        result: dict with ``names`` and a square ``matrix`` (as returned by
            :func:`correlation_matrix`).
        cutoff: absolute correlation above which a pair is considered redundant.

    Returns:
        dict with keys:
            drop: dropped names, in the order they were dropped.
            kept: remaining names, in input order.
            pairs: one dict per decision, ``{"a", "b", "corr"}`` with the two names in
                input order and the signed correlation of that pair.

    Example:
        >>> res = {"names": ["a", "b"], "matrix": [[1.0, 0.95], [0.95, 1.0]]}
        >>> correlation_filter(res, cutoff=0.9)["drop"]
        ['b']
    """
    names = list(result["names"])
    mat = result["matrix"]
    active = list(range(len(names)))
    dropped: list[int] = []
    pairs: list[dict[str, Any]] = []

    def mean_abs(k: int) -> float:
        vals = [abs(mat[k][m]) for m in active if m != k and mat[k][m] is not None]
        return math.fsum(vals) / len(vals) if vals else 0.0

    while True:
        best: tuple[int, int] | None = None
        for ai, i in enumerate(active):
            for j in active[ai + 1 :]:
                c = mat[i][j]
                if c is None or abs(c) <= cutoff:
                    continue
                if best is None or abs(c) > abs(mat[best[0]][best[1]]):
                    best = (i, j)
        if best is None:
            break
        i, j = best
        mi, mj = mean_abs(i), mean_abs(j)
        drop = i if mi > mj else j  # tie: later column (j > i) in input order
        pairs.append({"a": names[i], "b": names[j], "corr": mat[i][j]})
        active.remove(drop)
        dropped.append(drop)
    return {
        "drop": [names[k] for k in dropped],
        "kept": [names[k] for k in active],
        "pairs": pairs,
    }


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P(X > x) for df >= 1, via the regularized upper gamma."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        return 1.0 - _gamma_p_series(a, z)
    return _gamma_q_cf(a, z)


def _gamma_p_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its power series (x < a + 1)."""
    ap = a
    delta = total = 1.0 / a
    for _ in range(100000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by Lentz's continued fraction (x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _cramers_pair(
    a: Sequence[Any], b: Sequence[Any], bias_corrected: bool
) -> tuple[float | None, float | None, int]:
    """Cramer's V, chi-square p-value and row count for two categorical columns (None = missing)."""
    rows = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(rows)
    if n < 2:
        return None, None, n
    levels_r = list(dict.fromkeys(x for x, _ in rows))
    levels_c = list(dict.fromkeys(y for _, y in rows))
    r, k = len(levels_r), len(levels_c)
    if r < 2 or k < 2:
        return None, None, n
    ri = {v: i for i, v in enumerate(levels_r)}
    ci = {v: j for j, v in enumerate(levels_c)}
    table = [[0] * k for _ in range(r)]
    for x, y in rows:
        table[ri[x]][ci[y]] += 1
    row_tot = [sum(row) for row in table]
    col_tot = [sum(table[i][j] for i in range(r)) for j in range(k)]
    chi2 = math.fsum(
        (table[i][j] - row_tot[i] * col_tot[j] / n) ** 2 / (row_tot[i] * col_tot[j] / n)
        for i in range(r)
        for j in range(k)
    )
    p_value = _chi2_sf(chi2, (r - 1) * (k - 1))
    phi2 = chi2 / n
    if bias_corrected:  # Bergsma (2013)
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_c = r - (r - 1) ** 2 / (n - 1)
        k_c = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_c - 1, r_c - 1)
    else:
        denom = min(k - 1, r - 1)
    v = None if denom <= 0 else min(1.0, math.sqrt(phi2 / denom))
    return v, p_value, n


def cramers_v_matrix(
    columns: Any, bias_corrected: bool = True, n_category: int = _N_CATEGORY
) -> dict[str, Any]:
    """Cramer's V and chi-square p-values between categorical columns.

    Only categorical columns (strings or booleans, or integer codes with at most
    ``n_category`` distinct values) take part. Each pair uses the rows where both
    values are observed; the rest are excluded and counted in ``n_pairs``. Bias
    correction follows Bergsma (2013). The chi-square p-value is the uncorrected
    test of independence.

    Args:
        columns: Dict of column name to values (or a table accepted by the data helpers).
        bias_corrected: apply the Bergsma bias correction to V.
        n_category: maximum distinct integer values for a numeric column to count as categorical.

    Returns:
        dict with keys:
            names: categorical column names, in input order.
            matrix: symmetric list of lists of float or None. Diagonal is 1.0 when the
                column has at least 2 levels and 2 rows. None when a table has fewer than
                2 levels or fewer than 2 rows, or when the bias-corrected denominator is zero.
            p_values: symmetric list of lists of float or None (same None rules).
            n_pairs: symmetric list of lists of int, rows used per pair.
            skipped: columns that are not categorical, in input order.

    Example:
        >>> out = cramers_v_matrix({"x": ["a", "a", "b", "b"], "y": ["u", "u", "v", "v"]},
        ...                        bias_corrected=False)
        >>> out["matrix"][0][1], round(out["p_values"][0][1], 6)
        (1.0, 0.0455)
    """
    cols = as_columns(columns)
    _n_rows(cols)
    names: list[str] = []
    skipped: list[str] = []
    for name, values in cols.items():
        (names if _kind(values, n_category) == "categorical" else skipped).append(name)
    data = {nm: [v if _observed(v) else None for v in cols[nm]] for nm in names}
    p = len(names)
    matrix: list[list[float | None]] = [[None] * p for _ in range(p)]
    p_values: list[list[float | None]] = [[None] * p for _ in range(p)]
    n_pairs = [[0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            v, pv, n = _cramers_pair(data[names[i]], data[names[j]], bias_corrected)
            if i == j:
                v = 1.0 if pv is not None else None
            matrix[i][j] = matrix[j][i] = v
            p_values[i][j] = p_values[j][i] = pv
            n_pairs[i][j] = n_pairs[j][i] = n
    return {
        "names": names,
        "matrix": matrix,
        "p_values": p_values,
        "n_pairs": n_pairs,
        "skipped": skipped,
    }


def _forward_solve(lower: Sequence[Sequence[float]], c: Sequence[float]) -> list[float]:
    """Solve L w = c for a lower-triangular L given as rows (row i has length i + 1)."""
    w: list[float] = []
    for i, row in enumerate(lower):
        w.append((c[i] - math.fsum(row[k] * w[k] for k in range(i))) / row[i])
    return w


def vif(columns: Any) -> dict[str, Any]:
    """Variance inflation factor of each numeric column on complete cases.

    Rows with a missing or non-finite value in any numeric column are excluded
    (counted in ``n_excluded``). VIF_j = 1 / (1 - R^2_j), where R^2_j comes from
    regressing column j on all the others with an intercept. A column that is
    an exact linear combination of the others gets ``inf``. Constant columns get
    None (VIF undefined) and are named in ``warning``. Regression is done on the
    centered correlation matrix with a Cholesky factor of the columns kept so far.

    Args:
        columns: Dict of column name to values (or a table accepted by the data helpers).

    Returns:
        dict with keys:
            vif: {name: float, inf or None} for each numeric column.
            n_used: complete rows used.
            n_excluded: rows left out because a numeric column was missing or non-finite.
            names: numeric column names, in input order.
            skipped: non-numeric column names, in input order.
            warning: None, or a message when VIF could not be computed (too few
                complete rows, or constant columns).

    Example:
        >>> out = vif({"x": [1, 2, 3, 4], "y": [1, 2, 3, 5]})
        >>> out["n_used"], round(out["vif"]["x"], 6) > 1
        (4, True)
    """
    cols = as_columns(columns)
    total = _n_rows(cols)
    names: list[str] = []
    skipped: list[str] = []
    for name, values in cols.items():
        (names if _is_numeric_column(values) else skipped).append(name)
    num = {nm: [float(v) if is_number(v) else None for v in cols[nm]] for nm in names}
    complete = [r for r in range(total) if all(num[nm][r] is not None for nm in names)]
    n_used = len(complete)
    p = len(names)
    out: dict[str, Any] = {
        "vif": {nm: None for nm in names},
        "n_used": n_used,
        "n_excluded": total - n_used,
        "names": names,
        "skipped": skipped,
        "warning": None,
    }
    if n_used < p + 1:
        out["warning"] = (
            f"only {n_used} complete rows for {p} numeric columns; "
            f"need at least {p + 1}; VIF not computed"
        )
        return out

    live: list[str] = []
    constant: list[str] = []
    centered: dict[str, list[float]] = {}
    for nm in names:
        x = [num[nm][r] for r in complete]
        if min(x) == max(x):
            constant.append(nm)
            continue
        m = math.fsum(x) / n_used
        centered[nm] = [v - m for v in x]
        live.append(nm)
    if constant:
        out["warning"] = f"constant column(s) {constant}: VIF undefined (None)"

    q = len(live)
    ss = {nm: math.fsum(c * c for c in centered[nm]) for nm in live}
    corr = [[1.0] * q for _ in range(q)]
    for i in range(q):
        for k in range(i + 1, q):
            ci, ck = centered[live[i]], centered[live[k]]
            r = math.fsum(a * b for a, b in zip(ci, ck)) / (
                math.sqrt(ss[live[i]]) * math.sqrt(ss[live[k]])
            )
            corr[i][k] = corr[k][i] = max(-1.0, min(1.0, r))

    def vif_of(j: int) -> float:
        # Greedily build a Cholesky factor of the other columns that are not
        # linearly dependent on the ones already kept; the dependent ones add nothing.
        basis: list[int] = []
        lower: list[list[float]] = []
        for k in range(q):
            if k == j:
                continue
            w = _forward_solve(lower, [corr[b][k] for b in basis])
            d = 1.0 - math.fsum(t * t for t in w)
            if d > _TOL:
                lower.append(w + [math.sqrt(d)])
                basis.append(k)
        z = _forward_solve(lower, [corr[b][j] for b in basis])
        resid = 1.0 - math.fsum(t * t for t in z)
        if resid <= _TOL:
            return math.inf
        return 1.0 / resid

    for idx, nm in enumerate(live):
        out["vif"][nm] = vif_of(idx)
    return out


def _auc_folded(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann-Whitney AUC with mean ranks for ties, folded to max(auc, 1 - auc)."""
    n1 = sum(1 for lab in labels if lab)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = _average_ranks([float(s) for s in scores])
    rank_sum = math.fsum(rk for rk, lab in zip(ranks, labels) if lab)
    auc = (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1.0 - auc)


def _eta2(groups_of: Sequence[Any], y: Sequence[float]) -> float | None:
    """Between-group share of the variance of y (eta squared)."""
    if min(y) == max(y):
        return None
    groups: dict[Any, list[float]] = {}
    for g, v in zip(groups_of, y):
        groups.setdefault(g, []).append(v)
    grand = math.fsum(y) / len(y)
    ss_tot = math.fsum((v - grand) ** 2 for v in y)
    ss_b = math.fsum(
        len(vs) * (math.fsum(vs) / len(vs) - grand) ** 2 for vs in groups.values()
    )
    return ss_b / ss_tot


def _leak_score(
    x: Sequence[Any],
    t: Sequence[Any],
    kind: str,
    binary: bool,
    positive: Any,
) -> tuple[float | None, str | None]:
    if not x:
        return None, None
    if binary:
        labels = [v == positive for v in t]
        if kind == "numeric":
            return _auc_folded(x, labels), "auc"
        events: dict[Any, list[int]] = {}
        for g, lab in zip(x, labels):
            cell = events.setdefault(g, [0, 0])
            cell[0] += int(lab)
            cell[1] += 1
        rates = [events[g][0] / events[g][1] for g in x]
        return _auc_folded(rates, labels), "auc_level_rate"
    yv = [float(v) for v in t]
    if kind == "numeric":
        r = _pearson([float(v) for v in x], yv)
        return (None if r is None else r * r), "r2"
    return _eta2(x, yv), "eta2"


def leakage_hints(
    columns: Any, y: Sequence[Any], threshold: float = 0.95
) -> list[dict[str, Any]]:
    """Single-feature predictive score per column, flagged when at or above a threshold.

    Scores by feature kind and target type:
        numeric feature, binary y: AUC by ranks, folded to max(auc, 1 - auc), metric ``"auc"``.
        numeric feature, numeric y: squared Pearson correlation, metric ``"r2"``.
        categorical feature, binary y: AUC of the in-sample level event rates, folded;
            metric ``"auc_level_rate"``. In-sample rates overstate the score for
            high-cardinality columns (no leave-one-out).
        categorical feature, numeric y: eta squared (between-group share of variance),
            metric ``"eta2"``.

    Binary y means exactly two distinct observed values. Rows with a missing or
    non-finite feature or y are excluded and counted in ``n_used``. Nothing is
    removed; this only flags.

    Args:
        columns: Dict of column name to values (or a table accepted by the data helpers).
        y: Target values, one per row.
        threshold: score at or above which a column is flagged.

    Returns:
        list of dicts, one per column in input order, with keys:
            name: column name.
            score: float, or None when it cannot be computed (constant y, no rows,
                one class among the used rows, or zero variance).
            metric: ``"auc"``, ``"auc_level_rate"``, ``"r2"``, ``"eta2"``, or None.
            flagged: True when score is not None and score >= threshold.
            n_used: rows used (feature and y both observed).

    Raises:
        ValueError: if y has more than two distinct values and is not numeric, or
            its length differs from the columns.

    Example:
        >>> hints = leakage_hints({"f": [1, 2, 3, 4, 5, 6]}, [0, 0, 0, 1, 1, 1])
        >>> hints[0]["score"], hints[0]["flagged"]
        (1.0, True)
    """
    cols = as_columns(columns)
    total = _n_rows(cols)
    y_list = list(y)
    if len(y_list) != total:
        raise ValueError("y must have one value per row of the columns")
    yuse = [v if _observed(v) else None for v in y_list]
    y_levels = list(dict.fromkeys(v for v in yuse if v is not None))
    if len(y_levels) > 2 and not all(is_number(v) for v in y_levels):
        raise ValueError("y must be binary or numeric")
    binary = len(y_levels) == 2
    positive = y_levels[1] if binary else None
    degenerate = len(y_levels) < 2

    out: list[dict[str, Any]] = []
    for name, values in cols.items():
        feat = [v if _observed(v) else None for v in values]
        rows = [r for r in range(total) if feat[r] is not None and yuse[r] is not None]
        x = [feat[r] for r in rows]
        t = [yuse[r] for r in rows]
        if degenerate:
            score, metric = None, None
        else:
            score, metric = _leak_score(x, t, _kind(values), binary, positive)
        out.append(
            {
                "name": name,
                "score": score,
                "metric": metric,
                "flagged": score is not None and score >= threshold,
                "n_used": len(rows),
            }
        )
    return out
