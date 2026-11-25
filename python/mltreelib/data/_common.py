"""Shared primitives for the data helpers (standard library only).

Missing means ``None`` or a float NaN. Infinities are neither missing nor
usable: helpers report them as ``n_inf`` and ignore them in statistics.
Nothing here modifies its input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def is_missing(value: Any) -> bool:
    """True for None and float NaN (and the pandas NA/NaT singletons)."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return type(value).__name__ in ("NAType", "NaTType")


def is_infinite(value: Any) -> bool:
    return isinstance(value, float) and math.isinf(value)


def is_number(value: Any) -> bool:
    """A finite real number (bool excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_observed(values: Sequence[Any]) -> list[float]:
    """Finite numeric values, in order, as floats (missing and infinities skipped)."""
    return [float(v) for v in values if is_number(v)]


def as_columns(table: Any) -> dict[str, list[Any]]:
    """Return a dict of column name to list of values for a dict or table-like input."""
    if isinstance(table, Mapping):
        if "columns" in table and isinstance(table["columns"], Mapping):  # load_dataset result
            table = table["columns"]
        return {str(k): list(v) for k, v in table.items()}
    from mltreelib._tables import to_rows

    rows, names = to_rows(table)
    width = len(rows[0]) if rows else len(names or [])
    names = names or [f"x{j}" for j in range(width)]
    return {name: [row[j] for row in rows] for j, name in enumerate(names)}


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (R type 7) of an already sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return math.nan
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
