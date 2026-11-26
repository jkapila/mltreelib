"""Duck-typed table input for models.

Models accept a list of rows, a NumPy array, a dict of columns, a pandas or
Polars-style data frame, or an Arrow table. Detection is by behaviour, so none
of those packages is imported. Missing values are passed through as they are
(NaN, None, null); the model routes them. Only the pandas ``NA`` and ``NaT``
singletons, which cannot be converted to a number, are turned into ``None`` at
this input boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _cell(value: Any) -> Any:
    if type(value).__name__ in ("NAType", "NaTType"):
        return None
    return value


def _columns(x: Any) -> tuple[list[str], list[list[Any]]] | None:
    if isinstance(x, Mapping):
        names = [str(k) for k in x]
        return names, [list(x[k]) for k in x]
    if callable(getattr(x, "to_pydict", None)):  # Arrow table
        d = x.to_pydict()
        return [str(k) for k in d], [list(v) for v in d.values()]
    cols = getattr(x, "columns", None)
    if cols is not None and hasattr(x, "__getitem__"):  # pandas / Polars data frame
        names = [str(c) for c in cols]
        return names, [list(x[c]) for c in cols]
    return None


def to_target_list(y: Any) -> list[Any]:
    """Convert a 1-D sequence, NumPy array, Polars Series, or Arrow Array to Python scalars."""
    if callable(getattr(y, "to_pylist", None)):
        return [_cell(v) for v in y.to_pylist()]
    if callable(getattr(y, "to_list", None)):
        return [_cell(v) for v in y.to_list()]
    if callable(getattr(y, "tolist", None)):
        vals = y.tolist()
        if isinstance(vals, list):
            return [_cell(v) for v in vals]
    return [_cell(v) for v in y]


def to_rows(x: Any) -> tuple[list[list[Any]], list[str] | None]:
    """Return ``(rows, column_names)``; names are ``None`` when the input has none."""
    predict_args = getattr(x, "predict_args", None)
    if callable(predict_args):  # a Dataset: its feature rows (and names) only
        args = predict_args()
        return args["x"], args.get("col_names")
    if isinstance(x, (list, tuple)):  # already rows: no copy, no scan
        if x and not isinstance(x[0], (list, tuple)):
            raise ValueError("X must be 2-D (rows, columns); got a 1-D sequence.")
        return x, None  # type: ignore[return-value]
    cols = _columns(x)
    if cols is None and callable(getattr(x, "tolist", None)):  # NumPy array
        if getattr(x, "ndim", None) == 1:
            raise ValueError(
                "X must be 2-D (rows, columns); got a 1-D array. Use X.reshape(-1, 1) for one feature."
            )
        return x.tolist(), None
    if cols is not None:
        names, data = cols
        lengths = {n: len(col) for n, col in zip(names, data)}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"all columns must have the same length; got {lengths}")
        n = len(data[0]) if data else 0
        return [[_cell(c[i]) for c in data] for i in range(n)], names
    return [[_cell(v) for v in row] for row in x], None
