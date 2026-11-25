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


def to_rows(x: Any) -> tuple[list[list[Any]], list[str] | None]:
    """Return ``(rows, column_names)``; names are ``None`` when the input has none."""
    predict_args = getattr(x, "predict_args", None)
    if callable(predict_args):  # a Dataset: its feature rows (and names) only
        args = predict_args()
        return args["x"], args.get("col_names")
    if isinstance(x, (list, tuple)):  # already rows: no copy, no scan
        return x, None  # type: ignore[return-value]
    cols = _columns(x)
    if cols is None and callable(getattr(x, "tolist", None)):  # NumPy array
        return x.tolist(), None
    if cols is not None:
        names, data = cols
        n = len(data[0]) if data else 0
        return [[_cell(c[i]) for c in data] for i in range(n)], names
    return [[_cell(v) for v in row] for row in x], None
