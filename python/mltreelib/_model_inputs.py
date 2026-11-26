"""Input conversion shared by clustering and anomaly models."""

from math import isfinite
from typing import Any

from mltreelib._tables import to_rows


def _numeric_rows(x: Any) -> list[list[float | None]]:
    """Convert numeric inputs while preserving missing cells.

    Accepts anything `mltreelib._tables.to_rows` understands (rows, NumPy,
    dict of columns, data frame, Arrow table).
    """
    rows, _ = to_rows(x)
    return [[_numeric_cell(value) for value in row] for row in rows]


def _numeric_cell(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if isfinite(result) else None
