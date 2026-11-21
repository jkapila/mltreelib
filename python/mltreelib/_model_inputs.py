"""Input conversion shared by clustering and anomaly models."""

from collections.abc import Sequence
from math import isfinite
from typing import Any


def _numeric_rows(x: Sequence[Sequence[Any]]) -> list[list[float | None]]:
    """Convert numeric inputs while preserving missing cells."""
    return [[_numeric_cell(value) for value in row] for row in x]


def _numeric_cell(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if isfinite(result) else None
