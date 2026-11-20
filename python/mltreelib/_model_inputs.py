"""Input conversion shared by clustering and anomaly models."""

from collections.abc import Sequence
from math import isfinite
from typing import Any


def _impute_float_matrix(
    x: Sequence[Sequence[Any]], col_means: list[float] | None = None
) -> tuple[list[list[float]], list[float]]:
    """Convert sequence of rows to float matrix with mean imputation for missing values."""
    if not x:
        return [], []
    n_cols = len(x[0])
    if col_means is None:
        sums = [0.0] * n_cols
        counts = [0] * n_cols
        for row in x:
            for j, v in enumerate(row):
                if v is not None and not (isinstance(v, float) and not isfinite(v)):
                    try:
                        sums[j] += float(v)
                        counts[j] += 1
                    except (ValueError, TypeError):
                        pass
        col_means = [
            sums[j] / counts[j] if counts[j] > 0 else 0.0 for j in range(n_cols)
        ]
    out = []
    for row in x:
        r = []
        for j, v in enumerate(row):
            if v is None or (isinstance(v, float) and not isfinite(v)):
                r.append(col_means[j])
            else:
                try:
                    r.append(float(v))
                except (ValueError, TypeError):
                    r.append(col_means[j])
        out.append(r)
    return out, col_means
