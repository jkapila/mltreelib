"""Distribution helpers and backwards-compatible tree imports."""

from collections.abc import Sequence

from mltreelib.anomaly import AnomalyTree
from mltreelib.cluster import ClusterTree

__all__ = [
    "AnomalyTree",
    "ClusterTree",
    "distributions_to_quantiles",
    "histograms_to_quantiles",
]


def histograms_to_quantiles(
    histograms: Sequence[Sequence[float]],
    bin_edges: Sequence[float] | None = None,
    n_quantiles: int = 50,
) -> list[list[float]]:
    """Convert probability histograms into quantile vectors for Wasserstein-2 optimal transport trees.

    Args:
        histograms: 2D array-like of shape (N, B) containing probability masses (sums to ~1.0).
        bin_edges: Optional (B + 1) boundaries of the bins. If None, defaults to uniform [0, 1, ..., B].
        n_quantiles: Number of quantile evaluation points in (0, 1). Default 50.

    Returns:
        2D list of shape (N, n_quantiles) containing the quantile function values.
    """
    out: list[list[float]] = []
    tau_grid = [(k + 0.5) / n_quantiles for k in range(n_quantiles)]
    for row in histograms:
        b = len(row)
        edges = list(bin_edges) if bin_edges is not None else [float(i) for i in range(b + 1)]
        total = sum(row)
        if total <= 0:
            out.append([0.0] * n_quantiles)
            continue
        cdf = [0.0] * (b + 1)
        running = 0.0
        for i in range(b):
            running += row[i] / total
            cdf[i + 1] = running
        cdf[b] = 1.0

        q_row: list[float] = []
        for tau in tau_grid:
            idx = 0
            while idx < b and cdf[idx + 1] < tau:
                idx += 1
            if idx >= b:
                q_row.append(edges[-1])
            else:
                p_low = cdf[idx]
                p_high = cdf[idx + 1]
                e_low = edges[idx]
                e_high = edges[idx + 1]
                if p_high - p_low > 1e-12:
                    val = e_low + (tau - p_low) / (p_high - p_low) * (e_high - e_low)
                else:
                    val = (e_low + e_high) / 2.0
                q_row.append(val)
        out.append(q_row)
    return out


def distributions_to_quantiles(
    samples: Sequence[Sequence[float]],
    n_quantiles: int = 50,
) -> list[list[float]]:
    """Convert empirical sample distributions into quantile vectors for Wasserstein-2 trees.

    Args:
        samples: 2D array-like of shape (N, M) where each row contains observations from a distribution.
        n_quantiles: Number of quantile evaluation points. Default 50.

    Returns:
        2D list of shape (N, n_quantiles) containing the empirical quantile values.
    """
    out: list[list[float]] = []
    tau_grid = [(k + 0.5) / n_quantiles for k in range(n_quantiles)]
    for row in samples:
        sorted_row = sorted([float(x) for x in row if x is not None])
        m = len(sorted_row)
        if m == 0:
            out.append([0.0] * n_quantiles)
            continue
        q_row: list[float] = []
        for tau in tau_grid:
            pos = tau * (m - 1)
            low = int(pos)
            high = min(low + 1, m - 1)
            frac = pos - low
            val = (1.0 - frac) * sorted_row[low] + frac * sorted_row[high]
            q_row.append(val)
        out.append(q_row)
    return out
