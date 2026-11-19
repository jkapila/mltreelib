"""Unified Anomaly and Clustering Tree Models.

Provides unified, scikit-learn-compatible facades:
- ``AnomalyTree``: Single interface for density estimation, extended isolation,
  half-space, and random-cut anomaly detection trees.
- ``ClusterTree``: Single interface for predictive clustering trees (PCT),
  CLTree, HDBSCAN, and Unsupervised Random Forest (URF) clustering.
"""

from collections.abc import Sequence
from typing import Any

from mltreelib._core import (
    HDBSCAN as _HDBSCAN,
)
from mltreelib._core import (
    ActiveAnomalyTree,
    ContextAnomalyTree,
)
from mltreelib._core import (
    CLTree as _CLTree,
)
from mltreelib._core import (
    DensityEstimationTree as _DET,
)
from mltreelib._core import (
    ExtendedIsolationTree as _EIF,
)
from mltreelib._core import (
    HalfSpaceTree as _HST,
)
from mltreelib._core import (
    PredictiveClusteringTree as _PCT,
)
from mltreelib._core import (
    RandomCutTree as _RCT,
)
from mltreelib._core import (
    UnsupervisedRandomForest as _URF,
)


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
                if v is not None and not (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
                    try:
                        sums[j] += float(v)
                        counts[j] += 1
                    except (ValueError, TypeError):
                        pass
        col_means = [sums[j] / counts[j] if counts[j] > 0 else 0.0 for j in range(n_cols)]
    out = []
    for row in x:
        r = []
        for j, v in enumerate(row):
            if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
                r.append(col_means[j])
            else:
                try:
                    r.append(float(v))
                except (ValueError, TypeError):
                    r.append(col_means[j])
        out.append(r)
    return out, col_means


class AnomalyTree:
    """Unified Tree-Based Anomaly Detector.

    Wraps multiple tree-based anomaly engines under a consistent API:
    - ``"eif"`` or ``"isolation"``: Extended Isolation Tree (oblique hyperplanes, mixed types).
    - ``"det"`` or ``"density"``: Density Estimation Tree (piecewise constant density, mixed types).
    - ``"hst"`` or ``"half_space"``: Streaming Half-Space Tree (mass-based scoring).
    - ``"rct"`` or ``"random_cut"``: Random Cut Tree (co-displacement scoring).
    - ``"active"``: Active Anomaly Tree with expert feedback adaptation.
    - ``"context"``: Variable-Order Context Anomaly Tree for discrete sequences.

    Parameters
    ----------
    kind : str, default="eif"
        Anomaly tree engine to use: ``"eif"``, ``"det"``, ``"hst"``, ``"rct"``, ``"active"``, or ``"context"``.
    max_depth : int, optional
        Maximum tree depth.
    min_rows : int, default=10
        Minimum leaf samples.
    contamination : float, default=0.05
        Expected proportion of anomalies in the dataset (used to calibrate binary predictions).
    seed : int, default=42
        Random seed for stochastic engines (EIF, HST, RCT, Active).
    **params
        Engine-specific hyperparameters (e.g. ``extension_level`` for EIF,
        ``window_size`` for HST, ``sample_size`` for RCT, ``eta`` for Active).
    """

    def __init__(
        self,
        kind: str = "eif",
        max_depth: int | None = None,
        min_rows: int = 10,
        contamination: float = 0.05,
        seed: int = 42,
        **params: Any,
    ):
        self.kind = kind.lower().strip()
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.contamination = contamination
        self.seed = seed
        self.params = params
        self._threshold: float | None = None
        self._threshold_strict: bool = False
        self._fitted = False
        self._col_means: list[float] | None = None
        self._inner: Any = None

        self._init_inner()

    def _init_inner(self):
        k = self.kind
        if k in ("eif", "isolation", "extended_isolation"):
            ext = self.params.get("extension_level", None)
            sub = self.params.get("subsample_size", None)
            self._inner = _EIF(
                max_depth=self.max_depth,
                extension_level=ext,
                subsample_size=sub,
                seed=self.seed,
            )
        elif k in ("det", "density", "density_estimation"):
            depth = self.max_depth or 8
            min_split = self.params.get("min_samples_split", self.min_rows * 2)
            alpha = self.params.get("alpha", 0.0)
            min_gain = self.params.get("min_density_gain", 0.0)
            self._inner = _DET(
                max_depth=depth,
                min_samples_split=min_split,
                min_leaf_samples=self.min_rows,
                min_density_gain=min_gain,
                alpha=alpha,
            )
        elif k in ("hst", "half_space"):
            depth = self.max_depth or 6
            wsize = self.params.get("window_size", 250)
            init_mass = self.params.get("initial_mass", 0.0)
            self._inner = _HST(
                max_depth=depth,
                window_size=wsize,
                initial_mass=init_mass,
                seed=self.seed,
            )
        elif k in ("rct", "random_cut"):
            ssize = self.params.get("sample_size", 256)
            self._inner = _RCT(sample_size=ssize, seed=self.seed)
        elif k in ("active", "aad"):
            depth = self.max_depth or 8
            eta = self.params.get("eta", 0.1)
            pw = self.params.get("prior_weight", 1.0)
            self._inner = ActiveAnomalyTree(
                max_depth=depth, eta=eta, prior_weight=pw, seed=self.seed
            )
        elif k in ("context", "sequence"):
            depth = self.max_depth or 5
            asize = self.params.get("alphabet_size", 256)
            smooth = self.params.get("smoothing_discount", 0.5)
            self._inner = ContextAnomalyTree(
                max_depth=depth, alphabet_size=asize, smoothing_discount=smooth
            )
        else:
            valid = ["eif", "det", "hst", "rct", "active", "context"]
            raise ValueError(
                f"Unknown anomaly tree kind '{self.kind}', expected one of {valid}"
            )

    def fit(
        self,
        x: Sequence[Sequence[Any]],
        col_names: Sequence[str] | None = None,
        dtypes: Sequence[str] | None = None,
    ) -> "AnomalyTree":
        """Fit the anomaly model on training data.

        Accepts native python lists (floats, ints, strings, None for missing values).
        """
        k = self.kind
        if k in (
            "eif",
            "isolation",
            "extended_isolation",
            "det",
            "density",
            "density_estimation",
        ):
            cnames = list(col_names) if col_names is not None else None
            dts = list(dtypes) if dtypes is not None else None
            self._inner.fit(x, col_names=cnames, dtypes=dts)
        elif k in ("context", "sequence"):
            # Expects sequence of integers
            self._inner.fit(x)
        else:
            # Float matrix algorithms (HST, RCT, Active)
            float_mat, self._col_means = _impute_float_matrix(x)
            self._inner.fit(float_mat)

        self._fitted = True

        # Calibrate threshold on training data if contamination > 0
        if self.contamination > 0.0 and k not in ("context", "sequence"):
            scores = self.score_samples(x)
            sorted_scores = sorted(scores)
            # Find score cutoff strictly for the top contamination fraction
            n_anom = max(1, round(self.contamination * len(sorted_scores)))
            # Index of first anomaly from the tail
            idx = max(0, len(sorted_scores) - n_anom)
            self._threshold = sorted_scores[idx]
            # Deterministic tie behavior: if threshold ties with inliers below idx,
            # use strict inequality to prevent large inlier clumps from falsely becoming anomalies.
            self._threshold_strict = idx > 0 and sorted_scores[idx] == sorted_scores[idx - 1]

        return self

    def score_samples(self, x: Sequence[Sequence[Any]]) -> list[float]:
        """Compute continuous anomaly scores for data points.

        Higher scores indicate greater degree of anomalousness.
        """
        if not self._fitted:
            raise ValueError("AnomalyTree is not fitted yet.")

        k = self.kind
        if k in ("eif", "isolation", "extended_isolation"):
            return self._inner.anomaly_score(x)
        elif k in ("det", "density", "density_estimation"):
            return self._inner.predict_anomaly_score(x)
        elif k in ("hst", "half_space"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            return self._inner.anomaly_score(float_mat)
        elif k in ("rct", "random_cut"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            return self._inner.codisp(float_mat)
        elif k in ("active", "aad"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            return self._inner.anomaly_score(float_mat)
        elif k in ("context", "sequence"):
            raise ValueError(
                "Use surprise_score(context, symbol) for ContextAnomalyTree"
            )
        return []

    def predict(self, x: Sequence[Sequence[Any]]) -> list[int]:
        """Predict binary anomaly labels (1 = anomaly, 0 = inlier / normal)."""
        scores = self.score_samples(x)
        if self._threshold is not None:
            if getattr(self, "_threshold_strict", False):
                return [1 if s > self._threshold else 0 for s in scores]
            return [1 if s >= self._threshold else 0 for s in scores]
        return [1 if s >= 0.5 else 0 for s in scores]

    def predict_density(self, x: Sequence[Sequence[Any]]) -> list[float]:
        """Compute estimated probability density (only for density trees)."""
        if not hasattr(self._inner, "predict_density"):
            raise ValueError(
                f"predict_density is not supported by {self.kind} anomaly tree"
            )
        return self._inner.predict_density(x)

    def surprise_score(self, context: Sequence[int], symbol: int) -> float:
        """Compute surprise score -log2 P(symbol | context) for sequence context tree."""
        if not hasattr(self._inner, "surprise_score"):
            raise ValueError(
                f"surprise_score is only supported by 'context' anomaly tree, got kind='{self.kind}'"
            )
        return self._inner.surprise_score(list(context), symbol)

    def conditional_prob(self, context: Sequence[int], symbol: int) -> float:
        """Compute conditional probability P(symbol | context) for sequence context tree."""
        if not hasattr(self._inner, "conditional_prob"):
            raise ValueError(
                f"conditional_prob is only supported by 'context' anomaly tree, got kind='{self.kind}'"
            )
        return self._inner.conditional_prob(list(context), symbol)

    def update_feedback(
        self, point: Sequence[float], is_anomaly: bool, eta: float = 0.5
    ) -> "AnomalyTree":
        """Update model weights based on expert feedback (for Active anomaly tree)."""
        if not hasattr(self._inner, "update_feedback"):
            raise ValueError(
                f"update_feedback is only supported by 'active' anomaly tree, got kind='{self.kind}'"
            )
        pt = [float(v) for v in point]
        self._inner.update_feedback(pt, is_anomaly, eta)
        return self

    def feedback(
        self, point: Sequence[float], is_anomaly: bool, eta: float = 0.5
    ) -> "AnomalyTree":
        """Alias for update_feedback."""
        return self.update_feedback(point, is_anomaly, eta)

    @property
    def threshold(self) -> float | None:
        """Contamination threshold score separating normal points from anomalies."""
        return self._threshold

    def __repr__(self) -> str:
        return f"AnomalyTree(kind='{self.kind}', max_depth={self.max_depth}, min_rows={self.min_rows}, contamination={self.contamination})"


class ClusterTree:
    """Unified Tree-Based Clustering Model.

    Wraps divisive and density-based tree clustering engines under a consistent API:
    - ``"pct"`` or ``"predictive"``: Predictive Clustering Trees (unsupervised or multi-target supervised, rules).
    - ``"cltree"`` or ``"density"``: CLTree (density-based spatial clustering tree with bounding boxes).
    - ``"hdbscan"`` or ``"hierarchical"``: HDBSCAN mutual reachability MST hierarchy.
    - ``"urf"`` or ``"proximity"``: Unsupervised Random Forest proximity-based clustering.

    Parameters
    ----------
    kind : str, default="pct"
        Clustering engine: ``"pct"``, ``"cltree"``, ``"hdbscan"``, or ``"urf"``.
    n_clusters : int, optional
        Target cluster count (required for URF, auto-detected in HDBSCAN / CLTree).
    max_depth : int, default=6
        Maximum tree depth for divisive clustering trees.
    min_rows : int, default=5
        Minimum leaf samples / cluster size.
    metric : str, default="euclidean"
        Distance metric (for HDBSCAN: ``"euclidean"``, ``"manhattan"``, ``"chebyshev"``).
    seed : int, default=42
        Random seed for stochastic clustering algorithms (URF).
    **params
        Engine-specific hyperparameters (e.g. ``target_weights`` for PCT,
        ``min_density`` for CLTree, ``cluster_selection_method`` for HDBSCAN, ``linkage`` for URF).
    """

    def __init__(
        self,
        kind: str = "pct",
        n_clusters: int | None = None,
        max_depth: int = 6,
        min_rows: int = 5,
        metric: str = "euclidean",
        seed: int = 42,
        **params: Any,
    ):
        self.kind = kind.lower().strip()
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.metric = metric
        self.seed = seed
        self.params = params
        self._fitted = False
        self._inner: Any = None
        self._col_means: list[float] | None = None
        self._labels: list[int | None] | None = None

        self._init_inner()

    def _init_inner(self):
        k = self.kind
        if k in ("pct", "predictive", "clus"):
            min_split = self.params.get("min_samples_split", self.min_rows * 2)
            min_vr = self.params.get("min_variance_reduction", 1e-4)
            tw = self.params.get("target_weights", None)
            self._inner = _PCT(
                max_depth=self.max_depth,
                min_samples_split=min_split,
                min_samples_leaf=self.min_rows,
                min_variance_reduction=min_vr,
                target_weights=tw,
            )
        elif k in ("cltree", "density"):
            min_density = self.params.get("min_density", 0.6)
            self._inner = _CLTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_rows,
                min_density=min_density,
            )
        elif k in ("hdbscan", "hierarchical"):
            eps = self.params.get("cluster_selection_epsilon", 0.0)
            max_c = self.params.get("max_cluster_size", None)
            csm = self.params.get("cluster_selection_method", "eom")
            single = self.params.get("allow_single_cluster", False)
            self._inner = _HDBSCAN(
                min_cluster_size=self.min_rows,
                min_samples=self.params.get("min_samples", self.min_rows),
                cluster_selection_epsilon=eps,
                max_cluster_size=max_c,
                metric=self.metric,
                cluster_selection_method=csm,
                allow_single_cluster=single,
            )
        elif k in ("urf", "proximity", "forest"):
            ntrees = self.params.get("n_trees", 50)
            mfeat = self.params.get("max_features", None)
            linkage = self.params.get("linkage", "average")
            self._inner = _URF(
                n_trees=ntrees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_rows,
                max_features=mfeat,
                linkage=linkage,
                seed=self.seed,
            )
        else:
            valid = ["pct", "cltree", "hdbscan", "urf"]
            raise ValueError(
                f"Unknown cluster tree kind '{self.kind}', expected one of {valid}"
            )

    def fit(
        self,
        x: Sequence[Sequence[Any]],
        y: Sequence[Sequence[float]] | None = None,
        col_names: Sequence[str] | None = None,
        dtypes: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> "ClusterTree":
        """Fit the clustering tree on dataset x (and optional targets y for PCT)."""
        k = self.kind
        if k in ("pct", "predictive", "clus"):
            cnames = list(col_names) if col_names is not None else None
            dts = list(dtypes) if dtypes is not None else None
            tnames = list(target_names) if target_names is not None else None
            y_mat = [list(r) for r in y] if y is not None else None
            self._inner.fit(
                x, y=y_mat, col_names=cnames, dtypes=dts, target_names=tnames
            )
            self._labels = self._inner.predict_cluster(x)
        elif k in ("cltree", "density"):
            float_mat, self._col_means = _impute_float_matrix(x)
            cnames = list(col_names) if col_names is not None else None
            self._inner.fit(float_mat, col_names=cnames)
            self._labels = self._inner.predict_cluster(float_mat)
        elif k in ("hdbscan", "hierarchical"):
            float_mat, self._col_means = _impute_float_matrix(x)
            self._inner.fit(float_mat)
            self._labels = self._inner.labels
        elif k in ("urf", "proximity", "forest"):
            float_mat, self._col_means = _impute_float_matrix(x)
            n_c = self.n_clusters if self.n_clusters is not None else 2
            self._inner.fit(float_mat, n_clusters=n_c)
            self._labels = self._inner.labels
            self._train_encodings = self._inner.apply(float_mat)

        self._fitted = True
        return self

    def predict(self, x: Sequence[Sequence[Any]]) -> list[int | None]:
        """Predict cluster index for query points x."""
        if not self._fitted:
            raise ValueError("ClusterTree is not fitted yet.")

        k = self.kind
        if k in ("pct", "predictive", "clus"):
            return self._inner.predict_cluster(x)
        elif k in ("cltree", "density"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            return self._inner.predict_cluster(float_mat)
        elif k in ("hdbscan", "hierarchical"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            return self._inner.predict(float_mat)
        elif k in ("urf", "proximity", "forest"):
            float_mat, _ = _impute_float_matrix(x, self._col_means)
            if hasattr(self, "_train_encodings") and self._labels:
                query_enc = self._inner.apply(float_mat)
                preds: list[int | None] = []
                for q in query_enc:
                    best_match = 0
                    best_prox = -1
                    for idx, t_enc in enumerate(self._train_encodings):
                        matches = sum(1 for a, b in zip(q, t_enc) if a == b)
                        if matches > best_prox:
                            best_prox = matches
                            best_match = idx
                    preds.append(self._labels[best_match])
                return preds
            if len(float_mat) == len(self._labels or []):
                return self._labels or []
            return [0] * len(float_mat)
        return []

    @property
    def labels_(self) -> list[int | None]:
        """Cluster labels for each training sample (-1 / None = noise)."""
        if not self._fitted:
            raise ValueError("ClusterTree is not fitted yet.")
        return self._labels or []

    @property
    def n_clusters_(self) -> int:
        """Discovered or configured number of clusters."""
        if hasattr(self._inner, "n_clusters"):
            return self._inner.n_clusters
        if self._labels:
            valid = {c for c in self._labels if c is not None and c >= 0}
            return len(valid)
        return self.n_clusters or 0

    def rules(self) -> list[dict[str, Any]]:
        """Extract interpretable cluster rule descriptions (PCT / CLTree)."""
        if hasattr(self._inner, "cluster_rules"):
            raw = self._inner.cluster_rules()
            return [
                {"cluster_id": r[0], "rule": r[1], "prototype": r[2], "size": r[3]}
                for r in raw
            ]
        if hasattr(self._inner, "cluster_bounding_boxes"):
            raw = self._inner.cluster_bounding_boxes()
            return [
                {
                    "cluster_id": r[0],
                    "bounding_box": r[1],
                    "size": r[2],
                    "density": r[3],
                }
                for r in raw
            ]
        return []

    def __repr__(self) -> str:
        return f"ClusterTree(kind='{self.kind}', n_clusters={self.n_clusters}, max_depth={self.max_depth}, min_rows={self.min_rows})"


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
