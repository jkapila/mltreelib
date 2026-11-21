"""Tree-based clustering."""

from collections.abc import Sequence
from typing import Any, Literal, overload

from mltreelib._core import HDBSCAN as _HDBSCAN
from mltreelib._core import CLTree as _CLTree
from mltreelib._core import PredictiveClusteringTree as _PCT
from mltreelib._core import UnsupervisedRandomForest as _URF
from mltreelib._model_inputs import _impute_float_matrix
from mltreelib._unsupervised_snapshot import _UnsupervisedSnapshot

__all__ = ["ClusterTree"]


class ClusterTree(_UnsupervisedSnapshot):
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
    min_samples_split : int, optional
        PCT minimum samples per split. Defaults to twice ``min_rows``.
    min_variance_reduction : float, default=1e-4
        PCT minimum reduction in target variance.
    target_weights : sequence of float, optional
        Per-target weights for PCT.
    min_density : float, default=0.6
        CLTree minimum cluster density.
    min_samples : int, optional
        HDBSCAN neighbor count for core distances. Defaults to ``min_rows``.
    cluster_selection_epsilon : float, default=0
        HDBSCAN distance threshold for merging clusters.
    max_cluster_size : int, optional
        HDBSCAN upper limit on cluster size.
    cluster_selection_method : str, default="eom"
        HDBSCAN cluster selection method, ``"eom"`` or ``"leaf"``.
    allow_single_cluster : bool, default=False
        Allow HDBSCAN to select a single cluster.
    n_trees : int, default=50
        Number of trees in the URF proximity forest.
    max_features : int, optional
        Number of URF features to consider per split.
    linkage : str, default="average"
        URF linkage method, ``"average"``, ``"single"``, or ``"complete"``.
    """

    _snapshot_format = "mltreelib.cluster_tree"
    _snapshot_engines = ("pct", "predictive", "clus")

    def __init__(
        self,
        kind: str = "pct",
        n_clusters: int | None = None,
        max_depth: int = 6,
        min_rows: int = 5,
        metric: str = "euclidean",
        seed: int = 42,
        *,
        min_samples_split: int | None = None,
        min_variance_reduction: float = 1e-4,
        target_weights: Sequence[float] | None = None,
        min_density: float = 0.6,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int | None = None,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        n_trees: int = 50,
        max_features: int | None = None,
        linkage: str = "average",
        **params: Any,
    ):
        self.kind = kind.lower().strip()
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.metric = metric
        self.seed = seed
        self.params = {
            **params,
            "min_samples_split": min_samples_split,
            "min_variance_reduction": min_variance_reduction,
            "target_weights": target_weights,
            "min_density": min_density,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "max_cluster_size": max_cluster_size,
            "cluster_selection_method": cluster_selection_method,
            "allow_single_cluster": allow_single_cluster,
            "n_trees": n_trees,
            "max_features": max_features,
            "linkage": linkage,
        }
        self._fitted = False
        self._inner: Any = None
        self._col_means: list[float] | None = None
        self._labels: list[int | None] | None = None

        k = self.kind
        if k in ("pct", "predictive", "clus"):
            self._inner = _PCT(
                max_depth=self.max_depth,
                min_samples_split=self.min_rows * 2
                if min_samples_split is None
                else min_samples_split,
                min_samples_leaf=self.min_rows,
                min_variance_reduction=min_variance_reduction,
                target_weights=target_weights,
            )
        elif k in ("cltree", "density"):
            self._inner = _CLTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_rows,
                min_density=min_density,
            )
        elif k in ("hdbscan", "hierarchical"):
            self._inner = _HDBSCAN(
                min_cluster_size=self.min_rows,
                min_samples=self.min_rows if min_samples is None else min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                max_cluster_size=max_cluster_size,
                metric=self.metric,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster,
            )
        elif k in ("urf", "proximity", "forest"):
            self._inner = _URF(
                n_trees=n_trees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_rows,
                max_features=max_features,
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

        self._fitted = True
        return self

    def _snapshot_config(self) -> dict[str, Any]:
        return {
            **self.params,
            "kind": self.kind,
            "n_clusters": self.n_clusters,
            "max_depth": self.max_depth,
            "min_rows": self.min_rows,
            "metric": self.metric,
            "seed": self.seed,
        }

    def _snapshot_state(self) -> dict[str, Any]:
        return {"labels": self.labels_}

    def _restore_snapshot_state(self, state: dict[str, Any]) -> None:
        labels = state["labels"]
        n_clusters = self.n_clusters_
        if (
            not isinstance(labels, list)
            or len(labels) != self._inner._snapshot_rows
            or any(
                type(label) is not int or not 0 <= label < n_clusters
                for label in labels
            )
        ):
            raise ValueError("Invalid predictive clustering labels.")
        self._labels = labels

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: Literal["cluster"] = "cluster"
    ) -> list[int | None]: ...

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: Literal["distances"]
    ) -> list[list[float]]: ...

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: str
    ) -> list[int | None] | list[list[float]]: ...

    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: str = "cluster"
    ) -> list[int | None] | list[list[float]]:
        """Predict cluster labels or URF distances from each query to each training row."""
        if not self._fitted:
            raise ValueError("ClusterTree is not fitted yet.")

        k = self.kind
        valid = (
            ["cluster", "distances"]
            if k in ("urf", "proximity", "forest")
            else ["cluster"]
        )
        if kind not in valid:
            raise ValueError(
                f"Unknown prediction kind {kind!r} for {self.kind} cluster tree. Use one of {valid}."
            )
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
            if kind == "distances":
                return self._inner.distances(float_mat)
            return self._inner.predict(float_mat)
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
