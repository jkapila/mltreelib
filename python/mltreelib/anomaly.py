"""Tree-based anomaly detection."""

from collections.abc import Sequence
from typing import Any, Literal, overload

from mltreelib._core import ActiveAnomalyTree, ContextAnomalyTree
from mltreelib._core import DensityEstimationTree as _DET
from mltreelib._core import ExtendedIsolationTree as _EIF
from mltreelib._core import HalfSpaceTree as _HST
from mltreelib._core import RandomCutTree as _RCT
from mltreelib._model_inputs import _impute_float_matrix
from mltreelib._unsupervised_snapshot import _UnsupervisedSnapshot

__all__ = ["AnomalyTree"]


class AnomalyTree(_UnsupervisedSnapshot):
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
    extension_level, subsample_size : int, optional
        EIF hyperplane order and training subsample size.
    min_samples_split : int, optional
        DET minimum samples per split. Defaults to twice ``min_rows``.
    alpha, min_density_gain : float
        DET density smoothing and minimum gain.
    window_size : int, default=250
        HST stream window size.
    initial_mass : float, default=0
        HST initial node mass.
    sample_size : int, default=256
        RCT sample size.
    eta, prior_weight : float
        Active anomaly learning rate and prior leaf weight.
    alphabet_size : int, default=256
        Context tree alphabet size.
    smoothing_discount : float, default=0.5
        Context tree discount for smoothing symbol probabilities.
    """

    _snapshot_format = "mltreelib.anomaly_tree"
    _snapshot_engines = ("eif", "isolation", "extended_isolation")

    def __init__(
        self,
        kind: str = "eif",
        max_depth: int | None = None,
        min_rows: int = 10,
        contamination: float = 0.05,
        seed: int = 42,
        *,
        extension_level: int | None = None,
        subsample_size: int | None = None,
        min_samples_split: int | None = None,
        alpha: float = 0.0,
        min_density_gain: float = 0.0,
        window_size: int = 250,
        initial_mass: float = 0.0,
        sample_size: int = 256,
        eta: float = 0.1,
        prior_weight: float = 1.0,
        alphabet_size: int = 256,
        smoothing_discount: float = 0.5,
        **params: Any,
    ):
        self.kind = kind.lower().strip()
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.contamination = contamination
        self.seed = seed
        self.params = {
            **params,
            "extension_level": extension_level,
            "subsample_size": subsample_size,
            "min_samples_split": min_samples_split,
            "alpha": alpha,
            "min_density_gain": min_density_gain,
            "window_size": window_size,
            "initial_mass": initial_mass,
            "sample_size": sample_size,
            "eta": eta,
            "prior_weight": prior_weight,
            "alphabet_size": alphabet_size,
            "smoothing_discount": smoothing_discount,
        }
        self._threshold: float | None = None
        self._threshold_strict: bool = False
        self._fitted = False
        self._col_means: list[float] | None = None
        self._inner: Any = None

        k = self.kind
        if k in ("eif", "isolation", "extended_isolation"):
            self._inner = _EIF(
                max_depth=self.max_depth,
                extension_level=extension_level,
                subsample_size=subsample_size,
                seed=self.seed,
            )
        elif k in ("det", "density", "density_estimation"):
            depth = self.max_depth or 8
            self._inner = _DET(
                max_depth=depth,
                min_samples_split=self.min_rows * 2
                if min_samples_split is None
                else min_samples_split,
                min_leaf_samples=self.min_rows,
                min_density_gain=min_density_gain,
                alpha=alpha,
            )
        elif k in ("hst", "half_space"):
            depth = self.max_depth or 6
            self._inner = _HST(
                max_depth=depth,
                window_size=window_size,
                initial_mass=initial_mass,
                seed=self.seed,
            )
        elif k in ("rct", "random_cut"):
            self._inner = _RCT(sample_size=sample_size, seed=self.seed)
        elif k in ("active", "aad"):
            depth = self.max_depth or 8
            self._inner = ActiveAnomalyTree(
                max_depth=depth, eta=eta, prior_weight=prior_weight, seed=self.seed
            )
        elif k in ("context", "sequence"):
            depth = self.max_depth or 5
            self._inner = ContextAnomalyTree(
                max_depth=depth,
                alphabet_size=alphabet_size,
                smoothing_discount=smoothing_discount,
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
            self._threshold_strict = (
                idx > 0 and sorted_scores[idx] == sorted_scores[idx - 1]
            )

        return self

    def _snapshot_config(self) -> dict[str, Any]:
        return {
            **self.params,
            "kind": self.kind,
            "max_depth": self.max_depth,
            "min_rows": self.min_rows,
            "contamination": self.contamination,
            "seed": self.seed,
        }

    def _snapshot_state(self) -> dict[str, Any]:
        return {"threshold": self._threshold, "strict": self._threshold_strict}

    def _restore_snapshot_state(self, state: dict[str, Any]) -> None:
        threshold = state["threshold"]
        strict = state["strict"]
        if (
            type(strict) is not bool
            or (
                threshold is not None
                and (type(threshold) not in (float, int) or not 0 <= threshold <= 1)
            )
            or (strict and threshold is None)
        ):
            raise ValueError("Invalid anomaly threshold state.")
        self._threshold = threshold
        self._threshold_strict = strict

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

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: Literal["binary"] = "binary"
    ) -> list[int]: ...

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: Literal["score", "path_length"]
    ) -> list[float]: ...

    @overload
    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: str
    ) -> list[int] | list[float]: ...

    def predict(
        self, x: Sequence[Sequence[Any]], *, kind: str = "binary"
    ) -> list[int] | list[float]:
        """Predict binary labels, anomaly scores, or EIF path lengths.

        Context trees expose sequence predictions via ``surprise_score`` and
        ``conditional_prob``.
        """
        if not self._fitted:
            raise ValueError("AnomalyTree is not fitted yet.")
        if self.kind in ("context", "sequence"):
            raise ValueError(
                "Use surprise_score(context, symbol) or conditional_prob(context, symbol) "
                "for ContextAnomalyTree"
            )
        valid = ["binary", "score"]
        if self.kind in ("eif", "isolation", "extended_isolation"):
            valid.append("path_length")
        if kind not in valid:
            raise ValueError(
                f"Unknown prediction kind {kind!r} for {self.kind} anomaly tree. Use one of {valid}."
            )
        if kind == "path_length":
            return self._inner.path_length(x)
        scores = self.score_samples(x)
        if kind == "score":
            return scores
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
