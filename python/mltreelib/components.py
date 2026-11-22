"""Component analysis, matrix factorization, and dimensionality reduction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from mltreelib._core import ComponentModel as _CoreComponentModel
from mltreelib._model_inputs import _numeric_rows
from mltreelib.report import ModelReport

__all__ = ["ComponentModel", "ComponentResult"]


@dataclass
class ComponentResult:
    """Fitted component analysis result and projection outputs."""

    components: list[list[float]]
    loadings: list[list[float]]
    scores: list[list[float]]
    explained_variance: list[float] | None
    explained_variance_ratio: list[float] | None
    reconstruction: list[list[float]] | None
    reconstruction_errors: list[float] | None
    report: ModelReport


def _clean_dense_rows(x: Sequence[Sequence[Any]]) -> list[list[float]]:
    rows = _numeric_rows(x)
    dense: list[list[float]] = []
    for i, row in enumerate(rows):
        clean_row: list[float] = []
        for j, val in enumerate(row):
            if val is None:
                raise ValueError(f"Missing or non-finite value at row {i}, column {j}")
            clean_row.append(val)
        dense.append(clean_row)
    return dense


class ComponentModel:
    """Consolidated Component Analysis, Factor Analysis, and Dimensionality Reduction Model.

    Provides a unified facade across multiple decomposition and factorization methods:
    - ``"pca"``: Classical Principal Component Analysis (centered and standardized).
    - ``"svd"`` or ``"truncated_svd"``: Truncated SVD without centering.
    - ``"whitening"``: ZCA and PCA covariance decorrelation preserving Mahalanobis geometry.
    - ``"ica"`` or ``"fastica"``: Independent Component Analysis via FastICA (negentropy/logcosh).
    - ``"nmf"``: Non-negative Matrix Factorization (MU, projected ALS, HALS coordinate descent with L1/L2 penalties).
    - ``"fa"`` or ``"factor_analysis"``: Exploratory Factor Analysis via Principal Axis Factoring (PAF / Minres),
      estimating shared communalities h^2 and uniquenesses psi = 1 - h^2 with Bartlett/Thomson scoring
      and Schmid-Leiman hierarchical bifactor decomposition.

    Parameters
    ----------
    kind : str, default="pca"
        Decomposition engine: ``"pca"``, ``"svd"``, ``"whitening"``, ``"ica"``, ``"nmf"``, or ``"fa"``.
    n_components : int, optional
        Number of components or factors to extract. Defaults to min(n_samples, n_features).
    standardize : bool, default=True
        For PCA and Factor Analysis: center and scale columns to zero mean and unit variance.
    max_iter : int, default=200
        Maximum iterations for iterative algorithms (ICA, NMF, FA).
    tol : float, default=1e-4
        Convergence tolerance for iterative algorithms.
    seed : int, default=42
        Random seed for stochastic initialization.
    whitening_kind : str, default="zca"
        For whitening: ``"zca"`` (zero-phase, minimal distortion) or ``"pca"`` whitening.
    eps : float, default=1e-5
        Ridge regularization term added to eigenvalues in whitening.
    rotation : str, default="none"
        Factor/PCA rotation method aligned with R stats:
        - ``"none"``: Standard unrotated orthogonal eigenvectors.
        - ``"varimax"``: Orthogonal Varimax (gamma=1.0, R stats::varimax default).
        - ``"quartimax"``: Orthogonal Quartimax (gamma=0.0).
        - ``"equamax"``: Orthogonal Equamax (gamma=k/2).
        - ``"parsimax"``: Orthogonal Parsimax.
        - ``"biquartimax"``: Orthogonal Biquartimax (gamma=0.5).
        - ``"promax"``: Oblique Promax Procrustes rotation (R stats::promax).
    promax_m : float, default=4.0
        Promax target exponent (default 4.0 in R stats::promax).
    solver : str, default="hals"
        Solver engine for NMF or Factor Analysis:
        - For NMF: ``"hals"`` (Hierarchical ALS), ``"als"`` (Projected ALS), ``"mu"`` (Multiplicative Updates).
        - For FA: ``"paf"`` (Principal Axis Factoring / Minres).
    l1_reg : float, default=0.0
        L1 sparsity penalty for NMF HALS.
    l2_reg : float, default=0.0
        L2 ridge penalty for NMF HALS/ALS.
    """

    _snapshot_format = "mltreelib.component_model"

    def __init__(
        self,
        kind: str = "pca",
        n_components: int | None = None,
        *,
        standardize: bool = True,
        max_iter: int = 200,
        tol: float = 1e-4,
        seed: int = 42,
        whitening_kind: str = "zca",
        eps: float = 1e-5,
        rotation: str = "none",
        promax_m: float = 4.0,
        solver: str = "hals",
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
    ) -> None:
        self.kind = kind.lower().strip()
        self.n_components = n_components
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.whitening_kind = whitening_kind.lower().strip()
        self.eps = eps
        self.rotation = rotation.lower().strip()
        self.promax_m = promax_m
        self.solver = solver.lower().strip()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self._fitted = False
        self._inner: _CoreComponentModel | None = None

    def fit(self, x: Sequence[Sequence[Any]]) -> ComponentModel:
        """Fit the component analysis model on input rows."""
        dense = _clean_dense_rows(x)
        self._inner = _CoreComponentModel.fit(
            data=dense,
            kind=self.kind,
            n_components=self.n_components,
            standardize=self.standardize,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
            whitening_kind=self.whitening_kind,
            eps=self.eps,
            rotation=self.rotation,
            promax_m=self.promax_m,
            solver=self.solver,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
        )
        self._fitted = True
        return self

    def transform(self, x: Sequence[Sequence[Any]]) -> list[list[float]]:
        """Project input rows into component coordinates."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        dense = _clean_dense_rows(x)
        return self._inner.transform(dense)

    def fit_transform(self, x: Sequence[Sequence[Any]]) -> list[list[float]]:
        """Fit model on x and return transformed component scores."""
        return self.fit(x).transform(x)

    def inverse_transform(self, scores: Sequence[Sequence[Any]]) -> list[list[float]]:
        """Map component scores back to the original feature coordinates."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        dense = _clean_dense_rows(scores)
        return self._inner.inverse_transform(dense)

    def reconstruct(self, x: Sequence[Sequence[Any]]) -> list[list[float]]:
        """Reconstruct observations by projecting into component space and back."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        dense = _clean_dense_rows(x)
        return self._inner.reconstruct(dense)

    def score_samples(self, x: Sequence[Sequence[Any]]) -> list[float]:
        """Compute row-wise squared reconstruction error ||x - x_hat||^2."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        dense = _clean_dense_rows(x)
        return self._inner.score_samples(dense)

    @property
    def components_(self) -> list[list[float]]:
        """Fitted component/loading direction vectors (shape: n_components x n_features)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.components

    @property
    def loadings_(self) -> list[list[float]]:
        """Factor loadings matrix Lambda of shape (n_features x n_components)."""
        comps = self.components_
        if not comps:
            return []
        p = len(comps[0])
        k = len(comps)
        return [[comps[i][j] for i in range(k)] for j in range(p)]

    @property
    def explained_variance_(self) -> list[float] | None:
        """Variance explained by each component (PCA and SVD)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.explained_variance

    @property
    def explained_variance_ratio_(self) -> list[float] | None:
        """Percentage of variance explained by each component (PCA and SVD)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.explained_variance_ratio

    @property
    def rotation_matrix_(self) -> list[list[float]] | None:
        """Rotation transformation matrix T if rotation was applied, else None."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.rotation_matrix

    @property
    def factor_correlations_(self) -> list[list[float]] | None:
        """Factor correlation matrix Phi (for oblique Promax rotations in PCA and Factor Analysis)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.factor_correlations

    @property
    def uniquenesses_(self) -> list[float] | None:
        """Specific variances psi_i per feature (Factor Analysis)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.uniquenesses

    @property
    def communalities_(self) -> list[float] | None:
        """Common variance shares h_i^2 = 1 - psi_i per feature (Factor Analysis)."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.communalities

    def schmid_leiman(self) -> tuple[list[list[float]], list[float]]:
        """Compute Schmid-Leiman hierarchical bifactor decomposition.

        Transforms oblique factor loadings into an orthogonal general factor g
        and independent group factors.

        Returns
        -------
        bifactor_loadings : list of list of float
            Shape (p, k + 1) where column 0 is General factor g, and columns 1..=k are group factors.
        variance_shares : list of float
            [general_factor_variance_share, group_factors_variance_share].
        """
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        res = self._inner.schmid_leiman()
        if res is None:
            raise ValueError("Schmid-Leiman decomposition is only available for Factor Analysis with oblique rotations.")
        return res

    @property
    def reconstruction_error_(self) -> float | None:
        """Reconstruction error for NMF models."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._inner.reconstruction_error

    @property
    def report(self) -> ModelReport:
        """Return the structured ModelReport with diagnostics and complexity metrics."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return ModelReport(self._inner.report())

    def config(self) -> dict[str, Any]:
        """Return effective configuration dictionary."""
        return {
            "kind": self.kind,
            "n_components": self.n_components,
            "standardize": self.standardize,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "seed": self.seed,
            "whitening_kind": self.whitening_kind,
            "eps": self.eps,
            "rotation": self.rotation,
            "promax_m": self.promax_m,
            "solver": self.solver,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
        }

    def to_json(self) -> str:
        """Return fitted model as versioned JSON string."""
        if not self._fitted or self._inner is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return json.dumps(
            {
                "format": self._snapshot_format,
                "format_version": 1,
                "configuration": self.config(),
                "native": json.loads(self._inner.to_json()),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> ComponentModel:
        """Restore fitted model from versioned JSON string."""
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("format") != cls._snapshot_format:
            raise ValueError(f"Expected a {cls.__name__} snapshot.")
        cfg = payload["configuration"]
        model = cls(**cfg)
        model._inner = _CoreComponentModel.from_json(json.dumps(payload["native"]))
        model._fitted = True
        return model

    def to_bytes(self) -> bytes:
        """Return fitted model as UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> ComponentModel:
        """Restore fitted model from UTF-8 JSON bytes."""
        return cls.from_json(data.decode("utf-8"))

    def save(self, path: str | Path) -> None:
        """Save fitted model snapshot to disk atomically."""
        dest = Path(path)
        data = self.to_bytes()
        temp = None
        try:
            with NamedTemporaryFile(dir=dest.parent, delete=False) as stream:
                temp = Path(stream.name)
                stream.write(data)
            temp.replace(dest)
        except BaseException:
            if temp is not None and temp.exists():
                temp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> ComponentModel:
        """Load fitted model snapshot from disk."""
        return cls.from_bytes(Path(path).read_bytes())
