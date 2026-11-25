"""Feature transforms with fit, transform and inverse_transform (standard library only).

Transforms that complement the Rust scalers: ``log1p``, ``power``, ``mean_diff``,
``median_diff``, ``boxcox`` and ``yeojohnson``. Each transform works on one column
(a sequence of values). Missing values (None or NaN) and infinities pass through
``transform`` and ``inverse_transform`` unchanged, so the output has one entry per
input row. Statistics (means, medians, lambdas) use finite observed values only.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from mltreelib.data._common import is_infinite, is_missing, quantile

__all__ = [
    "FeatureTransformer",
    "GroupedFeatureTransformer",
    "boxcox_lambda",
    "log1p",
    "yeojohnson_lambda",
]

_KINDS = ("log1p", "power", "mean_diff", "median_diff", "boxcox", "yeojohnson")
_ACCEPTED_PARAMS = {"power": ("p",), "boxcox": ("lmbda",), "yeojohnson": ("lmbda",)}
_LAMBDA_KINDS = ("boxcox", "yeojohnson")
_LAMBDA_LO, _LAMBDA_HI = -5.0, 5.0
_LAMBDA_TOL = 1e-8
_NEAR_ZERO = 1e-12
_INVPHI = (math.sqrt(5.0) - 1.0) / 2.0


def _scan(values: Sequence[Any]) -> list[str]:
    """Classify each row: 'm' missing, 'i' infinite, 'o' observed finite number."""
    kinds: list[str] = []
    for v in values:
        if is_missing(v):
            kinds.append("m")
        elif is_infinite(v):
            kinds.append("i")
        elif (
            isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)
        ):
            kinds.append("o")
        else:
            raise TypeError(
                f"values must be numbers, None or NaN; got {v!r} at index {len(kinds)}"
            )
    return kinds


def _is_odd_int(p: float) -> bool:
    return float(p).is_integer() and int(p) % 2 == 1


def _forward(kind: str, x: float, par: float | None) -> float:
    """Transform one finite value. Raises ValueError when x is outside the domain."""
    if kind == "log1p":
        return math.log1p(x)
    if kind == "power":
        if x < 0 and not _is_odd_int(par):
            raise ValueError("negative x needs an odd integer p")
        return math.pow(x, par)
    if kind in ("mean_diff", "median_diff"):
        return x - par
    if kind == "boxcox":
        log_x = math.log(x)
        return log_x if abs(par) < _NEAR_ZERO else math.expm1(par * log_x) / par
    # yeojohnson
    if x >= 0:
        log_x = math.log1p(x)
        return log_x if abs(par) < _NEAR_ZERO else math.expm1(par * log_x) / par
    log_x = math.log1p(-x)
    if abs(2.0 - par) < _NEAR_ZERO:
        return -log_x
    return -math.expm1((2.0 - par) * log_x) / (2.0 - par)


def _inverse(kind: str, y: float, par: float | None) -> float:
    """Invert one finite value. Raises ValueError when y has no real preimage."""
    if kind == "log1p":
        return math.expm1(y)
    if kind == "power":
        if _is_odd_int(par):
            return math.copysign(math.pow(abs(y), 1.0 / par), y)
        return math.pow(y, 1.0 / par)
    if kind in ("mean_diff", "median_diff"):
        return y + par
    if kind == "boxcox":
        if abs(par) < _NEAR_ZERO:
            return math.exp(y)
        return math.exp(math.log1p(par * y) / par)
    # yeojohnson
    if y >= 0:
        if abs(par) < _NEAR_ZERO:
            return math.expm1(y)
        return math.expm1(math.log1p(par * y) / par)
    if abs(2.0 - par) < _NEAR_ZERO:
        return -math.expm1(-y)
    return -math.expm1(math.log1p(-(2.0 - par) * y) / (2.0 - par))


class _RowError(ValueError):
    """A domain error carrying the offending row position, so callers can remap it."""

    def __init__(self, problem: str, value: Any, row: int) -> None:
        super().__init__(f"{problem}; value {value!r} at index {row}")
        self.problem = problem
        self.value = value
        self.row = row


def _map_obs(
    values: Sequence[Any],
    kinds: Sequence[str],
    fn: Callable[[float], float],
    problem: str,
) -> list[Any]:
    """Apply fn to observed rows; missing and infinite rows are returned unchanged.

    A ValueError from fn is re-raised as a _RowError naming the first offending index.
    """
    out: list[Any] = []
    for i, (v, k) in enumerate(zip(values, kinds)):
        if k != "o":
            out.append(v)
            continue
        try:
            out.append(fn(float(v)))
        except ValueError as exc:
            raise _RowError(problem, v, i) from exc
    return out


def _golden_max(f: Callable[[float], float], lo: float, hi: float, tol: float) -> float:
    """Maximise a unimodal f on [lo, hi] by golden-section search."""
    a, b = lo, hi
    c = b - _INVPHI * (b - a)
    d = a + _INVPHI * (b - a)
    fc, fd = f(c), f(d)
    while (b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _INVPHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INVPHI * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _profile_loglik(
    obs: Sequence[float], lam: float, kind: str, log_jacobian: float
) -> float:
    """Profile log-likelihood of lambda: -n/2 log(var) + (lambda - 1) * sum(log Jacobian)."""
    try:
        ys = [_forward(kind, x, lam) for x in obs]
    except (OverflowError, ValueError):
        return -math.inf
    n = len(ys)
    mean = math.fsum(ys) / n
    var = math.fsum((y - mean) ** 2 for y in ys) / n
    if not var > 0.0:
        return -math.inf
    return -0.5 * n * math.log(var) + (lam - 1.0) * log_jacobian


def _fit_lambda(obs: Sequence[float], kind: str) -> float:
    """Maximum-likelihood lambda on [-5, 5]. NaN with no data; 1.0 when values are constant."""
    n = len(obs)
    if n == 0:
        return math.nan
    if n < 2 or min(obs) == max(obs):
        return 1.0
    if kind == "boxcox":
        log_jacobian = math.fsum(math.log(x) for x in obs)
    else:
        log_jacobian = math.fsum(math.copysign(math.log1p(abs(x)), x) for x in obs)
    return _golden_max(
        lambda lam: _profile_loglik(obs, lam, kind, log_jacobian),
        _LAMBDA_LO,
        _LAMBDA_HI,
        _LAMBDA_TOL,
    )


_DOMAIN_TEXT = {
    "log1p": "need x > -1",
    "boxcox": "need x > 0",
    "power": "need x >= 0, or x < 0 when p is an odd integer",
}


class FeatureTransformer:
    """Fit-then-apply transform for one column.

    Args:
        kind: One of ``'log1p'``, ``'power'``, ``'mean_diff'``, ``'median_diff'``,
            ``'boxcox'`` or ``'yeojohnson'``.
        **params: ``p`` for ``'power'`` (default 2.0, nonzero). ``lmbda`` for
            ``'boxcox'`` and ``'yeojohnson'`` fixes lambda instead of fitting it.

    Attributes set by ``fit``:
        params_ (dict): ``{}`` for log1p; ``{'p': p}`` for power; ``{'mean': m}``;
            ``{'median': m}``; ``{'lambda': lam}`` for boxcox and yeojohnson. NaN
            when there are no observed values to fit.
        n_fit_ (int): observed finite values used for fitting.
        n_missing_ (int): None and NaN rows (not used, kept in the output).
        n_inf_ (int): infinite rows (not used, kept in the output).

    Rules:
        Missing and infinite rows come back unchanged. A value outside a
        transform's domain raises ValueError naming the first offending index.

        Domains and formulas:
        - ``log1p``: x > -1; y = log(1 + x); inverse x = expm1(y).
        - ``power``: p is finite and nonzero. y = x**p, where a negative x is
          allowed only when p is an odd integer (then the inverse is the
          sign-preserving real root). Otherwise x >= 0, and x > 0 when p < 0.
          The inverse is y**(1/p) and needs y >= 0 (y > 0 when p < 0). Even
          integer p is rejected for negative x because it is not invertible there.
        - ``mean_diff`` / ``median_diff``: y = x - centre; inverse adds it back.
        - ``boxcox``: x > 0; y = (x**l - 1) / l, or log(x) when |l| < 1e-12.
          The inverse is x = (l*y + 1)**(1/l), or exp(y) when |l| < 1e-12.
        - ``yeojohnson``: any real x; y = ((x+1)**l - 1) / l for x >= 0 and
          -((1-x)**(2-l) - 1) / (2-l) for x < 0, with log forms at l = 0 and l = 2.
          The inverse is exact for every lambda, including l = 0 and l = 2.

    Lambda is fitted by golden-section search on [-5, 5] to tolerance 1e-8 of the
    profile log-likelihood. The search assumes a unimodal likelihood; a maximum at
    the boundary means the data wants lambda outside the range.

    Example:
        >>> t = FeatureTransformer("mean_diff").fit([1, 2, 3, None])
        >>> t.params_
        {'mean': 2.0}
        >>> t.transform([4, None])
        [2.0, None]
    """

    def __init__(self, kind: str, **params: float | None) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {_KINDS}")
        accepted = _ACCEPTED_PARAMS.get(kind, ())
        unknown = [name for name in params if name not in accepted]
        if unknown:
            raise TypeError(
                f"kind {kind!r} does not accept {unknown[0]!r}; accepted: {accepted or 'none'}"
            )
        self.kind = kind
        self.params = dict(params)
        self._fixed: dict[str, float] = {}
        if kind == "power":
            p = float(params.get("p", 2.0))  # type: ignore[arg-type]
            if not math.isfinite(p) or p == 0.0:
                raise ValueError("power parameter p must be finite and nonzero")
            self._fixed = {"p": p}
        elif kind in _LAMBDA_KINDS and params.get("lmbda") is not None:
            lam = float(params["lmbda"])  # type: ignore[arg-type]
            if not math.isfinite(lam):
                raise ValueError("lmbda must be finite")
            self._fixed = {"lambda": lam}
        self._fitted = False

    def _par(self) -> float | None:
        """The single fitted parameter (mean, median, lambda or p); None for log1p."""
        return next(iter(self.params_.values()), None)

    def fit(self, values: Iterable[Any]) -> FeatureTransformer:
        """Fit parameters on the finite observed values. Returns self."""
        values = list(values)
        kinds = _scan(values)
        obs = [float(v) for v, k in zip(values, kinds) if k == "o"]
        self.n_fit_ = len(obs)
        self.n_missing_ = kinds.count("m")
        self.n_inf_ = kinds.count("i")

        if self.kind in _DOMAIN_TEXT:
            probe = next(iter(self._fixed.values()), 1.0)
            _map_obs(
                values,
                kinds,
                lambda x: _forward(self.kind, x, probe),
                f"{self.kind} {_DOMAIN_TEXT.get(self.kind, 'no domain limit')}",
            )

        if self.kind == "log1p":
            self.params_: dict[str, float] = {}
        elif self.kind == "power":
            self.params_ = dict(self._fixed)
        elif self.kind == "mean_diff":
            self.params_ = {"mean": math.fsum(obs) / len(obs) if obs else math.nan}
        elif self.kind == "median_diff":
            self.params_ = {"median": quantile(sorted(obs), 0.5) if obs else math.nan}
        else:
            lam = self._fixed["lambda"] if self._fixed else _fit_lambda(obs, self.kind)
            self.params_ = {"lambda": lam}
        self._fitted = True
        return self

    def _apply(self, values: Iterable[Any], inverse: bool) -> list[Any]:
        if not self._fitted:
            raise RuntimeError("FeatureTransformer is not fitted; call fit first")
        values = list(values)
        kinds = _scan(values)
        par = self._par()
        if par is not None and not math.isfinite(par) and "o" in kinds:
            raise ValueError(f"{self.kind}: transformer was fit on no observed values")
        if inverse:
            return _map_obs(
                values,
                kinds,
                lambda y: _inverse(self.kind, y, par),
                f"{self.kind} inverse has no real preimage",
            )
        return _map_obs(
            values,
            kinds,
            lambda x: _forward(self.kind, x, par),
            f"{self.kind} {_DOMAIN_TEXT.get(self.kind, 'no domain limit')}",
        )

    def transform(self, values: Iterable[Any]) -> list[Any]:
        """Transform one value per row; missing and infinite rows are returned unchanged."""
        return self._apply(values, inverse=False)

    def inverse_transform(self, values: Iterable[Any]) -> list[Any]:
        """Invert ``transform`` row by row; missing and infinite rows are returned unchanged."""
        return self._apply(values, inverse=True)

    def fit_transform(self, values: Iterable[Any]) -> list[Any]:
        """Fit on ``values`` and return their transform."""
        values = list(values)
        return self.fit(values).transform(values)


def _group_indices(groups: Sequence[Any]) -> dict[Any, list[int]]:
    """Row positions per group label, groups in order of first appearance. None is a group."""
    index: dict[Any, list[int]] = {}
    for i, g in enumerate(groups):
        if isinstance(g, float) and math.isnan(g):
            raise ValueError(f"NaN is not a valid group label (row {i})")
        index.setdefault(g, []).append(i)
    return index


def _with_rows(idx: Sequence[int], call: Callable[[Any], Any], sub: Any) -> Any:
    """Run call(sub) on a group's rows; map a domain error's position back to the input row."""
    try:
        return call(sub)
    except _RowError as exc:
        raise _RowError(exc.problem, exc.value, idx[exc.row]) from exc


class GroupedFeatureTransformer:
    """One FeatureTransformer per group label.

    Args:
        kind: Transform kind, as for :class:`FeatureTransformer`.
        fallback: ``None`` (default) raises KeyError for a group unseen at fit time.
            ``'global'`` uses a transformer fitted on all rows for unseen groups.
        **params: Passed to each per-group FeatureTransformer.

    Domain errors name the offending input row. Groups are checked in order of first
    appearance, so the reported row is the first bad row in the first bad group.

    Attributes set by ``fit``:
        transformers_ (dict): group label to fitted FeatureTransformer, in first-appearance order.
        global_ (FeatureTransformer | None): fallback transformer when ``fallback='global'``.

    Example:
        >>> g = GroupedFeatureTransformer("mean_diff").fit([1, 3, 10, 20], ["a", "a", "b", "b"])
        >>> g.transform([1, 3, 10, 20], ["a", "a", "b", "b"])
        [-1.0, 1.0, -5.0, 5.0]
    """

    def __init__(
        self, kind: str, *, fallback: str | None = None, **params: float | None
    ) -> None:
        if fallback not in (None, "global"):
            raise ValueError("fallback must be None or 'global'")
        FeatureTransformer(kind, **params)  # validate kind and params early
        self.kind = kind
        self.params = dict(params)
        self.fallback = fallback
        self._fitted = False

    def _new(self) -> FeatureTransformer:
        return FeatureTransformer(self.kind, **self.params)

    def fit(
        self, values: Iterable[Any], groups: Iterable[Any]
    ) -> GroupedFeatureTransformer:
        """Fit one transformer per group on that group's rows. Returns self."""
        values, groups = list(values), list(groups)
        if len(values) != len(groups):
            raise ValueError("values and groups must have the same length")
        self.transformers_: dict[Any, FeatureTransformer] = {}
        for g, idx in _group_indices(groups).items():
            sub = [values[i] for i in idx]
            self.transformers_[g] = _with_rows(idx, self._new().fit, sub)
        self.global_ = self._new().fit(values) if self.fallback == "global" else None
        self._fitted = True
        return self

    def _apply(
        self, values: Iterable[Any], groups: Iterable[Any], inverse: bool
    ) -> list[Any]:
        if not self._fitted:
            raise RuntimeError(
                "GroupedFeatureTransformer is not fitted; call fit first"
            )
        values, groups = list(values), list(groups)
        if len(values) != len(groups):
            raise ValueError("values and groups must have the same length")
        out: list[Any] = [None] * len(values)
        for g, idx in _group_indices(groups).items():
            tr = self.transformers_.get(g)
            if tr is None:
                if self.global_ is None:
                    raise KeyError(g)
                tr = self.global_
            sub = [values[i] for i in idx]
            call = tr.inverse_transform if inverse else tr.transform
            res = _with_rows(idx, call, sub)
            for i, r in zip(idx, res):
                out[i] = r
        return out

    def transform(self, values: Iterable[Any], groups: Iterable[Any]) -> list[Any]:
        """Transform each row with its group's parameters. Unseen groups raise KeyError unless fallback='global'."""
        return self._apply(values, groups, inverse=False)

    def inverse_transform(
        self, values: Iterable[Any], groups: Iterable[Any]
    ) -> list[Any]:
        """Invert each row with its group's parameters."""
        return self._apply(values, groups, inverse=True)

    def fit_transform(self, values: Iterable[Any], groups: Iterable[Any]) -> list[Any]:
        """Fit per group on ``values`` and return their transform."""
        values, groups = list(values), list(groups)
        return self.fit(values, groups).transform(values, groups)


def log1p(values: Iterable[Any]) -> list[Any]:
    """Return ``log(1 + x)`` per row (x > -1 for observed values; missing passes through).

    Example:
        >>> log1p([0.0, None])
        [0.0, None]
    """
    return FeatureTransformer("log1p").fit_transform(values)


def boxcox_lambda(values: Iterable[Any]) -> float:
    """Box-Cox lambda maximising the profile log-likelihood on observed values (x > 0).

    Returns NaN when there are no observed values and 1.0 when fewer than two distinct values.

    Example:
        >>> boxcox_lambda([])
        nan
    """
    return float(FeatureTransformer("boxcox").fit(values).params_["lambda"])


def yeojohnson_lambda(values: Iterable[Any]) -> float:
    """Yeo-Johnson lambda maximising the profile log-likelihood on observed values.

    Returns NaN when there are no observed values and 1.0 when fewer than two distinct values.

    Example:
        >>> yeojohnson_lambda([2.0, 2.0])
        1.0
    """
    return float(FeatureTransformer("yeojohnson").fit(values).params_["lambda"])
