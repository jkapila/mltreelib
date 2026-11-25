"""Row and feature accessors for the dataset dict returned by ``load_dataset``.

These helpers turn a dict of columns into the ``x`` rows and ``y``/``time``/
``event``/``treatment``/``outcome`` vectors that the models take, so docs and
examples stop rebuilding feature matrices by hand. Pure Python, standard
library only.

Values are passed through unchanged: ``None`` stays ``None``, NaN stays NaN,
infinities stay infinite, and rows are never dropped, even when the target is
missing. Missing counts are reported under ``n_*_missing`` keys. Event and
treatment codes are passed through as well: coding conventions differ between
datasets (for example 1 or 2 for death) and are not normalised here.

Every returned row is a new list, so a model that mutates its input cannot
change the caller's columns. Inputs are never modified.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mltreelib.data._common import as_columns, is_missing

__all__ = ["select_rows", "survival_xy", "uplift_xy", "xy"]

_DTYPES = ("numeric", "categorical")
# Count keys and the per-row field each one counts.
_MISSING_COUNTS = {
    "n_y_missing": "y",
    "n_time_missing": "time",
    "n_event_missing": "event",
    "n_treatment_missing": "treatment",
    "n_outcome_missing": "outcome",
}


def _resolve(columns: Any) -> tuple[dict[str, list[Any]], dict[str, str]]:
    """Return ``(columns, declared_dtypes)``; declared dtypes come from a load_dataset result."""
    declared: dict[str, str] = {}
    if isinstance(columns, Mapping) and isinstance(columns.get("columns"), Mapping):
        declared = dict(columns.get("dtypes") or {})
    return as_columns(columns), declared


def _n_rows(cols: dict[str, list[Any]]) -> int:
    lengths = {name: len(values) for name, values in cols.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"ragged columns, lengths differ: {lengths}")
    return next(iter(lengths.values()), 0)


def _names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _require(cols: dict[str, list[Any]], name: str, role: str) -> None:
    if name not in cols:
        raise KeyError(f"{role} column {name!r} is not in columns")


def _infer_dtype(values: Sequence[Any]) -> str:
    """Categorical if any observed value is a string or bool, otherwise numeric."""
    for v in values:
        if not is_missing(v) and isinstance(v, (str, bool)):
            return "categorical"
    return "numeric"


def _feature_names(
    cols: dict[str, list[Any]],
    reserved: Sequence[str],
    features: Iterable[str] | None,
    exclude: Iterable[str] | None,
) -> list[str]:
    """Feature names in input order (or in the order given by ``features``)."""
    excluded = _names(exclude)
    for name in excluded:
        _require(cols, name, "exclude")
    if features is None:
        return [c for c in cols if c not in reserved and c not in excluded]
    chosen = _names(features)
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate names in features: {chosen}")
    for name in chosen:
        _require(cols, name, "feature")
        if name in reserved:
            raise ValueError(
                f"{name!r} is a target or outcome column and cannot be a feature"
            )
    return [c for c in chosen if c not in excluded]


def _rows(cols: dict[str, list[Any]], names: list[str], n: int) -> list[list[Any]]:
    if not names:
        return [[] for _ in range(n)]
    return [list(row) for row in zip(*(cols[c] for c in names))]


def _dtypes(
    cols: dict[str, list[Any]], names: list[str], declared: dict[str, str]
) -> list[str]:
    out = []
    for name in names:
        dt = declared.get(name)
        out.append(dt if dt in _DTYPES else _infer_dtype(cols[name]))
    return out


def _count_missing(values: Sequence[Any]) -> int:
    return sum(1 for v in values if is_missing(v))


def xy(
    columns: Any,
    target: str,
    features: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Build model inputs ``x`` (rows), ``y`` and feature metadata from columns.

    Features default to every column except the target and ``exclude``, in
    input order. ``features`` sets an explicit list and order instead; names in
    ``exclude`` are removed from it. Rows with a missing target are kept; the
    count is reported in ``n_y_missing``.

    Args:
        columns: A dict of column name to list, or a ``load_dataset`` result.
            For a load_dataset result its ``dtypes`` are used; otherwise dtypes
            are inferred (any str or bool observed value means categorical).
        target: Name of the target column.
        features: Optional explicit feature names, in the order wanted.
        exclude: Column names to leave out of the features.

    Returns:
        A dict with keys ``x`` (one new list per row), ``y`` (list),
        ``names`` (feature names), ``dtypes`` (``"numeric"`` or
        ``"categorical"`` per feature) and ``n_y_missing`` (int).

    Raises:
        KeyError: If the target, a feature or an excluded name is not a column.
        ValueError: If columns have different lengths, a feature is repeated,
            or the target is also listed as a feature.

    Example:
        >>> cols = {"a": [1, 2, None], "b": ["u", "v", "u"], "y": [0.5, None, 1.5]}
        >>> out = xy(cols, "y")
        >>> out["x"]
        [[1, 'u'], [2, 'v'], [None, 'u']]
        >>> out["dtypes"], out["n_y_missing"]
        (['numeric', 'categorical'], 1)
    """
    cols, declared = _resolve(columns)
    n = _n_rows(cols)
    _require(cols, target, "target")
    names = _feature_names(cols, [target], features, exclude)
    y = list(cols[target])
    return {
        "x": _rows(cols, names, n),
        "y": y,
        "names": names,
        "dtypes": _dtypes(cols, names, declared),
        "n_y_missing": _count_missing(y),
    }


def survival_xy(
    columns: Any,
    time: str,
    event: str,
    features: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Build survival inputs: ``x`` rows, ``time`` and ``event`` vectors.

    ``time`` and ``event`` are left out of the default features. The event
    column is passed through unchanged; coding differs between datasets (1 or
    2 for death, or three outcomes in one trial), so map it to the model's
    convention yourself. Rows with a missing time or event are kept and counted.

    Args:
        columns: A dict of column name to list, or a ``load_dataset`` result.
        time: Name of the follow-up time column.
        event: Name of the event column.
        features: Optional explicit feature names, in the order wanted.
        exclude: Column names to leave out of the features.

    Returns:
        A dict with keys ``x``, ``time``, ``event``, ``names``, ``dtypes``,
        ``n_time_missing`` and ``n_event_missing``.

    Raises:
        KeyError, ValueError: As for ``xy``.

    Example:
        >>> cols = {"t": [5.0, None, 9.0], "died": [1, 2, 0], "age": [60, 70, 50]}
        >>> out = survival_xy(cols, "t", "died")
        >>> out["event"], out["n_time_missing"]
        ([1, 2, 0], 1)
    """
    cols, declared = _resolve(columns)
    n = _n_rows(cols)
    _require(cols, time, "time")
    _require(cols, event, "event")
    names = _feature_names(cols, [time, event], features, exclude)
    t = list(cols[time])
    e = list(cols[event])
    return {
        "x": _rows(cols, names, n),
        "time": t,
        "event": e,
        "names": names,
        "dtypes": _dtypes(cols, names, declared),
        "n_time_missing": _count_missing(t),
        "n_event_missing": _count_missing(e),
    }


def uplift_xy(
    columns: Any,
    treatment: str,
    outcome: str,
    features: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Build uplift inputs: ``x`` rows, ``treatment`` and ``outcome`` vectors.

    The treatment column is passed through unchanged (no recoding to 0/1).
    Rows with a missing treatment or outcome are kept and counted; this
    helper does not decide which rows to analyse.

    Args:
        columns: A dict of column name to list, or a ``load_dataset`` result.
        treatment: Name of the treatment column.
        outcome: Name of the outcome column.
        features: Optional explicit feature names, in the order wanted.
        exclude: Column names to leave out of the features.

    Returns:
        A dict with keys ``x``, ``treatment``, ``outcome``, ``names``,
        ``dtypes``, ``n_treatment_missing`` and ``n_outcome_missing``.

    Raises:
        KeyError, ValueError: As for ``xy``.

    Example:
        >>> cols = {"treat": [1, 0, None], "conv": [0.0, 1.0, 1.0], "age": [30, 40, 50]}
        >>> out = uplift_xy(cols, "treat", "conv")
        >>> out["names"], out["n_treatment_missing"]
        (['age'], 1)
    """
    cols, declared = _resolve(columns)
    n = _n_rows(cols)
    _require(cols, treatment, "treatment")
    _require(cols, outcome, "outcome")
    names = _feature_names(cols, [treatment, outcome], features, exclude)
    w = list(cols[treatment])
    o = list(cols[outcome])
    return {
        "x": _rows(cols, names, n),
        "treatment": w,
        "outcome": o,
        "names": names,
        "dtypes": _dtypes(cols, names, declared),
        "n_treatment_missing": _count_missing(w),
        "n_outcome_missing": _count_missing(o),
    }


def select_rows(result: Mapping[str, Any], indices: Iterable[int]) -> dict[str, Any]:
    """Subset the row-aligned fields of an ``xy``/``survival_xy``/``uplift_xy`` result.

    ``x``, ``y``, ``time``, ``event``, ``treatment`` and ``outcome`` are
    subset by ``indices`` (negative indices follow Python rules; out-of-range
    raises IndexError). ``names`` and ``dtypes`` are copied unchanged. The
    ``n_*_missing`` counts are recomputed for the selected rows, so they
    always match the returned vectors. Rows are new lists.

    Args:
        result: A dict returned by one of the accessors in this module.
        indices: Row positions to keep, in the order wanted; repeats allowed.

    Returns:
        A new dict with the same keys as ``result``.

    Example:
        >>> res = xy({"a": [1, 2, 3], "y": [0, None, 1]}, "y")
        >>> sub = select_rows(res, [2, 0])
        >>> sub["x"], sub["y"], sub["n_y_missing"]
        ([[3], [1]], [1, 0], 0)
    """
    idx = [operator.index(i) for i in indices]
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key in ("names", "dtypes"):
            out[key] = list(value)
        elif key == "x":
            out[key] = [list(value[i]) for i in idx]
        elif isinstance(value, list):
            out[key] = [value[i] for i in idx]
        else:
            out[key] = value
    for count_key, field in _MISSING_COUNTS.items():
        if count_key in out and field in out:
            out[count_key] = _count_missing(out[field])
    return out
