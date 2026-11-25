"""A dataset that holds its columns and their roles, with the data helpers as methods.

``Dataset`` is a thin layer over the helper functions in this package. It
holds the columns once, records which columns are the target, survival time
and event, uplift treatment, groups and weights, and exposes the helpers as
methods, so callers do not have to remember which function takes which
arguments. Every method delegates to the function it names; no logic is
duplicated here.

A ``Dataset`` never changes after construction. Splits, folds and ``take``
return new objects. Missing values (None and NaN) are never imputed, dropped,
recoded or reordered, and the caller's data is never modified.

Column kinds (``"numeric"`` or ``"categorical"``) are decided once, at
construction, from the declared dtypes of a ``load_dataset`` result or from
the inference in ``mltreelib.data.access``. Every child dataset keeps those
kinds, so a train subset never re-infers a column differently from the full
data.

Example:
    >>> ds = Dataset({"a": [1.0, 2.0, None, 4.0], "y": [0, 1, 1, 0]}, target="y")
    >>> ds.names, ds.dtypes, ds.n_rows
    (['a'], ['numeric'], 4)
    >>> train, test = ds.split(test_size=0.5, seed=0)
    >>> train.n_rows, test.n_rows
    (2, 2)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mltreelib.data import assoc as _assoc
from mltreelib.data import binning as _binning
from mltreelib.data import profile as _profile
from mltreelib.data._common import is_missing
from mltreelib.data.access import _feature_names, _infer_dtype, _n_rows, _resolve, _rows
from mltreelib.data.split import kfold as _kfold
from mltreelib.data.split import take as _take
from mltreelib.data.split import train_test_split as _train_test_split
from mltreelib.data.split import train_val_test_split as _train_val_test_split

__all__ = ["Dataset"]

_KINDS = ("numeric", "categorical")
# stratify=True on a numeric column with more distinct values than this would make
# (almost) every value a class of one, so the test side comes out empty.
_MAX_AUTO_LEVELS = 20


class Dataset:
    """Columns plus their roles, with the data helpers as methods.

    Args:
        columns: A dict of column name to list, a ``load_dataset`` result, or
            any table ``mltreelib.data._common.as_columns`` accepts (for example
            a data frame).
        target: Name of the target column for supervised tasks.
        features: Explicit feature names, in the order wanted. Role columns
            cannot be features.
        exclude: Column names to leave out of the default features.
        event: Survival event column.
        time: Survival follow-up time column.
        treatment: Uplift treatment column.
        outcome: Uplift outcome column. It is the target, so it cannot be
            combined with a different ``target``.
        groups: Column of group ids, for example for grouped splits.
        weights: Column of per-row sample weights.
        name: Optional label. A ``load_dataset`` result supplies its own name.

    Raises:
        KeyError: If a role or ``exclude`` names a column that does not exist.
            The message names the column.
        ValueError: If a feature is also a role, one column is used for two
            roles, ``target`` and ``outcome`` differ, or the columns have
            different lengths.

    Example:
        >>> ds = Dataset({"age": [30, 40], "t": [5.0, 9.0], "died": [1, 0]},
        ...              time="t", event="died")
        >>> ds.names
        ['age']
    """

    def __init__(
        self,
        columns: Any,
        target: str | None = None,
        *,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = (),
        event: str | None = None,
        time: str | None = None,
        treatment: str | None = None,
        outcome: str | None = None,
        groups: str | None = None,
        weights: str | None = None,
        name: str | None = None,
    ) -> None:
        cols, declared = _resolve(columns)
        if name is None and _is_load_result(columns):
            name = columns.get("name")
        n = _n_rows(cols)
        kinds: dict[str, str] = {}
        for col, values in cols.items():
            dt = declared.get(col)
            kinds[col] = dt if dt in _KINDS else _infer_dtype(values)

        if target is not None and outcome is not None and target != outcome:
            raise ValueError(
                f"target={target!r} and outcome={outcome!r} name different columns; "
                "for uplift pass the outcome as outcome= (it is the target)"
            )
        roles: dict[str, str | None] = {
            "target": target if target is not None else outcome,
            "time": time,
            "event": event,
            "treatment": treatment,
            "groups": groups,
            "weights": weights,
        }
        owner: dict[str, str] = {}
        for role, col in roles.items():
            if col is None:
                continue
            if col not in cols:
                raise KeyError(f"{role} column {col!r} is not in columns")
            if col in owner:
                raise ValueError(
                    f"column {col!r} is used as both {owner[col]} and {role}"
                )
            owner[col] = role

        chosen = None if features is None else _as_list(features)
        for col in chosen or ():
            if col in owner:
                raise ValueError(
                    f"{col!r} is the {owner[col]} column and cannot also be a feature"
                )
        names = _feature_names(cols, list(owner), chosen, exclude)
        self._setup(cols, kinds, names, n, roles, name)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        name: str,
        target: str | None = None,
        *,
        auto_download: bool = True,
        **roles: Any,
    ) -> Dataset:
        """Load a benchmark dataset with ``load_dataset`` and attach roles.

        Args:
            name: Dataset name, as ``mltreelib.datasets.load_dataset`` takes it.
            target: Target column, as in the constructor.
            auto_download: Passed to ``load_dataset``.
            **roles: Any other constructor argument (``time``, ``event``,
                ``treatment``, ``outcome``, ``features``, ``exclude``, ...).

        Returns:
            A new ``Dataset``.

        Raises:
            ValueError: If the dataset is unknown or not available locally.
            ImportError: If ``pyarrow`` is missing.
        """
        from mltreelib.datasets import load_dataset  # lazy: avoids an import cycle

        return cls(load_dataset(name, auto_download=auto_download), target, **roles)

    def _setup(
        self,
        cols: dict[str, list[Any]],
        kinds: dict[str, str],
        names: list[str],
        n: int,
        roles: dict[str, str | None],
        name: str | None,
    ) -> None:
        self._cols = cols
        self._kinds = kinds
        self._names = list(names)
        self._n = n
        self._roles = dict(roles)
        self._name = name
        self._x = _rows(cols, self._names, n)

    def _subset(self, positions: Any) -> Dataset:
        idx = list(positions)
        child = Dataset.__new__(Dataset)
        child._setup(
            _take(self._cols, idx),
            self._kinds,
            self._names,
            len(idx),
            self._roles,
            self._name,
        )
        return child

    # -- read-only views ----------------------------------------------------

    @property
    def columns(self) -> dict[str, list[Any]]:
        """Every column, as fresh lists (roles and features alike)."""
        return {name: list(values) for name, values in self._cols.items()}

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return self._n

    @property
    def names(self) -> list[str]:
        """Feature names, in order."""
        return list(self._names)

    @property
    def dtypes(self) -> list[str]:
        """``"numeric"`` or ``"categorical"`` per feature, in ``names`` order."""
        return [self._kinds[c] for c in self._names]

    @property
    def x(self) -> list[list[Any]]:
        """Feature rows: one new list per row, missing values kept."""
        return [list(row) for row in self._x]

    @property
    def y(self) -> list[Any] | None:
        """Target values, or None without a target role."""
        return self._role_values("target")

    @property
    def time(self) -> list[Any] | None:
        """Survival time values, or None."""
        return self._role_values("time")

    @property
    def event(self) -> list[Any] | None:
        """Survival event codes, passed through unchanged, or None."""
        return self._role_values("event")

    @property
    def treatment(self) -> list[Any] | None:
        """Uplift treatment codes, passed through unchanged, or None."""
        return self._role_values("treatment")

    @property
    def groups(self) -> list[Any] | None:
        """Group ids, or None."""
        return self._role_values("groups")

    @property
    def weights(self) -> list[Any] | None:
        """Sample weights, or None."""
        return self._role_values("weights")

    def _role_values(self, role: str) -> list[Any] | None:
        col = self._roles[role]
        return None if col is None else list(self._cols[col])

    def _feature_cols(self) -> dict[str, list[Any]]:
        return {name: self._cols[name] for name in self._names}

    def _need_target(self, what: str) -> list[Any]:
        if self._roles["target"] is None:
            raise ValueError(f"{what} needs a target role; pass target= (or outcome=)")
        return self.y  # type: ignore[return-value]

    # -- printable results (delegate to the helpers) -----------------------

    def describe(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Per-column statistics: ``profile.describe`` on every column."""
        return _profile.describe(self._cols, **kwargs)

    def missing(self, **kwargs: Any) -> dict[str, Any]:
        """Missingness by column, row and pattern: ``profile.missing_profile``."""
        return _profile.missing_profile(self._cols, **kwargs)

    def missing_vs_target(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Target mean when each feature is missing or observed (needs a target)."""
        y = self._need_target("missing_vs_target")
        return _profile.missing_vs_target(self._feature_cols(), y, **kwargs)

    def checks(self) -> dict[str, Any]:
        """Feature-level checks, computed on the features only.

        Returns:
            A dict with ``constant_columns``, ``near_zero_variance``,
            ``duplicate_rows`` (``n_groups``, ``n_extra_rows`` = rows that repeat
            an earlier row, and ``groups`` of row indices) and ``class_balance``
            (None unless the target is categorical or has exactly two values).
        """
        feats = self._feature_cols()
        dup = _profile.duplicate_rows(feats) if feats else []
        out: dict[str, Any] = {
            "constant_columns": _profile.constant_columns(feats) if feats else [],
            "near_zero_variance": _profile.near_zero_variance(feats) if feats else [],
            "duplicate_rows": {
                "n_groups": len(dup),
                "n_extra_rows": sum(len(group) - 1 for group in dup),
                "groups": dup,
            },
            "class_balance": None,
        }
        target = self._roles["target"]
        if target is not None:
            y = self.y
            distinct = {v for v in y if not is_missing(v)}  # type: ignore[union-attr]
            if self._kinds[target] == "categorical" or len(distinct) == 2:
                out["class_balance"] = _profile.class_balance(y)  # type: ignore[arg-type]
        return out

    def associations(self) -> dict[str, Any]:
        """Association checks on the features (plus leakage hints when there is a target).

        Returns:
            A dict with ``correlation_matrix``, ``cramers_v_matrix``, ``vif`` and
            ``leakage_hints`` (None without a target).
        """
        feats = self._feature_cols()
        out: dict[str, Any] = {
            "correlation_matrix": _assoc.correlation_matrix(feats),
            "cramers_v_matrix": _assoc.cramers_v_matrix(feats),
            "vif": _assoc.vif(feats),
            "leakage_hints": None,
        }
        if self._roles["target"] is not None:
            out["leakage_hints"] = _assoc.leakage_hints(feats, self.y)  # type: ignore[arg-type]
        return out

    def iv(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Information value per feature. The target must be binary.

        The event level is the larger of the two target values, as in
        ``binning._event_flags`` (for strings, the alphabetically larger one).
        """
        y = self._need_target("iv")
        return _binning.information_value_table(self._feature_cols(), y, **kwargs)

    def schema(self) -> dict[str, Any]:
        """Schema of every column, for later validation: ``profile.schema_capture``."""
        return _profile.schema_capture(self._cols)

    def summary(self) -> str:
        """Aligned text table of ``describe()``.

        Numeric columns show the mean, std, min, median and max. Categorical
        columns show the distinct count and the mode. The other cells are blank.
        """
        header = [
            "name",
            "kind",
            "n",
            "missing%",
            "mean",
            "std",
            "min",
            "median",
            "max",
            "distinct",
            "mode",
        ]
        body = []
        for row in self.describe():
            if row["mean"] is not None:
                stats = [
                    row["mean"],
                    row["std"],
                    row["min"],
                    row["median"],
                    row["max"],
                    None,
                    None,
                ]
            else:
                stats = [None, None, None, None, None, row["n_distinct"], row["mode"]]
            body.append(
                [
                    row["name"],
                    row["kind"],
                    _cell(row["n"]),
                    f"{row['pct_missing']:.1f}",
                    *(_cell(v) for v in stats),
                ]
            )
        table = [header, *body]
        widths = [max(len(line[i]) for line in table) for i in range(len(header))]
        return "\n".join(
            "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
            for line in table
        )

    def __str__(self) -> str:
        label = self._name if self._name is not None else "(unnamed)"
        lines = [f"Dataset {label}: {self._n} rows, {len(self._names)} features"]
        roles = [
            f"{role}={col}" for role, col in self._roles.items() if col is not None
        ]
        lines.append("roles: " + (", ".join(roles) if roles else "none"))
        lines.append("features:")
        width = max((len(c) for c in self._names), default=0)
        for col in self._names:
            missing = sum(1 for v in self._cols[col] if is_missing(v))
            lines.append(
                f"  {col.ljust(width)}  {self._kinds[col]:<11}  missing {missing}"
            )
        return "\n".join(lines)

    # -- splitting ----------------------------------------------------------

    def take(self, indices: Sequence[int]) -> Dataset:
        """New dataset with these rows, in the order given. Repeats are allowed."""
        return self._subset(indices)

    def split(
        self,
        test_size: float = 0.2,
        seed: int = 0,
        stratify: bool | str | Sequence[str] | None = None,
        groups: str | Sequence[str] | None = None,
        time: bool | str | None = None,
        val_size: float | None = None,
    ) -> tuple[Dataset, Dataset] | tuple[Dataset, Dataset, Dataset]:
        """Split into train and test, or train, validation and test.

        Args:
            test_size: Fraction in (0, 1), or an int count of test rows.
            seed: Seed for the shuffle. The same seed gives the same split.
            stratify: ``True`` stratifies on the first of these roles that is set:
                ``treatment`` (uplift), then ``event`` (survival), then ``target``.
                A column name, or a list of names, stratifies on those columns.
                ``True`` raises if the chosen column is numeric with more than 20
                distinct values, since every value would be a class of one.
            groups: A column name, or a list of names, whose groups never split.
                Not taken from the ``groups`` role automatically.
            time: A column name (rows ordered by it, latest rows go to test) or
                ``True`` (row order). No shuffling. Cannot combine with stratify
                or groups.
            val_size: If given, a validation set is also cut from the rows left
                after the test split (a fraction of all rows).

        Returns:
            ``(train, test)``, or ``(train, val, test)`` when ``val_size`` is given.
            Rows are never dropped or changed; missing values stay missing.

        Raises:
            ValueError: For an invalid mode (see the helpers), a continuous column
                under ``stratify=True``, or no role to stratify on.
            KeyError: If a named column does not exist.
        """
        strat = self._stratify_labels(stratify)
        grp = None if groups is None else self._labels_for(groups, "groups")
        order = self._time_keys(time)
        if val_size is None:
            train, test = _train_test_split(
                self._n,
                test_size=test_size,
                seed=seed,
                stratify=strat,
                groups=grp,
                time_order=order,
            )
            return self._subset(train), self._subset(test)
        train, val, test = _train_val_test_split(
            self._n,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            stratify=strat,
            groups=grp,
            time_order=order,
        )
        return self._subset(train), self._subset(val), self._subset(test)

    def kfold(
        self,
        k: int = 5,
        seed: int = 0,
        stratify: bool | str | Sequence[str] | None = None,
        groups: str | Sequence[str] | None = None,
        time: bool | str | None = None,
    ) -> list[tuple[Dataset, Dataset]]:
        """``k`` (train, test) pairs. Arguments as in :meth:`split`.

        Every row is in exactly one test fold. With ``time``, the folds form an
        expanding window: block 0 is never tested.
        """
        strat = self._stratify_labels(stratify)
        grp = None if groups is None else self._labels_for(groups, "groups")
        order = self._time_keys(time)
        folds = _kfold(
            self._n, k=k, seed=seed, stratify=strat, groups=grp, time_order=order
        )
        return [(self._subset(train), self._subset(test)) for train, test in folds]

    def _stratify_labels(self, stratify: Any) -> list[Any] | None:
        if stratify is None or stratify is False:
            return None
        if stratify is not True:
            return self._labels_for(stratify, "stratify")
        for role in ("treatment", "event", "target"):
            col = self._roles[role]
            if col is not None:
                break
        else:
            raise ValueError(
                "stratify=True needs a treatment, event or target role; "
                "or pass the column name(s) to stratify on"
            )
        distinct = {v for v in self._cols[col] if not is_missing(v)}
        if self._kinds[col] == "numeric" and len(distinct) > _MAX_AUTO_LEVELS:
            raise ValueError(
                f"stratify=True on numeric column {col!r} with {len(distinct)} distinct "
                "values would make classes of one; stratify on a categorical column "
                "name instead"
            )
        return self._labels_for(col, "stratify")

    def _labels_for(self, spec: Any, what: str) -> list[Any]:
        names = _as_list(spec)
        if not names:
            raise ValueError(f"{what} needs at least one column name")
        for col in names:
            if col not in self._cols:
                raise KeyError(f"{what} column {col!r} is not in columns")
        if len(names) == 1:
            return list(self._cols[names[0]])
        return [
            tuple(None if is_missing(v) else v for v in row)
            for row in zip(*(self._cols[c] for c in names))
        ]

    def _time_keys(self, time: Any) -> list[Any] | None:
        if time is None or time is False:
            return None
        if time is True:
            return list(range(self._n))
        if isinstance(time, str):
            return self._labels_for(time, "time")
        raise TypeError("time must be a column name or True (row order)")

    # -- model hand-off -----------------------------------------------------

    def fit_args(self) -> dict[str, Any]:
        """Keyword arguments for ``ModelTree.fit``: ``model.fit(**ds.fit_args())``.

        ``y`` is the time column when a time role is set (survival; ``event=`` is
        then passed too), otherwise the target. ``treatment=`` and
        ``sample_weight=`` are passed when those roles are set. Feature rows,
        ``col_names`` and ``dtypes`` always are.

        Raises:
            ValueError: If there is neither a target nor a time role.
        """
        y = self.time if self._roles["time"] is not None else self.y
        if y is None:
            raise ValueError("fit_args needs a target or time role")
        args: dict[str, Any] = {
            "x": self.x,
            "y": y,
            "col_names": self.names,
            "dtypes": self.dtypes,
        }
        if self._roles["event"] is not None:
            args["event"] = self.event
        if self._roles["treatment"] is not None:
            args["treatment"] = self.treatment
        if self._roles["weights"] is not None:
            args["sample_weight"] = self.weights
        return args

    def predict_args(self) -> dict[str, Any]:
        """Keyword arguments for ``ModelTree.predict``: ``model.predict(**ds.predict_args())``."""
        return {"x": self.x}


def _is_load_result(columns: Any) -> bool:
    return isinstance(columns, Mapping) and isinstance(columns.get("columns"), Mapping)


def _as_list(spec: Any) -> list[Any]:
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
