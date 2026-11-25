"""One-shot, printable evaluation report built on :class:`mltreelib.Metrics`.

:class:`MetricsReport` runs the ``Metrics`` methods for one task and presents
every scalar and table they return: an overview, aligned tables and notes. It
computes no metric itself. Missing values (None or NaN) are counted and shown,
never imputed. Where ``Metrics`` would drop, recode or hang on a missing value,
the report refuses the input and names the argument instead.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any

from mltreelib._core import Metrics as _Metrics

__all__ = ["MetricsReport", "metrics_report"]

_TASKS = ("regression", "classification", "time_series", "survival", "uplift")

# Keyword arguments accepted per task.
_ARGS: dict[str, tuple[str, ...]] = {
    "regression": ("y_true", "y_pred", "groups"),
    "classification": ("y_true", "y_pred", "proba", "groups", "threshold"),
    "time_series": ("y_true", "y_pred", "y_train", "seasonal_period", "std_pred"),
    "survival": ("durations", "events", "risk", "surv_curves", "times"),
    "uplift": ("y_true", "treatment", "tau_pred"),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "regression": ("y_true", "y_pred"),
    "classification": ("y_true",),
    "time_series": ("y_true", "y_pred"),
    "survival": ("durations", "events", "risk"),
    "uplift": ("y_true", "treatment", "tau_pred"),
}
# Columns that print as whole numbers.
_COUNT_KEYS = frozenset({"n", "decile", "bucket", "class", "support", "events"})
# Text tables show the first 20 and last 5 rows of longer tables; .tables keeps all.
_TEXT_HEAD, _TEXT_TAIL = 20, 5


def _task_key(task: Any) -> str:
    if not isinstance(task, str):
        raise TypeError(f"task must be a string, got {type(task).__name__}")
    key = task.strip().lower()
    if key not in _ARGS:
        raise ValueError(f"unknown task {task!r}; valid tasks are: {', '.join(_TASKS)}")
    return key


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, numbers.Real) and math.isnan(value))


def _is_seq(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _as_list(name: str, value: Any) -> list[Any]:
    if not _is_seq(value):
        raise TypeError(f"{name} must be a sequence, got {type(value).__name__}")
    return list(value)


def _arg(
    task: str,
    name: str,
    value: Any,
    *,
    allow_missing: bool = False,
    integral: bool = False,
) -> list[Any]:
    """Copy a per-row argument; None becomes NaN (the missing marker Metrics reads)."""
    out: list[Any] = []
    for i, v in enumerate(_as_list(name, value)):
        if _missing(v):
            out.append(math.nan)
            continue
        if not isinstance(v, numbers.Real):
            raise TypeError(f"{name}[{i}] must be a number or None/NaN, got {v!r}")
        if integral and not float(v).is_integer():
            raise TypeError(f"{name}[{i}] must be an integer class label, got {v!r}")
        out.append(v)
    missing = sum(1 for v in out if _missing(v))
    if missing and not allow_missing:
        raise ValueError(
            f"{task} report: {name} has {missing} missing value(s) (None/NaN). Metrics would "
            "drop, recode or hang on them for this task, so the report refuses; remove those rows first."
        )
    return out


def _length(name: str, values: list[Any], n: int, ref: str = "y_true") -> None:
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} rows but {ref} has {n}")


def _only(name: str, values: list[Any], allowed: tuple[Any, ...]) -> None:
    bad = sorted({v for v in values if not _missing(v) and v not in allowed}, key=repr)
    if bad:
        raise ValueError(f"{name} must be one of {allowed}, found {bad}")


def _pairs(rows: list[Any], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [dict(zip(names, row)) for row in rows]


def _scalars(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, (list, dict))}


def _fmt(key: str, value: Any) -> str:
    """Text form of one value: 4 significant digits for floats, whole numbers for counts."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, numbers.Real):
        if (key in _COUNT_KEYS or key.startswith("n_")) and math.isfinite(value):
            return str(round(value))
        if isinstance(value, numbers.Integral):
            return str(int(value))
        x = float(value)
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.4g}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        x = float(value)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return str(value)


def _prepare(task: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate the arguments for ``task`` and return normalised columns."""
    extra = sorted(k for k, v in args.items() if v is not None and k not in _ARGS[task])
    if extra:
        raise TypeError(
            f"{task} report does not take {', '.join(extra)}; valid arguments: {', '.join(_ARGS[task])}"
        )
    args = {**dict.fromkeys(_ARGS[task]), **args}
    for name in _REQUIRED[task]:
        if args.get(name) is None:
            raise ValueError(f"{task} report requires {name}")
    if task == "classification" and args["y_pred"] is None and args["proba"] is None:
        raise ValueError("classification report requires y_pred (labels) or proba")

    if task == "regression":
        return _prep_regression(args)
    if task == "classification":
        return _prep_classification(args)
    if task == "time_series":
        return _prep_time_series(args)
    if task == "survival":
        return _prep_survival(args)
    return _prep_uplift(args)


def _check_any_pair(task: str, yt: list[Any], yp: list[Any]) -> None:
    if not any(not _missing(a) and not _missing(b) for a, b in zip(yt, yp)):
        raise ValueError(
            f"{task} report: no row has both y_true and y_pred observed "
            f"(y_true missing: {sum(map(_missing, yt))}, y_pred missing: {sum(map(_missing, yp))}, "
            f"of {len(yt)} rows); nothing to score"
        )


def _input_row(name: str, values: list[Any]) -> dict[str, Any]:
    return {"argument": name, "n": len(values), "n_missing": sum(map(_missing, values))}


def _prep_regression(args: dict[str, Any]) -> dict[str, Any]:
    t = "regression"
    yt = _arg(t, "y_true", args["y_true"], allow_missing=True)
    if not yt:
        raise ValueError("regression report: y_true is empty")
    yp = _arg(t, "y_pred", args["y_pred"], allow_missing=True)
    _length("y_pred", yp, len(yt))
    _check_any_pair(t, yt, yp)
    rows = [_input_row("y_true", yt), _input_row("y_pred", yp)]
    groups = None
    if args["groups"] is not None:
        groups, keys_missing = _group_keys(args["groups"], len(yt))
        rows.append({"argument": "groups", "n": len(groups), "n_missing": keys_missing})
    return {
        "y_true": yt,
        "y_pred": yp,
        "groups": groups,
        "input_rows": rows,
        "input_scalars": {},
    }


def _group_keys(value: Any, n: int) -> tuple[list[Any], int]:
    keys = _as_list("groups", value)
    _length("groups", keys, n)
    out = [None if _missing(k) else str(k) for k in keys]
    return out, sum(1 for k in out if k is None)


def _prep_classification(args: dict[str, Any]) -> dict[str, Any]:
    t = "classification"
    proba_given = args["proba"] is not None
    yt = _arg(t, "y_true", args["y_true"], allow_missing=not proba_given, integral=True)
    if not yt:
        raise ValueError("classification report: y_true is empty")
    n = len(yt)
    input_scalars: dict[str, Any] = {}
    rows = [_input_row("y_true", yt)]
    proba: list[Any] | None = None
    multi = False
    yp: list[Any] | None = None
    if args["y_pred"] is not None:
        yp = _arg(
            t, "y_pred", args["y_pred"], allow_missing=not proba_given, integral=True
        )
        _length("y_pred", yp, n)
    if args["proba"] is not None:
        proba, multi = _proba(args["proba"], n)
        rows.append({"argument": "proba", "n": n, "n_missing": 0})
    if args["threshold"] is not None and not proba_given:
        raise ValueError("classification report: threshold only applies with proba")
    if proba is not None and yp is None:
        if multi:
            raise ValueError(
                "classification report: multiclass proba needs y_pred; threshold only applies to binary proba"
            )
        threshold = 0.5 if args["threshold"] is None else args["threshold"]
        if not isinstance(threshold, numbers.Real) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}")
        yp = [1 if p >= threshold else 0 for p in proba]
        input_scalars["threshold"] = threshold
    elif args["threshold"] is not None:
        raise ValueError(
            "classification report: give y_pred or threshold, not both (threshold derives y_pred)"
        )
    assert yp is not None  # guaranteed by the required-argument checks
    _check_any_pair(t, yt, yp)
    rows.insert(1, _input_row("y_pred", yp))
    classes = sorted({v for v in yt + yp if not _missing(v)})
    if proba is not None and not multi:
        _only("labels with binary proba", yt + yp, (0, 1, 0.0, 1.0))
    if proba is not None and multi and len(proba[0]) != len(classes):
        raise ValueError(
            f"proba rows have {len(proba[0])} classes but y_true and y_pred contain {len(classes)} distinct labels"
        )
    groups = None
    if args["groups"] is not None:
        groups, keys_missing = _group_keys(args["groups"], n)
        rows.append({"argument": "groups", "n": n, "n_missing": keys_missing})
    return {
        "y_true": yt,
        "y_pred": yp,
        "proba": proba,
        "multi": multi,
        "classes": classes,
        "groups": groups,
        "input_rows": rows,
        "input_scalars": input_scalars,
    }


def _proba(value: Any, n: int) -> tuple[list[Any], bool]:
    rows = _as_list("proba", value)
    if len(rows) != n:
        raise ValueError(f"proba has {len(rows)} rows but y_true has {n}")
    if all(not _is_seq(r) for r in rows):
        probs = _arg("classification", "proba", rows)
        if any(not 0.0 <= p <= 1.0 for p in probs):
            raise ValueError("proba values must lie in [0, 1]")
        return probs, False
    if all(_is_seq(r) for r in rows):
        mat = [
            _arg("classification", f"proba row {i}", _as_list("proba", r))
            for i, r in enumerate(rows)
        ]
        if len({len(r) for r in mat}) != 1:
            raise ValueError("every proba row must have the same number of classes")
        if any(not 0.0 <= p <= 1.0 for r in mat for p in r):
            raise ValueError("proba values must lie in [0, 1]")
        return mat, True
    raise TypeError(
        "proba must be probabilities of the positive class (binary) or one row of class probabilities per row (multiclass)"
    )


def _prep_time_series(args: dict[str, Any]) -> dict[str, Any]:
    t = "time_series"
    yt = _arg(t, "y_true", args["y_true"])
    if not yt:
        raise ValueError("time_series report: y_true is empty")
    yp = _arg(t, "y_pred", args["y_pred"])
    _length("y_pred", yp, len(yt))
    rows = [_input_row("y_true", yt), _input_row("y_pred", yp)]
    scalars: dict[str, Any] = {}
    ytrain = None
    if args["y_train"] is not None:
        ytrain = _arg(t, "y_train", args["y_train"])
        rows.append(_input_row("y_train", ytrain))
    std = None
    if args["std_pred"] is not None:
        std = _arg(t, "std_pred", args["std_pred"])
        _length("std_pred", std, len(yt))
        rows.append(_input_row("std_pred", std))
    season = args["seasonal_period"]
    if season is not None:
        if (
            isinstance(season, bool)
            or not isinstance(season, numbers.Integral)
            or season < 1
        ):
            raise ValueError(
                f"seasonal_period must be a whole number >= 1, got {season!r}"
            )
        scalars["seasonal_period"] = int(season)
    return {
        "y_true": yt,
        "y_pred": yp,
        "y_train": ytrain,
        "std_pred": std,
        "seasonal_period": season,
        "input_rows": rows,
        "input_scalars": scalars,
    }


def _prep_survival(args: dict[str, Any]) -> dict[str, Any]:
    t = "survival"
    dur = _arg(t, "durations", args["durations"])
    if not dur:
        raise ValueError("survival report: durations is empty")
    n = len(dur)
    ev = _arg(t, "events", args["events"])
    _length("events", ev, n, "durations")
    _only("events (1 = event, 0 = censored)", ev, (0, 1, 0.0, 1.0))
    rk = _arg(t, "risk", args["risk"])
    _length("risk", rk, n, "durations")
    rows = [
        _input_row("durations", dur),
        _input_row("events", ev),
        _input_row("risk", rk),
    ]
    tm = None
    if args["times"] is not None:
        tm = _arg(t, "times", args["times"])
        rows.append(_input_row("times", tm))
    sc = None
    if args["surv_curves"] is not None:
        curves = _as_list("surv_curves", args["surv_curves"])
        _length("surv_curves", curves, n, "durations")
        sc = [_arg(t, f"surv_curves row {i}", r) for i, r in enumerate(curves)]
        if tm is not None and any(len(r) != len(tm) for r in sc):
            raise ValueError(
                f"every surv_curves row must have one value per entry of times ({len(tm)})"
            )
        rows.append({"argument": "surv_curves", "n": n, "n_missing": 0})
    return {
        "durations": dur,
        "events": ev,
        "risk": rk,
        "surv_curves": sc,
        "times": tm,
        "input_rows": rows,
        "input_scalars": {},
    }


def _prep_uplift(args: dict[str, Any]) -> dict[str, Any]:
    t = "uplift"
    yt = _arg(t, "y_true", args["y_true"])
    if not yt:
        raise ValueError("uplift report: y_true is empty")
    tr = _arg(t, "treatment", args["treatment"])
    _length("treatment", tr, len(yt))
    _only("treatment (1 = treated, 0 = control)", tr, (0, 1, 0.0, 1.0))
    tau = _arg(t, "tau_pred", args["tau_pred"])
    _length("tau_pred", tau, len(yt))
    rows = [
        _input_row("y_true", yt),
        _input_row("treatment", tr),
        _input_row("tau_pred", tau),
    ]
    return {
        "y_true": yt,
        "treatment": tr,
        "tau_pred": tau,
        "input_rows": rows,
        "input_scalars": {},
    }


class MetricsReport:
    """Printable evaluation report for one task, built from :class:`mltreelib.Metrics`.

    The primary interface. ``MetricsReport(task, y_true=..., y_pred=..., **inputs)``
    runs the matching ``Metrics`` methods and keeps every scalar and table they
    return. Use it as a value: ``print(rep)``, ``rep["rmse"]``, ``rep.to_dict()``.

    Tasks and their inputs (keyword names as shown):

    - ``regression``: ``y_true``, ``y_pred``, optional ``groups`` (one label per row).
    - ``classification``: ``y_true`` (integer labels), ``y_pred`` (labels) or ``proba``,
      optional ``groups`` and ``threshold``. ``proba`` is the positive-class
      probability per row (binary, labels 0 and 1) or one row of class
      probabilities per row (multiclass, with ``y_pred``). Without ``y_pred``,
      labels are ``proba >= threshold`` (default 0.5, binary only).
    - ``time_series``: ``y_true``, ``y_pred``, optional ``y_train``,
      ``seasonal_period``, ``std_pred``.
    - ``survival``: ``durations``, ``events`` (1 = event, 0 = censored), ``risk``
      (higher = shorter expected survival), optional ``surv_curves`` (one row
      of S(t) per subject) and ``times`` (the grid those rows refer to).
    - ``uplift``: ``y_true`` (outcome), ``treatment`` (0/1), ``tau_pred``.

    Missing values (None or NaN) are never imputed or recoded. Regression and
    label-based classification drop incomplete pairs inside ``Metrics`` and show
    the count (``n_missing``). Other arguments with missing values are refused
    with an error naming the argument, because ``Metrics`` would hang or give
    wrong numbers on them.

    Attributes:
        task: The task name, lower-cased.
        scalars: The overview values (a dict).
        tables: Table name to list of row dicts.

    Example:
        >>> rep = MetricsReport("regression", y_true=[3.0, 5.0, 2.5, 7.0],
        ...                     y_pred=[2.5, 5.0, 3.0, 8.0])
        >>> rep["n"], rep["mae"]
        (4, 0.5)
        >>> print(rep.to_text().splitlines()[0])
        Metrics report: regression
    """

    def __init__(
        self, task: str, y_true: Any = None, y_pred: Any = None, **inputs: Any
    ) -> None:
        self.task = _task_key(task)
        args: dict[str, Any] = {"y_true": y_true, "y_pred": y_pred, **inputs}
        cols = _prepare(self.task, args)
        self._metrics = _Metrics()
        self._notes: list[str] = []
        self._sections: list[dict[str, Any]] = []
        self._all: dict[str, Any] = {}
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._rows = len(cols.get("y_true", cols.get("durations", [])))
        self._add(
            "Inputs",
            cols["input_scalars"],
            [
                (
                    "inputs",
                    "Rows per argument (n_missing = None/NaN)",
                    cols["input_rows"],
                )
            ],
        )
        getattr(self, f"_build_{self.task}")(cols)
        self.scalars: dict[str, Any] = next(
            (dict(s["scalars"]) for s in self._sections if s["title"] == "Overview"), {}
        )

    # -- public surface -------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Return one scalar by name, e.g. ``rep["rmse"]``."""
        if key not in self._all:
            raise KeyError(
                f"no scalar {key!r} in this {self.task} report; see .scalars"
            )
        return self._all[key]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as nested plain Python data, safe for ``json.dumps``.

        NaN becomes None and infinities become the strings ``"inf"`` and ``"-inf"``.
        """
        return _json_safe(
            {
                "task": self.task,
                "scalars": self.scalars,
                "sections": [
                    {
                        "title": s["title"],
                        "scalars": s["scalars"],
                        "tables": [name for name, _ in s["tables"]],
                    }
                    for s in self._sections
                ],
                "tables": self.tables,
                "notes": self._notes,
            }
        )

    def to_json(self) -> str:
        """Return ``to_dict()`` as an indented JSON string."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavoured Markdown (every table in full)."""
        out = [f"# Metrics report: {self.task}", ""]
        for sec in self._sections:
            if not sec["scalars"] and not sec["tables"]:
                continue
            out += [f"## {sec['title']}", ""]
            if sec["scalars"]:
                out += _md_table(
                    ["metric", "value"],
                    [[k, _fmt(k, v)] for k, v in sec["scalars"].items()],
                )
                out.append("")
            for name, caption in sec["tables"]:
                out += [f"**{caption}**", ""]
                out += _md_rows(self.tables[name])
                out.append("")
        if self._notes:
            out += ["## Notes", ""] + [f"- {note}" for note in self._notes] + [""]
        return "\n".join(out).rstrip() + "\n"

    def to_text(self) -> str:
        """Render the report as aligned plain text (R/statsmodels style)."""
        title = f"Metrics report: {self.task}"
        out = [title, "=" * len(title), ""]
        for sec in self._sections:
            if not sec["scalars"] and not sec["tables"]:
                continue
            out += [sec["title"], "-" * len(sec["title"])]
            if sec["scalars"]:
                out += _text_scalars(sec["scalars"])
                out.append("")
            for name, caption in sec["tables"]:
                out += [f"{caption}:"] + _text_table(self.tables[name], name) + [""]
        if self._notes:
            out += ["Notes", "-----"] + [f"  - {note}" for note in self._notes] + [""]
        return "\n".join(out).rstrip() + "\n"

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"MetricsReport(task={self.task!r}, rows={self._rows})"

    # -- internals ------------------------------------------------------------

    def _add(
        self,
        title: str,
        scalars: dict[str, Any] | None = None,
        tables: list[tuple[str, str, list[dict[str, Any]]]] | tuple[()] = (),
    ) -> None:
        scalars = dict(scalars or {})
        self._sections.append(
            {
                "title": title,
                "scalars": scalars,
                "tables": [(name, caption) for name, caption, _ in tables],
            }
        )
        for name, _, rows in tables:
            self.tables[name] = rows
        self._all.update(scalars)

    def _run(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._metrics, method)(*args, **kwargs)
        except ValueError as exc:
            raise ValueError(
                f"{self.task} report: Metrics.{method} could not score the input: {exc}"
            ) from exc

    def _optional(self, method: str, *args: Any) -> Any:
        """Call a ranking metric that is undefined for some inputs; record a note instead of failing."""
        try:
            return getattr(self._metrics, method)(*args)
        except ValueError as exc:
            self._notes.append(f"{method} is undefined for this input: {exc}")
            return None

    def _build_regression(self, c: dict[str, Any]) -> None:
        r = self._run("regression", c["y_true"], c["y_pred"])
        self._add("Overview", _scalars(r))
        self._add(
            "Error by decile",
            tables=[
                ("error_deciles", "Error profile by target decile", r["deciles"]),
            ],
        )
        if c["groups"] is not None:
            g = self._run("regression_by_group", c["groups"], c["y_true"], c["y_pred"])
            self._add(
                "By group",
                tables=[
                    (
                        "by_group",
                        "Regression report per group",
                        [{"group": k, **_scalars(v)} for k, v in g.items()],
                    ),
                ],
            )
            self._group_note(c)

    def _group_note(self, c: dict[str, Any]) -> None:
        missing = sum(1 for k in c["groups"] if k is None)
        if missing:
            self._notes.append(
                f"{missing} row(s) with a missing group key are excluded from the by-group table "
                "(counted in the groups row of the inputs table)."
            )

    def _build_classification(self, c: dict[str, Any]) -> None:
        yt, yp, pr, multi = c["y_true"], c["y_pred"], c["proba"], c["multi"]
        binary = pr is not None and not multi
        if binary:
            rep = self._run("classification", yt, yp, probabilities=pr)
        else:
            rep = self._run("classification", yt, yp)
        overview = _scalars(rep)
        if not binary:
            for key in ("auc", "logloss", "brier"):
                overview.pop(key, None)
            if pr is not None:
                self._notes.append(
                    "Metrics computes log loss, Brier score and ROC AUC for binary probabilities only; "
                    "multiclass probabilities are scored one-vs-rest in the table below."
                )
        self._add("Overview", overview)
        if any(_missing(v) for v in yt + yp):
            self._notes.append(
                "classification_scores (balanced accuracy, MCC, micro and weighted averages, Jaccard) "
                "is skipped: Metrics.classification_scores does not accept missing labels."
            )
        else:
            score = self._run("classification_scores", yt, yp)
            self._add(
                "Scorecard",
                {
                    k: v
                    for k, v in score.items()
                    if k not in ("n", "n_classes", "accuracy")
                },
            )
        labels = [row["class"] for row in rep["per_class"]]
        confusion = [
            {"actual": lab, **{f"pred {p}": int(v) for p, v in zip(labels, row)}}
            for lab, row in zip(labels, rep["confusion"])
        ]
        self._add(
            "Per class",
            tables=[
                (
                    "per_class",
                    "Precision, recall, F1 and support per class",
                    rep["per_class"],
                ),
                (
                    "confusion",
                    "Confusion matrix (rows = actual, columns = predicted)",
                    confusion,
                ),
            ],
        )
        if binary:
            self._binary_curves(yt, pr)
        elif pr is not None:
            self._one_vs_rest(yt, pr, c["classes"])
        if c["groups"] is not None:
            g = self._run("classification_by_group", c["groups"], yt, yp)
            rows = [{"group": k, **_scalars(v)} for k, v in g.items()]
            for row in (
                rows
            ):  # per-group reports get no probabilities, so these are never computed
                for key in ("auc", "logloss", "brier"):
                    row.pop(key, None)
            self._add(
                "By group",
                tables=[("by_group", "Classification report per group", rows)],
            )
            self._group_note(c)

    def _binary_curves(self, yt: list[Any], pr: list[Any]) -> None:
        pr_curve = self._optional("pr_curve", yt, pr)
        ks = self._optional("ks_statistic", yt, pr)
        roc = self._optional("roc_curve", yt, pr)
        lift = self._optional("lift_curve", yt, pr)
        gain = self._optional("gain_curve", yt, pr)
        cal = self._optional("calibration", yt, pr)
        dec = self._optional("decile_table", yt, pr)
        scalars = {
            "average_precision": pr_curve["average_precision"] if pr_curve else None,
            "ks": ks["ks"] if ks else None,
            "score_at_ks": ks["score_at_ks"] if ks else None,
            "tpr_at_ks": ks["tpr_at_ks"] if ks else None,
            "fpr_at_ks": ks["fpr_at_ks"] if ks else None,
        }
        tables = []
        if roc:
            tables.append(
                (
                    "roc_points",
                    "ROC curve points",
                    _pairs(roc["points"], ("fpr", "tpr")),
                )
            )
        if pr_curve:
            tables.append(
                (
                    "pr_points",
                    "Precision-recall curve points",
                    _pairs(pr_curve["points"], ("recall", "precision")),
                )
            )
        if ks:
            tables.append(
                ("ks_points", "KS sweep", _pairs(ks["points"], ("fpr", "tpr", "score")))
            )
        if dec is not None:
            tables.append(("decile", "Decile table (decile 1 = highest scores)", dec))
        if lift:
            tables.append(
                (
                    "lift_points",
                    "Cumulative lift curve",
                    _pairs(lift["points"], ("population_fraction", "lift")),
                )
            )
        if gain:
            tables.append(
                (
                    "gain_points",
                    "Cumulative gains curve",
                    _pairs(gain["points"], ("population_fraction", "capture_rate")),
                )
            )
        if cal is not None:
            tables.append(("calibration", "Calibration bins", cal["bins"]))
        self._add("Ranking and calibration", scalars, tables)

    def _one_vs_rest(
        self, yt: list[Any], pr: list[list[Any]], classes: list[Any]
    ) -> None:
        rows = []
        for k, cls in enumerate(classes):
            ind = [float(t == cls) for t in yt]
            scores = [row[k] for row in pr]
            roc = self._optional("roc_curve", ind, scores)
            pr_c = self._optional("pr_curve", ind, scores)
            brier = self._optional("brier_score", ind, scores)
            rows.append(
                {
                    "class": cls,
                    "n_positive": sum(1 for t in yt if t == cls),
                    "auc": roc["auc"] if roc else None,
                    "average_precision": pr_c["average_precision"] if pr_c else None,
                    "brier": brier,
                }
            )
        self._add(
            "One-vs-rest probabilities",
            tables=[
                (
                    "one_vs_rest",
                    "Per-class ROC AUC, average precision and Brier score",
                    rows,
                ),
            ],
        )

    def _build_time_series(self, c: dict[str, Any]) -> None:
        r = self._run(
            "time_series",
            c["y_true"],
            c["y_pred"],
            y_train=c["y_train"],
            seasonal_period=c["seasonal_period"],
            std_pred=c["std_pred"],
        )
        self._add("Overview", _scalars(r))

    def _build_survival(self, c: dict[str, Any]) -> None:
        r = self._run(
            "survival",
            c["durations"],
            c["events"],
            c["risk"],
            surv_curves=c["surv_curves"],
            times=c["times"],
        )
        self._add(
            "Overview",
            _scalars(r),
            [
                (
                    "brier_at_horizons",
                    "Brier score per horizon",
                    _pairs(r["brier_at_horizons"], ("time", "brier")),
                )
            ]
            if "brier_at_horizons" in r
            else [],
        )

    def _build_uplift(self, c: dict[str, Any]) -> None:
        r = self._run("uplift", c["y_true"], c["treatment"], c["tau_pred"])
        self._add(
            "Overview",
            _scalars(r),
            [
                (
                    "qini_curve",
                    "Qini curve points",
                    _pairs(r["qini_curve"], ("population_fraction", "qini")),
                ),
                (
                    "cumulative_gain",
                    "Cumulative gain points",
                    _pairs(r["cumulative_gain"], ("population_fraction", "gain")),
                ),
            ],
        )
        self._notes.append(
            "Metrics.uplift gives no per-decile uplift table; top_decile_uplift and "
            "top_quintile_uplift are the decile-level figures it returns."
        )


def _text_scalars(scalars: dict[str, Any]) -> list[str]:
    width = max(len(k) for k in scalars)
    values = {k: _fmt(k, v) for k, v in scalars.items()}
    vwidth = max(len(v) for v in values.values())
    return [f"  {k:<{width}}  {values[k]:>{vwidth}}" for k in scalars]


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    return cols


def _text_table(rows: list[dict[str, Any]], name: str) -> list[str]:
    if not rows:
        return ["  (no rows)"]
    shown: list[dict[str, Any]] = rows
    omitted = 0
    if len(rows) > _TEXT_HEAD + _TEXT_TAIL:
        omitted = len(rows) - _TEXT_HEAD - _TEXT_TAIL
        shown = rows[:_TEXT_HEAD] + rows[-_TEXT_TAIL:]
    cols = _columns(rows)
    cells = [[_fmt(c, r.get(c)) for c in cols] for r in shown]
    widths = [
        max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)
    ]
    numeric = [
        all(r.get(c) is None or isinstance(r.get(c), numbers.Real) for r in rows)
        for c in cols
    ]

    def line(parts: list[str]) -> str:
        return (
            "  "
            + "  ".join(
                p.rjust(widths[i]) if numeric[i] else p.ljust(widths[i])
                for i, p in enumerate(parts)
            ).rstrip()
        )

    out = [line(cols)] + [line(row) for row in cells]
    if omitted:
        out.insert(
            1 + _TEXT_HEAD,
            f"  ... {omitted} rows omitted; full table in .tables[{name!r}]",
        )
    return out


def _md_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    cols = _columns(rows)
    numeric = [
        all(r.get(c) is None or isinstance(r.get(c), numbers.Real) for r in rows)
        for c in cols
    ]
    body = [[_fmt(c, r.get(c)) for c in cols] for r in rows]
    return _md_table(cols, body, numeric)


def _md_table(
    header: list[str], body: list[list[str]], numeric: list[bool] | None = None
) -> list[str]:
    numeric = numeric or [False] * len(header)

    def esc(s: Any) -> str:
        return str(s).replace("|", "\\|")

    align = ["---:" if num else ":---" for num in numeric]
    out = [
        "| " + " | ".join(esc(h) for h in header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    out += ["| " + " | ".join(esc(cell) for cell in row) + " |" for row in body]
    return out


def metrics_report(
    task: str, y_true: Any = None, y_pred: Any = None, **inputs: Any
) -> MetricsReport:
    """Build a :class:`MetricsReport`; a thin alias of ``MetricsReport(...)``.

    Example:
        >>> metrics_report("regression", y_true=[1.0, 2.0], y_pred=[1.0, 2.0])["rmse"]
        0.0
    """
    return MetricsReport(task, y_true, y_pred, **inputs)
