"""Rule extraction, pruning, and decision list modeling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from mltreelib._core import ModelTree as _CoreModelTree
from mltreelib._core import RuleSet as _CoreRuleSet
from mltreelib._model_inputs import _numeric_rows
from mltreelib.report import ModelReport, build_report

__all__ = ["RuleModel", "RuleResult"]


@dataclass
class RuleResult:
    """Fitted rule-set inference outputs."""

    rules: list[str]
    predictions: list[float]
    default_prediction: float
    default_confidence: float
    report: ModelReport


class RuleModel:
    """Unified Rule-Set Extraction and Pruning Model.

    Extracts, simplifies, and prunes transparent modular IF-THEN rules from decision trees
    using Quinlan's pessimistic error bound.

    Parameters
    ----------
    kind : str, default="c50"
        Rule mining family: ``"c50"``, ``"c45"``, or ``"tree"``.
    max_depth : int, default=6
        Maximum tree depth for the underlying decision tree.
    min_rows : int, default=20
        Minimum leaf samples per rule.
    prune : bool, default=True
        Whether to greedily prune rule preconditions using pessimistic error bounds.
    cf : float, default=0.25
        Quinlan's confidence factor CF for pessimistic pruning (default 0.25).
    """

    _snapshot_format = "mltreelib.rule_model"

    def __init__(
        self,
        kind: str = "c50",
        *,
        max_depth: int = 6,
        min_rows: int = 20,
        prune: bool = True,
        cf: float = 0.25,
    ) -> None:
        self.kind = kind.lower().strip()
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.prune = prune
        self.cf = cf

        self._fitted = False
        self._ruleset: _CoreRuleSet | None = None
        self._tree: _CoreModelTree | None = None
        self._n_samples = 0
        self._n_features = 0

    def fit(
        self,
        x: Sequence[Sequence[Any]] | _CoreModelTree,
        y: Sequence[float] | None = None,
    ) -> RuleModel:
        """Fit rule set on training data (X, y) or extract rules from an existing tree."""
        if hasattr(x, "extract_rules"):
            self._tree = x
            self._ruleset = x.extract_rules(prune=self.prune, cf=self.cf)
            self._fitted = True
            return self

        if y is None:
            raise ValueError("RuleModel.fit requires target values y when X is a feature matrix.")

        x_dense = _numeric_rows(x)
        self._n_samples = len(x_dense)
        if self._n_samples == 0:
            raise ValueError("RuleModel requires at least one training row.")
        self._n_features = len(x_dense[0])

        tree = _CoreModelTree(
            max_depth=self.max_depth,
            min_rows=self.min_rows,
        )
        tree.fit(x_dense, [float(v) for v in y])
        self._tree = tree
        self._ruleset = tree.extract_rules(
            prune=self.prune,
            cf=self.cf,
        )
        self._fitted = True
        return self

    def predict(self, x: Sequence[Sequence[Any]]) -> list[float]:
        """Predict target using the extracted rule set."""
        if not self._fitted or self._ruleset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        dense = _numeric_rows(x)
        return self._ruleset.predict(dense)

    @property
    def rules_(self) -> list[str]:
        """Extracted IF-THEN rules."""
        if not self._fitted or self._ruleset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return [str(r).strip() for r in self._ruleset.rules]

    @property
    def default_prediction_(self) -> float:
        """Fallback prediction when no rule fires."""
        if not self._fitted or self._ruleset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._ruleset.default_prediction

    @property
    def default_confidence_(self) -> float:
        """Confidence of the default fallback rule."""
        if not self._fitted or self._ruleset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return self._ruleset.default_confidence

    @property
    def report(self) -> ModelReport:
        """Return the structured ModelReport."""
        if not self._fitted or self._ruleset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        n_rules = len(self._ruleset)
        return build_report(
            "Rule Set Model",
            objective_name="Quinlan C4.5/C5.0 Rule Extraction and Pessimistic Pruning",
            objective_direction="minimize",
            objective_form="min pessimistic_error_bound(e, n, z(CF))",
            learner_family="Rule Induction",
            learner_kind=f"Rule Set ({self.kind.upper()})",
            learner_representation="Conjunctive IF-THEN propositional rules with default fallback",
            structure={
                "rule_count": n_rules,
                "default_prediction": f"{self.default_prediction_:.4f}",
                "default_confidence": f"{self.default_confidence_ * 100.0:.1f}%",
            },
            nominal_complexity=n_rules,
            capabilities=["predict", "rules"],
            controller_kind="rule_model",
            config=self.config(),
            converged=True,
            stop_reason="Rule extraction and greedy pruning completed",
            metrics={"rule_count": float(n_rules)},
        )

    def config(self) -> dict[str, Any]:
        """Return effective configuration dictionary."""
        return {
            "kind": self.kind,
            "max_depth": self.max_depth,
            "min_rows": self.min_rows,
            "prune": self.prune,
            "cf": self.cf,
        }

    def to_json(self) -> str:
        """Return fitted model snapshot as JSON string."""
        if not self._fitted or self._tree is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        return json.dumps(
            {
                "format": self._snapshot_format,
                "format_version": 1,
                "configuration": self.config(),
                "tree_json": self._tree.to_json(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> RuleModel:
        """Restore fitted model from versioned JSON string."""
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("format") != cls._snapshot_format:
            raise ValueError(f"Expected a {cls.__name__} snapshot.")
        cfg = payload["configuration"]
        model = cls(**cfg)
        tree = _CoreModelTree.from_json(payload["tree_json"])
        model._tree = tree
        model._ruleset = tree.extract_rules(prune=model.prune, cf=model.cf)
        model._fitted = True
        return model

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> RuleModel:
        return cls.from_json(data.decode("utf-8"))

    def save(self, path: str | Path) -> None:
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
    def load(cls, path: str | Path) -> RuleModel:
        return cls.from_bytes(Path(path).read_bytes())

    def __repr__(self) -> str:
        return f"RuleModel(kind={self.kind!r}, prune={self.prune}, cf={self.cf})"

    def __str__(self) -> str:
        if self._ruleset is not None:
            return str(self._ruleset)
        return repr(self)
