"""Unified Model Report Python Surface."""

from __future__ import annotations

from typing import Any

from mltreelib._core import ModelReport as _CoreModelReport

__all__ = ["ModelReport"]


class ModelReport:
    """Unified Model Report containing Objective, Learner, Controller, Diagnostics, and optional Inference.

    Wraps the core Rust `ModelReport` representation and renders to text (R/statsmodels-style),
    Markdown, and dictionary/JSON forms.
    """

    def __init__(self, core_report: _CoreModelReport) -> None:
        self._inner = core_report

    @property
    def title(self) -> str:
        """Report title."""
        return self._inner.title

    def to_text(self) -> str:
        """Render report as R/statsmodels-style aligned console text."""
        return self._inner.to_text()

    def to_markdown(self) -> str:
        """Render report as GitHub-flavored Markdown."""
        return self._inner.to_markdown()

    def to_json(self) -> str:
        """Render report as formatted JSON string."""
        return self._inner.to_json()

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return self._inner.to_dict()

    @classmethod
    def from_json(cls, json_str: str) -> ModelReport:
        """Create ModelReport from JSON string."""
        return cls(_CoreModelReport.from_json(json_str))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelReport:
        """Create ModelReport from dictionary structure."""
        import json
        return cls.from_json(json.dumps(data))

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<ModelReport title={self.title!r}>"


def build_report(
    title: str,
    *,
    objective_name: str,
    objective_direction: str = "minimize",
    objective_form: str | None = None,
    final_loss: float | None = None,
    objective_components: list[tuple[str, float]] | None = None,
    units: str | None = None,
    learner_family: str,
    learner_kind: str,
    learner_representation: str,
    structure: dict[str, Any] | None = None,
    nominal_complexity: int = 0,
    effective_complexity: float | None = None,
    capabilities: list[str] | None = None,
    controller_kind: str,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    iteration_budget: int | None = None,
    evaluation_budget: int | None = None,
    missing_policy: str = "error_on_missing",
    converged: bool | None = None,
    stop_reason: str | None = None,
    fit_time_sec: float | None = None,
    eval_count: int | None = None,
    metrics: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> ModelReport:
    """Helper to construct a compliant 4-section ModelReport from Python models."""
    struct_map = {k: str(v) for k, v in (structure or {}).items()}
    cfg_map = {k: str(v) for k, v in (config or {}).items()}
    payload = {
        "title": title,
        "version": 1,
        "objective": {
            "name": objective_name,
            "math_form": objective_form,
            "direction": objective_direction,
            "final_value": final_loss,
            "components": objective_components or [],
            "units": units,
        },
        "learner": {
            "family": learner_family,
            "kind": learner_kind,
            "representation": learner_representation,
            "structure": struct_map,
            "nominal_complexity": nominal_complexity,
            "effective_complexity": effective_complexity,
            "capabilities": capabilities or [],
        },
        "controller": {
            "kind": controller_kind,
            "config": cfg_map,
            "seed": seed,
            "iteration_budget": iteration_budget,
            "evaluation_budget": evaluation_budget,
            "missing_policy": missing_policy,
            "fallback_reason": None,
        },
        "diagnostics": {
            "converged": converged,
            "stop_reason": stop_reason,
            "fit_time_sec": fit_time_sec,
            "eval_count": eval_count,
            "metrics": metrics or {},
            "warnings": warnings or [],
        },
        "inference": None,
    }
    return ModelReport.from_dict(payload)
