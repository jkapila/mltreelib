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

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<ModelReport title={self.title!r}>"
