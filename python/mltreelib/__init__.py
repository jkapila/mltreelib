"""mltreelib: tree-based machine learning with a Rust core.

The heavy lifting lives in ``mltreelib-core`` (Rust); this package exposes it
with a scikit-learn-shaped API without depending on scikit-learn.
"""

from mltreelib._core import (
    DatasetBuilder,
    GroupedScaler,
    IndexMapper,
    VariableScaler,
    calendar_features,
    decomposed_transform,
    generate,
    grouped_scale,
    grouped_transform,
    lag_by_days,
    summarize,
    trended_transform,
    windowed_transform,
)
from mltreelib.datasets import list_datasets, load_dataset

__all__ = [
    "DatasetBuilder",
    "GroupedScaler",
    "IndexMapper",
    "VariableScaler",
    "calendar_features",
    "decomposed_transform",
    "generate",
    "grouped_scale",
    "grouped_transform",
    "lag_by_days",
    "list_datasets",
    "load_dataset",
    "summarize",
    "trended_transform",
    "windowed_transform",
]
__version__ = "0.1.0"
