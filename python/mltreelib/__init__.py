"""mltreelib: tree-based machine learning with a Rust core.

The heavy lifting lives in ``mltreelib-core`` (Rust); this package exposes it
with a scikit-learn-shaped API without depending on scikit-learn.
"""

from mltreelib._core import (
    DatasetBuilder,
    GroupedScaler,
    IndexMapper,
    generate,
    grouped_scale,
    grouped_transform,
    summarize,
)

__all__ = [
    "DatasetBuilder",
    "GroupedScaler",
    "IndexMapper",
    "generate",
    "grouped_scale",
    "grouped_transform",
    "summarize",
]
__version__ = "0.1.0"
