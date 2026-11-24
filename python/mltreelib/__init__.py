"""mltreelib: tree-based machine learning with a Rust core.

The heavy lifting lives in ``mltreelib-core`` (Rust); this package exposes it
with a scikit-learn-shaped API without depending on scikit-learn.

External modules like ``datasets`` (requires pyarrow) and ``plots`` (requires matplotlib)
are lazily loaded so the core package remains lean and dependency-free.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from mltreelib._controls import GrowthControl
from mltreelib._core import (
    HDBSCAN,
    ActiveAnomalyTree,
    BaseNEncoder,
    CatBoostEncoder,
    CLTree,
    ContextAnomalyTree,
    CountEncoder,
    DatasetBuilder,
    DensityEstimationTree,
    EmbeddingEncoder,
    ExtendedIsolationTree,
    GritBot,
    GroupedScaler,
    HalfSpaceTree,
    HashingEncoder,
    IndexMapper,
    JamesSteinEncoder,
    LeafControl,
    LeaveOneOutEncoder,
    MarsRegressor,
    MEstimatorEncoder,
    Metrics,
    ModelTree,
    OneHotEncoder,
    PredictiveClusteringTree,
    QuantileEncoder,
    RandomCutTree,
    Rule,
    RuleSet,
    SumEncoder,
    TargetEncoder,
    TfidfEncoder,
    TrieKNN,
    UnsupervisedRandomForest,
    VariableScaler,
    WoEEncoder,
    calendar_features,
    competing_risks_calibration,
    competing_risks_report,
    d_calibration,
    decomposed_transform,
    detect_quantile_crossings,
    fast_interaction_screen,
    generate,
    grouped_scale,
    grouped_transform,
    integrated_calibration_index,
    lag_by_days,
    one_calibration,
    summarize,
    trended_transform,
    windowed_transform,
    winnow,
)
from mltreelib.components import ComponentModel, ComponentResult
from mltreelib.models import (
    AnomalyModel,
    AnomalyResult,
    AnomalyTree,
    ClusterModel,
    ClusterResult,
    ClusterTree,
    RuleModel,
    RuleResult,
    distributions_to_quantiles,
    histograms_to_quantiles,
)
from mltreelib.report import ModelReport

if TYPE_CHECKING:
    from mltreelib import plots
    from mltreelib.datasets import dataset_path, list_datasets, load_dataset

# Lazy-loaded attributes for optional dataset and plotting extras
_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "list_datasets": ("mltreelib.datasets", "list_datasets"),
    "load_dataset": ("mltreelib.datasets", "load_dataset"),
    "dataset_path": ("mltreelib.datasets", "dataset_path"),
    "plots": ("mltreelib.plots", None),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        mod_name, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr) if attr is not None else mod
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "HDBSCAN",
    "ActiveAnomalyTree",
    "AnomalyModel",
    "AnomalyResult",
    "AnomalyTree",
    "BaseNEncoder",
    "CLTree",
    "CatBoostEncoder",
    "ClusterModel",
    "ClusterResult",
    "ClusterTree",
    "ComponentModel",
    "ComponentResult",
    "ContextAnomalyTree",
    "CountEncoder",
    "DatasetBuilder",
    "DensityEstimationTree",
    "EmbeddingEncoder",
    "ExtendedIsolationTree",
    "GritBot",
    "GroupedScaler",
    "GrowthControl",
    "HalfSpaceTree",
    "HashingEncoder",
    "IndexMapper",
    "JamesSteinEncoder",
    "LeafControl",
    "LeaveOneOutEncoder",
    "MEstimatorEncoder",
    "MarsRegressor",
    "Metrics",
    "ModelReport",
    "ModelTree",
    "OneHotEncoder",
    "PredictiveClusteringTree",
    "QuantileEncoder",
    "RandomCutTree",
    "Rule",
    "RuleModel",
    "RuleResult",
    "RuleSet",
    "SumEncoder",
    "TargetEncoder",
    "TfidfEncoder",
    "TrieKNN",
    "UnsupervisedRandomForest",
    "VariableScaler",
    "WoEEncoder",
    "calendar_features",
    "competing_risks_calibration",
    "competing_risks_report",
    "d_calibration",
    "dataset_path",
    "decomposed_transform",
    "detect_quantile_crossings",
    "distributions_to_quantiles",
    "fast_interaction_screen",
    "generate",
    "grouped_scale",
    "grouped_transform",
    "histograms_to_quantiles",
    "integrated_calibration_index",
    "lag_by_days",
    "list_datasets",
    "load_dataset",
    "one_calibration",
    "plots",
    "summarize",
    "trended_transform",
    "windowed_transform",
    "winnow",
]
__version__ = "0.4.14"
