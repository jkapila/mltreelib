"""Normalize structured model configuration before the native constructor."""

from difflib import get_close_matches
from functools import cache
from inspect import signature

_GROWTH_FIELDS = frozenset(
    [
        "max_depth",
        "min_rows",
        "minsplit",
        "minbucket",
        "minprob",
        "max_leaf_nodes",
        "min_gain",
        "purity_gate",
        "purity_gate_metric",
        "tail_quantile",
        "gain",
        "split_precision",
        "split_cols",
        "col_sample",
        "seed",
        "cat_order",
        "cat_search",
        "tree_monotone",
        "col_weights",
        "interaction_groups",
        "alpha",
        "imbalance_penalty",
        "prune",
        "cp_alpha",
        "cp",
        "honesty",
        "honesty_fraction",
        "maxsurrogate",
        "split_test",
        "split_alpha",
        "bonferroni",
        "preselect",
        "min_relevance",
        "split_eval",
        "presort",
        "mmd_kernel",
        "mmd_bandwidth",
        "fairness_penalty",
        "protected_col",
        "fairness_metric",
        "sporf_d",
        "sporf_projections",
        "extratrees",
        "ebm_fast",
        "target_cause",
        "n_classes",
        "max_bins",
        "min_bin_rows",
    ]
)
_LEAF_FIELDS = frozenset(
    [
        "leaf",
        "family",
        "leaf_family",
        "solver",
        "leaf_solver",
        "leaf_cols",
        "l1",
        "l2",
        "leaf_digest",
        "tau",
        "huber_delta",
        "expectile_tau",
        "non_negative",
        "non_negative_intercept",
        "nb_alpha",
        "beta_phi",
        "group_l2",
        "groups",
        "leaf_monotone",
        "mars_degree",
        "mars_terms",
        "mars_penalty",
        "tweedie_p",
        "tobit_threshold",
        "knn_k",
        "knn_k_validity",
        "knn_variant",
        "knn_metric",
        "knn_gamma",
        "knn_p",
        "events",
        "treatments",
        "shrinkage",
        "max_classes",
        "leaf_na_policy",
        "aft_dist",
        "cure_fraction",
        "spline_knots",
    ]
)


def _check_fields(options, fields, owner):
    unknown = options.keys() - fields
    if unknown:
        name = min(unknown, key=str)
        suggestion = get_close_matches(str(name), sorted(fields), n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown {owner} parameter {name!r}.{hint}")


class GrowthControl:
    """Options for splitting, stopping, pruning, and routing a model tree.

    Omitted options inherit the ModelTree defaults.
    """

    __slots__ = ("_options",)

    def __init__(self, **options):
        _check_fields(options, _GROWTH_FIELDS, "growth control")
        self._options = options

    def config(self):
        """Return the explicitly configured growth options."""
        return self._options.copy()

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_options")[name]
        except KeyError:
            raise AttributeError(name) from None


@cache
def _constructor_parameters(factory):
    return signature(factory).parameters


def _normalize_model_options(factory, args, kwargs):
    kwargs = dict(kwargs or {})
    parameters = _constructor_parameters(factory)
    _check_fields(kwargs, set(parameters) | {"growth_control"}, "ModelTree")
    positional = set(list(parameters)[: len(args)])

    growth = kwargs.pop("growth_control", None)
    if isinstance(growth, GrowthControl):
        growth = growth.config()
    if growth is not None:
        if not isinstance(growth, dict):
            raise TypeError("growth_control must be a GrowthControl or dict")
        _check_fields(growth, _GROWTH_FIELDS, "growth control")
        _merge(kwargs, growth, positional)

    leaf = kwargs.get("leaf_control")
    if isinstance(leaf, dict):
        if "n_classes" in leaf:
            raise ValueError("Supply n_classes in growth_control or as a flat keyword")
        _check_fields(leaf, _LEAF_FIELDS, "leaf control")
        if kwargs.get("control") is not None or "control" in positional:
            raise ValueError("Supply either control or leaf_control")
        kwargs.pop("leaf_control")
        leaf = leaf.copy()
        for alias, canonical in (("leaf_family", "family"), ("leaf_solver", "solver")):
            if alias in leaf:
                if canonical in leaf:
                    raise ValueError(f"Supply either {alias} or {canonical}")
                leaf[canonical] = leaf.pop(alias)
        _merge(kwargs, leaf, positional)
    return kwargs


def _merge(kwargs, options, positional):
    supplied = set(kwargs) | positional
    for alias, canonical in (("leaf_family", "family"), ("leaf_solver", "solver")):
        if alias in supplied or canonical in supplied:
            supplied.update((alias, canonical))
    duplicates = supplied & options.keys()
    if duplicates:
        raise ValueError(f"Parameter {min(duplicates)!r} was supplied more than once")
    kwargs.update(options)
