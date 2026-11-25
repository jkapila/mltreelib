"""Defaults must be identical on every surface: Rust core, constructor, controls."""

from mltreelib import LeafControl, ModelTree


def test_rust_core_defaults_equal_constructor_defaults():
    constructor = ModelTree().config()
    core = ModelTree.core_defaults()
    different = {k: (constructor[k], core[k]) for k in constructor if k in core and constructor[k] != core[k]}
    assert different == {}


def test_leaf_control_defaults_equal_constructor_defaults():
    constructor = ModelTree().config()
    control = LeafControl()
    different = {
        k: (getattr(control, k), constructor[k])
        for k in constructor
        if hasattr(control, k) and getattr(control, k) != constructor[k]
    }
    assert different == {}


def test_documented_headline_defaults():
    cfg = ModelTree().config()
    assert (cfg["max_depth"], cfg["min_rows"]) == (6, 15)
