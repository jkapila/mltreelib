"""Archived legacy ModelTree variations used in v0.4.x Elo arena runs.

Kept for historical reference, ablations, and replication studies.
Superseded in v0.4.8+ by ModelTree(default) and ModelTree(mean).
"""

from typing import Any
from mltreelib import ModelTree


def make_legacy_regression_models(roles: list[str] | None = None) -> dict[str, Any]:
    """Return dictionary of legacy regression ModelTree builders."""
    ms = {}

    def make_mt(leaf: str, leaf_cols: str, min_relevance: float | None = None):
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            dtypes = (
                roles
                if roles and len(roles) == p
                else [
                    ("categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric")
                    for j in range(p)
                ]
            )
            kw = {"leaf": leaf, "leaf_cols": leaf_cols}
            if min_relevance is not None:
                kw["min_relevance"] = min_relevance
            m = ModelTree(**kw)
            m.fit(xtr, [float(y) for y in ytr], dtypes=dtypes)
            return m.predict(xte)

        return run

    ms["ModelTree(all)"] = make_mt("linear", "all")
    ms["ModelTree(path)"] = make_mt("linear", "path")
    ms["ModelTree(winnow)"] = make_mt("linear", "all", min_relevance=0.05)

    def make_mt_mars(mars_degree: int = 1, max_depth: int = 3):
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            dtypes = (
                roles
                if roles and len(roles) == p
                else [
                    ("categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric")
                    for j in range(p)
                ]
            )
            m = ModelTree(leaf="mars", max_depth=max_depth, mars_degree=mars_degree)
            m.fit(xtr, [float(y) for y in ytr], dtypes=dtypes)
            return m.predict(xte)

        return run

    ms["ModelTree(leaf=mars,deg=1)"] = make_mt_mars(mars_degree=1)
    ms["ModelTree(leaf=mars,deg=2)"] = make_mt_mars(mars_degree=2)

    return ms
