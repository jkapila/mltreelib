"""ModelTree(all) factory extracted from the v0.4.7 Elo registry."""
import time
from mltreelib import ModelTree


def make_models(meta):
    roles = list(meta["columns"].values()) if (meta and "columns" in meta) else None

    def make_mt(leaf: str, leaf_cols: str, min_relevance: float | None = None):
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            dtypes = (
                roles
                if roles and len(roles) == p
                else [
                    (
                        "categorical"
                        if any(isinstance(r[j], str) for r in xtr)
                        else "numeric"
                    )
                    for j in range(p)
                ]
            )
            kw = {"leaf": leaf, "leaf_cols": leaf_cols}
            if min_relevance is not None:
                kw["min_relevance"] = min_relevance
            m = ModelTree(**kw)
            t0 = time.perf_counter()
            m.fit(xtr, [float(y) for y in ytr], dtypes=dtypes)
            pred = m.predict(xte)
            return pred, time.perf_counter() - t0

        return run
    return {"ModelTree(all)": make_mt("linear", "all")}
