"""ModelTree(all) factory extracted from the v0.4.7 Elo registry."""
import time
from mltreelib import ModelTree


def make_models(meta):
    classes = list(meta["classes"]) if (meta and "classes" in meta) else ["0", "1"]

    roles = list(meta["columns"].values()) if (meta and "columns" in meta) else None

    def make_mt(leaf: str, leaf_cols: str = "all", min_relevance: float | None = None):
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

            if len(classes) == 2:
                pos_label = str(classes[1])
                y_bin = [1.0 if str(y) == pos_label else 0.0 for y in ytr]

                # High-dimensional safeguards: when p >= 50, 'all' dense linear leaf
                # creates massive O(p^2) Gram matrices that cause memory / runtime explosion.
                # Auto-guard 'all' to 0D 'mean' for p >= 100, or to 'path' for 50 <= p < 100.
                effective_leaf = leaf
                effective_leaf_cols = leaf_cols
                if p >= 100 and leaf == "linear" and leaf_cols == "all":
                    effective_leaf = "mean"
                elif p >= 50 and leaf == "linear" and leaf_cols == "all":
                    effective_leaf_cols = "path"

                kw = {
                    "leaf": effective_leaf,
                    "leaf_family": "binomial",
                    "gain": "deviance",
                    "leaf_cols": effective_leaf_cols,
                    "seed": 42,
                }
                if min_relevance is not None:
                    kw["min_relevance"] = min_relevance
                m = ModelTree(**kw)
                t0 = time.perf_counter()
                m.fit(xtr, y_bin, dtypes=dtypes)
                probs = m.predict_proba(xte)
                fit_s = time.perf_counter() - t0
                aligned = [[1.0 - p, p] for p in probs]
                return aligned, fit_s
            else:
                # Multiclass (Softmax / Multinomial)
                class_map = {str(c): float(i) for i, c in enumerate(classes)}
                y_multi = [class_map[str(y)] for y in ytr]

                effective_leaf = leaf
                effective_leaf_cols = leaf_cols
                if p >= 50 and leaf == "linear" and leaf_cols == "all":
                    effective_leaf_cols = "path"

                kw = {
                    "leaf": effective_leaf,
                    "leaf_family": "softmax",
                    "n_classes": len(classes),
                    "gain": "entropy",
                    "leaf_cols": effective_leaf_cols,
                    "seed": 42,
                }
                if min_relevance is not None:
                    kw["min_relevance"] = min_relevance
                m = ModelTree(**kw)
                t0 = time.perf_counter()
                m.fit(xtr, y_multi, dtypes=dtypes)
                aligned = m.predict(xte, kind="proba")
                fit_s = time.perf_counter() - t0
                return aligned, fit_s

        return run
    return {"ModelTree(all)": make_mt("linear", "all")}
