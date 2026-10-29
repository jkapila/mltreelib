"""Live edge-case playground for mltreelib — run locally:

    uv run --with marimo marimo edit playground/edges.marimo.py

Every cell is a knob you can flip to surface edge cases: typed ingest
(categoricals as levels, NAs as-is), the split-precision grid, categorical
split search, gain criteria (incl. a custom callable), t-digest leaf
quantiles, and the training log. Nothing here is imported by the library —
it is a testing surface.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _intro():
    import marimo as mo
    mo.md("""
    # mltreelib edge-case playground

    Flip knobs below and re-run top-to-bottom. The **training log** cell
    shows candidate generation live (grid vs exact, categorical search,
    gain taken).
    """)
    return (mo,)


@app.cell
def _data(mo):
    import random
    n = mo.ui.slider(60, 2000, value=400, label="rows")
    na_rate = mo.ui.slider(0.0, 0.4, value=0.05, step=0.01, label="NA rate")
    cat_card = mo.ui.slider(2, 12, value=5, label="categorical levels")
    mo.md(f"## 1. Synthetic typed data — rows={n.value}, NA={na_rate.value}, levels={cat_card.value}")
    return n, na_rate, cat_card, random


@app.cell
def _frame(n, na_rate, cat_card, random):
    random.seed(42)
    xs, ys, dtypes = [], [], ["numeric", "categorical"]
    for _ in range(n.value):
        num = random.uniform(0, 3)
        lev = float(random.randrange(cat_card.value))
        # truth: piecewise in the level codes, linear in the numeric
        y = 2.0 * num + (5.0 if lev >= cat_card.value - 2 else 0.0)
        xs.append([
            None if random.random() < na_rate.value else num,
            None if random.random() < na_rate.value else lev,
        ])
        ys.append(y)
    return xs, ys, dtypes


@app.cell
def _knobs(mo):
    max_depth = mo.ui.slider(1, 8, value=3, label="max_depth")
    split_precision = mo.ui.dropdown(
        options={"0.1 grid (default)": 0.1, "0.01 grid": 0.01, "exact (None)": None},
        value="0.1 grid (default)", label="split_precision",
    )
    cat_search = mo.ui.dropdown(
        options=["cumulative", "single_vs_rest", "both"], value="cumulative",
        label="cat_search",
    )
    cat_order = mo.ui.dropdown(
        options=["frequency_desc", "target_mean", [2.0, 1.0, 0.0]],
        value="frequency_desc", label="cat_order (last = explicit)",
    )
    gain = mo.ui.dropdown(
        options=["sse", "entropy", "gini", "gain_ratio", "sdr", "twoing",
                 "custom (weighted |mean gap|)"], value="sse",
        label="gain",
    )
    leaf_digest = mo.ui.checkbox(value=True, label="leaf_digest (t-digest per leaf)")
    alpha = mo.ui.slider(0.0, 0.49, value=0.0, step=0.01, label="alpha (min child weight share)")
    imbalance_penalty = mo.ui.slider(0.0, 1000.0, value=0.0, step=10.0, label="imbalance_penalty")
    col_sample = mo.ui.dropdown(
        options={"all columns": None, "half": 0.5, "third": 0.33}, value="all columns",
        label="col_sample (per-node random columns)",
    )
    prune = mo.ui.dropdown(options=["none", "bic", "mdl"], value="none", label="prune")
    mo.md("## 2. Growth knobs")
    return (max_depth, split_precision, cat_search, cat_order, gain, leaf_digest,
            alpha, imbalance_penalty, col_sample, prune)


@app.cell
def _fit(mo, xs, ys, dtypes, max_depth, split_precision, cat_search, cat_order,
         gain, leaf_digest, alpha, imbalance_penalty, col_sample, prune):
    from mltreelib import ModelTree

    def custom_gain(left, right):
        # pairs ABI: (y, w) rows per side
        lm = sum(y * w for y, w in left) / max(sum(w for _, w in left), 1e-9)
        rm = sum(y * w for y, w in right) / max(sum(w for _, w in right), 1e-9)
        return abs(lm - rm)

    g = custom_gain if gain.value.startswith("custom") else gain.value
    model = ModelTree(
        max_depth=max_depth.value,
        min_rows=5,
        split_precision=split_precision.value,
        cat_search=cat_search.value,
        cat_order=cat_order.value,
        gain=g,
        leaf_digest=leaf_digest.value,
        alpha=alpha.value,
        imbalance_penalty=imbalance_penalty.value,
        col_sample=col_sample.value,
        prune=prune.value,
        verbose=2,
    )
    model.fit(xs, ys, dtypes=dtypes)
    mo.md(f"**fitted** — leaves: {len(model.leaves())}, splits: {len(model.splits())}")
    return model, custom_gain


@app.cell
def _log(mo, model):
    mo.md("## 3. Training log (verbose=2: scan config + per node)")
    mo.accordion({"training_log()": "\n".join(model.training_log())})


@app.cell
def _quantiles(mo, model):
    probes = [[0.5, 2.0], [None, 0.0], [2.5, None], [0.0, 0.0]]
    taus = [0.05, 0.25, 0.5, 0.9, 0.99]
    rows = {
        f"x={p}": [model.quantile([p], t)[0] for t in taus]
        for p in probes
    }
    mo.md("## 4. Conditional quantiles from the leaf t-digests (any tau, no refit)")
    mo.ui.table(rows)


@app.cell
def _edge(mo, model):
    # classic edge rows: all-NA, unseen level, huge value
    edge_rows = [[None, None], [123456.0, 99.0], [0.0, None]]
    mo.md(f"## 5. Edge rows — predictions never block: {model.predict(edge_rows)}")


if __name__ == "__main__":
    app.run()
