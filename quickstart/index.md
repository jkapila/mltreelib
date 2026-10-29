# Quickstart

Fit a tree, read it, see it. The full script lives at [`examples/quickstart.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/quickstart.py>).

```
from mltreelib import ModelTree, Metrics

# Toy data: a step function plus a linear trend (deterministic).
x = [[float(i % 8)] for i in range(80)]
y = [2.0 * (i % 8) + (5.0 if i % 8 > 4 else 0.0) + ((i * 13) % 7) * 0.05 for i in range(80)]

# Defaults are already tuned (regularized linear leaves, shallow depth).
model = ModelTree()
model.fit(x, y)

pred = model.predict([[float(v)] for v in range(8)])
print("predictions:", [round(p, 2) for p in pred])
print("metrics:", {k: round(v, 3) for k, v in
                   Metrics().regression(y_test, pred).items() if k in ("rmse", "mae", "r2")})

print(model)                                  # rpart-style tree text
dot = model.to_dot()                         # graphviz does the layout
pgv.AGraph(string=dot).draw("docs/assets/quickstart_tree.svg", prog="dot")
```

Output:

```
predictions: [0.16, 2.18, 4.17, 6.17, 8.12, 15.12, 17.14, 19.16]
metrics: {'mae': 0.151, 'rmse': 0.153, 'r2': 0.999}

n= 80

node), split, n, deviance, yval
      * denotes terminal node

 1) root 80 3639.19987 9.02625
   2) x0< 4 50 397.26680 4.15800
     4) x0< 0 10 0.12400 0.16000 *
     5) x0>=0 40 197.34275 5.15750
      10) x0< 1 10 0.09600 2.18000 *
      11) x0>=1 30 79.04000 6.15000
        22) x0< 2 10 0.08025 4.16500 *
        23) x0>=2 20 19.85637 7.14250 *
   3) x0>=4 30 81.95200 17.14000
     6) x0< 6 20 20.62200 16.13000
      12) x0< 5 10 0.09600 15.12000 *
      13) x0>=5 10 0.12400 17.14000 *
     7) x0>=6 10 0.12400 19.16000 *

wrote docs/assets/quickstart_tree.svg (rendered by graphviz)
```

Three things to notice:

- **The tree prints like rpart** — same numbering, indentation, deviance and yval columns, `*` on leaves. If you know `rpart`, you already know how to read this.
- **`to_dot()` describes the tree, graphviz draws it** — one box per node with the split condition, `n`, and fitted value; edges labeled with thresholds. Layout belongs to graphviz (via pygraphviz); we don't reinvent it.
- **Defaults are tuned** — `leaf="linear"` with a small ridge (`l2`) penalty at shallow depth is the configuration that won our [benchmark bake-off](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>).

Next: the [end-to-end walkthrough](<https://jkapila.github.io/mltreelib/examples/walkthrough/index.md>) for a real end-to-end problem, or the [API reference](<https://jkapila.github.io/mltreelib/reference/model/index.md>) for every parameter.
