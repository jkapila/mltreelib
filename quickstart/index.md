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

x_test = [[float(v)] for v in range(8)]
y_test = [2.0 * v + (5.0 if v > 4 else 0.0) for v in range(8)]
pred = model.predict(x_test)
print("predictions:", [round(p, 2) for p in pred])
print("metrics:", {k: round(v, 3) for k, v in
                   Metrics().regression(y_test, pred).items() if k in ("rmse", "mae", "r2")})

print(model)                                  # rpart-style tree text
dot = model.to_dot()                         # graphviz does the layout
print(dot)  # Install pygraphviz and Graphviz to render this DOT string.
```

Output:

```
predictions: [0.25, 2.09, 4.21, 6.15, 8.09, 15.17, 17.14, 19.11]
metrics: {'mae': 0.151, 'rmse': 0.161, 'r2': 0.999}

n= 80 

node), split, n, deviance, yval
      * denotes terminal node

1) root 80 3639.1999 9.02625  
  2) signal_step <= 4 50 397.2668 4.15800  
    4) signal_step <= 1 20 20.6220 1.17000 *
    5) signal_step > 1 30 79.0400 6.15000 *
  3) signal_step > 4 30 81.9520 17.14000 *

DOT output omitted
```

Three things to notice:

- **The tree prints like rpart** — same numbering, indentation, deviance and yval columns, `*` on leaves. If you know `rpart`, you already know how to read this.
- **`to_dot()` describes the tree, graphviz draws it** — one box per node with the split condition, `n`, and fitted value; edges labeled with thresholds. Layout belongs to graphviz (via pygraphviz); we don't reinvent it.
- **Defaults are tuned** — `leaf="linear"` with a small ridge (`l2`) penalty at shallow depth is the configuration that won our [benchmark bake-off](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>).

Next: the [end-to-end walkthrough](<https://jkapila.github.io/mltreelib/examples/walkthrough/index.md>) for a real end-to-end problem, or the [API reference](<https://jkapila.github.io/mltreelib/reference/model/index.md>) for every parameter.
