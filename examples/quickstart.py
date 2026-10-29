"""mltreelib in 30 seconds: fit a tree, read it, see it.

    uv run python examples/quickstart.py
"""

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
print("metrics:", {k: round(v, 3) for k, v in Metrics().regression(y_test, pred).items()
                   if k in ("rmse", "mae", "r2")})

print()  # rpart-style tree text
print(model)

dot = model.to_dot()  # graphviz does layout; we only describe the graph
try:
    import pygraphviz as pgv

    pgv.AGraph(string=dot).draw("docs/assets/quickstart_tree.svg", prog="dot")
    print("wrote docs/assets/quickstart_tree.svg (rendered by graphviz)")
except ImportError:
    open("quickstart_tree.dot", "w").write(dot)
    print("wrote quickstart_tree.dot (render: dot -Tsvg quickstart_tree.dot -o tree.svg)")
