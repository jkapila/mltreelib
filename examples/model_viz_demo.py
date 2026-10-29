"""Model building and tree visualization: read the tree, see the tree.

    uv run python examples/model_viz_demo.py
"""

from mltreelib import ModelTree

# A step function a linear leaf cannot absorb - forces real splits.
x = [[(i % 20) / 20.0, ((i * 7) % 13) / 13.0] for i in range(300)]
y = [10.0 * (r[0] > 0.5) + 6.0 * (r[1] > 0.6) + (i % 5) * 0.02 for i, r in enumerate(x)]

model = ModelTree(max_depth=3, min_rows=15, leaf="mean")
model.fit(x, y)

print(f"leaves: {model.n_leaves()}")
print()
print(model)  # rpart-style text

print("splits:", [(s["col"], round(s["threshold"], 3)) for s in model.splits()])
print("split paths (column names per root-to-leaf path):")
for p in model.split_paths():
    print("   ", p)

# Leaf models: fitted values + row counts per terminal node.
for leaf in model.leaves():
    print(f"leaf n={leaf['n']}  yval={leaf['mean']:.3f}")

open("viz_tree.svg", "w").write(model.to_svg())
print("wrote viz_tree.svg")

# Mean vs linear leaves on the same data (step effects resist both equally).
for kind in ("mean", "linear"):
    m = ModelTree(max_depth=3, min_rows=15, leaf=kind, l1=0.0, l2=0.0)
    m.fit(x, y)
    print(f"leaf={kind:<6} leaves={m.n_leaves()}")
