"""Monotone and interaction-group constraints - and how to verify them.

    uv run python examples/constraints_demo.py
"""

from mltreelib import ModelTree

# --- monotone: the target DECREASES in x0; mono=+1 forbids that ---------
x = [[(i % 25) / 25.0, ((i * 11) % 17) / 17.0] for i in range(300)]
y = [-8.0 * r[0] + 0.5 * r[1] + (i % 3) * 0.01 for i, r in enumerate(x)]
grid = [[t / 24.0, 0.5] for t in range(25)]

for label, kw in [("unconstrained", {}), ("monotone +1 on x0", {"monotone": [1, 0]})]:
    m = ModelTree(max_depth=3, min_rows=10, l1=0.0, l2=0.0, **kw)
    m.fit(x, y)
    p = m.predict(grid)
    ok = all(b >= a - 1e-9 for a, b in zip(p, p[1:]))
    print(f"{label:<20} predictions non-decreasing along x0: {ok}")

# --- interaction groups: no path may mix groups ------------------------
# x2 only matters deep inside x0 > 0.75; unconstrained the tree mixes
# groups along some path, constrained every path stays within one group.
x2 = [[(i % 30) / 30.0, ((i * 7) % 11) / 11.0, ((i * 13) % 17) / 17.0] for i in range(300)]
y2 = [5.0 * r[0] + (8.0 if r[0] > 0.75 and r[2] > 0.5 else 0.0) for r in x2]
groups = [[0, 1], [2]]
g_of = {"x0": 0, "x1": 0, "x2": 1}

for label, kw in [("unconstrained", {}), ("groups [[0,1],[2]]", {"interaction_groups": groups})]:
    m = ModelTree(max_depth=4, min_rows=10, l1=0.0, l2=0.0, **kw)
    m.fit(x2, y2)
    mixed = [p for p in m.split_paths() if len({g_of[c] for c in p}) > 1]
    print(f"{label:<20} paths mixing groups: {len(mixed)}")
