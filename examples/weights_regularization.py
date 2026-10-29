"""Row weights, leaf regularization, and relevance screening.

    uv run python examples/weights_regularization.py
"""

from mltreelib import ModelTree

x = [[(i % 20) / 20.0, ((i * 9) % 13) / 13.0] for i in range(300)]
y = [4.0 * r[0] + 2.0 * r[1] + (i % 5) * 0.02 for i, r in enumerate(x)]

# --- row weights: double the weight of the top half of x0 --------------
w = [2.0 if r[0] > 0.5 else 1.0 for r in x]
m_plain, m_weighted = ModelTree(max_depth=2, min_rows=10, leaf="mean"), ModelTree(
    max_depth=2, min_rows=10, leaf="mean"
)
m_plain.fit(x, y)
m_weighted.fit(x, y, w)
print("unweighted leaf means:", [round(l["mean"], 2) for l in m_plain.leaves()])
print("weighted leaf means:  ", [round(l["mean"], 2) for l in m_weighted.leaves()])

# --- regularization on linear leaves (ridge / lasso / elastic net) -----
# Two correlated predictors: lasso keeps one, ridge shares, elastic between.
xc = [[r[0], r[0] + 0.01 * r[1]] for r in x]  # x1 ~ x0 (collinear)
yc = [3.0 * r[0] for r in x]
for label, l1, l2 in [("ridge  (l2)", 0.0, 1.0), ("lasso  (l1)", 1.0, 0.0),
                      ("elastic", 1.0, 1.0)]:
    m = ModelTree(max_depth=0, min_rows=5, leaf="linear", l1=l1, l2=l2)
    m.fit(xc, yc)
    leaf = m.leaves()[0]
    coefs = [round(c, 3) for c in leaf["coef"]]
    print(f"{label:<12} leaf coefficients: {coefs}")

# --- relevance screening: drop junk columns before the split search ----
import random

random.seed(3)
xj = [r + [random.random()] for r in x]  # x2 = pure noise
mj = ModelTree(max_depth=3, min_rows=10, min_relevance=0.05)
mj.fit(xj, y)
used = {s["col"] for s in mj.splits()}
print(f"min_relevance=0.05 -> splits use columns {sorted(used)} (noise column 2 gone: {2 not in used})")
