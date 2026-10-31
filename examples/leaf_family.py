"""Distributional leaves: exact grouped MLEs and categorical classification.

Run: uv run python examples/leaf_family.py
"""

from mltreelib import ModelTree

# Each binary predictor group contains both outcomes. The independent MLEs
# are 1/4 and 3/4, hence logit intercept=-ln(3), slope=2*ln(3).
x = [[0.0]] * 4 + [[1.0]] * 4
y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
binary = ModelTree(leaf_family="binomial", gain="deviance", max_depth=0,
                   min_rows=1, l2=0.0)
binary.fit(x, y)
p = binary.predict_proba([[0.0], [1.0]])
assert abs(p[0] - 0.25) < 1e-6 and abs(p[1] - 0.75) < 1e-6
print("binomial P(1):", [round(v, 6) for v in p])
print("binomial P(Y <= 0):", [round(v, 6) for v in binary.cdf([[0.0], [1.0]], 0.0)])

# Log-link interpolation: group means 1 and 4 imply mean 2 at x=0.5.
counts = ModelTree(leaf_family="poisson", gain="deviance", max_depth=0,
                   min_rows=1, l2=0.0)
counts.fit([[0.0], [0.0], [1.0], [1.0]], [0.0, 2.0, 2.0, 6.0])
mu = counts.predict([[0.0], [0.5], [1.0]])
assert all(abs(a - b) < 1e-6 for a, b in zip(mu, [1.0, 2.0, 4.0]))
print("poisson means:", [round(v, 6) for v in mu])
print("poisson P(Y <= 0):", [round(v, 6) for v in counts.cdf([[0.0], [1.0]], 0.0)])

# Native level partitions and learned missing routing; no encoding.
segments = ModelTree(leaf_family="binomial", leaf="mean", gain="deviance",
                     min_rows=2, max_depth=1, prune="none")
segments.fit([["low"]] * 8 + [["high"]] * 8 + [[None]] * 4,
             [0.0] * 8 + [1.0] * 12, dtypes=["categorical"])
p = segments.predict_proba([["low"], ["high"], [None]])
assert p == [0.0, 1.0, 1.0]
print("categorical + NA P(1):", p)
