"""Fit a weighted Gaussian or generalized linear model without growing a tree.

Pass a complete numeric matrix and one response per row. Use nonnegative
case weights with positive total weight. An intercept is always included
and excluded from the penalty.

Use `direct` for Gaussian OLS or ridge, and `ccd` for Gaussian fits with
L1 or coefficient signs. Use `lbfgs` for smooth GLMs with L2. Use `irls`
for GLMs with L1 or sign constraints. `cmaes`, `shade`, `nes`, and `ga`
also support L1, L2, and coefficient signs across all families. The
stochastic solvers use `seed` for reproducibility.

CCD and GLM IRLS apply L1 and L2 to scaled coefficients when L1 is positive.
The remaining solvers penalize coefficients on the original scale.
"""

from mltreelib._core import FitResult
from mltreelib._core import fit_glm as fit

__all__ = ["FitResult", "fit"]
