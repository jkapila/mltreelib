"""Unified Solver Architecture Example in mltreelib.

Demonstrates the statistical and evolutionary regression solvers implemented in mltreelib-core:
1. Exact OLS / Ridge (Cholesky factorization + QR fallback)
2. Cyclic Coordinate Descent (CCD with active set for Lasso / ElasticNet / Monotone)
3. Quasi-Newton L-BFGS (for non-linear GLM deviance minimization)
4. Evolutionary Optimization (CMA-ES, SHADE, sNES, RCGA for non-convex loss)

How Solvers Are Exposed:
Currently, solvers are configured per leaf via `leaf_solver` on `ModelTree`:
- `ModelTree(leaf='linear', leaf_solver='irls' | 'ccd' | 'lbfgs' | 'cmaes' | 'shade' | 'nes' | 'ga')`
- Setting `max_depth=0` fits a global standalone linear or GLM model using that solver.
- Leaf models output comprehensive R `summary(lm)` statistics via `model.leaf_summary(id)`.
"""

from __future__ import annotations

import random
from mltreelib import ModelTree


def run_solvers_demonstration() -> None:
    print("=" * 70)
    print("mltreelib Unified Solver Suite Overview")
    print("=" * 70)

    # Generate synthetic regression data: y = 3.5*x0 - 2.0*x1 + 0.5*x2 + noise
    random.seed(42)
    n = 120
    X = [[random.uniform(-2.0, 2.0), random.uniform(-1.0, 1.0), random.uniform(0.0, 5.0)] for _ in range(n)]
    y = [3.5 * r[0] - 2.0 * r[1] + 0.5 * r[2] + random.gauss(0.0, 0.2) for r in X]

    # 1. Exact Normal OLS Solver
    print("\n1. Exact OLS / Ridge (leaf_solver='irls'):")
    ols_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="irls", l2=0.0)
    ols_tree.fit(X, y)
    leaf_id = ols_tree.leaves()[0]["id"]
    print(ols_tree.leaf_summary(leaf_id))

    # 2. Coordinate Descent (L1 Lasso / ElasticNet)
    print("\n2. Cyclic Coordinate Descent (leaf_solver='ccd', l1=0.1, l2=0.01):")
    ccd_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="ccd", l1=0.1, l2=0.01)
    ccd_tree.fit(X, y)
    ccd_coef = ccd_tree.leaves()[0]["coef"]
    print(f"CCD Coefficients: {[round(c, 4) for c in ccd_coef]}")

    # 3. Quasi-Newton L-BFGS Solver
    print("\n3. Quasi-Newton L-BFGS (leaf_solver='lbfgs'):")
    lbfgs_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="lbfgs", l2=0.01)
    lbfgs_tree.fit(X, y)
    lbfgs_coef = lbfgs_tree.leaves()[0]["coef"]
    print(f"L-BFGS Coefficients: {[round(c, 4) for c in lbfgs_coef]}")

    # 4. Evolutionary Solver: CMA-ES
    print("\n4. Covariance Matrix Adaptation Evolution Strategy (leaf_solver='cmaes'):")
    cmaes_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="cmaes")
    cmaes_tree.fit(X, y)
    cmaes_coef = cmaes_tree.leaves()[0]["coef"]
    print(f"CMA-ES Coefficients: {[round(c, 4) for c in cmaes_coef]}")

    # 5. Evolutionary Solver: SHADE (Success-History Adaptive DE)
    print("\n5. Adaptive Differential Evolution (leaf_solver='shade'):")
    shade_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="shade")
    shade_tree.fit(X, y)
    shade_coef = shade_tree.leaves()[0]["coef"]
    print(f"SHADE Coefficients: {[round(c, 4) for c in shade_coef]}")


if __name__ == "__main__":
    run_solvers_demonstration()
