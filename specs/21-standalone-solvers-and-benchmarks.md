# Spec 21 · Standalone fitting and solver benchmarks

Status: proposed

## Goal

Fit a weighted linear or generalized linear model directly from Python using the Rust solver code that also fits tree leaves. Make it possible to compare optimizers on the same objective and data.

## Proposed interface

```python
from mltreelib.solvers import fit

result = fit(X, y, family="binomial", solver="lbfgs", l2=0.1, sample_weight=w)
probabilities = result.predict(X_new)
```

- Start with the supported Gaussian and GLM objectives. Document which solvers, penalties, and constraints each objective supports.
- Accept weights, applicable penalties, convergence tolerances, and explicit seeds for stochastic algorithms. Validate target domains, dimensions, and incompatible options in Rust.
- Return coefficients, intercept, predictions on the response scale, and convergence information. Return inference statistics only where the solver computes them.
- Reuse the same objective and optimizer dispatch for standalone fitting and linear tree leaves. Preserve the existing leaf NA policies and fit defaults.

## Adding solvers

Use the existing direct, IRLS, CCD, L-BFGS, SHADE, CMA-ES, GA, and sNES paths as baselines. Evaluate proximal Newton or a similar method for sparse penalized GLMs. Evaluate a second-order method for difficult smooth objectives only when a supported family supplies the required derivatives.

Add a solver when it enables a supported objective or penalty, improves convergence reliability, or improves time at comparable held-out quality. Record convergence failures and run one multicore fit at a time.

## Acceptance

- Check weighted Gaussian and GLM fits against known solutions or documented reference results.
- Compare standalone and single-leaf fits with the same design, penalty, and solver.
- Report training objective, held-out deviance, fit time, and convergence status for each candidate and baseline.
- Verify deterministic runs when the seed and inputs are fixed.

Generic Python objective callbacks and plugin solvers are outside this proposal.
