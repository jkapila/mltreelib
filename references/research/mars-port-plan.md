# Rust MARS (earth-compatible) — implementation plan

Design extracted via codex consult (2026-10-29; codex is the standing buddy per
AGENTS.md). Target: a native Rust MARS (Friedman 1991, `earth` semantics)
replacing the slow `mars-earth` Python port in the benchmark field, validated
against R `earth` as oracle.

## Why

R `earth` fits the field's sets in 0.05s median (max 1.1s); `mars-earth`
times out at 180s on the wide sets. The algorithm is cheap (hinge-basis
expansion + OLS + GCV pruning) and maps onto machinery we already own:
hinge features are extra design columns through our Gram accumulation, the
fit is `solve_normal`/elastic, backward GCV is bookkeeping.

## Gate 1 — fixed-design numerical

Export earth's basis definitions and matrix, selected terms, coefficients,
RSS, predictions. Evaluate those exact bases in Rust and fit them
independently. For well-conditioned fixtures: scale-normalized prediction/RSS
agreement ≈ 1e-8. Compare coefficients only when the design is full-rank and
well conditioned. Compare the optimized candidate scorer against brute-force
OLS on tiny datasets. Earth exposes the oracle fields (earth.object docs).

## Gate 2 — end-to-end search

Pin the R package version; explicitly match degree, penalty, spans, term
budgets, response scaling, linear-term behavior. Disable Fast MARS first;
validate it separately. Check:

- forward RSS never increases; backward RSS never decreases as terms delete;
- GCV is recomputed correctly and the selected subset minimizes it along the
  recorded path (**GCV itself need not be monotonic**);
- basis counts / selected sizes fall within fixture-specific expected ranges
  (no universal tolerance);
- held-out RMSE and prediction curves comparable — initial engineering gate
  ≤1% median RMSE gap across fixed datasets/splits (not a theoretical
  guarantee); absolute tolerances near zero RMSE.

Fixtures (generate data ONCE, share across languages — same seeds do NOT
give same RNG streams): constant target; exact linear; one hinge; multi-knot
1D spline; additive hinges + irrelevant predictors; two-variable hinge
product (verify degree 2 improves on degree 1); Friedman-1 clean + noisy;
repeated values, constant/duplicate columns, near-collinear predictors, tiny
samples, extreme scales; several existing regression benchmarks with
identical preprocessing and splits. Measure fit time, predict time, peak
memory, completion rates; fits serialized under the repo's multicore rule.

## v1 scope

Continuous predictors, one regression response, piecewise-linear bases,
general degree limit (default 1; validate degrees 1 and 2 thoroughly).
Factor lists make higher degree a small extension (search cost grows).

Landing sequence:

1. **Core representation + reference path** — `mars.rs`: parameters, factor
   lists, basis evaluation, deterministic exhaustive search, OLS, backward
   GCV. Store COMPLETE factor definitions so pruning a parent cannot
   invalidate a surviving interaction.
2. **Performance path** — sorted predictor indices, span filtering, running
   candidate statistics, reusable factorization/scratch. Require agreement
   with the reference scorer. Earth-compatible linear additions; parent
   shortlisting only after exhaustive parity passes.
3. **Public surface + evidence** — PyO3 binding, thin Python
   `MarsRegressor` (`fit`/`predict`, basis/coefficients inspection,
   equation-style printing), Rust behavior tests, pytest, R-oracle fixtures,
   docs/README/benchmark registry updates.

Exposed knobs: `degree`, `max_terms`, `max_pruned_terms`, `penalty`,
`thresh`, `minspan`, `endspan` (+ Fast MARS controls when implemented).

**Missingness**: earth supports only `na.fail`. v1 rejects unresolved missing
training values with a clear error; explicitly requested upstream imputation
is fine; propagate missing predictions when required inputs are unavailable.
Tree default directions do not define semantics for a sum of hinge products.

**Deferred**: categorical bases, learned missing-hinge behavior, weights,
multiple responses, GLMs, cubic smoothing.

## References

- Friedman 1991, "Multivariate Adaptive Regression Splines" (Ann. Stat.).
- R `earth`: earth.object docs, `?earth` (missing-data contract).
- Existing survey entry: references/algorithms/00-index.md (MARS/MSP line).
