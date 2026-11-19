## Problem Statement

Our most accurate leaf family is **linear** — it beats mean leaves 16–2 across the 18-dataset defaults matrix — but fitting it costs 10–1000× more than mean-leaf trees, rpart, or LightGBM on the same data (superconductivity: ~140s vs 0.08s mean-leaf; food delivery: ~25s vs 0.15s). The cause is structural: the split scan refits a full least-squares problem at **every candidate split, per column, per node** (per-candidate Gram solve). The result: the winning family cannot be fielded inside the 180s Elo budget, HPO sweeps crawl, and the deep small-leaf ("boosting-shape") profile that wins on wide physics data is unaffordable in its linear form. Users must choose between the accurate family and fast iteration.

## Solution

Keep the scan's shape (per-value Gram groups, running left/right aggregates, preallocated scratch) but **maintain the left/right Gram factorizations incrementally across the candidate sweep**: moving one value group across the cut is a low-rank update/downdate of the maintained factorizations, and each candidate's fit becomes two triangular solves (O(k²)) instead of a fresh factorization (O(k³)). Expected 10–50× on linear scans, to be measured on the shape fixtures. Splits, leaf models, and predictions stay identical to floating-point tolerance — the optimization is invisible except in wall-clock and fit-time columns.

## User Stories

1. As a data scientist, I want linear-leaf model trees to fit in seconds on wide datasets, so that I can iterate on features without waiting minutes per fit.
2. As an Elo benchmark runner, I want the linear family to fit inside the 180s per-dataset budget, so that ModelTree can enter scoring rounds as a real field player.
3. As an HPO researcher, I want per-fit cost low enough for hundreds of trials, so that parameter studies finish in minutes instead of hours.
4. As a practitioner, I want predictions identical before and after this optimization, so that existing results and analyses stay valid.
5. As a maintainer, I want a brute-force oracle test asserting the scan picks the same splits, so that future refactors cannot silently change tree structure.
6. As a practitioner with categorical columns, I want the level-partition scan to benefit too, so that mixed-type datasets get the same speedup.
7. As a practitioner with missing values, I want NA surrogate routing and both-side scoring untouched, so that NA-first-class semantics hold exactly as documented.
8. As a practitioner using row weights `w`, I want weighted Grams updated by the same machinery, so that weighted fits stay correct and fast.
9. As a practitioner with monotone constraints, I want constrained splits to remain exactly correct even if their speedup is deferred, so that constraints never produce wrong trees.
10. As a practitioner with elastic-net leaves (`l1`/`l2`), I want the regularized path to benefit too, so that shrinkage is not a speed trap.
11. As a practitioner using screening knobs (`split_cols`, `col_sample`, `preselect`), I want the speedup to compose with them multiplicatively, so that screened scans are near-instant.
12. As a purity-gate user, I want deep small-leaf fits affordable, so that the boosting-shape profile is usable in practice on wide data.
13. As a docs reader, I want an example printing real fit timings and accuracy across representative shapes, so that the speed claim is verifiable output, not marketing.
14. As a CI maintainer, I want the full suite green with byte-identical golden assertions, so that I trust the optimization changed nothing observable.
15. As a user with collinear columns, I want the rank-revealing singular-Gram fallback retained, so that degenerate designs neither crash nor mispredict.
16. As a user with very wide leaves (k ≈ 2·columns), I want zero per-candidate heap allocation, so that the scan's memory behavior stays flat as width grows.
17. As a reproducibility-focused researcher, I want deterministic outputs (no wall-clock or unordered iteration in results), so that runs remain comparable across machines.
18. As a cross-platform user, I want the equivalence tolerance stated explicitly (predictions within 1e-9 relative), so that x86 and ARM both pass the oracle honestly.
19. As an Elo driver author, I want both mean and linear flavors affordable, so that I can field a fast stock player and an accuracy player in the same round.
20. As the future leaf-family developer (distributional leaves), I want the scan's solve seam kept clean, so that the incremental machinery can be reused for new link functions.
21. As a speed-skeptical reviewer, I want the before/after timing table committed alongside the change, so that the claimed speedup is measured evidence.
22. As a learner reading the codebase, I want the scan module's loop-shape documentation updated to describe the incremental factorization, so that the reference notes stay accurate.
23. As a maintainer of the defaults, I want this change to require no knob or default changes, so that the just-rebaked defaults remain untouched.
24. As a dataset curator working with the three data handlers (integer / float-with-precision / categorical-levels), I want candidate grids unchanged, so that the same cut semantics (`v < a` / `v in {a,b,c}`) hold regardless of speed work.

## Implementation Decisions

- The linear scan keeps its externally visible shape: per-value Gram groups from the three data handlers, running left/right aggregates, preallocated solve scratch, single-threaded below the existing work threshold. Candidate grids are **not** altered — the optimization is purely in how per-candidate fits are computed.
- The per-candidate full Gram solve is replaced by **maintained factorizations**: as the sweep moves one value group across the cut, the leaving side receives a low-rank downdate and the entering side a low-rank update (rank = rows in the group, applied as a rank-1 sequence); each candidate's per-side SSE then costs two triangular solves plus quadratic forms.
- The update side is numerically safe by construction (positive-semidefinite addition). The downdate side is **guarded**: on factorization failure or ill-conditioning, fall back to a one-off refactorization from the maintained Gram and continue. Correctness beats speed at every candidate; the fallback bounds the worst case.
- Groups larger than a small row-count threshold use direct refactorization instead of per-row updates, since update cost is rank·O(k²) and stops paying past that rank.
- The elastic-net path (`l1 > 0`, coordinate descent on the standardised Gram) **warm-starts from the previous candidate's coefficients** instead of cold-starting — same solution, far fewer descent iterations.
- Monotone-constrained candidates keep the existing full-solve path initially (correctness first); both paths are covered by the oracle test so a later speedup of the constrained path is safe.
- The rank-revealing elimination fallback for singular Grams is retained unchanged; ridge adjustment keeps the intercept unpenalized exactly as today.
- Leaf-model extraction at the chosen split is untouched — nothing downstream of the scan changes. The scan statistics/trace fields keep populating (candidate counts, timings).
- No new knobs, no default changes: this is implementation speed only, invisible in the API surface.
- Module docs describing the loop shape (currently citing the grouped-accumulator reference) are updated to describe the incremental factorization as the divergence from that reference.

## Testing Decisions

- Good tests assert **external behavior only**: which split is chosen, tree structure, predictions, and leaf models. Factorization internals (update/downdate sequences, refactor triggers) are never asserted directly — they are implementation detail.
- One seam: the existing model-tree behavior suite. No new seams are introduced.
- The existing golden assertions (byte-exact rpart-layout print, exact structure on fixed fixtures) must pass **unchanged** — they are the regression net proving observable behavior did not move.
- New **oracle test** at the same seam: a brute-force scorer written inside the test enumerates every candidate split and computes per-side OLS SSE independently (naive normal equations, deliberately not sharing code with the implementation), then asserts the fitted tree chose the same splits and that predictions match within 1e-9 relative tolerance. Fixtures cover: random signal, collinear columns, NA-heavy columns, weighted rows, categorical level columns, and both unregularized and elastic-net leaves.
- Timing evidence lives in an **example script** (speed-scan) printing real fit times and test rmse across the shape fixtures (wide, long, categorical) — printed output in the docs tradition, not wall-clock assertions in CI (shared runners make those flaky).
- Prior art for the patterns: the byte-exact printer test (golden), the Chow-gate and BIC-prune tests (behavioral properties), and the knob-tour example (real printed output).
- The full suite (Rust core + Python behavior tests) runs green before and after; the new oracle tests are additive behavior tests, not re-pinnings.

## Out of Scope

- Distributional leaf families (issue #4) and the prediction-API/docs work (issue #5) — both build on this seam later.
- Binned/histogram candidate approximation (border_count-style): an accuracy tradeoff requiring its own measured gate, not this ticket.
- Mean-leaf scan changes (already O(1) per candidate).
- Elo re-scoring runs and board updates (issue #2) — this ticket unblocks ModelTree's entry but does not run it.
- Parallelism/rayon threshold changes.
- Any knob additions, default changes, or API surface changes.

## Further Notes

- Motivation is the measured defaults matrix (candidate C vs old, mean vs linear): linear wins the family comparison 16–2 and the only blocker is fit cost. The superconductivity deep-profile result (mean, ~1164 leaves, 0.39s, best accuracy on that set) shows the boosting-shape regime is real — this ticket makes its linear counterpart affordable.
- Loop-shape lineage: grouped accumulators and per-candidate solves on preallocated scratch follow the linear tree-learner reference our module docs cite; the incremental maintained factorization is the step beyond that reference, and the docs must say so.
- Expected effect [INFERENCE, to be measured]: 10–50× on linear scans (superconductivity ~140s → single-digit seconds). The committed before/after table on the shape fixtures is the evidence artifact; the re-run of the 18-dataset matrix after landing should show an identical rmse column and a transformed fit-time column.
- Sequencing decision (user): this ticket lands **before** the leaf-family expansion (#4). After it lands, re-score ModelTree flavors in Elo and unexclude them from the field.
