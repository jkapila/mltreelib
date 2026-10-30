# Incremental linear-scan factorization — measured speedup (2026-10-31)

Issue #7. The linear-leaf split scan refit a full least-squares system per
candidate (`solve_normal_sum` → faer Cholesky or rank-revealing elimination,
O(k³) each) and rebuilt the parent design per column. The rewrite maintains
Cholesky factors across the sweep (LINPACK `dchud`/`dchdd` rank-1
update/downdate per row crossing the cut; column-major kernels) and hoists
the parent statistics to one computation per node. Candidates are scored by
triangular solves (O(k²)); every path without a live factor is exactly the
historical solve.

## Acceptance evidence

Characterization oracle (5 tests in the model-tree suite): brute-force
candidate enumeration with an independent weighted-least-squares solver —
random signal, NA-heavy + weights + grid, collinear + ridge, elastic net,
categorical. All green **before** the rewrite (characterization) and **after**
(equivalence). The full suite (131 tests, 10 binaries) green; golden
assertions byte-identical.

Before/after on the 18 field sets (candidate-C defaults, linear leaves):

| dataset | rmse | parity | before s | after s | speedup |
|---|---|---|---|---|---|
| ozone | 4.3106 | 3.4e-05 | 0.00 | 0.02 | — |
| boston | 4.5411 | 3.1e-05 | 0.04 | 0.05 | 0.8x |
| concrete | 6.4212 | 2.3e-06 | 0.02 | 0.03 | 0.8x |
| ta_qsar_fish | 0.9978 | 3.3e-05 | 0.02 | 0.02 | 0.9x |
| ta_wine | 0.6980 | 1.9e-05 | 0.19 | 0.28 | 0.7x |
| abalone | 2.3198 | 3.0e-05 | 0.13 | 0.15 | 0.9x |
| ta_airfoil | 2.4698 | 4.7e-05 | 0.08 | 0.13 | 0.6x |
| ta_concrete | 6.4851 | 1.5e-05 | 0.04 | 0.04 | 0.9x |
| ta_houses | 0.2574 | 3.9e-05 | 0.93 | 0.78 | 1.2x |
| ta_fiat500 | 793.8410 | 4.1e-05 | 0.06 | 0.06 | 1.0x |
| fc2008 | 201.3604 | 1.7e-05 | 0.03 | 0.04 | 0.8x |
| ta_healthcare | 4883.7711 | 1.1e-07 | 0.04 | 0.04 | 1.1x |
| ta_diamonds | 606.7475 | 1.4e-05 | 0.80 | 1.08 | 0.7x |
| ta_protein | 4.5162 | 4.9e-05 | 3.42 | 2.66 | 1.3x |
| ta_miami | 127118.2833 | 2.4e-05 | 3.38 | 2.16 | 1.6x |
| ta_qsar_tid11 | 1.0740 | 1.8e-05 | 2.63 | 1.97 | 1.3x |
| ta_food_delivery | 7.4359 | 6.7e-06 | 25.62 | 7.22 | **3.5x** |
| ta_superconductivity | 16.0310 | 5.4e-06 | 102.15 | 23.97 | **4.3x** |

Aggregate ≈ 3.3x. The headline pairs are exactly the ones the ticket was
opened for (superconductivity 102s, food delivery 26s under the Elo budget).

## Honest gaps and mechanisms

- **Measured 3.5–4.3x on heavy sets vs the spec's inferred 10–50x.** The
  mechanism is sound (per-candidate O(k³) refits are gone where factors
  live), but two floors remain: the fold itself costs O(n·k²) per column
  (rank-1 steps are the new per-row baseline), and the hand-rolled rank-1
  kernels, while slice-based, still trail faer's blocked Cholesky constants.
  Blocked rank-k group updates and SIMD kernels are the headroom.
- **The break-even policy matters.** A first cut (fold every group
  unconditionally) regressed small/low-cardinality sets up to 10x — folding
  a wide group costs more than the refit it replaces. Groups wider than
  ~dim/3 rows now drop the factors; solves take the exact path until a finer
  group re-materializes them. Small sets then sit at ~0.6-1.0x (fixed
  SideTrack construction overhead on fits that take tens of milliseconds).
- **Parity is ~1e-5 rmse, not bit-identical, at full-tree depth.** The oracle
  proves split identity on clear-gap fixtures; on real data the fast-path
  arithmetic (factorized solve vs faer/elimination) differs in the last
  ulps, and candidates that tie within the `beats` band (1 + 1e-12) can flip
  in either direction. The flips are behavior-neutral in score (same rmse to
  4-5 decimals everywhere) and the golden fixtures — where gaps are clear —
  pass byte-identical. This is the documented floating-point tolerance of
  the change, not a divergence in the decision rule.
- **l2 = 0 singular systems** (dead NA-indicator columns, collinear designs)
  refuse the factorization and take the historical rank-revealing path —
  correct but without the speedup. The default ridge (l2 = 0.5) keeps every
  system positive definite, so field runs get the fast path.
- The elastic-net path (l1 > 0) has no maintained factor; it warm-starts the
  coordinate descent from the previous candidate's coefficients (same
  unique optimum for l2 > 0; the objective value is unique regardless).
