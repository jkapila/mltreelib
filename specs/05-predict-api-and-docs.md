# [PREDICT+DOCS] Prediction API family, docs/examples completion, small extensions

## Problem Statement

The tree predicts values only — no probabilities, intervals, or
distributional outputs — while the library now has leaf t-digests, robust
and quantile leaves, and (soon) distributional families that could answer
richer questions. Separately, recent surfaces (median/quantile/huber leaves,
parts of the criterion family, several growth knobs) exist and are tested but
are not demonstrated on any docs page, and a few small extensions are owed.

## Solution

A `predict` API family that answers the questions a fitted tree can already
answer (probabilities for 0/1 targets, conditional quantiles, prediction
intervals, per-row source info) with an extension path for the family-based
outputs of issue #4. Plus: every documented surface gets a real example page
with real output, and the small extensions land.

## User Stories

1. As a user, I want `predict_proba` for 0/1 targets, so that classification outputs are probabilities, not hard values.
2. As a user, I want prediction intervals per row, so that uncertainty is visible where the tree already knows it (leaf digests).
3. As a user, I want conditional quantiles through the predict family, so that quantile leaves and digests share one entry point.
4. As a user, I want per-row source info (which leaf, its n/stats), so that predictions are inspectable without a separate explain call.
5. As a user, I want distribution parameters per row where the leaf family provides them, so that issue #4 has an output surface ready to plug into.
6. As a user, I want thresholded class labels with a configurable threshold, so that decisions layer cleanly on probabilities.
7. As a user, I want the predict family to compose with leaf kinds (mean/linear/median/quantile/huber) and future families, so that one API serves all.
8. As a maintainer, I want docs pages demonstrating median/quantile/huber leaves with real output, so that the surface is visible, not just tested.
9. As a maintainer, I want examples covering the full criterion family with real output, so that docs never claim what no run shows.
10. As a maintainer, I want the growth-knob examples (grid, cat search, col_sample, prune) demonstrated on an examples page.
11. As a maintainer, I want the docs pages regenerated-verified against the built module, so that signatures never drift (existing gen_api_doc discipline).
12. As a maintainer, I want the small owed extensions folded in (Elo page time-tier note, playground knob refresh), so that nothing dangles.

## Implementation Decisions

- The predict family's exact shape is OPEN — to be decided at kickoff with
  the user's inputs (they hold further ideas). Candidate shapes recorded:
  (a) separate methods (`predict_proba`, `predict_interval`, `predict_dist`,
  `predict_source`), (b) one `predict(..., return_=...)` switch, (c) a
  `Prediction` result object. The spec assumes (a) unless the user's inputs
  say otherwise.
- Probability source for 0/1 targets: the leaf's fitted probability (the
  event rate from the leaf target stats) — no family machinery required;
  issue #4 later plugs binomial-family probabilities into the same method.
- Prediction intervals: from the per-leaf t-digest quantiles at the requested
  coverage (works for every leaf kind that keeps `leaf_digest`).
- Quantiles: the existing `quantile(x, tau)` is folded into the family as the
  same contract, not duplicated.
- Source info reuses the existing leaf records (n, mean, stats, digest
  presence) — no new internals.
- Class labels + threshold: `predict_label(x, threshold=0.5)` (name open at
  kickoff) layered strictly on the probability output.
- Docs/examples: the example scripts remain the source of every pasted
  output (the established real-output discipline); new page(s) under
  Examples + cross-links from the API pages.

## Testing Decisions

- External behavior only: probabilities in [0,1] and separating a planted
  regime; intervals ordered and covering at the advertised rate on a planted
  distribution; quantile family consistency with the existing digest tests;
  source info matching the leaf the row lands in.
- Single seam: the existing core integration tests file (prior art: the
  quantile/leaf-kind tests). Python surface via the throwaway-smoke
  convention.

## Out of Scope

- Distributional leaf families themselves (issue #4 owns the fits; this spec
  owns the output API shell).
- The growth-knob implementations (issue #3).
- Calibration tooling, per-class thresholds beyond one threshold argument,
  multi-output targets.

## Further Notes

- Kickoff protocol (user directive): **ask the user for their predict-API
  ideas at the start of this ticket** — they hold further inputs on shape and
  coverage; fold them in before implementation begins.
