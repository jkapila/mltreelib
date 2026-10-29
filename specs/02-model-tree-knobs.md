# [KNOBS] Model-tree capability batch: significance splits, honesty, surrogates, pruning, EFB, purity gate

## Problem Statement

Our model tree grows by raw gain maximization with size-based stopping, so
statistically-aware rivals (lmtree/glmTree) out-grow it: splits are chosen
where SSE drops most, not where the leaf model actually changes, and no
multiple-testing correction applies. Missing capabilities the maintainer
named: significance-gated splitting, separate leaf/split variable sets,
honest leaf estimation, NA surrogate splits, a real pruning family
(rpart CP / ccp_alpha, C5 pessimistic, cost-matrix), ctree-style
minsplit/minbucket/minprob, EFB-style pre-selection beyond winnow, and an
experimental purity gate for leaf complexity.

## Solution

One batch of growth knobs — all default-off, all measured by a round before
any default flips: `split_test="chow"` with `split_alpha` + `bonferroni`
growth-time significance gating; explicit `leaf_cols`/`split_cols`;
`honesty` sample splitting; surrogate splits for NA routing; the pruning
family (`prune` extended with cp/ccp_alpha, c5, cost-matrix); the
minsplit/minbucket/minprob trio; EFB-style pre-selection; and the purity
gate (`leaf_gate` = bic|cv with the DAR y-tail exception).

## User Stories

1. As a maintainer, I want splits accepted only when the leaf-model change is significant (`split_test="chow"`), so that the tree splits where the model breaks, not where noise yields gain.
2. As a maintainer, I want `split_alpha` and Bonferroni correction over candidate tests, so that growth is multiple-testing aware like mob/ctree.
3. As a maintainer, I want explicit `leaf_cols` and `split_cols` lists, so that I can split on interactions while fitting leaf models on a chosen subset (lmtree's `y ~ x | z`).
4. As a maintainer, I want an `honesty` option, so that leaf models are fitted on rows not used to choose splits (grf's mechanic, pre-building the forest phase).
5. As a maintainer, I want NA surrogate splits, so that missing values route by the best surrogate's agreement instead of a single learned direction (rpart-style) — the biggest missing-data capability.
6. As a maintainer, I want cost-complexity pruning (`prune="cp"` with `ccp_alpha`), so that tree size is priced like rpart/sklearn.
7. As a maintainer, I want C5-style pessimistic error pruning with a confidence factor, so that classification-shaped pruning is available.
8. As a maintainer, I want cost-matrix pruning support, so that asymmetric mispricing is expressible (the niche-but-real case).
9. As a maintainer, I want `minsplit`, `minbucket`, and `minprob` separated (ctree trio), so that split eligibility and leaf size are different criteria, not one `min_rows`.
10. As a maintainer, I want EFB-style pre-selection, so that exclusive/bundled features are screened before scanning (we only have winnow-style relevance today).
11. As a maintainer, I want a purity gate (`leaf_gate`), so that pure leaves skip the model fit and save time.
12. As a DAR maintainer, I want leaves in the target's y-tails to always keep their fit, so that extrapolation/interpolation behavior in the tails survives the gate.
13. As a maintainer, I want `gain="cv"` as a split criterion, so that coefficient-of-variation reduction can drive splits as well as gate leaves.
14. As a maintainer, I want every knob default-off, so that no board number moves until a round measures it.
15. As a maintainer, I want the round manifests to record each knob's state, so that knob effects are attributable.
16. As a maintainer, I want categorical partitions to keep their binary level-set semantics through all new gates, so that nothing regresses there.
17. As a maintainer, I want surrogates to degrade gracefully when no surrogate beats a threshold, so that NA routing never gets worse than today.
18. As a maintainer, I want the training log to show gate decisions, so that I can see why a leaf became constant.

## Implementation Decisions

- Growth gate: a Chow/F test comparing pooled-vs-split model fit; the
  statistic falls out of the SSEs the linear scan already computes
  (prototype-verified formula: `F = ((SSE_pooled − SSE_split)/k) /
  (SSE_split/(n − 2k))`). The F tail probability uses a small
  incomplete-beta implementation in core (std-only).
- Naming (confirmed with the user): the significance level is
  `split_alpha` (the existing `alpha` stays the grf-style min-child-weight
  share); lasso stays `l1`.
- Surrogates: each split stores up to two surrogate columns + cut points
  chosen by agreement with the primary partition; NA routing = surrogate
  agreement, falling back to `missing_left` (which stays the learned
  default). Surrogate records live in a side table like the categorical
  partitions, keeping `Split` copy-sized.
- Pruning family extends the existing `PruneCost` concept: `cp`/`ccp_alpha`
  cost-complexity path, `c5` pessimistic pruning with a confidence factor
  knob, `cost` cost-matrix variant. All post-growth, bottom-up, like the
  existing bic/mdl.
- The minsplit/minbucket/minprob trio maps onto the existing gate checks in
  all three scans; `min_rows` remains as the simple alias while defaults
  hold.
- Purity gate: `leaf_gate=None|"bic"|"cv"` with `leaf_gate_threshold` and
  `tail_quantile` (y-tail exception). BIC is the PILOT-style node-local
  rule (the evidence-backed choice from the leaf-purity research).
- EFB-style pre-selection composes with the existing `min_relevance`
  screening as a separate opt-in stage.

## Testing Decisions

- Tests assert external behavior only: which splits form, where NA routes,
  tree sizes after pruning, leaf outputs, prediction values.
- Single seam: the existing core integration tests (the file where the
  criterion, leaf-kind, weight, column-sample and pruning tests already
  live). Prior art: the bic-prune and huber-leaf tests.
- rpart-parity fixtures are added only where behavior claims rpart
  equivalence (CP pruning path, surrogate routing).
- The Python binding keeps the established throwaway-smoke convention; no
  new permanent test seam.

## Out of Scope

- Distributional / leaf_family work (separate issue).
- Forest/committee ensembling; multiway categorical partitions (reserved).
- Elo rescoring (the knob batch is measured in the big round after landing).

## Further Notes

- The user's order is authoritative: chow gate → leaf/split cols → honesty →
  surrogates → pruning → minsplit trio → EFB → purity gate.
