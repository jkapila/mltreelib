# [CAT-BUG] Categorical split path: repro harness + fix (ta_diamonds failure)

## Problem Statement

The categorical (level-set) split path misbehaves on `ta_diamonds` — a
dataset dominated by categorical predictors (cut / color / clarity). Two
symptoms, possibly one root cause:

1. An earlier round logged `ModelTree(all)` `rmse=0.000` on `ta_diamonds` —
   near-perfect prediction on a hard dataset strongly suggests leakage in
   the categorical path (a split or leaf seeing information it must not).
2. The overnight round (2026-10-30) recorded `ModelTree(all) TIMEOUT (300s,
   killed)` on `ta_diamonds` — the categorical scan may also be blowing up
   candidate counts / fit time on that shape (3 cat cols, ~53k rows).

User directive: **build smaller data and test examples to trace the bug**
before touching the algorithm — find the smallest failing slice first.

## Solution

A reduced-repro harness for the categorical path, a traced diagnosis from
example tests, then the fix with a regression test.

## User Stories

1. As a maintainer, I want small categorical fixtures (tens of rows, known
   ground truth) that reproduce the failure, so that the bug is traceable.
2. As a maintainer, I want the smallest failing slice of ta_diamonds
   (subsampled progressively), so that the trigger is isolated.
3. As a maintainer, I want example tests that trace the split search on
   categorical columns (candidate counts, chosen partitions, gain values,
   leaf fits per regime) via the training log, so that the wrong step is
   visible.
4. As a maintainer, I want the rmse=0.000 leak explained and fixed, so
   that field numbers on categorical datasets are trustworthy.
5. As a maintainer, I want the timeout cause identified (candidate
   explosion vs pathological fit), so that the budget story stays honest.
6. As a maintainer, I want a regression test pinning the fixed behavior on
   the reduced fixtures.

## Implementation Decisions

- Method (user): smaller data first — subsample `ta_diamonds` (full data
  via `load_dataset`) to n = 1k / 5k / 10k rows and to single-column
  triggers; construct parallel synthetic fixtures from the generator with
  categorical-only signal and planted ground truth.
- Trace via existing tooling first: `verbose=2` training log (candidate
  counts, dtype search, chosen gain), `break_down`, per-leaf stats; only
  then instrument internals.
- The repro fixtures live as examples (the established real-output
  discipline) and graduate into the behavior-test seam once the fix lands.
- If the leak is in the evaluation prep (dataset split leakage in
  `benchmarks/`), not the tree: fix the prep and say so — the spec owns
  the honest answer either way.

## Testing Decisions

- Reduced fixtures must fail before the fix and pass after (both pinned).
- One regression test in the existing core integration seam; example tests
  keep the trace visible in docs.

## Out of Scope

- New categorical features (cat×cat combinations remain in the roadmap).
- Knob work (#3) and predict API (#5) — this bug hunt goes first when we
  next work categorical behavior.

## Further Notes

- Companion facts: the Elo anomaly entry in the 2026-10-30 handoff
  (must-do #1) is consolidated here; the timeout row in
  `benchmarks/elo_run.log` ("ta_diamonds ModelTree(all) TIMEOUT (300s)") is
  the second symptom.
