# [ELO] Finish the field: new model entries, time-budget tiers, overnight full scoring

## Problem Statement

The Elo field is missing whole model schools (Bayesian BART, the TRUST linear
model tree, random-forest variants, statistical boosting), fits can run
unbounded (an 811s diamonds fit proved it), and there is no way to wake up to
a fully scored board. The maintainer needs every candidate model scored on
all 18 field datasets overnight, under a strict time budget, with a
provenance record of what produced each number.

## Solution

Wire the remaining field entries into the scorers, enforce the three-tier
time budget on every fit (30s slow / 180s critical / 300s hard kill), and run
the full field + round chain overnight so the morning board is complete.
Skip-if-scored makes the run incremental: only new (dataset, model) pairs
execute.

## User Stories

1. As a maintainer, I want BART (dbarts) and bartMachine scored on the field, so that the Bayesian school has a measured place on the board.
2. As a maintainer, I want the TRUST linear model tree (trust-free) scored, so that our tree is compared against its closest published rival.
3. As a maintainer, I want random-forest variants (sklearn RF, ranger, fru) scored, so that the averaging school is represented beyond grf.
4. As a maintainer, I want mboost (blackboost) scored, so that statistical gradient boosting is on the board.
5. As a maintainer, I want every fit bounded by a 300s hard kill, so that one pathological dataset cannot stall the field.
6. As a maintainer, I want a critical log line when any fit exceeds 180s, so that slow models are visible without reading every row.
7. As a maintainer, I want a slow-marker when a fit exceeds 30s, so that we can see who is off the TabArena-toppers pace.
8. As a maintainer, I want killed fits to leave the previous row untouched and an attempts-ledger record, so that timeouts never poison the scores.
9. As a maintainer, I want the run to be incremental, so that restarting never re-scores finished pairs.
10. As a maintainer, I want the round to record a provenance manifest per run, so that every number is attributable to settings + git SHA.
11. As a reader, I want the morning board regenerated automatically (interactive table + ASCII ranking), so that no manual step is needed.
12. As a maintainer, I want all run events in stdlib-logging files, so that "what ran when" has one place to look.
13. As a maintainer, I want the diamonds rmse=0.000 anomaly investigated before the candidates are judged, so that no decision rests on a broken row.
14. As a maintainer, I want one multi-core job at a time, so that the field does not oversubscribe cores.

## Implementation Decisions

- New field entries live in the existing scorer registries: the R scorer's
  model switch gains bartMachine/dbarts-BART, ranger, fru, mboost; the python
  scorer's model map gains trust-free and sklearn RandomForestRegressor via
  its lazy `attempt(name, builder)` pattern so missing optional deps degrade
  to a recorded failure, not a crash.
- mboost entry is `blackboost` with tree base-learners, stock config (the
  field's stock-configs-only rule).
- trust-free is consumed as its published Python package (the local
  references checkout is documentation, not the install path).
- Time tiers are shared constants: 30s slow, 180s critical, 300s hard kill;
  the hard kill goes through the existing subprocess harness
  (process-group death); round fits run in forked children so a kill takes
  the rayon threads with it.
- The field driver keeps skip-if-scored and the attempts ledger as the
  audit record; the round keeps settings-manifests + log lines.
- The overnight chain is strictly serialized: field, then round, then board
  generation.

## Testing Decisions

- Good tests assert external behavior only: scored rows, budget enforcement,
  and ledger records — never how a scorer builds its model internally.
- The scorer contract seam: attempts-ledger entries + scores-file rows (the
  existing convention used by the budget harness sleeper test).
- The budget harness keeps its existing sleeper-based kill test as prior art;
  new tier warnings are covered by the same shape.
- Rust/Python model code is compile-checked and smoke-run (throwaway), per
  the established convention.

## Out of Scope

- The capability knobs batch (separate issue).
- The leaf distribution family (separate issue).
- Re-scoring already-scored pairs; Elo protocol changes; multiway splits.

## Further Notes

- The anomaly (a near-zero test error on diamonds under the candidate
  config) must be root-caused first; if it is data leakage in the prepared
  split, the fix lands before the big round.
