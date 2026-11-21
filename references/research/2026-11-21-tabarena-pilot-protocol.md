# TabArena exact-split pilot protocol

Date: 2026-11-21.

## Sources and fixed facts

- OpenML Study 457 is the `TabArena-v0.1` suite. It contains 51 curated tasks.
  The [OpenML study record](https://www.openml.org/api/v1/json/study/457) is the
  source of the task list.
- TabArena documents a shared model protocol with an official 8-fold by 1-set
  inner validation protocol. See the [TabArena repository](https://github.com/autogluon/tabarena)
  and its [validation protocol documentation](https://github.com/autogluon/tabarena/blob/main/AGENTS.md).
- TabArena's public artifacts include processed results and raw predictions.
  The upstream examples state that `results_per_split.csv` is generated locally
  from the Results tier and is not hosted as a standalone file. See the
  [artifact examples](https://github.com/autogluon/tabarena/blob/main/examples/README.md).
  Therefore, this pilot does not assume that exact public per-fold losses exist
  for CatBoost or LightGBM.

## Reproducible pilot

`examples/benchmark_tabarena_pilot.py` loads OpenML task metadata and data with
`openml.tasks.get_task`, then uses the task's `get_train_test_split_indices` for
the same train/test rows across ModelTree, CatBoost, and LightGBM. It records
task ID, fold, metric, loss, fit time, prediction time, and serialized size as
JSON. The optional `--include-forest` flag adds `LinearForest` on regression
tasks. Forest snapshots are not serialized by the current runner, so their
`serialized_bytes` field is null.

The default pilot uses task 363612 (`airfoil_self_noise`, regression) and task
363629 (`diabetes`, binary classification), fold 0. These tasks are numeric-only,
so the pilot avoids inventing a categorical preprocessing policy. Add a task or
fold explicitly with repeated `--task` or `--fold` flags.

```sh
OPENML_CACHE_DIR=/tmp/mltreelib-pilot-openml \
PYTHONPATH=/tmp/mltreelib-pilot-deps \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
uv run --offline --no-cache --no-sync python \
  examples/benchmark_tabarena_pilot.py --fold 0 --include-forest \
  --output references/research/2026-11-21-tabarena-pilot-results.json
```

The command uses the pinned local OpenML cache and dependency directory. It
sets one thread for each native backend. Record package versions and the
command environment beside the JSON output.

The runner is a paired split pilot. It is not an official TabArena leaderboard
entry because it does not reproduce TabArena's inner validation, tuning budget,
preprocessing, or ensembling. Do not compare its raw errors with leaderboard
Elo. The pilot also does not match any tabular foundation-model variant.

## Result storage

The JSON result file must include one row per entrant, task, and fold. Add a
short dated result section below this protocol after the run. Include exact
losses, fit and prediction seconds, serialized sizes, package versions, and any
failed entrant. Keep the raw JSON artifact in `references/research/`.

No result is published in this note until the runner completes with the pinned
cache and environment above.

## Reuse decision

The upstream artifact description supports reusing aggregate and processed
results, but it does not establish a stable downloadable per-fold CSV for each
released entrant. Until a pinned Results artifact is obtained and its task,
fold, metric, and imputation columns are checked, rerun the local baselines on
the same OpenML task splits and label peer results as unavailable.

