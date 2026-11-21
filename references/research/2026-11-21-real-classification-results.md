# Real classification split and calibration results

Date: 2026-11-21.

The paired runner is `examples/benchmark_real_classification.py`. It loads the
repository Parquet files, creates one immutable row-index payload, and runs the
same payload in isolated Python processes. It uses `RAYON_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Each timing is the median of
three fits and predictions. The recipe is `max_depth=6`, `min_rows=30`,
`gain="gini"`, `leaf="mean"`, `prune="none"`, `seed=42`; it uses `softmax` for
SDSS17 and `binomial` for Diabetes.

## Inputs and provenance

- SDSS17: 78,053 rows, 9 numeric predictors, three classes. The deterministic
  index order is `(i * 1103515245 + 12345) & 0x7fffffff`, then 75% of the first
  80% is fit, the remaining 25% is calibration, and the final 20% is test.
- Diabetes: OpenML task 363629 fold 0. The official train rows split by row
  order into 256 fit and 256 calibration rows. The official test rows contain
  256 rows. This keeps calibration separate from both fit and test.
- SDSS17 SHA-256: `668a363a27266198aa6c0da81e338e52aa91d1ca0f06a3bfc487f041f392f042`.
- Diabetes SHA-256: `4bd1b10df159f3212a7f82546105388cf77abb6239abc591be3c9a6cdf4691f3`.
- Branch extension SHA-256: `b62539ebed7d665bd0386cfb4f52e2cebf7668d6370e5cccb22d90f67f30c313`.
- Baseline: wheel `/tmp/mltreelib-0.4.11-wheel.VaMmc5`, version 0.4.11.
- Branch: local checkout, version 0.4.11, extension path under
  `/mnt/d/Programs/mltreelib/python/mltreelib`.

## Results

| data | package | split | log loss | accuracy | fit median (s) | predict median (s) |
|---|---|---|---:|---:|---:|---:|
| SDSS17 | v0.4.11 wheel | fast | 0.12026166 | 0.96278265 | 0.293321 | 0.020601 |
| SDSS17 | branch | fast | 0.12026166 | 0.96278265 | 0.313350 | 0.027601 |
| SDSS17 | branch | histogram | 0.11975565 | 0.96355134 | 0.226868 | 0.024784 |
| Diabetes | v0.4.11 wheel | fast | 0.57255319 | 0.68359375 | 0.000431 | 0.000094 |
| Diabetes | branch | fast | 0.57255319 | 0.68359375 | 0.000596 | 0.000105 |
| Diabetes | branch | fast + isotonic | 0.55142922 | 0.68359375 | — | — |

The baseline does not accept `split_eval="histogram"`, so no baseline
histogram row exists. The split change preserved fast predictions on this
SDSS17 payload. Histogram improved measured SDSS17 loss and timing in this run,
but this is one fixed recipe and split, not a general accuracy claim.

The branch isotonic calibrator improved Diabetes test log loss on this separate
validation fit. The baseline wheel has no `mltreelib.calibration` module, so its
isotonic result is unavailable. These results do not test ensembles or solvers.

## Exact commands

```sh
.venv/bin/python examples/benchmark_real_classification.py \
  --make-payload /tmp/sdss.json --data datasets/SDSS17.parquet
.venv/bin/python examples/benchmark_real_classification.py \
  --make-payload /tmp/diabetes.json --data datasets/diabetes.parquet \
  --split-file /tmp/mltreelib-pilot-openml/org/openml/www/tasks/363629/datasplits.arff
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python examples/benchmark_real_classification.py \
  --payload /tmp/sdss.json --mode fast --output /tmp/branch-sdss-fast.json
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/mltreelib-0.4.11-wheel.VaMmc5/bin/python examples/benchmark_real_classification.py \
  --payload /tmp/sdss.json --mode fast --output /tmp/base-sdss-fast.json
```

The same command uses `--mode histogram` for the branch-only histogram row.
The focused candidate-cap proof is `cargo test -p mltreelib-core
--test multiclass_split histogram_mode_limits_candidates_and_preserves_aligned_class_boundary`.
The paired SDSS17 fast probability hash matched between baseline and branch.
The branch Diabetes raw Brier score was `0.38550195`; isotonic Brier was
`0.37284569`. The runner records payload, binary, trial, and probability hashes
in each JSON result.
