# Real-data unsupervised results

The paired benchmark uses the published `mltreelib` 0.4.11 wheel and a
local release build. Both builds run
[`examples/real_unsupervised_workflow.py`](../../examples/real_unsupervised_workflow.py)
with the same model settings and ordered inputs. The
[raw results](2026-11-21-real-unsupervised-results.json) record every trial,
the extension SHA-256 for each build, and hashes for the Parquet files,
split indices, model inputs, predictions, fitted state, and snapshots.

## Data and timing

- Bank Customer Churn uses seven mixed-type predictors. The first 6,000
  rows after `Random(2026).shuffle` are fit rows. Positions 8,000–8,599 are
  holdout rows. EIF scores and PCT cluster labels use no outcome labels.
  [OpenML 46911](https://www.openml.org/d/46911)
- APS Failure uses eight numeric sensors with different missing rates.
  The first 1,200 shuffled rows are fit rows. The next 1,200 are holdout rows.
  No label informed the split or feature selection. The holdout contains 23
  failures (1.92%). Labels enter only the tie-aware average precision metric.
  [OpenML 46908](https://www.openml.org/d/46908)

APS predictors are `aa_000`, `ab_000`, `ac_000`, `ad_000`, `ag_005`,
`ah_000`, `an_000`, and `ao_000`. Each engine receives the same sensor
values and natural missing cells for its dataset.

The machine uses an AMD Ryzen 9 5900HS. Both Python processes use
Python 3.12.3, pandas 3.1.1, NumPy 2.5.5, and PyArrow 26.0.1. Each trial
creates a fresh model. Fit includes threshold calibration on training data
for the anomaly engines. Predict measures anomaly scores or cluster labels.
Save and load measure in-memory `to_bytes` and `from_bytes` independently.
All results use five serial repetitions and one Rayon, OpenMP, and OpenBLAS
thread. The runner checks that fit and holdout indices are disjoint.

The Bank Churn partitions contain no missing cells. APS contains 1,286
missing fit cells and 1,357 missing holdout cells. Seven of eight predictors
contain missing values in each partition.

## Measured results

| Data | Engine | Build | Fit (ms) | Predict (ms) | Save (ms) | Load (ms) | Snapshot (bytes) |
|---|---|---|---:|---:|---:|---:|---:|
| Bank Churn | EIF | 0.4.11 | 17.020 | 1.021 | 7.041 | 2.934 | 144,423 |
| Bank Churn | EIF | Branch | 17.071 | 0.795 | 6.938 | 2.792 | 144,423 |
| Bank Churn | PCT | 0.4.11 | 15.860 | 0.637 | 2.131 | 0.835 | 59,697 |
| Bank Churn | PCT | Branch | 21.621 | 0.796 | 2.008 | 1.358 | 59,697 |
| APS | EIF | 0.4.11 | 1.357 | 0.588 | 2.148 | 0.816 | 36,020 |
| APS | EIF | Branch | 1.553 | 0.480 | 1.824 | 0.878 | 36,020 |
| APS | HST | 0.4.11 | 5.371 | 1.743 | — | — | — |
| APS | HST | Branch | 4.041 | 2.752 | 1.920 | 0.688 | 27,764 |
| APS | CLTree | 0.4.11 | 4.316 | 1.021 | — | — | — |
| APS | CLTree | Branch | 4.786 | 1.672 | 1.415 | 0.731 | 25,930 |

Numbers are medians. Bank PCT fit was 15.860 ms on the wheel and
21.621 ms on the branch in this session. The single split and five
repetitions support workload measurements only.

EIF on both datasets and Bank PCT matched in predictions, fitted state,
and snapshot bytes across builds. Bank PCT assigned 124, 147, 177, and 152
holdout rows to clusters 0–3. APS average precision was 0.10216 for EIF and
0.18118 for HST on both builds. The 23 holdout failures give a prevalence
baseline of 0.01917.

The release imputes missing HST and CLTree inputs with training-column means.
The branch preserves missing values and routes them left. CLTree accounts
for those rows when evaluating a split. On this APS partition, HST still
produced the same scores and threshold. That match applies to this input.
CLTree assigned five clusters and 28 noise rows on the wheel, then four
clusters and 33 noise rows on the branch.

The release wheel can fit and predict with both HST and CLTree. Its
snapshot calls return these `ValueError` messages:

```text
JSON snapshots are available for eif, isolation, extended_isolation, got 'hst'.
JSON snapshots are available for pct, predictive, clus, got 'cltree'.
```

The branch restored both snapshots in every repetition. The runner checks
exact held-out scores, hard predictions, and thresholds for anomaly engines.
For clustering, it checks exact holdout predictions, fitted labels, and
cluster count. Predictions, state, and snapshot bytes also match across
fresh fits within each build where snapshots are available.

## Reproduction

Build the checkout with `uv run maturin develop --release`. Install the
published 0.4.11 wheel in a separate Python 3.12 environment. Use identical
versions of NumPy, pandas, and PyArrow in both environments. This run used
one shared dependency directory.

```sh
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/tmp/mltreelib-pilot-deps

for data in bank aps; do
  /tmp/mltreelib-0.4.11-wheel.VaMmc5/bin/python examples/real_unsupervised_workflow.py \
    --dataset "$data" --reps 5 --release-baseline > "/tmp/${data}-wheel.json"
  .venv/bin/python examples/real_unsupervised_workflow.py \
    --dataset "$data" --reps 5 > "/tmp/${data}-branch.json"
done
```

This comparison uses paired runs of the released wheel and branch. The
earlier Wine PCT and synthetic EIF example uses a different workload.
[Earlier example](../../examples/unsupervised_trees.py)
