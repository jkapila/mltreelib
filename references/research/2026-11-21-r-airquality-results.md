# R ModelTree on airquality — 2026-11-21

The source package installed and passed `R CMD check --no-manual --no-vignettes` from outside the repository on Ubuntu 24.04 x86-64 with R 4.3.3. The check reported zero errors, zero warnings, and one note for the 15.4 MB installed size. Cargo used cached dependencies in offline mode. R could not reach the CRAN index, and its package dependency check passed. The base-R tests cover model fit and prediction, categorical level mapping, registered native calls with malformed input, and external pointer handling.

## Holdout

The example uses the 153 rows in `datasets::airquality`. Ozone is present in 116 rows. The fit uses 61 of those rows from May through July. The holdout uses 26 August rows. The remaining 29 eligible rows fall in September. Predictors are Solar.R, Wind, Temp, Month, and Day. Two training rows and three holdout rows have missing Solar.R. The example keeps those values. [Dataset reference](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html)

The example fits a tree with a maximum depth of 6 and a minimum of two rows per child. It scores five fits against the same holdout. All five runs produced the same 26 finite predictions. Errors are in ozone parts per billion. [Example](../../r/airquality_model_tree.R)

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Branch R ModelTree | 22.1771 | 32.6901 |
| Training mean | 32.8985 | 43.9131 |
| Python ModelTree, released v0.4.11 or branch, with default leaf NA policy | 23.1300 | 34.5695 |
| Python ModelTree, released v0.4.11 or branch, with leaf NA indicators | 22.1771 | 32.6901 |

There is no released v0.4.11 R package for comparison. The Python runs use the same numeric rows, column order, depth, and minimum child size. Python defaults to `leaf_na_policy="drop_fallback"`. The R binding inherits the core default `Indicator`. Setting `leaf_na_policy="indicator"` in Python produced exact agreement with all R predictions for the released wheel and the branch. [Released Python build](https://pypi.org/project/mltreelib/0.4.11/)

The R timing includes input conversion and the native call. The five elapsed fit samples in seconds were `0.003, 0.001, 0.001, 0.001, 0.001`. The five prediction samples were `0.001, 0.001, 0.000, 0.001, 0.000`. Both medians were 0.001 seconds. Prediction falls at or below the resolution of this R timer, so the samples give no precise latency estimate.

## Reproduce the R check and example

Run the following block from the checkout. Cargo needs the dependencies in its cache when the offline setting is active.

```sh
out=$(mktemp -d)
r/build-r-package.sh "$out"
repo=$(pwd)
cd "$out"
export CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CARGO_NET_OFFLINE=true
R CMD check --no-manual --no-vignettes mltreelib_0.4.12.tar.gz
R_LIBS="$out/mltreelib.Rcheck" Rscript --vanilla "$repo/r/airquality_model_tree.R"
```
