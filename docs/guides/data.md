# Prepare data for modeling

**Path:** Define the prediction time → Describe the columns → Check missingness → Run quality checks → Measure associations → Split rows → Bin and transform on training rows → Hand the frame to the model → Apply the same schema to new data.

This guide follows one real dataset from raw columns to a model input. It covers the data helpers in `mltreelib.data`. It does not cover categorical encoders or time-series features; see the links in step 8.

| Decision | Check |
|---|---|
| Split | Keep future observations, customers or groups out of training when they define the deployment setting. |
| Numeric | Check units, impossible values, missingness and whether scaling is appropriate for the chosen model. |
| Categorical | Preserve the distinction between labels and ordered numeric measurements. Fit any mapping on training data. |
| Target | Exclude the outcome and features that become available only after the prediction time. |
| New data | Preserve column order, names, types and the fitted mapping. Test unknown levels and missing values. |

The example is the Give Me Some Credit table (`GiveMeSomeCredit`, 150,000 rows). The target `FinancialDistressNextTwoYears` says whether a borrower was in financial distress within two years. Each feature is assumed to be known when the lending decision is made; confirm that with the data owner before a real deployment. Two columns, `MonthlyIncome` and `NumberOfDependents`, have missing values. Every helper below keeps them as they are.

## 1. Describe each column

`describe` gives one row per column: its kind, how much is missing, the summary statistics, and outlier counts. Its statistics skip missing and infinite values and never fill them.

<!-- run -->
```python
from mltreelib import load_dataset
from mltreelib.data import describe, infer_types

d = load_dataset("GiveMeSomeCredit")
cols = d["columns"]
target = "FinancialDistressNextTwoYears"
cols[target] = [1 if v == "Yes" else 0 for v in cols[target]]  # hand recode: helpers never recode

print(infer_types(cols)[target])
rows = describe(cols, kinds={target: "categorical"})
print(f"{'column':<40}{'kind':<14}{'missing':>8}{'mean':>12}{'trim_mean':>12}{'mad':>10}{'iqr_out':>9}")
for r in rows:
    print(f"{r['name']:<40}{r['kind']:<14}{r['n_missing']:>8}{r['mean']:>12.3f}"
          f"{r['trimmed_mean']:>12.3f}{r['mad']:>10.3f}{r['n_outliers_iqr']:>9}")
```

<!-- output -->
```text
{'kind': 'ordinal_candidate', 'n_levels': 2, 'ordinal_candidate': True}
column                                  kind           missing        mean   trim_mean       mad  iqr_out
FinancialDistressNextTwoYears           categorical          0       0.067       0.000     0.000    10026
RevolvingUtilizationOfUnsecuredLines    numeric              0       6.048       0.274     0.148      763
age                                     numeric              0      52.295      51.969    11.000       46
NumberOfTime30-59DaysPastDueNotWorse    categorical          0       0.421       0.075     0.000    23982
DebtRatio                               numeric              0     353.005      51.655     0.246    31311
MonthlyIncome                           numeric          29731    6670.221    5787.565  2317.000     4879
NumberOfOpenCreditLinesAndLoans         numeric              0       8.453       7.963     3.000     3980
NumberOfTimes90DaysLate                 categorical          0       0.266       0.000     0.000     8338
NumberRealEstateLoansOrLines            numeric              0       1.018       0.877     1.000      793
NumberOfTime60-89DaysPastDueNotWorse    categorical          0       0.240       0.000     0.000     7604
NumberOfDependents                      categorical       3924       0.757       0.538     0.000    13336
```

Read the table this way:

- `kind` is a hint, not a type check. An integer column with fewer than six distinct values is `ordinal_candidate`: a category code or a 1-to-5 rating. The first line of the output shows the target with that hint, because its two codes are integers. Passing `kinds={target: "categorical"}` overrides the hint, and the table row then reports `kind_source="user"`. Ordered levels suit monotone constraints, so decide which columns are ordinal on purpose.
- `trimmed_mean` removes the top and bottom 10% by default (`trim=0.1`), so it is less affected by extreme values than `mean`. The gap is large here: `RevolvingUtilizationOfUnsecuredLines` has a mean of 6.048 but a trimmed mean of 0.274, and `MonthlyIncome` has 6,670 against 5,788.
- `mad` (median absolute deviation from the median, unscaled) is a robust spread. Compare it with the standard deviation to see whether a column is heavy-tailed.
- `n_outliers_iqr` counts values outside the Tukey fences. It describes them; nothing is removed. Read it with care for sparse columns: when the middle half is constant, the IQR is zero and every non-zero value counts as an outlier. The target (10,026 ones) and the delinquency counts show this, so their counts carry no meaning.
- Pass `inf_as_missing=True` to count infinite values as missing in `n_missing` and `pct_missing`. The statistics ignore infinities either way.

**Next:** the missingness step counts where the missing values are and checks whether they relate to the target, because a column with many gaps may carry a signal of its own.

## 2. Check missingness

`missing_profile` counts missing values per column and per row, groups rows by which columns they miss together, and gives a co-missing matrix. `missing_vs_target` compares the mean of the target when a column is missing with its mean when the column is observed.

<!-- run -->
```python
from mltreelib.data import missing_profile, missing_vs_target

mp = missing_profile(cols)
print("complete rows:", mp["n_complete_rows"], "of", mp["n_rows"])
for p in mp["patterns"]:
    if p["columns"]:
        print("missing together:", p["columns"], p["count"])
names = mp["co_missing"]["names"]
i, j = names.index("MonthlyIncome"), names.index("NumberOfDependents")
print("rows missing both:", mp["co_missing"]["matrix"][i][j])

mv = missing_vs_target(cols, cols[target])
for name in ["MonthlyIncome", "NumberOfDependents"]:
    r = mv[name]
    print(name, r["n_missing"], r["y_mean_when_missing"], r["y_mean_when_observed"])
```

<!-- output -->
```text
complete rows: 120269 of 150000
missing together: ('MonthlyIncome',) 25807
missing together: ('MonthlyIncome', 'NumberOfDependents') 3924
rows missing both: 3924
MonthlyIncome 29731 0.05613669234132723 0.06948590243537403
NumberOfDependents 3924 0.04561671763506626 0.0674101152824557
```

Two findings. First, the gaps overlap. 25,807 rows miss only `MonthlyIncome`, and 3,924 rows miss both columns. No row misses `NumberOfDependents` alone, so every missing dependents count also lacks income. The two gaps probably come from one source. 120,269 of the 150,000 rows are complete.

Second, the gaps relate to the target. The mean target is 0.0561 where income is missing and 0.0695 where it is observed; for dependents, 0.0456 against 0.0674. The rate differs, so the gap carries information, and the question for the data owner is why the field is blank.

Neither helper imputes or drops a row. The missing rows stay in every later step of this guide: the tree routes them natively in step 8, and any other model needs its own explicit choice. Check the same pattern by source, group and time when those fields exist.

**Next:** the quality checks look for columns and rows that add no information or that will break a later step, such as constant columns, duplicates and a target with too few cases of one class.

## 3. Run quality checks

These checks flag problems and change nothing. `class_balance` reports the target's share of each class. `constant_columns` and `near_zero_variance` find columns that barely vary. `duplicate_rows` groups identical rows. `outlier_flags` marks values outside the IQR fences for one column. `schema_capture` records each column's kind, range and levels, and `schema_validate` checks a later batch against that record.

<!-- run -->
```python
from mltreelib.data import (class_balance, constant_columns, duplicate_rows,
                            near_zero_variance, outlier_flags, schema_capture,
                            schema_validate, take)

print("constant:", constant_columns(cols))
print("near zero variance:", [r["name"] for r in near_zero_variance(cols)])
groups = duplicate_rows(cols)
print("duplicate groups:", len(groups), "rows in groups:", sum(len(g) for g in groups))
print("target balance:", class_balance(cols[target]))
flags = outlier_flags(cols["MonthlyIncome"])
print("MonthlyIncome flagged:", sum(f is True for f in flags), "missing:", sum(f is None for f in flags))

schema = schema_capture(cols)
batch = take(cols, range(1000))  # a copy: the original columns are not changed
batch["age"][0] = 200
print("schema problems:", schema_validate(batch, schema))
```

<!-- output -->
```text
constant: []
near zero variance: ['NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']
duplicate groups: 351 rows in groups: 960
target balance: {'counts': {0: 139974, 1: 10026}, 'proportions': {0: 0.93316, 1: 0.06684}, 'imbalance_ratio': 13.961101137043686, 'n_missing': 0, 'n': 150000}
MonthlyIncome flagged: 4879 missing: 29731
schema problems: [{'column': 'age', 'problem': 'out_of_range', 'detail': '1 value(s) outside [0.0, 109.0], e.g. 200.0'}]
```

Read the results with the data in mind:

- `near_zero_variance` flags `NumberOfTimes90DaysLate` and `NumberOfTime60-89DaysPastDueNotWorse` because most of their values are 0. Both are among the strongest columns in step 6, so the flag marks a sparse column, not a useless one. Do not drop a column on this flag alone.
- 351 duplicate groups hold 960 rows. Two borrowers can share every recorded value. Decide with the data owner whether a duplicate is a repeated record or a real match, and only then remove rows.
- The target has 139,974 zeros and 10,026 ones, an imbalance ratio of 13.96. A model that always predicts 0 would be right 93.3% of the time, so judge models on the rare class. Step 8 reports AUC for that reason.
- 4,879 incomes lie outside the IQR fences. Each flag is a value to inspect, not to delete: some high incomes are real and some are entry errors. The 29,731 missing incomes are not flagged; they stay `None`.
- `schema_validate` reports the one changed age as `out_of_range`: 200.0 is outside the captured range [0.0, 109.0]. Missing values pass unless `allow_missing=False`. Run it on every new batch before scoring.

**Next:** the association step looks at pairs of columns. Strong correlation or association between features costs accuracy and interpretability, and a feature that is almost the target can be a leak.

## 4. Measure associations

`correlation_matrix` gives pairwise correlation, using only the rows where both values are observed, and `correlation_filter` applies the greedy rule that drops columns until no remaining pair is above a cutoff. `cramers_v_matrix` gives the same kind of pairwise table for categorical columns. `vif` gives the variance inflation factor of each numeric column on complete rows. `leakage_hints` scores each feature alone against the target and flags high scores; it removes nothing.

<!-- run -->
```python
from mltreelib.data import correlation_filter, correlation_matrix, cramers_v_matrix, leakage_hints, vif

features = {k: v for k, v in cols.items() if k != target}
corr = correlation_matrix(features, method="spearman")
flt = correlation_filter(corr, cutoff=0.9)
print("correlation drop:", flt["drop"])
names = corr["names"]
pairs = [(abs(corr["matrix"][i][j]), names[i], names[j])
         for i in range(len(names)) for j in range(i + 1, len(names))
         if corr["matrix"][i][j] is not None]
top = max(pairs)
print("largest pair:", round(top[0], 3), top[1], top[2])

cv = cramers_v_matrix(cols)
print("categorical columns:", cv["names"])
for name, row in zip(cv["names"], cv["matrix"]):
    print(f"{name:<40}", [None if x is None else round(x, 3) for x in row])

out = vif(features)
print("VIF rows used:", out["n_used"], "excluded:", out["n_excluded"])
print({k: (None if v is None else round(v, 2)) for k, v in out["vif"].items()})

hints = leakage_hints(features, cols[target])
print([(h["name"], None if h["score"] is None else round(h["score"], 3), h["flagged"]) for h in hints])
```

<!-- output -->
```text
correlation drop: []
largest pair: 0.473 NumberOfOpenCreditLinesAndLoans NumberRealEstateLoansOrLines
categorical columns: ['FinancialDistressNextTwoYears', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
FinancialDistressNextTwoYears            [1.0, 0.289, 0.356, 0.289, 0.047]
NumberOfTime30-59DaysPastDueNotWorse     [0.289, 1.0, 0.372, 0.42, 0.021]
NumberOfTimes90DaysLate                  [0.356, 0.372, 1.0, 0.423, 0.011]
NumberOfTime60-89DaysPastDueNotWorse     [0.289, 0.42, 0.423, 1.0, 0.011]
NumberOfDependents                       [0.047, 0.021, 0.011, 0.011, 1.0]
VIF rows used: 120269 excluded: 29731
{'RevolvingUtilizationOfUnsecuredLines': 1.0, 'age': 1.09, 'NumberOfTime30-59DaysPastDueNotWorse': 26.77, 'DebtRatio': 1.0, 'MonthlyIncome': 1.02, 'NumberOfOpenCreditLinesAndLoans': 1.28, 'NumberOfTimes90DaysLate': 49.21, 'NumberRealEstateLoansOrLines': 1.25, 'NumberOfTime60-89DaysPastDueNotWorse': 60.64, 'NumberOfDependents': 1.07}
[('RevolvingUtilizationOfUnsecuredLines', 0.778, False), ('age', 0.635, False), ('NumberOfTime30-59DaysPastDueNotWorse', 0.69, False), ('DebtRatio', 0.524, False), ('MonthlyIncome', 0.576, False), ('NumberOfOpenCreditLinesAndLoans', 0.545, False), ('NumberOfTimes90DaysLate', 0.657, False), ('NumberRealEstateLoansOrLines', 0.537, False), ('NumberOfTime60-89DaysPastDueNotWorse', 0.622, False), ('NumberOfDependents', 0.547, False)]
```

What to read here:

- `correlation_matrix` uses Spearman ranks here, which are robust to the long right tails of the income and debt columns. It skips non-numeric columns and reports the rows used for each pair in `n_pairs`.
- `correlation_filter` drops the column with the larger mean absolute correlation to the others, and ties drop the later column. At cutoff 0.9 it drops nothing: the largest pair is open credit lines against real-estate loans, at 0.473. A pairwise rule can miss collinearity among several columns, which VIF catches.
- `cramers_v_matrix` treats integer columns with at most 20 distinct values as categorical, and it reports a bias-corrected V and a chi-square p-value for each pair. The target has V of 0.289 with the 30-59 day count, 0.356 with the 90-day count and 0.289 with the 60-89 day count, and 0.047 with dependents. The three delinquency counts also associate with each other at 0.37 to 0.42.
- VIF uses 120,269 complete rows and excludes the 29,731 rows with missing income. The three delinquency counts have VIF of 26.77, 49.21 and 60.64; every other numeric column is between 1.0 and 1.3. Those three move together, so decide which of them to keep before reading any coefficient that uses all three. VIF is also expensive: each numeric column needs its own regression on all the others, so the cost grows quickly as columns are added. Screen the columns first, then compute VIF on the shortlist.
- `leakage_hints` scores each feature alone against the target and flags scores at or above the threshold (0.95 here). Nothing is flagged; the highest score is 0.778 for `RevolvingUtilizationOfUnsecuredLines`. A flag is a question for the data owner: does this column exist at the prediction time? It is not a verdict.

**Next:** the split step fixes which rows train the model and which evaluate it. Every association and screening result above used all rows for description; from here on, only training rows should inform choices that the model will use.

## 5. Split rows

`train_test_split` returns index lists. Pass `stratify` to keep the class share the same in both sides, `groups` to keep all rows of a group on one side, or `time_order` to put the latest rows in the test set. `kfold` does the same for cross-validation. `take` subsets a dict of columns and `select_rows` subsets an `xy` result, both by row index and neither changes a missing value.

<!-- run -->
```python
from mltreelib.data import kfold, select_rows, take, train_test_split, xy

n = len(cols[target])
train_idx, test_idx = train_test_split(n, test_size=0.2, seed=2026, stratify=cols[target])
train_cols, test_cols = take(cols, train_idx), take(cols, test_idx)
print("stratified sizes:", len(train_idx), len(test_idx))
print("target share train:", round(class_balance(train_cols[target])["proportions"][1], 4),
      "test:", round(class_balance(test_cols[target])["proportions"][1], 4))

# Grouped: the dataset has no household id, so NumberOfDependents stands in to show the mode.
g_train, g_test = train_test_split(n, test_size=0.2, seed=2026, groups=cols["NumberOfDependents"])
shared = {cols["NumberOfDependents"][i] for i in g_train} & {cols["NumberOfDependents"][i] for i in g_test}
print("groups shared between sides:", len(shared))

# Time-ordered: the dataset has no date column, so row order stands in for time.
t_train, t_test = train_test_split(n, test_size=0.2, time_order=list(range(n)))
print("time-ordered test starts at row:", min(t_test), "of", n)

folds = kfold(n, k=5, seed=2026, stratify=cols[target])
print("fold test sizes:", [len(test) for _, test in folds])

frame = xy(cols, target)
tr, te = select_rows(frame, train_idx), select_rows(frame, test_idx)

def missing_cells(f):
    return sum(v is None for row in f["x"] for v in row)

print("missing feature cells, whole table:", missing_cells(frame))
print("missing feature cells, train plus test:", missing_cells(tr) + missing_cells(te))
print("train rows:", len(tr["y"]), "test rows:", len(te["y"]), "names:", tr["names"][:3])
```

<!-- output -->
```text
stratified sizes: 120000 30000
target share train: 0.0668 test: 0.0668
groups shared between sides: 0
time-ordered test starts at row: 120000 of 150000
fold test sizes: [30000, 30000, 30000, 30000, 30000]
missing feature cells, whole table: 33655
missing feature cells, train plus test: 33655
train rows: 120000 test rows: 30000 names: ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse']
```

Notes on the split:

- The stratified split gives each class its exact share, within one row. Both sides show a target share of 0.0668, the share of the whole table.
- A group is never split across the two sides: no group is shared here. Missing group labels form their own group. This dataset has no household or customer id, so `NumberOfDependents` is used only to show the mode. Use a real id in practice.
- `time_order` makes the test set the last rows, with no shuffling. Use a real event date in practice; the row order here is not a claim about when the rows were recorded.
- `kfold` with `stratify` gives five test folds of 30,000 rows each. `kfold` with `time_order` instead expands the training window fold by fold.
- No split drops a row, and `take` and `select_rows` do not change a value. Both sides together hold 33,655 missing feature cells, the same as the whole table. The split moved the gaps and changed none.

**Next:** the binning step measures how strongly each numeric column separates the two outcomes. It must use training rows only, because a bin edge or an information value fitted on the test rows would leak the answer into the model.

## 6. Bin a numeric column and measure information value

`iv_bins` cuts a numeric column into ordered bins, merges them to respect a monotone trend, and reports the information value (IV). `binner` gives the bin assignment for each row. `iv_levels` does the same for a categorical column, and `information_value_table` runs the screen over every column.

<!-- run -->
```python
from mltreelib.data import binner, information_value_table, iv_bins, iv_levels

y_tr = train_cols[target]
util = train_cols["RevolvingUtilizationOfUnsecuredLines"]
r = iv_bins(util, y_tr, n_bins=10, monotone="increasing")
print("IV:", round(r["iv"], 4), "trend:", r["trend"], "ordered bins:", len(r["bins"]))
b = binner(util, n_bins=10)
print("bin counts:", b["counts"], "missing:", b["n_missing"])
print("levels IV:", round(iv_levels(train_cols["NumberOfDependents"], y_tr)["iv"], 4))
features_tr = {k: v for k, v in train_cols.items() if k != target}
for row in information_value_table(features_tr, y_tr, n_bins=10):
    print(f"{row['name']:<40}{round(row['iv'], 4):>8}  {row['strength']}")
```

<!-- output -->
```text
IV: 1.0998 trend: increasing ordered bins: 8
bin counts: [12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000] missing: 0
levels IV: 0.0366
RevolvingUtilizationOfUnsecuredLines      1.1212  suspicious
NumberOfTimes90DaysLate                   0.8271  suspicious
NumberOfTime30-59DaysPastDueNotWorse      0.7458  suspicious
NumberOfTime60-89DaysPastDueNotWorse      0.5779  suspicious
age                                       0.2598  medium
DebtRatio                                 0.0801  weak
MonthlyIncome                             0.0738  weak
NumberOfOpenCreditLinesAndLoans           0.0613  weak
NumberRealEstateLoansOrLines              0.0545  weak
NumberOfDependents                        0.0352  weak
```

The greedy merge above is fast but not guaranteed optimal. `optimize="dp"` finds the best set of cut points under the same constraints with an exact dynamic programme over fine pre-bins. It also takes a trend (`"increasing"`, `"decreasing"`, `"peak"`, `"valley"` or `"auto"`), a minimum bin share, a maximum number of bins, and `special_values` that get their own bins outside the trend.

<!-- run -->
```python
exact = iv_bins(util, y_tr, optimize="dp", trend="auto", max_bins=8, min_bin_frac=0.05)
print("greedy IV:", round(r["iv"], 4), "| dp IV:", round(exact["iv"], 4))
print("dp trend:", exact["trend"], "bins:", len(exact["bins"]), "relaxed:", exact["constraint_relaxed"])
for b_ in exact["bins"]:
    print(f"({b_['lo']:.3f}, {b_['hi']:.3f}]  n={b_['n']:>6}  event rate={b_['event_rate']:.3f}  woe={b_['woe']:+.3f}")
```

<!-- output -->
```text
greedy IV: 1.0998 | dp IV: 1.114
dp trend: increasing bins: 8 relaxed: False
(-inf, 0.064]  n= 43200  event rate=0.017  woe=-1.396
(0.064, 0.137]  n= 14400  event rate=0.021  woe=-1.183
(0.137, 0.218]  n=  9600  event rate=0.031  woe=-0.814
(0.218, 0.301]  n=  7200  event rate=0.039  woe=-0.577
(0.301, 0.490]  n= 12000  event rate=0.059  woe=-0.142
(0.490, 0.698]  n=  9600  event rate=0.096  woe=+0.392
(0.698, 0.881]  n=  7200  event rate=0.145  woe=+0.860
(0.881, inf]  n= 16800  event rate=0.222  woe=+1.379
```

What to read here:

- `iv_bins` with `monotone="increasing"` cuts the 120,000 training rows into ten quantile bins of 12,000 rows each, merges them to eight ordered bins, and reports an IV of 1.0998. The event rate is forced to rise with the bin order. Without the argument the table below gives 1.1212 for the same column, because it applies no trend.
- The exact mode reaches an IV of 1.114 against 1.0998 for the greedy merge on the same column: a small gain, and it is guaranteed to be the best split under these constraints. It keeps eight bins, the event rate rises from 0.017 in the lowest bin to 0.222 in the highest, and `constraint_relaxed` is `False`, so every constraint was met.
- The table reads the strength labels as a rule of thumb: below 0.02 useless, below 0.1 weak, below 0.3 medium, below 0.5 strong, and above that suspicious. Four columns are above 0.5: utilization (1.1212), the 90-day count (0.8271), the 30-59 day count (0.7458) and the 60-89 day count (0.5779). Those are past-delinquency measures. A value this high is a reason to check leakage (step 4) and to confirm with the data owner that the history exists at the prediction time. `age` is medium at 0.2598, and the other five are weak.
- `NumberOfDependents` gives 0.0366 from `iv_levels`, which treats each value as its own level, and 0.0352 in the table, which bins the numeric column. The two methods differ by design.
- `binner` counts every row. Missing values get no bin and are counted in `n_missing`; they are never merged into a neighbouring bin. This column has no missing values, so its count is 0.
- Choose bins and trend on training rows. The test rows stay out of this step.

**Next:** the transform step fits scalings on training rows and applies them to both sides. Scalings and bin edges belong to the training data, so they travel with the model.

## 7. Transform numeric columns

`FeatureTransformer` fits one transform on the observed values of a column and applies it, with an exact inverse. The kinds are `log1p`, `boxcox`, `yeojohnson`, `mean_diff` and `median_diff`. A missing value passes through every kind unchanged. `log1p` needs no fit, so it is also available as a plain function. Box-Cox needs strictly positive values, so the snippet adds 1 to `MonthlyIncome` first; Yeo-Johnson accepts zero and negative values directly.

<!-- run -->
```python
from mltreelib.data import FeatureTransformer, log1p

inc = train_cols["MonthlyIncome"]
shifted = [None if v is None else v + 1.0 for v in inc]  # Box-Cox needs x > 0

def round_trip_ok(before, after):
    missing_kept = [a is None for a in before] == [b is None for b in after]
    return missing_kept and max(abs(a - b) for a, b in zip(before, after) if a is not None) < 1e-6

print("log1p first values:", log1p(inc[:3]))
for kind, values in [("boxcox", shifted), ("yeojohnson", inc), ("mean_diff", inc), ("median_diff", inc)]:
    t = FeatureTransformer(kind).fit(values)
    z = t.transform(values)
    print(f"{kind:<12} params={ {k: round(v, 4) for k, v in t.params_.items()} } "
          f"round trip ok={round_trip_ok(values, t.inverse_transform(z))}")
```

<!-- output -->
```text
log1p first values: [9.222466551774023, 8.955577146281511, None]
boxcox       params={'lambda': 0.3407} round trip ok=True
yeojohnson   params={'lambda': 0.3407} round trip ok=True
mean_diff    params={'mean': 6690.3986} round trip ok=True
median_diff  params={'median': 5400.0} round trip ok=True
```

What to read here:

- `params_` is what the fit learned: a lambda for the power transforms, a centre for the difference transforms. Fit it on the training column and reuse it on new data.
- `log1p` keeps the missing value: the first three training incomes give 9.2225, 8.9556 and a missing value.
- `boxcox` and `yeojohnson` both fit a lambda of 0.3407. For non-negative values, Yeo-Johnson equals Box-Cox of `x + 1`, so the two fits on this column must agree. The match is a consistency check, not a coincidence.
- The difference transforms centre the column on the training mean (6690.3986) or median (5400.0). They move the centre only, not the shape.
- Every round trip returns the original values, and the missing values sit in the same places. The check uses a tolerance of 1e-6, not exact equality.
- Box-Cox and log transforms have a domain. A value outside it raises `ValueError` and names the first bad row; it is not silently replaced. That is why the snippet shifts the income before Box-Cox.

**Next:** the hand-off step fits the tree on the untransformed frame from step 5. A tree separates rows by thresholds on the order of values, so a monotone transform such as a log does not change which rows a split separates. The transforms in this step matter for models that use distances or linear coefficients.

## 8. Hand the frame to the model

`xy` turns the columns into the feature rows, the target, the feature names and the dtypes that a model needs. Step 5 already built the training and test frames with `select_rows`, so this step passes them straight to the tree. Keep `dtypes` and `names` with the rows so that the column order at prediction time is the one the tree saw.

<!-- run -->
```python
from mltreelib import Metrics, ModelTree

tree = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance",
                 max_depth=3, min_rows=100, prune="none")
tree.fit(tr["x"], tr["y"], dtypes=tr["dtypes"], col_names=tr["names"])
scores = tree.predict_proba(te["x"])
roc = Metrics().roc_curve(te["y"], scores)
print("test AUC:", round(roc["auc"], 4))
print("test log loss:", round(Metrics().log_loss(te["y"], scores), 4))
```

<!-- output -->
```text
test AUC: 0.8194
test log loss: 0.1887
```

On the 30,000 test rows the tree reaches an AUC of 0.8194 and a log loss of 0.1887. The fit used the frame as it came out of step 5, with the 33,655 missing feature cells still in place. The tree routes them natively, so no imputation step is needed before this call. If a comparison model needs imputation, that step is a deliberate choice that belongs to that comparison, not to this frame.

### Choose the preparation

Models take categorical labels and missing predictors natively. The diagram shows the path from raw rows to the tree; the snippet below uses a categorical column with missing values and no separate encoder.

```mermaid
flowchart LR
    A["Raw Data Matrix\n(Python Lists / Dicts / Parquet)"] --> B["DatasetBuilder\n(NA Policy: Keep None & NaN)"]
    B --> C["Typed Frame & Schema\n(n_category Cardinality Check)"]
    C --> D["Categorical IndexMapper\n(Zero-Allocation Codes)"]
    D --> E["ModelTree Core Engine\n(Handles Missing & Categories Natively)"]

    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef buildBox fill:#0369a1,stroke:#0284c7,stroke-width:2px,color:#ffffff;
    classDef frameBox fill:#4338ca,stroke:#6366f1,stroke-width:2px,color:#ffffff;
    classDef mapBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef engineBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;

    class A dataBox;
    class B buildBox;
    class C frameBox;
    class D mapBox;
    class E engineBox;
```

<!-- run -->
```python
from mltreelib import ModelTree

x = [["north", 1.0], ["north", None], ["south", 1.0], ["south", None]] * 10
y = [1.0, 2.0, 9.0, 10.0] * 10
model = ModelTree(max_depth=2, min_rows=4, leaf="mean")
model.fit(x, y, dtypes=["categorical", "numeric"],
          col_names=["region", "amount"])
print(model.predict([["north", None], ["south", None]]))
print(model.splits())
```

<!-- output -->
```text
[1.5, 9.5]
[{'col': 'region', 'threshold': 0.0, 'missing_left': True, 'gain': 640.0}]
```

Use native categorical inputs for `ModelTree`. Use a separate encoder only when the downstream estimator or another workflow requires it, and fit it on training data only. Target-based encoders need leakage-safe training representations and an independently fitted evaluation pipeline. Missing-value behavior varies by operation. Test missing predictors, targets and unknown categories for the exact estimator before deployment.

- [Missing routing](../concepts/data-matrix-and-routing.md) explains `ModelTree` routing.
- [Categorical encoders](../concepts/data-matrix-and-routing.md) explains training leakage.
- [Data processing](../concepts/data-matrix-and-routing.md) shows typed ingestion, summaries and time features.
- [Time series forecasting](../concepts/timeseries-forecasting.md) shows prior-day features and a time split.
- [Data API](../reference/data.md) catalogs builders, encoders, transforms and the `mltreelib.data` helpers.

## What remains

The existing references do not yet give a checked missing and unseen-value matrix for every model and encoder. They also need a single end-to-end example that carries a fitted schema through deployment. The schema helpers in step 3 capture and validate a schema, but this guide stops before scoring a new batch with a fitted model. In a checked two-category example, an unseen category routed differently from a missing value. This conflicts with the stated unseen-as-missing contract and is recorded as a product gap for investigation. A checked `IndexMapper` maps an unseen label to missing. In a separate check, `OneHotEncoder` mapped missing and unseen labels to all-zero vectors, `CountEncoder` mapped them to missing and zero respectively, and `TargetEncoder` mapped them to missing and the fitted prior.
