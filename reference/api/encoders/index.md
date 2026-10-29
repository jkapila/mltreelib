# API reference — Encoders &amp; scalers

Pipeline tools for feeding downstream algorithms (sklearn and friends) — never invoked by the tree. Guide: [Data](<https://jkapila.github.io/mltreelib/reference/data/index.md>).

## IndexMapper

`pythonclass IndexMapper(())` Categorical string \<-\> stable code mapping (sklearn-style fit/transform).

### IndexMapper.fit

`pythonfit((self, /, name, values))` Learn the vocabulary of `name` (first-seen order). Re-fitting resets it.

### IndexMapper.get\_all\_names

`pythonget_all_names((self, /, name))` All fitted levels in code order.

Raises: ValueError: column not fitted.

### IndexMapper.get\_index

`pythonget_index((self, /, name, value))` Code for one level (None when the level is unseen).

Raises: ValueError: column not fitted.

### IndexMapper.get\_name

`pythonget_name((self, /, name, index))` Level for one code.

Raises: ValueError: column not fitted or code out of range.

### IndexMapper.inverse\_transform

`pythoninverse_transform((self, /, name, codes))` Map codes back to their string levels.

Args: name: fitted column name. codes: codes per row (None = missing).

Returns: string levels (None where the code is missing).

Raises: ValueError: column not fitted or code out of range.

### IndexMapper.transform

`pythontransform((self, /, name, values))` Codes for `values`; unseen levels map to None (missing).

## VariableScaler

`pythonclass VariableScaler(())` Per-column distributional scaling — each variable transformed independently (legacy `VariableTransformation` semantics).

Same strategy names as `GroupedScaler`, applied per column, with exact `inverse_transform`. None/NaN are missing and pass through as missing; degenerate scales give missing, never inf.

### VariableScaler.fit

`pythonfit((self, /, columns, strategy))` Fit all columns at once under the given strategy.

Args: columns: {name: values} for every column to scale (None/NaN = missing; counted in each column's `n_missing`). strategy: strategy name (see `GroupedScaler`).

Raises: ValueError: unknown strategy.

### VariableScaler.inverse\_transform

`pythoninverse_transform((self, /, name, values))` Exact inverse of `transform` under the fitted statistics.

Args: name: fitted column name. values: previously transformed values.

Returns: original-scale values (up to floating-point error).

Raises: ValueError: not fitted or unknown column.

### VariableScaler.transform

`pythontransform((self, /, name, values))` Scale one fitted column's values.

Args: name: fitted column name. values: numeric values (None/NaN = missing).

Returns: scaled values; None for missing inputs and degenerate scales.

Raises: ValueError: not fitted or unknown column.

## GroupedScaler

`pythonclass GroupedScaler(())` Fit/transform group distributional scaling with exact inverse (sklearn-shaped: fit once, then transform / inverse\_transform).

Strategies: centered family — "zscore", "min-max", "median-iqr", "mean-iqr", "median-qd", "median-std", "median-range", "mean-range", "min-qd", "min-iqr"; ratio family — "min", "max", "mean", "median", "iqr", "qd", "std", "range". Missing in -\> missing out; degenerate (zero) scales give missing, never inf.

### GroupedScaler.fit

`pythonfit((self, /, keys, values, strategy))` Fit group statistics of `values` under the given strategy.

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing; counted in each group's `n_missing`). strategy: strategy name (see the class docstring).

Raises: ValueError: unknown strategy or mismatched lengths.

### GroupedScaler.inverse\_transform

`pythoninverse_transform((self, /, keys, values))` Exact inverse of `transform` under the fitted statistics.

Args: keys: group label per row. values: previously transformed values.

Returns: original-scale values (up to floating-point error).

Raises: ValueError: not fitted or mismatched lengths.

### GroupedScaler.transform

`pythontransform((self, /, keys, values))` Scale values against the fitted group statistics.

Args: keys: group label per row. values: numeric value per row.

Returns: scaled values; None for missing key/value, unseen groups and degenerate scales.

Raises: ValueError: not fitted or mismatched lengths.

## BaseNEncoder

`pythonclass BaseNEncoder((base))` **Base-N / binary** encoding: each level becomes its base-`base` digits as indicator columns (`base = 2` is the classic Binary encoder). Task: **both** (unsupervised). Unseen and missing rows are all-zero.

### BaseNEncoder.fit

`pythonfit((self, /, values))` Fit the level vocabulary on `values`.

### BaseNEncoder.transform

`pythontransform((self, /, values))` Returns {digit name: indicator list}.

## CatBoostEncoder

`pythonclass CatBoostEncoder((prior, permutations, seed))` CatBoost-style ordered target statistics — the leakage-safe flavour.

`fit_transform` gives each row the running statistic of its level using only rows earlier in a shuffled order: `(sum_y_prefix + prior * global) / (n_prefix + 1)`, averaged over `permutations` seeded passes. The same level gets different values in different rows — that row-dependence is the anti-leakage mechanism. `transform` (new data) uses prior-smoothed fitted level means.

### CatBoostEncoder.fit

`pythonfit((self, /, values, y))` Fit prior-smoothed level means (used by `transform` on new data).

### CatBoostEncoder.fit\_transform

`pythonfit_transform((self, /, values, y))` Leakage-safe ordered encodings of the training rows themselves.

### CatBoostEncoder.transform

`pythontransform((self, /, values))` Encode new rows from the fitted level means; unseen -\> global mean, missing -\> None.

## CountEncoder

`pythonclass CountEncoder((normalize))` **Count / frequency** encoding: occurrence count per category (`normalize` gives the share of rows). Task: **both** (unsupervised). Unseen levels give 0.0; missing stays missing.

### CountEncoder.fit

`pythonfit((self, /, values))` Fit counts on `values`.

### CountEncoder.transform

`pythontransform((self, /, values))` Count (or share) per row; unseen -\> 0.0, missing -\> None.

## EmbeddingEncoder

`pythonclass EmbeddingEncoder((dim, lr, epochs, seed, task))` NN-style **learned embeddings** per category: a small embedding table and a linear head trained by SGD, one table per categorical feature. Task: **both** — softmax head for classification; for regression a ReLU output when all fitted targets are non-negative, linear otherwise. `transform` returns the embeddings, `predict` the trained head. Unseen levels give zeros; missing stays missing.

### EmbeddingEncoder.fit

`pythonfit((self, /, values, y, n_classes=None))` Train on (values, y); `n_classes` is inferred for classification when omitted, ignored for regression.

### EmbeddingEncoder.predict

`pythonpredict((self, /, values))` Predictions of the trained head (class label or activated value).

### EmbeddingEncoder.transform

`pythontransform((self, /, values))` Learned embedding per row (None = missing, zeros = unseen).

## HashingEncoder

`pythonclass HashingEncoder((n_buckets))` Hashed one-hot: levels are hashed into `n_buckets` indicator columns, so unseen levels still land somewhere. Deterministic.

### HashingEncoder.transform

`pythontransform((self, /, values))` Returns {bucket name: indicator list}.

## JamesSteinEncoder

`pythonclass JamesSteinEncoder(())` **James-Stein** encoding: level means shrunk toward the grand mean with `B = max(0, 1 - ((L - 3) * sigma2) / sum((mean_i - global)^2))`; fewer than three levels collapse to the grand mean. Task: **both**. Unseen -\> global; missing -\> None.

### JamesSteinEncoder.fit

`pythonfit((self, /, values, y))` Fit shrunk means on (values, y).

### JamesSteinEncoder.transform

`pythontransform((self, /, values))` Shrunk mean per row; unseen -\> global, missing -\> None.

## LeaveOneOutEncoder

`pythonclass LeaveOneOutEncoder(())` **Leave-one-out** encoding: `fit_transform` gives each row its level's mean target **without that row** (leakage-aware); `transform` on new data uses the full fitted level mean. Task: **both**. Unseen -\> global mean; missing -\> None.

### LeaveOneOutEncoder.fit

`pythonfit((self, /, values, y))` Fit level means on (values, y).

### LeaveOneOutEncoder.fit\_transform

`pythonfit_transform((self, /, values, y))` Row-level means excluding each row's own target.

### LeaveOneOutEncoder.transform

`pythontransform((self, /, values))` Full fitted level means for new rows.

## MEstimatorEncoder

`pythonclass MEstimatorEncoder((m))` **M-estimate** encoding: `(sum_y + m * global) / (n_level + m)` — target smoothing with fixed strength `m`. Task: **both** (leaky on training data; prefer `CatBoostEncoder` or `LeaveOneOutEncoder` there). Unseen -\> global; missing -\> None.

### MEstimatorEncoder.fit

`pythonfit((self, /, values, y))` Fit smoothed level means on (values, y).

### MEstimatorEncoder.transform

`pythontransform((self, /, values))` Smoothed mean per row; unseen -\> global, missing -\> None.

## OneHotEncoder

`pythonclass OneHotEncoder(())` One-hot expansion: one indicator column per fitted level (first-seen order). Missing values and levels unseen at fit produce an all-zero row.

### OneHotEncoder.fit

`pythonfit((self, /, values))` Learn the levels (first-seen order). Re-fitting resets them.

### OneHotEncoder.levels

`pythonlevels((self, /))` Fitted levels, in column order.

### OneHotEncoder.transform

`pythontransform((self, /, values))` Returns {level: indicator list}; unseen/missing rows are all-zero.

## QuantileEncoder

`pythonclass QuantileEncoder((q, smoothing))` **Quantile** encoding: each level becomes its smoothed target quantile (default median), `w * level_quantile + (1 - w) * global_quantile` with `w = n / (n + smoothing)`. Task: **regression**. Unseen -\> global quantile; missing -\> None.

### QuantileEncoder.fit

`pythonfit((self, /, values, y))` Fit quantiles on (values, y).

### QuantileEncoder.transform

`pythontransform((self, /, values))` Smoothed quantile per row; unseen -\> global, missing -\> None.

## SumEncoder

`pythonclass SumEncoder(())` **Sum contrast** coding: `k-1` columns; level `i < k-1` gets +1 in column `i`, the last level gets -1 across all columns. Task: **both** (unsupervised). Unseen rows give zeros; missing stays missing.

### SumEncoder.fit

`pythonfit((self, /, values))` Fit the level vocabulary on `values`.

### SumEncoder.transform

`pythontransform((self, /, values))` Contrast row per row (None = missing).

## TargetEncoder

`pythonclass TargetEncoder((smoothing))` Smoothed target encoding (fitted-mean flavour).

`w * level_mean + (1 - w) * global_mean` with `w = n / (n + smoothing)`. **Leaky on training data** (the row's own target is in its level mean) — prefer `CatBoostEncoder.fit_transform` for training features. Unseen levels fall back to the global mean; missing stays missing.

### TargetEncoder.fit

`pythonfit((self, /, values, y))` Fit level means on (values, y).

### TargetEncoder.prior

`pythonprior((self, /))` The global mean used as the unseen-level fallback.

### TargetEncoder.transform

`pythontransform((self, /, values))` Encode values; unseen -\> global mean, missing -\> None.

## TfidfEncoder

`pythonclass TfidfEncoder((normalize))` Class-level **TF-IDF**: each category becomes a vector over the target classes — `tf(v,j) = count(v,j)/count(v)` times `idf(j) = ln(1 + n/(1 + count(j))) + 1`; `normalize` scales each level vector to unit L2 norm. Task: **classification**. Unseen levels fall back to the global class shares; missing stays missing.

### TfidfEncoder.fit

`pythonfit((self, /, values, y, n_classes))` Fit class counts on (values, y, n\_classes).

### TfidfEncoder.transform

`pythontransform((self, /, values))` One class-space vector per row (None = missing).

## WoEEncoder

`pythonclass WoEEncoder((prior))` **Weight of Evidence** (credit-risk classic): one-vs-rest `ln(distr_j(v) / distr_not_j(v))` per category, smoothed with additive `prior`. Task: **classification**. Positive = over-representation of the class in that level. Unseen levels give neutral zeros; missing stays missing.

### WoEEncoder.fit

`pythonfit((self, /, values, y, n_classes))` Fit class counts on (values, y, n\_classes).

### WoEEncoder.transform

`pythontransform((self, /, values))` One WoE vector per row (None = missing).
