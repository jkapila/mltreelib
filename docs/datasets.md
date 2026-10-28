# Benchmark datasets

`datasets/` holds the full **TabArena-v0.1** suite: 51 curated tabular datasets
(87.4 MB, OpenML suite 457) — 7 regression, 44 classification, 4 small
(<1k rows), 40 medium, 7 large (up to 150k rows), including high-cardinality
categorical sets. Files are parquet with categoricals preserved as strings;
`datasets/MANIFEST.md` lists per-dataset target, size, license and source.

```python
from mltreelib import list_datasets, load_dataset

list_datasets()                 # ['APSFailure', 'Bank_Customer_Churn', ...]
ds = load_dataset("churn")
ds["n_rows"], ds["n_cols"]      # 5000, 20
ds["dtypes"]                    # {"state": "categorical", "account_length": "numeric", ...}
ds["na_counts"]                 # missing shown by default
```

`load_dataset` returns the same shape as `DatasetBuilder.build()`, so it feeds
straight back into the data layer.

!!! info "BeyondArena"
    The beyond-IID extension (temporal/grouped splits) ships via the separate
    `data-foundry` package and is intentionally not vendored; see the manifest
    for notes.
