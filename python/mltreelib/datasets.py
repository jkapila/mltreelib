"""Benchmark dataset loaders for the parquet suite in ``datasets/``.

The suite is TabArena-v0.1 (OpenML suite 457); see ``datasets/MANIFEST.md`` for
per-dataset metadata and re-download URLs. Parquet reading uses pyarrow —
an established core dependency, deliberately not hand-rolled.

`load_dataset` returns the same shape as `DatasetBuilder.build()`, so results
feed straight back into the data layer.
"""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


def dataset_path(name: str) -> Path:
    """Path of a dataset parquet file, validating the name."""
    path = _DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise ValueError(
            f"unknown dataset '{name}'; available: {list_datasets() or '(none — see datasets/MANIFEST.md)'}"
        )
    return path


def list_datasets() -> list[str]:
    """Names of the available benchmark datasets (parquet stems)."""
    return sorted(p.stem for p in _DATA_DIR.glob("*.parquet"))


def load_dataset(name: str) -> dict:
    """Load one dataset as ``name``/``columns``/``dtypes``/``n_rows``/``n_cols``.

    ``columns`` maps column name to a list with ``None`` for missing values;
    ``dtypes`` is "numeric" or "categorical" (strings/dictionaries).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path(name))
    columns: dict[str, list] = {}
    dtypes: dict[str, str] = {}
    for field in table.schema:
        values = table.column(field.name).to_pylist()  # None = missing
        if (
            pa.types.is_dictionary(field.type)
            or pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
        ):
            columns[field.name] = [None if v is None else str(v) for v in values]
            dtypes[field.name] = "categorical"
        else:
            # NaN is missing too (pandas/parquet float nulls)
            columns[field.name] = [
                None if v is None or (isinstance(v, float) and v != v) else v for v in values
            ]
            dtypes[field.name] = "numeric"
    return {
        "name": name,
        "columns": columns,
        "dtypes": dtypes,
        "na_counts": {
            col: sum(1 for v in values if v is None) for col, values in columns.items()
        },
        "n_rows": table.num_rows,
        "n_cols": table.num_columns,
    }
