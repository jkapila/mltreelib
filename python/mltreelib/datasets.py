"""Benchmark dataset loaders with local repository fallback and on-demand remote caching.

The suite includes TabArena-v0.1 (OpenML suite 457) and the SurvSet survival benchmarks.
Public CSV sources (FLCHAIN from Rdatasets, Hillstrom email uplift) are registered by URL.
Datasets are NOT bundled into package wheels; they are loaded from local repository
storage (`datasets/`) or downloaded on demand into `~/.cache/mltreelib/datasets/`
(override with `MLTREELIB_DATA_DIR`).
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
_CACHE_DIR = Path(
    os.environ.get("MLTREELIB_DATA_DIR", Path.home() / ".cache" / "mltreelib" / "datasets")
)

# SHA-256 of the published file, checked after each download (keyed by registry name).
DATASET_SHA256: dict[str, str] = {
    "flchain_rdatasets": "a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563",
    "hillstrom_email": "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
}

# OpenML ID (int) or direct file URL (str) registry for on-demand downloads
DATASET_REGISTRY: dict[str, int | str] = {
    # TabArena-v0.1 Curated Benchmarks
    "airfoil_self_noise": 46904,
    "Amazon_employee_access": 46905,
    "anneal": 46906,
    "Another-Dataset-on-used-Fiat-500": 46907,
    "APSFailure": 46908,
    "bank-marketing": 46910,
    "Bank_Customer_Churn": 46911,
    "Bioresponse": 46912,
    "blood-transfusion-service-center": 46913,
    "churn": 46915,
    "coil2000_insurance_policies": 46916,
    "concrete_compressive_strength": 46917,
    "credit-g": 46918,
    "credit_card_clients_default": 46919,
    "customer_satisfaction_in_airline": 46920,
    "diabetes": 46921,
    "Diabetes130US": 46922,
    "diamonds": 46923,
    "E-CommereShippingData": 46924,
    "Fitness_Club": 46927,
    "Food_Delivery_Time": 46928,
    "GiveMeSomeCredit": 46929,
    "hazelnut-spread-contaminant-detection": 46930,
    "healthcare_insurance_expenses": 46931,
    "heloc": 46932,
    "hiva_agnostic": 46933,
    "houses": 46934,
    "HR_Analytics_Job_Change_of_Data_Scientists": 46935,
    "in_vehicle_coupon_recommendation": 46937,
    "Is-this-a-good-customer": 46938,
    "jm1": 46979,
    "kddcup09_appetency": 46939,
    "Marketing_Campaign": 46940,
    "maternal_health_risk": 46941,
    "miami_housing": 46942,
    "MIC": 46980,
    "NATICUSdroid": 46969,
    "online_shoppers_intention": 46947,
    "physiochemical_protein": 46949,
    "polish_companies_bankruptcy": 46950,
    "qsar-biodeg": 46952,
    "QSAR-TID-11": 46953,
    "QSAR_fish_toxicity": 46954,
    "SDSS17": 46955,
    "seismic-bumps": 46956,
    "splice": 46958,
    "students_dropout_and_academic_success": 46960,
    "superconductivity": 46961,
    "taiwanese_bankruptcy_prediction": 46962,
    "website_phishing": 46963,
    "wine_quality": 46964,
    # SurvSet Benchmark Datasets
    "veteran": 43840,
    "pbc": 44101,
    "flchain": 44103,
    "colon": 44104,
    "rott2": 44105,
    "support": 44106,
    "gbsg2": 44107,
    # Public CSV sources (no Parquet offered); checksums in DATASET_SHA256
    "flchain_rdatasets": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv",
    "hillstrom_email": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
}

_SUFFIXES = (".parquet", ".csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(name: str, target_dir: Path | None = None) -> Path:
    """Download a dataset on demand into the cache directory.

    The file is written to ``<name><suffix>.part`` and renamed only after the
    download (and the SHA-256 check, when one is registered) succeeds.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"unknown dataset '{name}' for download; registered: {sorted(DATASET_REGISTRY.keys())}")

    out_dir = target_dir or _CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target_spec = DATASET_REGISTRY[name]
    suffix = ".csv" if isinstance(target_spec, str) and target_spec.endswith(".csv") else ".parquet"
    out_path = out_dir / f"{name}{suffix}"
    if out_path.exists():
        return out_path

    if isinstance(target_spec, int):
        url = f"https://data.openml.org/datasets/0004/{target_spec}/dataset_{target_spec}.pq"
    else:
        url = str(target_spec)

    part_path = out_path.with_name(out_path.name + ".part")
    try:
        urllib.request.urlretrieve(url, part_path)
        expected = DATASET_SHA256.get(name)
        if expected is not None:
            actual = _sha256(part_path)
            if actual != expected:
                raise ValueError(f"sha256 mismatch: expected {expected}, got {actual}")
        part_path.replace(out_path)
    except Exception as e:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download dataset '{name}' from {url}: {e}") from e

    return out_path


def _find_file(name: str, directory: Path) -> Path | None:
    for suffix in _SUFFIXES:
        candidate = directory / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def dataset_path(name: str, auto_download: bool = True) -> Path:
    """Locate a dataset file (Parquet or CSV), checking local repo, cache, or downloading."""
    # 1. Local repository datasets/ directory
    local_path = _find_file(name, _DATA_DIR)
    if local_path is not None:
        return local_path

    # 2. Local cache directory (~/.cache/mltreelib/datasets/)
    cache_path = _find_file(name, _CACHE_DIR)
    if cache_path is not None:
        return cache_path

    # 3. On-demand download if registered
    if auto_download and name in DATASET_REGISTRY:
        return download_dataset(name)
        
    raise ValueError(
        f"unknown dataset '{name}'; available locally: {list_datasets()} or registered: {list_available_datasets()}"
    )


def list_datasets() -> list[str]:
    """Names of currently available local datasets (parquet stems in repo or cache)."""
    found = set()
    if _DATA_DIR.exists():
        found.update(p.stem for p in _DATA_DIR.glob("*.parquet"))
    if _CACHE_DIR.exists():
        found.update(p.stem for p in _CACHE_DIR.glob("*.parquet"))
    return sorted(found)


def list_available_datasets() -> list[str]:
    """Names of all available benchmark datasets (local or on-demand downloadable)."""
    names = set(list_datasets())
    names.update(DATASET_REGISTRY.keys())
    return sorted(names)


def load_dataset(name: str, auto_download: bool = True) -> dict:
    """Load one dataset as ``name``/``columns``/``dtypes``/``n_rows``/``n_cols``.

    ``columns`` maps column name to a list with ``None`` for missing values;
    ``dtypes`` is "numeric" or "categorical" (strings/dictionaries).
    """
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "mltreelib.load_dataset requires 'pyarrow' (the 'dataset' optional extra). "
            "Install it with: pip install 'mltreelib[dataset]' or pip install pyarrow"
        ) from e

    path = dataset_path(name, auto_download=auto_download)
    if path.suffix == ".csv":
        table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(strings_can_be_null=True))
    else:
        table = pq.read_table(path)
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
                None if v is None or (isinstance(v, float) and math.isnan(v)) else v
                for v in values
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
