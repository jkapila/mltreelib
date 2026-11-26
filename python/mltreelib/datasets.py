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
import random
import urllib.request
import warnings
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
_CACHE_DIR = Path(
    os.environ.get("MLTREELIB_DATA_DIR", Path.home() / ".cache" / "mltreelib" / "datasets")
)

# SHA-256 of the published file, checked after each download (keyed by registry name).
# Rdatasets survival & uplift checksums recorded 2026-11-26.
DATASET_SHA256: dict[str, str] = {
    "veteran": "3fba5cb9b15a10ab94d54e28b54d2c39d95c27b0fcff1c57aee0eb7edfd93950",
    "pbc": "797ea9b6abfec34297ef07f361a2e0bfdd90c3c2def180adf8547ea30e75b613",
    "flchain": "a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563",
    "flchain_rdatasets": "a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563",
    "colon": "6f3472a64f696e3195daa198f054180c3e4c66408f7fb8c548c6f4c7b8f898ee",
    "rott2": "62703670d3be5d49c5476fb7131b020bdca9f1c25bb4522fd6abe8f9dab61f1b",
    "gbsg2": "9fa0fef0575d04d3273869be28405307f1b02993751498e8a0c844541cccd5d9",
    "mgus2": "5138ec8d8477f031f60e475c11b4fa6b048b1544682afb4fb675af74a5551567",
    "nsw": "3343cb4d20bd1dfd6b4940ca5610cdb2c177166658b2a070e135ec5c5c151020",
    "jobs_ii": "88d5cce12930f3d111597cf15b7c0cd3e7fa7f316f718de7024e147b9756fb9f",
    "hillstrom_email": "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
}

# Provenance, licence, and coding notes for curated survival and uplift suite datasets.
DATASET_METADATA: dict[str, dict[str, object]] = {
    "veteran": {
        "source": "R survival::veteran via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "survival",
        "coding_note": "status: 0=censored, 1=dead; time in days (N=137).",
    },
    "pbc": {
        "source": "R survival::pbc via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "competing_risks",
        "coding_note": "status: 0=censored, 1=transplant, 2=dead (event=1 maps to death; cause=status).",
    },
    "flchain": {
        "source": "R survival::flchain via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "survival",
        "coding_note": "futime in days, death 0/1; drops cause-of-death leakage column 'chapter'.",
    },
    "flchain_rdatasets": {
        "source": "R survival::flchain via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "survival",
        "coding_note": "Alias for flchain (futime in days, death 0/1; drops 'chapter').",
    },
    "colon": {
        "source": "R survival::colon via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "competing_risks",
        "coding_note": "2 rows per patient id (etype 1=recurrence, 2=death); grouped by id.",
    },
    "rott2": {
        "source": "R survival::rotterdam via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "survival",
        "coding_note": "Recurrence-free survival (rtime in days, recur 0/1; drops dtime/death).",
    },
    "gbsg2": {
        "source": "R survival::gbsg via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "survival",
        "coding_note": "Recurrence-free survival (rfstime in days, status 0/1).",
    },
    "mgus2": {
        "source": "R survival::mgus2 via Rdatasets",
        "licence": "GPL-2 | GPL-3 (R survival package)",
        "checksum_date": "2026-11-26",
        "task": "competing_risks",
        "coding_note": "Competing progression (ptime/pstat) vs death (futime/death); cause 0/1/2.",
    },
    "nsw": {
        "source": "R DAAG::nswdemo via Rdatasets (Dehejia-Wahba NSW experiment)",
        "licence": "GPL-2 | GPL-3 (R DAAG package)",
        "checksum_date": "2026-11-26",
        "task": "uplift",
        "randomized": True,
        "coding_note": "Randomized job training (trt 0/1); outcome re78 (1978 earnings).",
    },
    "jobs_ii": {
        "source": "R mediation::jobs via Rdatasets (JOBS II trial)",
        "licence": "GPL-2 | GPL-3 (R mediation package)",
        "checksum_date": "2026-11-26",
        "task": "uplift",
        "randomized": True,
        "coding_note": "Randomized assignment (treat 0/1) with non-compliance (comply 0/1); outcome depress2 or work1.",
    },
    "hillstrom_email": {
        "source": "Kevin Hillstrom MineThatData Email Analytics Challenge (2008)",
        "licence": "Public challenge dataset (MineThatData, free for research/benchmarks)",
        "checksum_date": "2026-11-26",
        "task": "uplift",
        "randomized": True,
        "coding_note": "64,000 randomized customers; treatment=(segment!='No E-Mail'); outcome visit/conversion/spend.",
    },
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
    # Survival & Competing-Risks Benchmark Datasets (Rdatasets CSV mirror, checksums in DATASET_SHA256)
    "veteran": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv",
    "pbc": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbc.csv",
    "flchain": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv",
    "colon": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv",
    "rott2": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rotterdam.csv",
    "gbsg2": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/gbsg.csv",
    "mgus2": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/mgus2.csv",
    "flchain_rdatasets": "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv",
    # Uplift Benchmark Datasets (public CSV sources, checksums in DATASET_SHA256)
    "nsw": "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nswdemo.csv",
    "jobs_ii": "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv",
    "hillstrom_email": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
}

_SUFFIXES = (".parquet", ".csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_offline() -> bool:
    """Return True when ``MLTREELIB_OFFLINE`` disables network downloads."""
    return os.environ.get("MLTREELIB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def dataset_metadata(name: str) -> dict[str, object]:
    """Return provenance, licence, and coding metadata for a registered suite dataset."""
    if name not in DATASET_METADATA:
        raise ValueError(f"no metadata registered for dataset '{name}'")
    return dict(DATASET_METADATA[name])


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
        expected = DATASET_SHA256.get(name)
        if expected is not None:
            actual = _sha256(out_path)
            if actual != expected:
                allow_mismatch = os.environ.get(
                    "MLTREELIB_ALLOW_CHECKSUM_MISMATCH", ""
                ).strip().lower() in {"1", "true", "yes"}
                if allow_mismatch:
                    warnings.warn(
                        f"sha256 mismatch for cached '{name}': expected {expected}, got {actual}; "
                        "continuing because MLTREELIB_ALLOW_CHECKSUM_MISMATCH is set.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise RuntimeError(
                        f"cached dataset '{name}' failed sha256 verification: expected {expected}, got {actual}"
                    )
        return out_path

    if is_offline():
        raise RuntimeError(
            f"cannot download dataset '{name}': offline mode is enabled (MLTREELIB_OFFLINE=1)"
        )

    if isinstance(target_spec, int):
        url = f"https://data.openml.org/datasets/0004/{target_spec}/dataset_{target_spec}.pq"
    else:
        url = str(target_spec)

    meta = DATASET_METADATA.get(name)
    if meta is not None:
        print(f"Downloading '{name}' from {url} (source: {meta['source']}; licence: {meta['licence']})")

    allow_mismatch = os.environ.get("MLTREELIB_ALLOW_CHECKSUM_MISMATCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    part_path = out_path.with_name(out_path.name + ".part")
    try:
        urllib.request.urlretrieve(url, part_path)
        expected = DATASET_SHA256.get(name)
        if expected is not None:
            actual = _sha256(part_path)
            if actual != expected:
                if allow_mismatch:
                    warnings.warn(
                        f"sha256 mismatch for '{name}' from {url}: expected {expected}, got {actual}; "
                        "continuing because MLTREELIB_ALLOW_CHECKSUM_MISMATCH is set.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"sha256 mismatch for '{name}' from {url}: expected {expected}, got {actual}"
                    )
        part_path.replace(out_path)
    except Exception as e:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download dataset '{name}' from {url}: {e}") from e

    return out_path


def _find_file(name: str, directory: Path) -> Path | None:
    aliases = (name, "flchain_rdatasets") if name == "flchain" else ((name, "flchain") if name == "flchain_rdatasets" else (name,))
    for candidate_name in aliases:
        for suffix in _SUFFIXES:
            candidate = directory / f"{candidate_name}{suffix}"
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
        if is_offline():
            raise RuntimeError(
                f"dataset '{name}' is not cached locally and offline mode is enabled (MLTREELIB_OFFLINE=1)"
            )
        return download_dataset(name)

    raise ValueError(
        f"unknown dataset '{name}'; available locally: {list_datasets()} or registered: {list_available_datasets()}"
    )


def list_datasets() -> list[str]:
    """Names of currently available local datasets (parquet or csv stems in repo or cache)."""
    found = set()
    for directory in (_DATA_DIR, _CACHE_DIR):
        if directory.exists():
            for suffix in _SUFFIXES:
                found.update(p.stem for p in directory.glob(f"*{suffix}"))
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


def _subsample_indices(n_rows: int, max_rows: int | None, seed: int) -> list[int]:
    if max_rows is None or max_rows >= n_rows:
        return list(range(n_rows))
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), max_rows))


def _take_columns(
    columns: dict[str, list],
    dtypes: dict[str, str],
    indices: list[int],
    drop: set[str],
) -> tuple[dict[str, list], dict[str, str]]:
    feat_cols: dict[str, list] = {}
    feat_dtypes: dict[str, str] = {}
    for col, values in columns.items():
        if col in drop:
            continue
        feat_cols[col] = [values[i] for i in indices]
        feat_dtypes[col] = dtypes[col]
    return feat_cols, feat_dtypes


def load_survival_dataset(
    name: str,
    *,
    auto_download: bool = True,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Load and normalise a survival or competing-risks suite dataset."""
    ds = load_dataset(name, auto_download=auto_download)
    cols: dict[str, list] = ds["columns"]
    dtypes: dict[str, str] = ds["dtypes"]
    n_total = int(ds["n_rows"])
    idx = _subsample_indices(n_total, max_rows, seed)

    cause: list[int] | None = None
    group: list | None = None
    task = "survival"

    if name in ("flchain", "flchain_rdatasets"):
        time = [float(cols["futime"][i]) for i in idx]
        event = [int(cols["death"][i]) for i in idx]
        drop = {"rownames", "futime", "death", "chapter"}
    elif name == "pbc":
        task = "competing_risks"
        time = [float(cols["time"][i]) for i in idx]
        cause = [int(cols["status"][i]) for i in idx]
        event = [1 if c == 2 else 0 for c in cause]
        drop = {"rownames", "id", "time", "status"}
    elif name == "mgus2":
        task = "competing_risks"
        time = [
            float(cols["ptime"][i]) if int(cols["pstat"][i]) == 1 else float(cols["futime"][i])
            for i in idx
        ]
        cause = [
            1 if int(cols["pstat"][i]) == 1 else (2 if int(cols["death"][i]) == 1 else 0)
            for i in idx
        ]
        event = [1 if c > 0 else 0 for c in cause]
        drop = {"rownames", "id", "ptime", "pstat", "futime", "death"}
    elif name == "gbsg2":
        time = [float(cols["rfstime"][i]) for i in idx]
        event = [int(cols["status"][i]) for i in idx]
        drop = {"rownames", "pid", "rfstime", "status"}
    elif name == "veteran":
        time = [float(cols["time"][i]) for i in idx]
        event = [int(cols["status"][i]) for i in idx]
        drop = {"rownames", "time", "status"}
    elif name == "colon":
        task = "competing_risks"
        time = [float(cols["time"][i]) for i in idx]
        event = [int(cols["status"][i]) for i in idx]
        cause = [int(cols["etype"][i]) for i in idx]
        group = [cols["id"][i] for i in idx]
        drop = {"rownames", "id", "study", "time", "status", "etype"}
    elif name == "rott2":
        time = [float(cols["rtime"][i]) for i in idx]
        event = [int(cols["recur"][i]) for i in idx]
        drop = {"rownames", "pid", "rtime", "recur", "dtime", "death"}
    else:
        raise ValueError(
            f"unknown survival suite dataset '{name}'; supported: "
            "flchain, flchain_rdatasets, pbc, mgus2, gbsg2, veteran, colon, rott2"
        )

    features, feat_dtypes = _take_columns(cols, dtypes, idx, drop)
    return {
        "name": name,
        "task": task,
        "time": time,
        "event": event,
        "cause": cause,
        "group": group,
        "features": features,
        "dtypes": feat_dtypes,
        "na_counts": {c: sum(1 for v in vals if v is None) for c, vals in features.items()},
        "n_rows": len(idx),
        "n_cols": len(features),
        "metadata": DATASET_METADATA.get(name, {}),
    }


def load_uplift_dataset(
    name: str,
    *,
    auto_download: bool = True,
    max_rows: int | None = None,
    seed: int = 42,
    outcome: str | None = None,
) -> dict[str, object]:
    """Load and normalise an uplift suite dataset with optional seeded subsampling."""
    ds = load_dataset(name, auto_download=auto_download)
    cols: dict[str, list] = ds["columns"]
    dtypes: dict[str, str] = ds["dtypes"]
    n_total = int(ds["n_rows"])
    idx = _subsample_indices(n_total, max_rows, seed)

    assignment: list[int] | None = None
    receipt: list[int] | None = None

    if name == "hillstrom_email":
        target_col = outcome or "visit"
        if target_col not in ("visit", "conversion", "spend"):
            raise ValueError(f"hillstrom_email outcome must be visit, conversion, or spend; got {target_col!r}")
        treatment = [0 if cols["segment"][i] == "No E-Mail" else 1 for i in idx]
        y_out = [float(cols[target_col][i]) for i in idx]
        drop = {"rownames", "segment", "visit", "conversion", "spend"}
    elif name == "nsw":
        target_col = outcome or "re78"
        if target_col != "re78":
            raise ValueError(f"nsw outcome must be 're78', got {target_col!r}")
        treatment = [int(cols["trt"][i]) for i in idx]
        y_out = [float(cols["re78"][i]) for i in idx]
        drop = {"rownames", "trt", "re78"}
    elif name == "jobs_ii":
        target_col = outcome or "work1"
        treatment = [int(cols["treat"][i]) for i in idx]
        assignment = list(treatment)
        receipt = [int(cols["comply"][i]) for i in idx]
        if target_col == "work1":
            y_out = [1.0 if cols["work1"][i] == "psyemp" else 0.0 for i in idx]
        elif target_col in ("depress2", "job_dich", "job_disc"):
            y_out = [float(cols[target_col][i]) for i in idx]
        else:
            raise ValueError(
                f"jobs_ii outcome must be work1, depress2, job_dich, or job_disc; got {target_col!r}"
            )
        drop = {"rownames", "treat", "comply", "control", "work1", "depress2", "job_dich", "job_disc"}
    else:
        raise ValueError(
            f"unknown uplift suite dataset '{name}'; supported: hillstrom_email, nsw, jobs_ii"
        )

    features, feat_dtypes = _take_columns(cols, dtypes, idx, drop)
    meta = DATASET_METADATA.get(name, {})
    return {
        "name": name,
        "task": "uplift",
        "randomized": bool(meta.get("randomized", True)),
        "treatment": treatment,
        "outcome": y_out,
        "assignment": assignment,
        "receipt": receipt,
        "features": features,
        "dtypes": feat_dtypes,
        "na_counts": {c: sum(1 for v in vals if v is None) for c, vals in features.items()},
        "n_rows": len(idx),
        "n_cols": len(features),
        "metadata": meta,
    }
