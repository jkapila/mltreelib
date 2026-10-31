"""Shared infrastructure for the Elo benchmarking framework.

Provides:
- Standardized file+console logging
- Dataset discovery and ELO_DATASETS filtering
- Metadata loading (<ds>.meta.json)
- Raw dataset reader (read_xy, read_split) preserving numbers and strings
- ScoreSheet: upserting scores and tracking completed (dataset, model) pairs
- AttemptsLedger: recording execution status and timing (ok, error, timeout)
- Canonical Elo rating calculation
- Standardized time budget tiers
"""

import csv
import itertools
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Standardized time tiers across the field
SLOW_S = 30.0        # TabArena-toppers line warning
CRITICAL_S = 180.0    # Critical slow warning
BUDGET_S = 300.0      # Hard kill deadline


def setup_logging(name: str, log_file: str = "benchmarks/elo_run.log") -> logging.Logger:
    """Setup standard file + console logging handler."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S")
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def discover_datasets(data_dir: Path, pattern: str = "*_train.csv") -> List[str]:
    """Discover dataset stems from data_dir, filtered by ELO_DATASETS if set."""
    ds = sorted(p.name[: -len(pattern[1:])] for p in data_dir.glob(pattern))
    if os.environ.get("ELO_DATASETS"):
        allowed = {d.strip() for d in os.environ["ELO_DATASETS"].split(",") if d.strip()}
        ds = [d for d in ds if d in allowed]
    return ds


def load_meta(data_dir: Path, ds: str) -> Optional[Dict[str, Any]]:
    """Load <ds>.meta.json if present."""
    meta_path = data_dir / f"{ds}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


def read_xy(path: Path, meta: Optional[Dict[str, Any]] = None, is_classification: bool = False) -> Tuple[List[List[Any]], List[Any]]:
    """Read raw feature rows and target list from CSV.
    
    Numeric features -> float or None (if missing).
    Categorical features -> str (empty string '' if missing).
    Target -> float (regression) or str (classification).
    Target is always the last column.
    """
    roles = None
    if meta and "columns" in meta:
        roles = list(meta["columns"].values())

    x, y = [], []
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return x, y
        width = len(header)
        for row in reader:
            if not row:
                continue
            if len(row) != width:
                raise ValueError(f"{path}: row width {len(row)} != header width {width}")
            
            cells = []
            for j, val in enumerate(row[:-1]):
                role = roles[j] if roles and j < len(roles) else None
                if role == "categorical":
                    cells.append(val)
                elif role == "numeric":
                    cells.append(None if val in ("", "N/A", "?", "nan", "NaN") else float(val))
                else:
                    # Inferred role: try float, fallback to str
                    if val in ("", "N/A", "?", "nan", "NaN"):
                        cells.append(None)
                    else:
                        try:
                            cells.append(float(val))
                        except ValueError:
                            cells.append(val)
            
            target_val = row[-1]
            if is_classification:
                y.append(target_val)
            else:
                y.append(float(target_val))
            x.append(cells)
            
    return x, y


def read_split(data_dir: Path, ds: str, is_classification: bool = False) -> Tuple[List[List[Any]], List[Any], List[List[Any]], List[Any], Optional[Dict[str, Any]]]:
    """Load train and test splits for a dataset."""
    meta = load_meta(data_dir, ds)
    xtr, ytr = read_xy(data_dir / f"{ds}_train.csv", meta, is_classification=is_classification)
    xte, yte = read_xy(data_dir / f"{ds}_test.csv", meta, is_classification=is_classification)
    return xtr, ytr, xte, yte, meta


# Language / ecosystem mapping across all models:
MODEL_LANG: Dict[str, str] = {
    # Rust core & Rust wheels
    "ModelTree(all)": "Rust",
    "ModelTree(path)": "Rust",
    "ModelTree(mean)": "Rust",
    "ModelTree(winnow)": "Rust",
    "forust (Rust)": "Rust",
    "perpetual (Rust)": "Rust",

    # R models
    "rpart": "R",
    "cubist": "R",
    "lmtree": "R",
    "glmtree": "R",
    "ctree": "R",
    "earth": "R",
    "mboost": "R",
    "ranger": "R",
    "bart": "R",
    "fru": "R",
    "grf_rf": "R",
    "grf_llf": "R",
    "c50_tree": "R",
    "c50_rules": "R",
    "C5.0 (tree)": "R",
    "C5.0 (rules)": "R",

    # Python models
    "LightGBM(200)": "Python",
    "CatBoost(200)": "Python",
    "CTBoost": "Python",
    "ChimeraBoost": "Python",
    "EBM": "Python",
    "HistGradientBoosting": "Python",
    "sklearn GBM": "Python",
    "ExtraTrees": "Python",
    "sklearn RF": "Python",
    "sklearn tree": "Python",
    "sklearn Linear": "Python",
    "sklearn LogReg": "Python",
    "KNN": "Python",
    "MARS (mars-earth)": "Python",
    "trust-free": "Python",
    "statsmodels OLS": "Python",
    "statsmodels RLM": "Python",
    "statsmodels QuantReg": "Python",
    "statsmodels MNLogit": "Python",
    "modeltrees lmtree": "Python",
    "modeltrees glmtree": "Python",
    "modeltrees multinomtree": "Python",
}


def get_model_lang(model_name: str) -> str:
    """Return 'Rust', 'Python', or 'R' for a model."""
    if model_name in MODEL_LANG:
        return MODEL_LANG[model_name]
    if "Rust" in model_name or model_name.startswith("ModelTree"):
        return "Rust"
    if model_name.startswith("sklearn") or model_name.startswith("statsmodels") or model_name.startswith("modeltrees"):
        return "Python"
    return "Python"


# Category handling metadata across all models in the Elo arena:
# - 'internal': native categorical handling (as-is; e.g. ModelTree, LightGBM, CatBoost, EBM, R factors)
# - 'external': driver-side ordinal encoding (e.g. sklearn, forust, perpetual)
# - 'none': numeric-only, refuses categorical columns (e.g. statsmodels, MARS, trust-free, grf)
MODEL_CAT_HANDLING: Dict[str, str] = {
    # 1. Native / Internal
    "ModelTree(all)": "internal",
    "ModelTree(path)": "internal",
    "ModelTree(mean)": "internal",
    "ModelTree(winnow)": "internal",
    "LightGBM(200)": "internal",
    "CatBoost(200)": "internal",
    "CTBoost": "internal",
    "ChimeraBoost": "internal",
    "EBM": "internal",
    "HistGradientBoosting": "internal",
    "modeltrees lmtree": "internal",
    "modeltrees glmtree": "internal",
    "modeltrees multinomtree": "internal",
    "rpart": "internal",
    "cubist": "internal",
    "c50_tree": "internal",
    "c50_rules": "internal",
    "C5.0 (tree)": "internal",
    "C5.0 (rules)": "internal",
    "lmtree": "internal",
    "glmtree": "internal",
    "ctree": "internal",
    "earth": "internal",
    "mboost": "internal",
    "ranger": "internal",
    "bart": "internal",
    "fru": "internal",

    # 2. External (driver-side ordinal encoding)
    "sklearn GBM": "external",
    "sklearn RF": "external",
    "sklearn tree": "external",
    "ExtraTrees": "external",
    "sklearn Linear": "external",
    "sklearn LogReg": "external",
    "KNN": "external",
    "forust (Rust)": "external",
    "perpetual (Rust)": "external",
    "MARS (mars-earth)": "external",

    # 3. None (numeric-only, refuses categoricals)
    "trust-free": "none",
    "statsmodels OLS": "none",
    "statsmodels RLM": "none",
    "statsmodels QuantReg": "none",
    "statsmodels MNLogit": "none",
    "grf_rf": "none",
    "grf_llf": "none",
}


def get_cat_handling(model_name: str) -> str:
    """Return 'internal', 'external', or 'none' for category handling."""
    if model_name in MODEL_CAT_HANDLING:
        return MODEL_CAT_HANDLING[model_name]
    for prefix, handling in [
        ("ModelTree", "internal"),
        ("CatBoost", "internal"),
        ("LightGBM", "internal"),
        ("sklearn", "external"),
        ("statsmodels", "none"),
        ("forust", "external"),
        ("perpetual", "external"),
    ]:
        if model_name.startswith(prefix):
            return handling
    return "unknown"


class ScoreSheet:
    """Manages reading and upserting benchmark results in a CSV file."""

    def __init__(self, path: Path, columns: List[str]):
        self.path = Path(path)
        self.columns = columns
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(self.columns)
        else:
            with open(self.path, "r", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                header = next(rdr, None)
            if header and header != self.columns:
                # Upgrade existing CSV header and preserve rows
                with open(self.path, "r", encoding="utf-8") as fh:
                    dict_rdr = csv.DictReader(fh)
                    existing_rows = list(dict_rdr)
                tmp_path = self.path.with_suffix(".tmp")
                with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=self.columns)
                    w.writeheader()
                    for r in existing_rows:
                        if "lang" in self.columns and not r.get("lang"):
                            r["lang"] = get_model_lang(r.get("model", ""))
                        if "cat_handling" in self.columns and not r.get("cat_handling"):
                            r["cat_handling"] = get_cat_handling(r.get("model", ""))
                        row_dict = {col: r.get(col, "") for col in self.columns}
                        w.writerow(row_dict)
                tmp_path.replace(self.path)

    def done_pairs(self) -> Set[Tuple[str, str]]:
        if not self.path.exists():
            return set()
        with open(self.path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return {(r["dataset"], r["model"]) for r in reader if "dataset" in r and "model" in r}

    def upsert(self, ds: str, model: str, **metrics):
        """Upsert a single (dataset, model) row."""
        rows = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                rows = [r for r in rdr if not (r.get("dataset") == ds and r.get("model") == model)]
        
        new_row = {"dataset": ds, "model": model}
        if "lang" in self.columns:
            new_row["lang"] = metrics.get("lang") or get_model_lang(model)
        if "cat_handling" in self.columns:
            new_row["cat_handling"] = metrics.get("cat_handling") or get_cat_handling(model)

        for k, v in metrics.items():
            if k in self.columns:
                if isinstance(v, float):
                    new_row[k] = f"{v:.4f}"
                else:
                    new_row[k] = str(v)
        
        # Ensure all columns present
        for col in self.columns:
            new_row.setdefault(col, "")

        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns)
            w.writeheader()
            for r in rows:
                if "lang" in self.columns and not r.get("lang"):
                    r["lang"] = get_model_lang(r.get("model", ""))
                if "cat_handling" in self.columns and not r.get("cat_handling"):
                    r["cat_handling"] = get_cat_handling(r.get("model", ""))
                row_dict = {col: r.get(col, "") for col in self.columns}
                w.writerow(row_dict)
            w.writerow(new_row)
        tmp_path.replace(self.path)


class AttemptsLedger:
    """Tracks attempts (ok, error, timeout) in benchmarks/elo_attempts.csv."""

    def __init__(self, path: Path = Path("benchmarks/elo_attempts.csv")):
        self.path = Path(path)
        self.columns = ["dataset", "model", "status", "seconds"]
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(self.columns)

    def attempted_pairs(self) -> Set[Tuple[str, str]]:
        if not self.path.exists():
            return set()
        with open(self.path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return {(r["dataset"], r["model"]) for r in reader if "dataset" in r and "model" in r}

    def record(self, ds: str, model: str, status: str, seconds: float):
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([ds, model, status, f"{seconds:.1f}"])


def regression_scores(pred: List[float], y_true: List[float]) -> Tuple[float, float]:
    """Compute RMSE and MAE ignoring non-finite values."""
    pairs = [(float(p), float(t)) for p, t in zip(pred, y_true) if p == p and not math.isinf(p)]
    if not pairs:
        raise RuntimeError("No finite predictions to compute scores")
    err = [p - t for p, t in pairs]
    rmse = math.sqrt(sum(e * e for e in err) / len(err))
    mae = sum(abs(e) for e in err) / len(err)
    return rmse, mae


def calculate_elo(
    rows: List[Dict[str, Any]],
    metric_col: str,
    lower_is_better: bool = True,
    k: float = 32.0,
    start_rating: float = 1000.0,
) -> Dict[str, float]:
    """Canonical Elo calculation: round-robin pairwise matchups per dataset."""
    by_ds_model = {}
    for r in rows:
        ds, m = r["dataset"], r["model"]
        val = r.get(metric_col)
        if val is not None and val != "":
            try:
                by_ds_model[(ds, m)] = float(val)
            except ValueError:
                pass

    models = sorted({m for (_, m) in by_ds_model})
    ratings = {m: start_rating for m in models}
    datasets = sorted({ds for (ds, _) in by_ds_model})

    for ds in datasets:
        here = {m: by_ds_model[(ds, m)] for m in models if (ds, m) in by_ds_model}
        for a, b in itertools.combinations(sorted(here), 2):
            va, vb = here[a], here[b]
            if lower_is_better:
                sa = 1.0 if va < vb else (0.0 if va > vb else 0.5)
            else:
                sa = 1.0 if va > vb else (0.0 if va < vb else 0.5)
            
            ea = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (sa - ea)
            ratings[b] += k * ((1.0 - sa) - (1.0 - ea))

    return ratings
