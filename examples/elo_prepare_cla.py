"""Prepare the classification Elo arena: 12 TabArena-v0.1 tasks, data AS-IS.

Selection (2026-10-31, this iteration's round): TabArena-v0.1 tasks ONLY
(no RuleQuest/demo sets), 8 binary + 4 multiclass,
spanning tiny->large rows, narrow->wide, balanced->heavily imbalanced, and
numeric/categorical-heavy mixes:

    credit-g, diabetes, blood-transfusion-service-center, heloc,
    Bank_Customer_Churn, online_shoppers_intention, jm1, bank-marketing,
    maternal_health_risk, students_dropout_and_academic_success, SDSS17,
    splice

Target columns are parsed from datasets/MANIFEST.md (single source of truth),
never hard-coded. Split: stratified 80/20 within class, seed 7.

Recipe (user-confirmed 2026-10-31): NO feature encoding here. Categoricals
stay strings, NA stays empty. Per-driver adaptation lives in the workers:
sklearn encodes (ordinal, fit on train only), R factors and
LightGBM/CatBoost/XGBoost/ModelTree consume raw.

Writes benchmarks/elo_cla_data/<ds>_{train,test}.csv (target LAST column) and
<ds>.meta.json (target, task, canonical class order, column roles).

Run:
    uv run --with pandas --with pyarrow python examples/elo_prepare_cla.py
"""

import json
import random
import re
from pathlib import Path

PQ = Path("datasets")
OUT = Path("benchmarks/elo_cla_data")
MANIFEST = PQ / "MANIFEST.md"
SEED = 7

# name -> task kind (kept explicit so the selection is reviewable in one place)
CL_TASKS = {
    # Original 8 binary datasets
    "credit-g": "binary",
    "diabetes": "binary",
    "blood-transfusion-service-center": "binary",
    "heloc": "binary",
    "Bank_Customer_Churn": "binary",
    "online_shoppers_intention": "binary",
    "jm1": "binary",
    "bank-marketing": "binary",
    # Additional 12 TabArena binary datasets
    "churn": "binary",
    "qsar-biodeg": "binary",
    "seismic-bumps": "binary",
    "Marketing_Campaign": "binary",
    "Is-this-a-good-customer": "binary",
    "Fitness_Club": "binary",
    "hazelnut-spread-contaminant-detection": "binary",
    "E-CommereShippingData": "binary",
    "taiwanese_bankruptcy_prediction": "binary",
    "NATICUSdroid": "binary",
    "coil2000_insurance_policies": "binary",
    "in_vehicle_coupon_recommendation": "binary",
    # Multiclass datasets
    "maternal_health_risk": "multiclass",
    "students_dropout_and_academic_success": "multiclass",
    "SDSS17": "multiclass",
    "splice": "multiclass",
}


def manifest_targets():
    """name -> (target column, task kind) from the MANIFEST table rows."""
    out = {}
    pat = re.compile(r"^\|\s*([^|]+?)\s*\|\s*\d+\s*\|\s*classification \((binary|multiclass)\)\s*\|\s*`([^`]+)`\s*\|")
    for line in MANIFEST.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = (m.group(3), m.group(2))
    return out


def stratified_indices(labels, seed=SEED):
    """80/20 within each class, deterministic shuffle per class."""
    by_class = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)
    rng = random.Random(seed)
    tr, te = [], []
    for lab in sorted(by_class, key=str):
        idx = by_class[lab][:]
        rng.shuffle(idx)
        cut = max(1, round(0.8 * len(idx)))
        tr.extend(idx[:cut])
        te.extend(idx[cut:])
    return sorted(tr), sorted(te)


def main():
    import pandas as pd

    targets = manifest_targets()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, task in CL_TASKS.items():
        if name not in targets:
            raise SystemExit(f"{name}: no classification row in {MANIFEST} — manifest is the source of truth, not guesses")
        target, manifest_task = targets[name]
        if manifest_task != task:
            raise SystemExit(f"{name}: task kind mismatch (manifest={manifest_task}, selection={task})")
        pq = PQ / f"{name}.parquet"
        if not pq.exists():
            raise SystemExit(f"{name}: {pq} missing — re-download from the MANIFEST OpenML id")
        df = pd.read_parquet(pq)
        if target not in df.columns:
            raise SystemExit(f"{name}: target {target!r} not in columns {list(df.columns)[:8]}…")
        before = len(df)
        df = df.dropna(subset=[target])
        y = df[target].astype(str)
        x = df.drop(columns=[target])
        # column roles by dtype: strings = categorical, numbers = numeric; data AS-IS
        roles = {c: ("categorical" if not pd.api.types.is_numeric_dtype(x[c]) else "numeric") for c in x.columns}
        tr, te = stratified_indices(y.tolist())
        xtr, ytr = x.iloc[tr], y.iloc[tr]
        xte, yte = x.iloc[te], y.iloc[te]
        for split, xs, ys in (("train", xtr, ytr), ("test", xte, yte)):
            frame = xs.copy()
            frame[target] = ys.values  # target LAST: matches the worker row contract
            frame.to_csv(OUT / f"{name}_{split}.csv", index=False, na_rep="")
        classes = sorted(y.unique().tolist(), key=str)
        meta = {
            "dataset": name,
            "target": target,
            "task": task,
            "classes": classes,
            "class_counts": {c: int((y == c).sum()) for c in classes},
            "n_train": len(tr),
            "n_test": len(te),
            "n_rows_dropped_na_target": before - len(df),
            "split": {"kind": "stratified_80_20", "seed": SEED},
            "columns": roles,
            "encoding": "none (raw as-is; sklearn-only ordinal encoding lives in the driver, fit on train)",
        }
        (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
        rows.append((name, task, len(classes), len(df), x.shape[1], sum(1 for v in roles.values() if v == "categorical")))
    print(f"{len(rows)} classification tasks written to {OUT}/ (target last, raw as-is)")
    print(f"{'dataset':<44} {'task':<11} {'K':>2} {'rows':>7} {'cols':>4} {'cat':>4}")
    for name, task, k, n, p, nc in rows:
        print(f"{name:<44} {task:<11} {k:>2} {n:>7} {p:>4} {nc:>4}")


if __name__ == "__main__":
    main()
