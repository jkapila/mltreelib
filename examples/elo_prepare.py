"""Prepare the Elo arena data for regression tasks preserving raw features.

- TabArena-v0.1 regression tasks read from datasets/*.parquet
- RuleQuest demo sets (kept for continuity with earlier runs)

Data is kept RAW:
- Numeric columns remain numeric
- Categorical columns remain raw strings (NO get_dummies / one-hot encoding)
- No artificial column variance trimming (the 300s budget handles wide data)
- Missing values written as empty string ("")
- Target column written as the last column named "y"
- Outputs <ds>_{train,test}.csv and <ds>.meta.json (column roles, target, split info)

Run:
    uv run --with pandas --with pyarrow python examples/elo_prepare.py
"""

import json
import random
from pathlib import Path
import pandas as pd

DATA = Path("references/sources/cubist-demo/CubistR2/Data")
PQ = Path("datasets")
OUT = Path("benchmarks/elo_data")
SEED = 7

# TabArena-v0.1 regression tasks: our name -> (parquet name, target column)
TABARENA = {
    "ta_airfoil": ("airfoil_self_noise", "scaled-sound-pressure"),
    "ta_fiat500": ("Another-Dataset-on-used-Fiat-500", "price"),
    "ta_concrete": ("concrete_compressive_strength", "ConcreteCompressiveStrength"),
    "ta_diamonds": ("diamonds", "price"),
    "ta_food_delivery": ("Food_Delivery_Time", "Time_taken(min)"),
    "ta_healthcare": ("healthcare_insurance_expenses", "charges"),
    "ta_houses": ("houses", "LnMedianHouseValue"),
    "ta_miami": ("miami_housing", "SALE_PRC"),
    "ta_protein": ("physiochemical_protein", "ResidualSize"),
    "ta_qsar_fish": ("QSAR_fish_toxicity", "LC50"),
    "ta_wine": ("wine_quality", "median_wine_quality"),
    "ta_qsar_tid11": ("QSAR-TID-11", "MEDIAN_PXC50"),
    "ta_superconductivity": ("superconductivity", "critical_temp"),
}

# RuleQuest demo sets: target column index, dropped columns, categoricals
RULEQUEST = {
    "concrete": {"target": 8, "drop": [], "cats": {}},
    "abalone": {"target": 8, "drop": [], "cats": {0: ["I", "M", "F"]}},
    "ozone": {"target": 8, "drop": [], "cats": {}},
    "boston": {
        "target": 7,
        "drop": [0, 1, 2, 3, 4, 5, 6, 19],
        "cats": {11: ["0", "1"]},
    },
    "fc2008": {
        "target": 9,
        "drop": [1],
        "cats": {
            0: ["CAR", "STATION WAGON", "SUV", "PICKUP TRUCK", "VAN"],
            4: ["Auto", "Manual"],
            6: ["F", "R", "4"],
            7: ["R", "P", "D", "E", "C"],
            8: ["T", "S", "-"],
        },
    },
}


def split_indices(n: int, seed: int = SEED):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(0.8 * n)
    return sorted(idx[:n_tr]), sorted(idx[n_tr:])


def write_dataset(name: str, x: pd.DataFrame, y: pd.Series, roles: dict, target_name: str = "y"):
    tr, te = split_indices(len(x))
    
    # Target always written as last column named 'y'
    for split, idx in (("train", tr), ("test", te)):
        frame = x.iloc[idx].copy()
        frame["y"] = y.iloc[idx].values
        frame.to_csv(OUT / f"{name}_{split}.csv", index=False, na_rep="")
    
    na_count = int(x.isna().sum().sum())
    meta = {
        "dataset": name,
        "target": target_name,
        "task": "regression",
        "n_train": len(tr),
        "n_test": len(te),
        "split": {"kind": "random_80_20", "seed": SEED},
        "columns": roles,
        "encoding": "none (raw as-is; native categoricals)",
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    
    cat_count = sum(1 for r in roles.values() if r == "categorical")
    print(f"{name:<20} {len(tr):>6} tr / {len(te):>5} te | "
          f"{x.shape[1]:>3} cols ({cat_count} cat) | {na_count} NA")


def load_tabarena():
    for name, (pq, target) in TABARENA.items():
        pq_path = PQ / f"{pq}.parquet"
        if not pq_path.exists():
            print(f"Skipping {name}: {pq_path} not found")
            continue
        
        df = pd.read_parquet(pq_path)
        if target not in df.columns:
            print(f"Skipping {name}: target {target} not in columns")
            continue
            
        df = df.dropna(subset=[target])
        y = df[target].astype(float)
        x = df.drop(columns=[target])
        
        roles = {}
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]):
                roles[c] = "numeric"
            else:
                roles[c] = "categorical"
                # Normalize string representations
                x[c] = x[c].astype(str).replace({"nan": None, "None": None, "<NA>": None})

        write_dataset(name, x, y, roles, target_name=target)


def load_rulequest():
    for name, cfg in RULEQUEST.items():
        data_path = DATA / f"{name}.data"
        if not data_path.exists():
            print(f"Skipping RuleQuest {name}: {data_path} not found")
            continue
            
        rows, y = [], []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            y.append(float(parts[cfg["target"]]))
            
            feats = {}
            for i, p in enumerate(parts):
                if i == cfg["target"] or i in cfg["drop"]:
                    continue
                if i in cfg["cats"]:
                    # Preserve raw category string (or None if missing)
                    feats[f"c{i}"] = None if p in ("N/A", "?", "") else p
                else:
                    feats[f"c{i}"] = None if p in ("N/A", "?", "") else float(p)
            rows.append(feats)
            
        x_df = pd.DataFrame(rows)
        roles = {}
        for c in x_df.columns:
            idx = int(c[1:])
            roles[c] = "categorical" if idx in cfg["cats"] else "numeric"
            
        write_dataset(name, x_df, pd.Series(y), roles, target_name=f"c{cfg['target']}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Preparing TabArena regression datasets (raw as-is)...")
    load_tabarena()
    print("\nPreparing RuleQuest datasets (raw as-is)...")
    load_rulequest()
    print("\nRegression dataset preparation complete -> benchmarks/elo_data/")


if __name__ == "__main__":
    main()
