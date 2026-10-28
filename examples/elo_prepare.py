"""Prepare the Elo arena data ONCE for every regression task:

- TabArena-v0.1's official regression tasks (13 in the MANIFEST; 11 run now),
  read from the local validated parquets in datasets/;
- the five RuleQuest demo sets (kept for continuity with earlier runs).

One 80/20 split each (seed 7), identical full one-hot encoding for every
scorer. Writes benchmarks/elo_data/<ds>_{train,test}.csv; scorers discover
datasets from those files, so adding data never touches the scorers.

Deferred (documented): superconductivity (82 cols) and QSAR-TID-11 (1025
cols) exceed the exact linear-leaf scan's cost today - they are the stress
cases for the preselect/binned-candidate phase.

Run:
    uv run --with pandas --with pyarrow python examples/elo_prepare.py
"""

import random
from pathlib import Path

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


def split_indices(n):
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    n_tr = int(0.8 * n)
    return sorted(idx[:n_tr]), sorted(idx[n_tr:])


def write_frames(name, x, y):
    import pandas as pd

    df = x.copy()
    df["y"] = y
    tr, te = split_indices(len(df))
    df.iloc[tr].to_csv(OUT / f"{name}_train.csv", index=False, na_rep="")
    df.iloc[te].to_csv(OUT / f"{name}_test.csv", index=False, na_rep="")
    na = int(df.drop(columns=["y"]).isna().sum().sum())
    print(f"{name:<16} {len(tr):>6} train / {len(te):>5} test, "
          f"{df.shape[1] - 1:>3} cols, {na} missing cells")


def load_tabarena():
    import pandas as pd

    for name, (pq, target) in TABARENA.items():
        df = pd.read_parquet(PQ / f"{pq}.parquet")
        y = df[target].astype(float)
        x = pd.get_dummies(df.drop(columns=[target]), dtype=float)
        if x.shape[1] > 40:
            # Tractability cap (documented): keep the 40 highest-variance
            # columns after encoding. Identical columns for every model, so
            # internal comparisons stay fair; wide-data handling is the
            # preselect layer's job.
            var = x.var()
            keep = list(var.sort_values(ascending=False).index[:40])
            x = x[sorted(keep)]
        write_frames(name, x, y)


def load_rulequest():
    import pandas as pd

    for name, cfg in RULEQUEST.items():
        rows, y = [], []
        for line in (DATA / f"{name}.data").read_text().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            y.append(float(parts[cfg["target"]]))
            feats = {}
            for i, p in enumerate(parts):
                if i == cfg["target"] or i in cfg["drop"]:
                    continue
                if i in cfg["cats"]:
                    for lev in cfg["cats"][i]:
                        feats[f"c{i}_{lev}"] = 1.0 if p == lev else 0.0
                else:
                    feats[f"c{i}"] = None if p in ("N/A", "?", "") else float(p)
            rows.append(feats)
        write_frames(name, pd.DataFrame(rows), pd.Series(y))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load_tabarena()
    load_rulequest()
    print("\ndeferred (wide-column stress for the preselect/binned-scan phase):")
    print("  superconductivity (82 cols), QSAR-TID-11 (1025 cols)")


if __name__ == "__main__":
    main()
