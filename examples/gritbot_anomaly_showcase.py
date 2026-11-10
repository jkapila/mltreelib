"""GritBot contextual anomaly showcase on Bank_Customer_Churn dataset.

Demonstrates Quinlan's GritBot: discovering surprising cases whose values deviate
strongly from expectation within conditional subgroups (e.g. balance, age, credit score).
"""

from pathlib import Path
import pyarrow.parquet as pq
from mltreelib import GritBot

def main():
    path = Path("datasets/Bank_Customer_Churn.parquet")
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    # Use PyArrow ParquetFile with context manager
    with pq.ParquetFile(path) as pf:
        table = pf.read()
        df = table.to_pandas()

    num_cols = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary",
    ]

    cols = [c for c in num_cols if c in df.columns]
    data = df[cols].values.tolist()

    print("=" * 70)
    print("GritBot Contextual Anomaly Engine (Quinlan 2010)")
    print(f"Scanning {len(data)} cases across {len(cols)} attributes: {cols}")
    print("=" * 70)

    bot = GritBot(
        min_items=35,
        max_norm=3.2,
        sig_threshold=0.01,
        max_depth=3,
    )

    anomalies = bot.scan(data, col_names=cols)

    pure_contextual = [a for a in anomalies if a["is_pure_contextual"]]
    global_outliers = [a for a in anomalies if not a["conditions"]]

    print(f"\nDiscovered {len(anomalies)} total anomalies (sig <= 0.01):")
    print(f"  • Pure Contextual Anomalies: {len(pure_contextual)} (hidden globally, exposed in subgroup)")
    print(f"  • Global Outliers:           {len(global_outliers)}")

    print("\n--- Top Pure Contextual Anomalies (Research Feature: Local vs Global Surprise) ---")
    for i, a in enumerate(pure_contextual[:5], start=1):
        print(f"{i}. {a['summary']}")
        for cond in a["conditions"]:
            print(f"      rule context: {cond['display']} (case has {cond['actual_val']:.2f})")
        print(f"      local z: {a['z_score']:.2f} vs global z: {a['global_z']:.2f} (context elevation: {a['context_ratio']:.2f}x)")
        print(f"      DET surprise score: {a['density_score']:.2f} nats, p-value: {a['significance']:.2e}")
        print()

    print("--- Top Global Outliers ---")
    for i, a in enumerate(global_outliers[:5], start=1):
        print(f"{i}. {a['summary']}")
        print(f"      z-score: {a['z_score']:.2f}, DET surprise: {a['density_score']:.2f} nats, p-value: {a['significance']:.2e}")
        print()

if __name__ == "__main__":
    main()
