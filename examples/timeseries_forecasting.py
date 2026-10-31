"""End-to-end time series feature engineering and forecasting with ModelTree.

Demonstrates:
1. Pure Rust time feature transforms:
   - calendar_features: weekday, weekend, quarter, season
   - lag_by_days: entity-grouped calendar day lags
   - windowed_transform: trailing rolling window means
2. Native missing value handling: initial rows with unobserved lags (None)
   route through ModelTree natively without manual imputation.
3. Forward holdout forecasting: evaluation on out-of-time future dates.
"""

from __future__ import annotations

from datetime import date, timedelta
from mltreelib import (
    Metrics,
    ModelTree,
    calendar_features,
    lag_by_days,
    windowed_transform,
)


def generate_panel_data():
    stores = ["store_nyc", "store_lon"]
    dates, keys, sales = [], [], []

    start = date(2025, 1, 1)
    for day in range(90):
        d_str = (start + timedelta(days=day)).isoformat()
        dow = (start + timedelta(days=day)).weekday()
        for s in stores:
            keys.append(s)
            dates.append(d_str)
            base = 100.0 + (50.0 if s == "store_nyc" else 0.0)
            trend = 0.5 * day
            season = 30.0 if dow in (5, 6) else 0.0
            sales.append(base + trend + season)
    return keys, dates, sales


def main():
    keys, dates, sales = generate_panel_data()
    print("== 1. Synthetic Store Sales Panel (90 days x 2 stores = 180 observations) ==")
    print(f"  dates: {dates[0]} to {dates[-1]}")

    # Feature extraction via Rust core
    cal = calendar_features(dates)
    lag1 = lag_by_days(keys, dates, sales, 1)
    lag7 = lag_by_days(keys, dates, sales, 7)
    win7_mean = windowed_transform(keys, dates, sales, 7, kind="mean")

    print("\n== 2. Engineered Time Features (first 4 rows) ==")
    print(f"  store      : {keys[:4]}")
    print(f"  weekday    : {cal['weekday'][:4]}")
    print(f"  weekend    : {cal['weekend'][:4]}")
    print(f"  lag_1d     : {lag1[:4]} (first day per store is None)")
    print(f"  win7_mean  : {[round(v, 2) for v in win7_mean[:4]]}")

    # Assemble feature matrix
    X = []
    for i in range(len(sales)):
        X.append([keys[i], cal["weekday"][i], cal["weekend"][i], lag1[i], lag7[i], win7_mean[i]])

    dtypes = ["categorical", "numeric", "numeric", "numeric", "numeric", "numeric"]

    # Temporal split: first 70 days (140 rows) for train, last 20 days (40 rows) for test
    split_idx = 140
    xtr, ytr = X[:split_idx], sales[:split_idx]
    xte, yte = X[split_idx:], sales[split_idx:]

    print("\n== 3. ModelTree Fit (Linear leaves with regularized trend) ==")
    tree = ModelTree(max_depth=3, leaf="linear", l2=0.1)
    tree.fit(xtr, ytr, dtypes=dtypes)

    preds = tree.predict(xte)
    metrics = Metrics().regression(yte, preds)

    print(f"  test RMSE : {metrics['rmse']:.3f}")
    print(f"  test MAE  : {metrics['mae']:.3f}")
    print(f"  test R2   : {metrics['r2']:.3f}")

    print("\n== 4. Out-of-Time Predictions vs Actuals (first 6 rows) ==")
    for i in range(6):
        print(f"  date={dates[split_idx + i]} store={keys[split_idx + i]:<10} actual={yte[i]:5.1f} predicted={preds[i]:5.1f}")


if __name__ == "__main__":
    main()
