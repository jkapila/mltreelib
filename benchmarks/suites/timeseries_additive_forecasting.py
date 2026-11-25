"""Suite 7: Time Series Feature Engineering & Additive Forecasting.

Demonstrates:
1. Pure Rust calendar feature extraction (calendar_features).
2. Entity-grouped calendar day lags (lag_by_days) with native missing routing.
3. Trailing window rolling aggregations (windowed_transform).
4. Additive trend & seasonal decomposition (decomposed_transform).
5. Out-of-time multi-horizon forecasting with regularized linear model trees.

Usage:
    uv run python benchmarks/suites/timeseries_additive_forecasting.py
"""

from datetime import date, timedelta
import math

from mltreelib import (
    Metrics,
    ModelTree,
    calendar_features,
    decomposed_transform,
    lag_by_days,
    windowed_transform,
)


def main():
    # 1. Generate realistic multi-store retail panel (120 days x 2 stores = 240 observations)
    start_date = date(2025, 1, 1)
    stores = ["store_nyc", "store_lon"]
    dates = []
    keys = []
    sales = []

    for d_idx in range(120):
        cur_date = (start_date + timedelta(days=d_idx)).isoformat()
        day_of_week = (start_date + timedelta(days=d_idx)).weekday()  # 0=Mon, 6=Sun
        weekend_boost = 35.0 if day_of_week >= 5 else 0.0

        for s_idx, store in enumerate(stores):
            base_demand = 140.0 if store == "store_nyc" else 95.0
            growth_trend = 0.45 * d_idx
            noise = math.sin(d_idx * 0.7 + s_idx * 1.5) * 4.0
            val = round(base_demand + growth_trend + weekend_boost + noise, 1)

            dates.append(cur_date)
            keys.append(store)
            sales.append(val)

    n_rows = len(dates)
    print(f"Generated retail sales panel: {n_rows} rows (120 days x 2 stores).")
    print(f"Date range: {dates[0]} to {dates[-1]}\n")

    # =========================================================================
    # Part 1: Pure Rust Feature Engineering Pipeline
    # =========================================================================
    print("=== 1. Pure Rust Time Feature Transforms ===")
    cal = calendar_features(dates, hemisphere="north")
    lag1 = lag_by_days(keys, dates, sales, 1)
    lag7 = lag_by_days(keys, dates, sales, 7)
    win7_mean = windowed_transform(keys, dates, sales, window_days=7, kind="mean")
    decomp = decomposed_transform(keys, dates, sales, season_granularity="weekday", trend_window=7)

    print("Engineered features in pure Rust (mltreelib-core):")
    print(f"  - calendar_features : weekday, weekend, month, day_of_year")
    print(f"  - lag_by_days       : 1-day lag, 7-day lag (initial {sum(1 for v in lag7 if v is None)} rows None)")
    print(f"  - windowed_transform: 7-day trailing mean")
    print(f"  - decomposed        : detrended, deseasonalized, residual\n")

    # Assemble feature matrix (use stationary autoregressive features and calendar cycles)
    feature_names = ["weekday", "weekend", "lag1", "lag7", "win7_mean"]
    X = [
        [
            cal["weekday"][i],
            cal["weekend"][i],
            lag1[i],
            lag7[i],
            win7_mean[i],
        ]
        for i in range(n_rows)
    ]
    y = sales

    # =========================================================================
    # Part 2: Out-of-Time Train / Test Split (First 90 days train, Last 30 days test)
    # =========================================================================
    split_idx = 180  # 90 days * 2 stores
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    dates_test = dates[split_idx:]
    keys_test = keys[split_idx:]

    print(f"Out-of-time evaluation split:")
    print(f"  Train: {split_idx} rows ({dates[0]} to {dates[split_idx - 1]})")
    print(f"  Test : {n_rows - split_idx} rows ({dates[split_idx]} to {dates[-1]})\n")

    # =========================================================================
    # Part 3: ModelTree Out-of-Time Forecasting
    # =========================================================================
    print("=== 2. ModelTree Out-of-Time Forecasting vs CART Step Function ===")
    cart = ModelTree(leaf="mean", max_depth=3, min_rows=15)
    cart.fit(X_train, y_train, col_names=feature_names)
    p_cart = cart.predict(X_test)

    # Linear model tree with ridge regularization for autoregressive trend extrapolation
    model_tree = ModelTree(leaf="linear", max_depth=3, min_rows=15, l2=0.05)
    model_tree.fit(X_train, y_train, col_names=feature_names)
    p_mt = model_tree.predict(X_test)

    m = Metrics()
    m_cart = m.regression(y_test, p_cart)
    m_mt = m.regression(y_test, p_mt)

    print(f"Standard CART (Mean) : Test RMSE = {m_cart['rmse']:.2f}, R2 = {m_cart['r2']:.3f}, MAE = {m_cart['mae']:.2f}")
    print(f"ModelTree (Linear)   : Test RMSE = {m_mt['rmse']:.2f}, R2 = {m_mt['r2']:.3f}, MAE = {m_mt['mae']:.2f}")
    err_reduction = (1.0 - m_mt["rmse"] / m_cart["rmse"]) * 100.0
    print(f"--> Out-of-time forecast error reduction: {err_reduction:.1f}%\n")

    # =========================================================================
    # Part 4: Generate Publication Multi-Panel Figure
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        c_teal = "#00897b"
        c_orange = "#f4511e"
        c_blue = "#0288d1"
        c_dark = "#263238"

        # Panel 1: Store Sales Trajectory & 7-Day Window
        nyc_indices = [i for i, k in enumerate(keys) if k == "store_nyc"]
        nyc_dates = [dates[i] for i in nyc_indices]
        nyc_sales = [sales[i] for i in nyc_indices]
        nyc_win = [win7_mean[i] for i in nyc_indices]

        ax1.plot(range(len(nyc_sales)), nyc_sales, color=c_blue, alpha=0.6, linewidth=1.2, label="Store NYC Actual Sales")
        ax1.plot(range(len(nyc_win)), nyc_win, color=c_dark, linewidth=2.0, label="7-Day Trailing Mean (windowed_transform)")
        ax1.axvline(90, color=c_orange, linestyle="--", linewidth=1.5, label="Out-of-Time Cutoff (Day 90)")
        ax1.set_title("Store Sales Panel & Pure Rust Trailing Window", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax1.set_xlabel("Elapsed Day", fontsize=10, color=c_dark)
        ax1.set_ylabel("Daily Units Sold", fontsize=10, color=c_dark)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 2: Seasonal Cycle Profile (Weekday vs Weekend)
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_means = [sum(sales[i] for i in range(n_rows) if cal["weekday"][i] == d) / sum(1 for i in range(n_rows) if cal["weekday"][i] == d) for d in range(7)]
        ax2.bar(weekdays, day_means, color=[c_blue]*5 + [c_orange]*2, edgecolor=c_dark, width=0.55)
        ax2.set_title("Day-of-Week Seasonality (calendar_features)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax2.set_xlabel("Day of Week", fontsize=10, color=c_dark)
        ax2.set_ylabel("Mean Sales Volume", fontsize=10, color=c_dark)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Panel 3: Additive Decomposition (Deseasonalized vs Residual)
        valid_res = [decomp["residual"][i] for i in range(len(decomp["residual"])) if decomp["residual"][i] is not None]
        ax3.hist(valid_res, bins=20, color=c_teal, alpha=0.75, edgecolor=c_dark)
        ax3.axvline(0, color=c_dark, linestyle="--")
        ax3.set_title("Additive Decomposition Residual (decomposed_transform)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax3.set_xlabel("Residual Demand Error", fontsize=10, color=c_dark)
        ax3.set_ylabel("Frequency", fontsize=10, color=c_dark)
        ax3.grid(True, linestyle=":", alpha=0.6)

        # Panel 4: Out-of-Time Forecast Horizon (Days 90-120)
        test_nyc_idx = [i for i, k in enumerate(keys_test) if k == "store_nyc"]
        y_test_nyc = [y_test[i] for i in test_nyc_idx]
        p_cart_nyc = [p_cart[i] for i in test_nyc_idx]
        p_mt_nyc = [p_mt[i] for i in test_nyc_idx]
        days_horizon = list(range(91, 91 + len(test_nyc_idx)))

        ax4.plot(days_horizon, y_test_nyc, marker="o", color=c_dark, linewidth=1.5, label="Actual Observed Demand")
        ax4.step(days_horizon, p_cart_nyc, where="post", color=c_orange, linewidth=2.0, alpha=0.8, label=f"CART Step (RMSE: {m_cart['rmse']:.2f})")
        ax4.plot(days_horizon, p_mt_nyc, marker="s", color=c_teal, linewidth=2.2, label=f"ModelTree Linear (RMSE: {m_mt['rmse']:.2f})")
        ax4.set_title("Out-of-Time Forecast Horizon (Store NYC, 30 Days)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax4.set_xlabel("Forecast Day", fontsize=10, color=c_dark)
        ax4.set_ylabel("Units Sold", fontsize=10, color=c_dark)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        plt.suptitle("mltreelib Suite 7: Time Series Feature Engineering & Additive Forecasting", fontsize=15, fontweight="bold", color=c_dark, y=0.98)
        plt.tight_layout()
        out_fig = "docs/assets/timeseries_decomposition.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"Generated publication asset: {out_fig}")
    except ImportError:
        print("Note: matplotlib not found; install mltreelib[plotting] to generate figures.")


if __name__ == "__main__":
    main()
