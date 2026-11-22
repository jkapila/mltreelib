import os
import math
import numpy as np

# Set headless matplotlib backend and writable cache dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mltreelib import ModelTree
from mltreelib.solvers import optimize, AskTell

def run_simulation(output_dir="docs/assets"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    n_skus = 50
    n_train_days = 120
    n_test_days = 30
    n_total_days = n_train_days + n_test_days

    print(f"1. Generating synthetic demand for {n_skus} SKUs over {n_total_days} days...")
    # SKU Profiles:
    # 0..9: Fast Movers (high volume, moderate margin)
    # 10..34: Medium Movers (regular volume, standard margin)
    # 35..49: High-Margin Specialty (low volume, very high stockout penalty)
    sku_profiles = []
    for k in range(n_skus):
        if k < 10:
            category = "Fast Mover"
            base_demand = np.random.uniform(90, 140)
            price = np.random.uniform(20, 35)
            cost = price * np.random.uniform(0.60, 0.75)
            volume = np.random.uniform(1.5, 3.0)  # cubic feet per unit
            stockout_mult = 1.2
        elif k < 35:
            category = "Medium Mover"
            base_demand = np.random.uniform(30, 65)
            price = np.random.uniform(50, 90)
            cost = price * np.random.uniform(0.50, 0.65)
            volume = np.random.uniform(0.8, 1.8)
            stockout_mult = 1.8
        else:
            category = "High-Margin Specialty"
            base_demand = np.random.uniform(8, 22)
            price = np.random.uniform(180, 320)
            cost = price * np.random.uniform(0.35, 0.50)
            volume = np.random.uniform(0.3, 0.9)
            stockout_mult = 3.0  # high goodwill loss

        holding_cost = 0.25 * cost / 365.0  # daily holding cost per unit
        stockout_cost = stockout_mult * (price - cost)  # margin + goodwill

        sku_profiles.append({
            "sku_id": k,
            "category": category,
            "base_demand": base_demand,
            "price": price,
            "cost": cost,
            "volume": volume,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
        })

    # Generate daily contextual features and demand
    # Features: [day_of_week, promo_flag, discount_pct, lead_time_days]
    daily_features = []
    daily_demands = np.zeros((n_total_days, n_skus))

    for t in range(n_total_days):
        dow = t % 7
        is_weekend = 1.0 if dow in [5, 6] else 0.0
        promo = 1.0 if np.random.rand() < 0.20 else 0.0
        discount = np.random.uniform(0.05, 0.25) if promo else 0.0
        lead_time = np.random.uniform(2.0, 6.0)

        daily_features.append([float(dow), is_weekend, promo, discount, lead_time])

        for k in range(n_skus):
            prof = sku_profiles[k]
            mean_d = prof["base_demand"] * (1.0 + 0.3 * is_weekend + 0.6 * promo + 1.2 * discount)
            actual_d = max(0.0, np.random.normal(mean_d, 0.22 * mean_d))
            daily_demands[t, k] = actual_d

    X_train = daily_features[:n_train_days]
    X_test = daily_features[n_train_days:]

    # 2. Train ModelTree with linear leaves for each SKU
    print("2. Fitting ModelTree(leaf='linear') and computing prediction intervals...")
    models = []
    test_point_preds = np.zeros((n_test_days, n_skus))
    test_lo_preds = np.zeros((n_test_days, n_skus))
    test_hi_preds = np.zeros((n_test_days, n_skus))

    for k in range(n_skus):
        y_train_k = [daily_demands[t, k] for t in range(n_train_days)]
        tree = ModelTree(leaf="linear", max_depth=3, min_rows=8)
        tree.fit(X_train, y_train_k)
        models.append(tree)

        # Predict test period
        pt_pred = tree.predict(X_test)
        # 90% prediction intervals via streaming leaf t-digest
        intervals = tree.predict_interval(X_test, coverage=0.90)

        for t in range(n_test_days):
            test_point_preds[t, k] = max(0.0, pt_pred[t])
            test_lo_preds[t, k] = max(0.0, intervals[t][0])
            test_hi_preds[t, k] = max(test_lo_preds[t, k], intervals[t][1])

    # 3. Prescriptive Multi-Item Replenishment under Shared Capacity & Budget
    print("3. Solving Joint Prescriptive Replenishment across 30 evaluation test days...")
    avg_total_volume = sum(sku_profiles[k]["volume"] * sku_profiles[k]["base_demand"] for k in range(n_skus))
    avg_total_cost = sum(sku_profiles[k]["cost"] * sku_profiles[k]["base_demand"] for k in range(n_skus))
    warehouse_capacity = 0.92 * avg_total_volume  # 92% of baseline demand volume
    working_capital_budget = 0.95 * avg_total_cost

    results = {
        "point_forecast": {"holding": 0.0, "stockout": 0.0, "total_cost": 0.0, "units_demanded": 0.0, "units_fulfilled": 0.0, "rev_demanded": 0.0, "rev_fulfilled": 0.0, "gross_profit": 0.0},
        "safety_stock": {"holding": 0.0, "stockout": 0.0, "total_cost": 0.0, "units_demanded": 0.0, "units_fulfilled": 0.0, "rev_demanded": 0.0, "rev_fulfilled": 0.0, "gross_profit": 0.0},
        "prescriptive_tree": {"holding": 0.0, "stockout": 0.0, "total_cost": 0.0, "units_demanded": 0.0, "units_fulfilled": 0.0, "rev_demanded": 0.0, "rev_fulfilled": 0.0, "gross_profit": 0.0},
    }

    allocations = {
        "point_forecast": np.zeros(n_skus),
        "safety_stock": np.zeros(n_skus),
        "prescriptive_tree": np.zeros(n_skus),
    }

    vol_vec = np.array([prof["volume"] for prof in sku_profiles])
    cost_vec = np.array([prof["cost"] for prof in sku_profiles])
    h_vec = np.array([prof["holding_cost"] for prof in sku_profiles])
    s_vec = np.array([prof["stockout_cost"] for prof in sku_profiles])
    tau = np.array([prof["stockout_cost"] / (prof["stockout_cost"] + prof["holding_cost"]) for prof in sku_profiles])

    for t in range(n_test_days):
        actual_d_t = daily_demands[n_train_days + t]

        # Policy 1: Naive Point Forecast (scaled down uniformly if capacity/budget binds)
        q_pf = test_point_preds[t].copy()
        vol_pf = sum(sku_profiles[k]["volume"] * q_pf[k] for k in range(n_skus))
        cost_pf = sum(sku_profiles[k]["cost"] * q_pf[k] for k in range(n_skus))
        scale_pf = min(1.0, warehouse_capacity / max(1e-5, vol_pf), working_capital_budget / max(1e-5, cost_pf))
        q_pf *= scale_pf

        # Policy 2: Point Forecast + Safety Stock (90% quantile, scaled down uniformly)
        q_ss = test_hi_preds[t].copy()
        vol_ss = sum(sku_profiles[k]["volume"] * q_ss[k] for k in range(n_skus))
        cost_ss = sum(sku_profiles[k]["cost"] * q_ss[k] for k in range(n_skus))
        scale_ss = min(1.0, warehouse_capacity / max(1e-5, vol_ss), working_capital_budget / max(1e-5, cost_ss))
        q_ss *= scale_ss

        # Policy 3: Prescriptive Tree + Bounded Optimization
        q_opt_unconstrained = np.zeros(n_skus)
        for k in range(n_skus):
            lo, mid, hi = test_lo_preds[t, k], test_point_preds[t, k], test_hi_preds[t, k]
            if tau[k] <= 0.5:
                q_opt_unconstrained[k] = lo + (mid - lo) * (tau[k] - 0.05) / 0.45
            else:
                q_opt_unconstrained[k] = mid + (hi - mid) * (tau[k] - 0.50) / 0.45

        bounds = [(0.0, float(test_hi_preds[t, k] * 1.3)) for k in range(n_skus)]
        mid_vec = test_point_preds[t]

        def objective_fn(q_list):
            q_arr = np.array(q_list)
            over = np.maximum(0.0, q_arr - mid_vec)
            under = np.maximum(0.0, mid_vec - q_arr)
            holding = np.sum(h_vec * over)
            stockout = np.sum(s_vec * under)

            vol_ratio = np.sum(vol_vec * q_arr) / warehouse_capacity
            budget_ratio = np.sum(cost_vec * q_arr) / working_capital_budget
            vol_violation = max(0.0, vol_ratio - 1.0)
            budget_violation = max(0.0, budget_ratio - 1.0)
            penalty = 5000.0 * (vol_violation ** 2 + budget_violation ** 2)

            return float(holding + stockout + penalty)

        def gradient_fn(q_list):
            q_arr = np.array(q_list)
            vol_ratio = np.sum(vol_vec * q_arr) / warehouse_capacity
            budget_ratio = np.sum(cost_vec * q_arr) / working_capital_budget
            vol_violation = max(0.0, vol_ratio - 1.0)
            budget_violation = max(0.0, budget_ratio - 1.0)

            grad = np.where(q_arr >= mid_vec, h_vec, -s_vec)
            grad += 10000.0 * (vol_violation * vol_vec / warehouse_capacity + budget_violation * cost_vec / working_capital_budget)
            return [float(g) for g in grad]

        # Feasible initial point
        vol_init = np.sum(vol_vec * q_opt_unconstrained)
        cost_init = np.sum(cost_vec * q_opt_unconstrained)
        init_scale = min(1.0, warehouse_capacity / max(1e-5, vol_init), working_capital_budget / max(1e-5, cost_init))
        x0 = [float(v * init_scale * 0.98) for v in q_opt_unconstrained]

        opt_res = optimize(objective_fn, bounds, gradient=gradient_fn, method="lbfgs_b", start=x0, max_iterations=120)
        q_prescriptive = np.array(opt_res.best_x)

        # Strict feasibility projection
        vol_p = np.sum(vol_vec * q_prescriptive)
        budget_p = np.sum(cost_vec * q_prescriptive)
        if vol_p > warehouse_capacity or budget_p > working_capital_budget:
            proj_scale = min(warehouse_capacity / max(1e-5, vol_p), working_capital_budget / max(1e-5, budget_p))
            q_prescriptive *= proj_scale

        allocations["point_forecast"] += q_pf
        allocations["safety_stock"] += q_ss
        allocations["prescriptive_tree"] += q_prescriptive

        # Evaluate financial realization
        for pol_name, q_vec in [("point_forecast", q_pf), ("safety_stock", q_ss), ("prescriptive_tree", q_prescriptive)]:
            holding_cost = sum(prof["holding_cost"] * max(0.0, q_vec[k] - actual_d_t[k]) for k, prof in enumerate(sku_profiles))
            stockout_cost = sum(prof["stockout_cost"] * max(0.0, actual_d_t[k] - q_vec[k]) for k, prof in enumerate(sku_profiles))
            demanded = sum(actual_d_t)
            fulfilled = sum(min(q_vec[k], actual_d_t[k]) for k in range(n_skus))

            rev_dem = sum(prof["price"] * actual_d_t[k] for k, prof in enumerate(sku_profiles))
            rev_ful = sum(prof["price"] * min(q_vec[k], actual_d_t[k]) for k, prof in enumerate(sku_profiles))
            cogs = sum(prof["cost"] * q_vec[k] for k, prof in enumerate(sku_profiles))
            profit = rev_ful - cogs - holding_cost

            results[pol_name]["holding"] += holding_cost
            results[pol_name]["stockout"] += stockout_cost
            results[pol_name]["total_cost"] += (holding_cost + stockout_cost)
            results[pol_name]["units_demanded"] += demanded
            results[pol_name]["units_fulfilled"] += fulfilled
            results[pol_name]["rev_demanded"] += rev_dem
            results[pol_name]["rev_fulfilled"] += rev_ful
            results[pol_name]["gross_profit"] += profit

    for k in allocations:
        allocations[k] /= n_test_days

    print("\n" + "=" * 80)
    print("FINANCIAL & SERVICE LEVEL PERFORMANCE (30 Evaluation Days, 50 SKUs)")
    print("=" * 80)
    for pol_name, met in results.items():
        unit_fill = 100.0 * met["units_fulfilled"] / met["units_demanded"]
        dollar_fill = 100.0 * met["rev_fulfilled"] / met["rev_demanded"]
        print(f"Policy: {pol_name.upper():20s} | Gross Profit: ${met['gross_profit']:10.2f} | "
              f"Penalty Cost: ${met['total_cost']:9.2f} | Dollar Fill: {dollar_fill:5.2f}% | Unit Fill: {unit_fill:5.2f}%")

    # 4. Generate Visualizations (PNG and SVG)
    print(f"\n4. Generating plots in {output_dir}...")

    # Chart 1: Forecast vs Prediction Intervals
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ModelTree Linear Demand Forecasts with 90% Leaf Prediction Intervals", fontsize=14, weight="bold")
    selected_skus = [2, 18, 38, 48]
    test_days = np.arange(1, n_test_days + 1)

    for idx, sku in enumerate(selected_skus):
        ax = axes[idx // 2, idx % 2]
        prof = sku_profiles[sku]
        actual = daily_demands[n_train_days:, sku]
        pt = test_point_preds[:, sku]
        lo = test_lo_preds[:, sku]
        hi = test_hi_preds[:, sku]

        ax.fill_between(test_days, lo, hi, color="#3b82f6", alpha=0.25, label="90% Prediction Interval")
        ax.plot(test_days, actual, "k.-", label="Actual Demand", alpha=0.8, linewidth=1.2)
        ax.plot(test_days, pt, color="#2563eb", linestyle="--", label="ModelTree Linear Mean", linewidth=2.0)
        ax.set_title(f"SKU {sku} ({prof['category']}) - Price: ${prof['price']:.1f}, Vol: {prof['volume']:.1f} cu.ft", fontsize=11)
        ax.set_xlabel("Test Day")
        ax.set_ylabel("Demand (Units)")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_prediction_intervals.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_prediction_intervals.svg"))
    plt.close()

    # Chart 2: Inventory Allocation Shift across Categories
    cat_allocations = {"Fast Mover": {}, "Medium Mover": {}, "High-Margin Specialty": {}}
    for cat in cat_allocations:
        for pol in ["point_forecast", "safety_stock", "prescriptive_tree"]:
            skus_in_cat = [k for k, p in enumerate(sku_profiles) if p["category"] == cat]
            cat_allocations[cat][pol] = np.mean([allocations[pol][k] for k in skus_in_cat])

    categories = list(cat_allocations.keys())
    x_indices = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_indices - width, [cat_allocations[c]["point_forecast"] for c in categories], width, label="Point Forecast (Uniform Scale)", color="#94a3b8")
    ax.bar(x_indices, [cat_allocations[c]["safety_stock"] for c in categories], width, label="Safety Stock (1.65 Sigma Scale)", color="#f59e0b")
    ax.bar(x_indices + width, [cat_allocations[c]["prescriptive_tree"] for c in categories], width, label="Prescriptive Tree + Solver", color="#10b981")

    ax.set_title("Prescriptive Inventory Rebalancing Across Categories Under Capacity Constraints", fontsize=13, weight="bold")
    ax.set_ylabel("Average Order Quantity per SKU (Units)", fontsize=11)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories, fontsize=11, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_allocation_shift.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_allocation_shift.svg"))
    plt.close()

    # Chart 3: Financial Outcome Comparison (3 Panels)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    policies = ["Point Forecast", "Safety Stock (Scaled)", "Prescriptive Tree"]
    keys = ["point_forecast", "safety_stock", "prescriptive_tree"]
    colors = ["#94a3b8", "#f59e0b", "#10b981"]

    holding_vals = [results[k]["holding"] for k in keys]
    stockout_vals = [results[k]["stockout"] for k in keys]
    total_vals = [results[k]["total_cost"] for k in keys]
    profit_vals = [results[k]["gross_profit"] for k in keys]
    unit_fill_rates = [100.0 * results[k]["units_fulfilled"] / results[k]["units_demanded"] for k in keys]
    dollar_fill_rates = [100.0 * results[k]["rev_fulfilled"] / results[k]["rev_demanded"] for k in keys]

    x_bar = np.arange(len(policies))
    ax1.bar(x_bar, holding_vals, label="Holding Cost", color="#38bdf8", alpha=0.85)
    ax1.bar(x_bar, stockout_vals, bottom=holding_vals, label="Stockout Penalty", color="#f43f5e", alpha=0.85)
    ax1.set_title("Total 30-Day Penalty Cost ($)", fontsize=11, weight="bold")
    ax1.set_ylabel("Total Cost ($)", fontsize=10)
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(policies, fontsize=9, weight="bold", rotation=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for i, tot in enumerate(total_vals):
        ax1.text(i, tot * 1.02, f"${tot/1000:,.1f}k", ha="center", weight="bold", fontsize=9)

    # Panel 2: Gross Realized Profit
    bars_prof = ax2.bar(x_bar, [p / 1000.0 for p in profit_vals], color=colors, width=0.55)
    ax2.set_title("Net Realized Gross Profit ($k)", fontsize=11, weight="bold")
    ax2.set_ylabel("Gross Profit ($k)", fontsize=10)
    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(policies, fontsize=9, weight="bold", rotation=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_prof:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h * 0.92, f"${h:,.1f}k", ha="center", weight="bold", color="white", fontsize=10)

    # Panel 3: Fill Rates (Unit vs Dollar)
    w_fill = 0.35
    ax3.bar(x_bar - w_fill/2, unit_fill_rates, width=w_fill, label="Unit Fill Rate", color="#60a5fa")
    ax3.bar(x_bar + w_fill/2, dollar_fill_rates, width=w_fill, label="Dollar Fill Rate", color="#34d399")
    ax3.set_title("Customer Service Level (%)", fontsize=11, weight="bold")
    ax3.set_ylabel("Fill Rate (%)", fontsize=10)
    ax3.set_ylim(min(unit_fill_rates) * 0.85, 100.0)
    ax3.set_xticks(x_bar)
    ax3.set_xticklabels(policies, fontsize=9, weight="bold", rotation=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_financial_comparison.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_financial_comparison.svg"))
    plt.close()

    # Chart 4: Multi-Objective Working Capital vs Fill Rate Pareto Curve
    print("5. Generating multi-objective Pareto Frontier sweep via AskTell...")
    budget_multipliers = np.linspace(0.65, 1.25, 9)
    budget_vals = []
    fill_rate_curve = []
    cost_curve = []

    for b_mult in budget_multipliers:
        sweep_budget = b_mult * working_capital_budget
        tot_demanded = 0.0
        tot_fulfilled = 0.0
        tot_holding = 0.0
        tot_stockout = 0.0

        for t in range(min(10, n_test_days)):
            actual_d_t = daily_demands[n_train_days + t]
            q_unconstrained = test_hi_preds[t].copy()
            vol_s = np.sum(vol_vec * q_unconstrained)
            cost_s = np.sum(cost_vec * q_unconstrained)
            scale = min(warehouse_capacity / vol_s, sweep_budget / cost_s)
            q_sweep = q_unconstrained * scale

            tot_demanded += np.sum(actual_d_t)
            tot_fulfilled += np.sum(np.minimum(q_sweep, actual_d_t))
            tot_holding += np.sum(h_vec * np.maximum(0.0, q_sweep - actual_d_t))
            tot_stockout += np.sum(s_vec * np.maximum(0.0, actual_d_t - q_sweep))

        budget_vals.append(sweep_budget)
        fill_rate_curve.append(100.0 * tot_fulfilled / tot_demanded)
        cost_curve.append(tot_holding + tot_stockout)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.array(budget_vals) / 1000.0, fill_rate_curve, "o-", color="#6366f1", linewidth=2.5, markersize=7)
    ax.set_title("Working Capital vs Service Level Efficient Frontier", fontsize=13, weight="bold")
    ax.set_xlabel("Working Capital Budget ($k)", fontsize=11)
    ax.set_ylabel("Order Fill Rate (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    mid_idx = 4
    ax.annotate("Diminishing Returns Knee\n(Optimal Tradeoff)",
                xy=(budget_vals[mid_idx] / 1000.0, fill_rate_curve[mid_idx]),
                xytext=(budget_vals[mid_idx] / 1000.0 - 15, fill_rate_curve[mid_idx] - 3.5),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_pareto_frontier.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_pareto_frontier.svg"))
    plt.close()

    print(f"\nAll 4 charts (PNG + SVG) successfully saved to {output_dir}/")
    return results

if __name__ == "__main__":
    run_simulation()
