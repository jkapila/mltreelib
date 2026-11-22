import os
import time
import numpy as np

# Set headless matplotlib backend and writable cache dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mltreelib import ModelTree

def get_sku_profiles(n_skus=50):
    np.random.seed(42)
    sku_profiles = []
    for k in range(n_skus):
        if k < 10:
            category = "Fast Mover"
            base_demand = np.random.uniform(90, 140)
            price = np.random.uniform(20, 35)
            cost = price * np.random.uniform(0.60, 0.75)
            volume = np.random.uniform(1.5, 3.0)  # cu.ft per unit
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
    return sku_profiles

def generate_synthetic_data(seed, sku_profiles, n_train_days=120, n_test_days=30):
    np.random.seed(seed)
    n_total_days = n_train_days + n_test_days
    n_skus = len(sku_profiles)

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
    return X_train, X_test, daily_demands

def solve_joint_dual(s_vec, h_vec, vol_vec, cost_vec, V_max, B_max, models, x_test_t):
    """
    Solve joint replenishment problem via Lagrangian multiplier dual search.
    The stationary fractile condition for each SKU is:
      F_k(q_k) = (s_k - lam * vol_k - mu * cost_k) / (s_k + h_k)
    """
    n_skus = len(s_vec)

    def get_q_k(k, tau):
        if tau <= 0.0001:
            return 0.0
        # Retrieve empirical conditional quantile directly from leaf t-digest
        return max(0.0, models[k].quantile(x_test_t, min(0.999, tau))[0])

    def eval_mult(lam, mu):
        q = np.zeros(n_skus)
        for k in range(n_skus):
            num = s_vec[k] - lam * vol_vec[k] - mu * cost_vec[k]
            if num > 0:
                tau = num / (s_vec[k] + h_vec[k])
                q[k] = get_q_k(k, tau)
            else:
                q[k] = 0.0
        return q, np.sum(vol_vec * q), np.sum(cost_vec * q)

    # 1. Check unconstrained (lam=0, mu=0)
    q0, v0, c0 = eval_mult(0.0, 0.0)
    if v0 <= V_max and c0 <= B_max:
        return q0, 0.0, 0.0

    # 2. Check if volume constraint alone suffices (mu = 0)
    lo_l, hi_l = 0.0, 100.0
    for _ in range(25):
        mid_l = 0.5 * (lo_l + hi_l)
        _, v_cur, _ = eval_mult(mid_l, 0.0)
        if v_cur > V_max:
            lo_l = mid_l
        else:
            hi_l = mid_l
    lam_v = hi_l
    q_v, v_v, c_v = eval_mult(lam_v, 0.0)
    if c_v <= B_max:
        return q_v, lam_v, 0.0

    # 3. Check if budget constraint alone suffices (lam = 0)
    lo_m, hi_m = 0.0, 10.0
    for _ in range(25):
        mid_m = 0.5 * (lo_m + hi_m)
        _, _, c_cur = eval_mult(0.0, mid_m)
        if c_cur > B_max:
            lo_m = mid_m
        else:
            hi_m = mid_m
    mu_b = hi_m
    q_b, v_b, c_b = eval_mult(0.0, mu_b)
    if v_b <= V_max:
        return q_b, 0.0, mu_b

    # 4. Both constraints bind: alternating bisection
    lam, mu = lam_v, 0.0
    for _ in range(15):
        lo_l, hi_l = 0.0, 100.0
        for _ in range(15):
            mid_l = 0.5 * (lo_l + hi_l)
            _, v_cur, _ = eval_mult(mid_l, mu)
            if v_cur > V_max:
                lo_l = mid_l
            else:
                hi_l = mid_l
        lam = 0.5 * (lo_l + hi_l)

        lo_m, hi_m = 0.0, 10.0
        for _ in range(15):
            mid_m = 0.5 * (lo_m + hi_m)
            _, _, c_cur = eval_mult(lam, mid_m)
            if c_cur > B_max:
                lo_m = mid_m
            else:
                hi_m = mid_m
        mu = 0.5 * (lo_m + hi_m)

    q_opt, _, _ = eval_mult(lam, mu)
    return q_opt, lam, mu

def run_simulation(n_seeds=20, output_dir="docs/assets"):
    os.makedirs(output_dir, exist_ok=True)
    n_skus = 50
    n_train_days = 120
    n_test_days = 30
    sku_profiles = get_sku_profiles(n_skus)

    vol_vec = np.array([p["volume"] for p in sku_profiles])
    cost_vec = np.array([p["cost"] for p in sku_profiles])
    h_vec = np.array([p["holding_cost"] for p in sku_profiles])
    s_vec = np.array([p["stockout_cost"] for p in sku_profiles])
    p_vec = np.array([p["price"] for p in sku_profiles])

    avg_total_volume = sum(sku_profiles[k]["volume"] * sku_profiles[k]["base_demand"] for k in range(n_skus))
    avg_total_cost = sum(sku_profiles[k]["cost"] * sku_profiles[k]["base_demand"] for k in range(n_skus))
    warehouse_capacity = 0.92 * avg_total_volume  # 92% of baseline volume
    working_capital_budget = 0.95 * avg_total_cost  # 95% of baseline capital

    print(f"Running 50-SKU replenishment benchmark across {n_seeds} seeds...")
    print(f"Capacity: Volume <= {warehouse_capacity:,.1f} cu.ft | Budget <= ${working_capital_budget:,.1f}")

    metrics = {
        "Policy A (Mean)": {"cost": [], "unit_fill": [], "dollar_fill": [], "profit": []},
        "Policy B (Fractile + Scale)": {"cost": [], "unit_fill": [], "dollar_fill": [], "profit": []},
        "Policy C (Joint Dual)": {"cost": [], "unit_fill": [], "dollar_fill": [], "profit": []},
    }

    c_beats_b_count = 0
    single_run_data = None  # To save plots from seed 0

    total_solve_time = 0.0
    total_days_solved = 0

    for seed in range(n_seeds):
        X_train, X_test, daily_demands = generate_synthetic_data(seed, sku_profiles, n_train_days, n_test_days)

        # Fit ModelTrees
        models = []
        for k in range(n_skus):
            tree = ModelTree(leaf="linear", max_depth=3, min_rows=8)
            tree.fit(X_train, daily_demands[:n_train_days, k].tolist())
            models.append(tree)

        res_a = {"holding": 0.0, "stockout": 0.0, "dem": 0.0, "ful": 0.0, "rev_dem": 0.0, "rev_ful": 0.0, "cogs": 0.0}
        res_b = {"holding": 0.0, "stockout": 0.0, "dem": 0.0, "ful": 0.0, "rev_dem": 0.0, "rev_ful": 0.0, "cogs": 0.0}
        res_c = {"holding": 0.0, "stockout": 0.0, "dem": 0.0, "ful": 0.0, "rev_dem": 0.0, "rev_ful": 0.0, "cogs": 0.0}

        alloc_a_seed = np.zeros(n_skus)
        alloc_b_seed = np.zeros(n_skus)
        alloc_c_seed = np.zeros(n_skus)

        seed_point_preds = np.zeros((n_test_days, n_skus))
        seed_lo_preds = np.zeros((n_test_days, n_skus))
        seed_hi_preds = np.zeros((n_test_days, n_skus))

        for t in range(n_test_days):
            x_test_t = X_test[t:t+1]
            actual_d = daily_demands[n_train_days + t]

            # 1. Predictions
            mean_preds = np.array([max(0.0, models[k].predict(x_test_t)[0]) for k in range(n_skus)])
            tau_unconstrained = s_vec / (s_vec + h_vec)
            fractile_preds = np.array([max(0.0, models[k].quantile(x_test_t, min(0.999, tau_unconstrained[k]))[0]) for k in range(n_skus)])

            if seed == 0:
                seed_point_preds[t] = mean_preds
                for k in range(n_skus):
                    seed_lo_preds[t, k] = max(0.0, models[k].quantile(x_test_t, 0.05)[0])
                    seed_hi_preds[t, k] = max(0.0, models[k].quantile(x_test_t, 0.95)[0])

            # Policy A: Order Mean + Uniform Downscaling
            scale_a = min(1.0, warehouse_capacity / max(1e-5, np.sum(vol_vec * mean_preds)), working_capital_budget / max(1e-5, np.sum(cost_vec * mean_preds)))
            q_a = mean_preds * scale_a

            # Policy B: Unconstrained Fractile + Uniform Downscaling
            scale_b = min(1.0, warehouse_capacity / max(1e-5, np.sum(vol_vec * fractile_preds)), working_capital_budget / max(1e-5, np.sum(cost_vec * fractile_preds)))
            q_b = fractile_preds * scale_b

            # Policy C: Joint Dual Solve
            t_start = time.perf_counter()
            q_c, _, _ = solve_joint_dual(s_vec, h_vec, vol_vec, cost_vec, warehouse_capacity, working_capital_budget, models, x_test_t)
            total_solve_time += (time.perf_counter() - t_start)
            total_days_solved += 1

            alloc_a_seed += q_a
            alloc_b_seed += q_b
            alloc_c_seed += q_c

            for pol, q_vec in [(res_a, q_a), (res_b, q_b), (res_c, q_c)]:
                pol["holding"] += float(np.sum(h_vec * np.maximum(0.0, q_vec - actual_d)))
                pol["stockout"] += float(np.sum(s_vec * np.maximum(0.0, actual_d - q_vec)))
                pol["dem"] += float(np.sum(actual_d))
                pol["ful"] += float(np.sum(np.minimum(q_vec, actual_d)))
                pol["rev_dem"] += float(np.sum(p_vec * actual_d))
                pol["rev_ful"] += float(np.sum(p_vec * np.minimum(q_vec, actual_d)))
                pol["cogs"] += float(np.sum(cost_vec * q_vec))

        alloc_a_seed /= n_test_days
        alloc_b_seed /= n_test_days
        alloc_c_seed /= n_test_days

        cost_a = res_a["holding"] + res_a["stockout"]
        cost_b = res_b["holding"] + res_b["stockout"]
        cost_c = res_c["holding"] + res_c["stockout"]

        metrics["Policy A (Mean)"]["cost"].append(cost_a)
        metrics["Policy A (Mean)"]["unit_fill"].append(100.0 * res_a["ful"] / res_a["dem"])
        metrics["Policy A (Mean)"]["dollar_fill"].append(100.0 * res_a["rev_ful"] / res_a["rev_dem"])
        metrics["Policy A (Mean)"]["profit"].append(res_a["rev_ful"] - res_a["cogs"] - res_a["holding"])

        metrics["Policy B (Fractile + Scale)"]["cost"].append(cost_b)
        metrics["Policy B (Fractile + Scale)"]["unit_fill"].append(100.0 * res_b["ful"] / res_b["dem"])
        metrics["Policy B (Fractile + Scale)"]["dollar_fill"].append(100.0 * res_b["rev_ful"] / res_b["rev_dem"])
        metrics["Policy B (Fractile + Scale)"]["profit"].append(res_b["rev_ful"] - res_b["cogs"] - res_b["holding"])

        metrics["Policy C (Joint Dual)"]["cost"].append(cost_c)
        metrics["Policy C (Joint Dual)"]["unit_fill"].append(100.0 * res_c["ful"] / res_c["dem"])
        metrics["Policy C (Joint Dual)"]["dollar_fill"].append(100.0 * res_c["rev_ful"] / res_c["rev_dem"])
        metrics["Policy C (Joint Dual)"]["profit"].append(res_c["rev_ful"] - res_c["cogs"] - res_c["holding"])

        if cost_c < cost_b:
            c_beats_b_count += 1

        if seed == 0:
            single_run_data = {
                "daily_demands": daily_demands,
                "seed_point_preds": seed_point_preds,
                "seed_lo_preds": seed_lo_preds,
                "seed_hi_preds": seed_hi_preds,
                "alloc_a": alloc_a_seed,
                "alloc_b": alloc_b_seed,
                "alloc_c": alloc_c_seed,
                "models": models,
                "X_test": X_test,
                "res_a": res_a,
                "res_b": res_b,
                "res_c": res_c,
            }

    avg_time_ms = 1000.0 * total_solve_time / total_days_solved
    print(f"\nAverage dual solve time per daily allocation: {avg_time_ms:.2f} ms across {total_days_solved} days.")
    print(f"Policy C achieved lower penalty cost than Policy B in {c_beats_b_count}/{n_seeds} seeds.\n")

    # Print Markdown Benchmark Table
    print("=" * 80)
    print("BENCHMARK RESULTS (20 Seeds, 50 SKUs, 30 Evaluation Days)")
    print("=" * 80)
    print("| Policy | Penalty Cost ($) | Unit Fill Rate (%) | Dollar Fill Rate (%) | Realized Profit ($) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for name in ["Policy A (Mean)", "Policy B (Fractile + Scale)", "Policy C (Joint Dual)"]:
        m = metrics[name]
        c_m, c_s = np.mean(m["cost"]), np.std(m["cost"])
        uf_m, uf_s = np.mean(m["unit_fill"]), np.std(m["unit_fill"])
        df_m, df_s = np.mean(m["dollar_fill"]), np.std(m["dollar_fill"])
        p_m, p_s = np.mean(m["profit"]), np.std(m["profit"])
        print(f"| {name} | ${c_m:,.0f} ± ${c_s:,.0f} | {uf_m:.2f}% ± {uf_s:.2f}% | {df_m:.2f}% ± {df_s:.2f}% | ${p_m:,.0f} ± ${p_s:,.0f} |")
    print("=" * 80)

    # Generate Publication Figures (Seed 0)
    print(f"\nGenerating plots in {output_dir}...")
    demands = single_run_data["daily_demands"]
    pt_preds = single_run_data["seed_point_preds"]
    lo_preds = single_run_data["seed_lo_preds"]
    hi_preds = single_run_data["seed_hi_preds"]

    # Chart 1: Forecast vs Leaf Quantile Intervals
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ModelTree Linear Demand Forecasts with 90% Leaf-Digest Prediction Intervals", fontsize=14, weight="bold")
    selected_skus = [2, 18, 38, 48]
    test_days = np.arange(1, n_test_days + 1)

    for idx, sku in enumerate(selected_skus):
        ax = axes[idx // 2, idx % 2]
        prof = sku_profiles[sku]
        actual = demands[n_train_days:, sku]
        pt = pt_preds[:, sku]
        lo = lo_preds[:, sku]
        hi = hi_preds[:, sku]

        ax.fill_between(test_days, lo, hi, color="#3b82f6", alpha=0.25, label="90% Leaf Prediction Interval")
        ax.plot(test_days, actual, "k.-", label="Synthetic Actual Demand", alpha=0.8, linewidth=1.2)
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
        skus_in_cat = [k for k, p in enumerate(sku_profiles) if p["category"] == cat]
        cat_allocations[cat]["Policy A (Mean)"] = np.mean(single_run_data["alloc_a"][skus_in_cat])
        cat_allocations[cat]["Policy B (Fractile + Scale)"] = np.mean(single_run_data["alloc_b"][skus_in_cat])
        cat_allocations[cat]["Policy C (Joint Dual)"] = np.mean(single_run_data["alloc_c"][skus_in_cat])

    categories = list(cat_allocations.keys())
    x_indices = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_indices - width, [cat_allocations[c]["Policy A (Mean)"] for c in categories], width, label="Policy A (Mean + Scale)", color="#94a3b8")
    ax.bar(x_indices, [cat_allocations[c]["Policy B (Fractile + Scale)"] for c in categories], width, label="Policy B (Fractile + Scale)", color="#f59e0b")
    ax.bar(x_indices + width, [cat_allocations[c]["Policy C (Joint Dual)"] for c in categories], width, label="Policy C (Joint Dual Solve)", color="#10b981")

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

    # Chart 3: Financial & Fill Rate Comparison (Seed 0 realization)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    policies = ["Policy A (Mean)", "Policy B (Fractile)", "Policy C (Dual)"]
    r_list = [single_run_data["res_a"], single_run_data["res_b"], single_run_data["res_c"]]
    colors = ["#94a3b8", "#f59e0b", "#10b981"]

    holding_vals = [r["holding"] for r in r_list]
    stockout_vals = [r["stockout"] for r in r_list]
    total_vals = [r["holding"] + r["stockout"] for r in r_list]
    profit_vals = [(r["rev_ful"] - r["cogs"] - r["holding"]) / 1000.0 for r in r_list]
    unit_fills = [100.0 * r["ful"] / r["dem"] for r in r_list]
    dollar_fills = [100.0 * r["rev_ful"] / r["rev_dem"] for r in r_list]

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
        ax1.text(i, tot * 1.02, f"${tot/1000:,.0f}k", ha="center", weight="bold", fontsize=9)

    bars_prof = ax2.bar(x_bar, profit_vals, color=colors, width=0.55)
    ax2.set_title("Net Realized Gross Profit ($k)", fontsize=11, weight="bold")
    ax2.set_ylabel("Gross Profit ($k)", fontsize=10)
    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(policies, fontsize=9, weight="bold", rotation=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_prof:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h * 0.92, f"${h:,.0f}k", ha="center", weight="bold", color="white", fontsize=10)

    w_fill = 0.35
    ax3.bar(x_bar - w_fill/2, unit_fills, width=w_fill, label="Unit Fill Rate", color="#60a5fa")
    ax3.bar(x_bar + w_fill/2, dollar_fills, width=w_fill, label="Dollar Fill Rate", color="#34d399")
    ax3.set_title("Customer Service Level (%)", fontsize=11, weight="bold")
    ax3.set_ylabel("Fill Rate (%)", fontsize=10)
    ax3.set_ylim(min(unit_fills) * 0.85, 100.0)
    ax3.set_xticks(x_bar)
    ax3.set_xticklabels(policies, fontsize=9, weight="bold", rotation=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_financial_comparison.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_financial_comparison.svg"))
    plt.close()

    # Chart 4: Working Capital vs Dollar Fill Rate Pareto Curve with Algorithmically Computed Knee
    print("Generating Pareto Frontier sweep across capital budgets...")
    budget_multipliers = np.linspace(0.65, 1.25, 13)
    budget_vals = []
    fill_rate_curve = []
    models_0 = single_run_data["models"]
    X_test_0 = single_run_data["X_test"]

    for b_mult in budget_multipliers:
        sweep_budget = b_mult * working_capital_budget
        tot_dem = 0.0
        tot_ful = 0.0

        for t in range(min(15, n_test_days)):
            x_test_t = X_test_0[t:t+1]
            actual_d = demands[n_train_days + t]
            q_sweep, _, _ = solve_joint_dual(s_vec, h_vec, vol_vec, cost_vec, warehouse_capacity, sweep_budget, models_0, x_test_t)
            tot_dem += np.sum(p_vec * actual_d)
            tot_ful += np.sum(p_vec * np.minimum(q_sweep, actual_d))

        budget_vals.append(sweep_budget)
        fill_rate_curve.append(100.0 * tot_ful / tot_dem)

    b_arr = np.array(budget_vals)
    f_arr = np.array(fill_rate_curve)

    # Compute knee via maximum perpendicular distance to the chord connecting endpoints
    b_norm = (b_arr - b_arr[0]) / (b_arr[-1] - b_arr[0])
    f_norm = (f_arr - f_arr[0]) / (f_arr[-1] - f_arr[0])
    # Perpendicular distance to line y = x in normalized coordinates
    distances = np.abs(b_norm - f_norm) / np.sqrt(2.0)
    knee_idx = int(np.argmax(distances))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(b_arr / 1000.0, f_arr, "o-", color="#6366f1", linewidth=2.5, markersize=7)
    ax.set_title("Working Capital vs Dollar Service Level Efficient Frontier", fontsize=13, weight="bold")
    ax.set_xlabel("Working Capital Budget ($k)", fontsize=11)
    ax.set_ylabel("Dollar Fill Rate (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.plot(b_arr[knee_idx] / 1000.0, f_arr[knee_idx], "ro", markersize=10, label="Algorithmic Knee Point")
    ax.annotate(f"Computed Knee: ${b_arr[knee_idx]/1000:.0f}k\n({f_arr[knee_idx]:.1f}% Dollar Fill)",
                xy=(b_arr[knee_idx] / 1000.0, f_arr[knee_idx]),
                xytext=(b_arr[knee_idx] / 1000.0 - 20, f_arr[knee_idx] - 3.0),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, weight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replenishment_pareto_frontier.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "replenishment_pareto_frontier.svg"))
    plt.close()

    print(f"\nAll 4 charts (PNG + SVG) successfully generated in {output_dir}/")
    return metrics

if __name__ == "__main__":
    run_simulation(n_seeds=20)
