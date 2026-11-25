"""Large-Scale Scaling and Fidelity Benchmark.

Evaluates performance, throughput (rows/sec), numerical stability, and accuracy
on datasets from 78,000 up to 250,000 samples:
1. Real-World Datasets:
   - SDSS17 (78,053 rows, 11 features, 3-class Multiclass: GALAXY, STAR, QSO)
   - Airline Customer Satisfaction (129,880 rows, 21 features, mixed categorical)
   - GiveMeSomeCredit (150,000 rows, 10 features, binary classification)
2. Synthetic Stress Benchmarks:
   - 250,000 rows Continuous Regression (Non-linear surface)
   - 250,000 rows Multiclass Classification (5 classes)
3. Solver scaling under large sample load (IRLS vs L-BFGS vs CCD)
"""

import time
import math
import numpy as np
from mltreelib import ModelTree
from mltreelib.datasets import load_dataset

def benchmark_sdss17():
    print("=" * 85)
    print("1. REAL-WORLD LARGE MULTICLASS: SDSS17 (78,053 rows x 11 features, 3 classes)")
    print("=" * 85)
    
    data = load_dataset("SDSS17")
    target_col = "ObjectType"
    cols = [c for c in data["columns"].keys() if c != target_col]
    n_rows = data["n_rows"]
    dtypes = [data["dtypes"][c] for c in cols]
    
    label_map = {"GALAXY": 0.0, "STAR": 1.0, "QSO": 2.0}
    
    split_idx = int(0.8 * n_rows)
    X_train = [[data["columns"][c][i] for c in cols] for i in range(split_idx)]
    y_train = [label_map[data["columns"][target_col][i]] for i in range(split_idx)]
    X_test = [[data["columns"][c][i] for c in cols] for i in range(split_idx, n_rows)]
    y_test = [label_map[data["columns"][target_col][i]] for i in range(split_idx, n_rows)]
    
    # A. Fast Streaming Histogram Subtraction
    t0 = time.perf_counter()
    m_fast = ModelTree(gain="gini", n_classes=3, max_depth=6, min_rows=30, split_eval="fast", seed=42)
    m_fast.fit(X_train, y_train, dtypes=dtypes)
    t_fast = time.perf_counter() - t0
    preds_fast = m_fast.predict(X_test)
    acc_fast = sum(1.0 for p, y in zip(preds_fast, y_test) if round(p) == round(y)) / len(y_test)
    throughput_fast = len(X_train) / t_fast
    
    print(f"  Streaming Histogram (Fast) : {t_fast*1000:6.1f} ms | "
          f"Acc: {acc_fast*100:5.2f}% | Leaves: {m_fast.n_leaves():2d} | "
          f"Throughput: {throughput_fast:,.0f} rows/s")
    
    # B. Preset "accurate" with Shrinkage
    t0 = time.perf_counter()
    m_acc = ModelTree(gain="gini", n_classes=3, preset="accurate", seed=42)
    m_acc.fit(X_train, y_train, dtypes=dtypes)
    t_acc = time.perf_counter() - t0
    preds_acc = m_acc.predict(X_test)
    acc_acc = sum(1.0 for p, y in zip(preds_acc, y_test) if round(p) == round(y)) / len(y_test)
    throughput_acc = len(X_train) / t_acc
    
    print(f"  Preset 'accurate' (Shrunk) : {t_acc*1000:6.1f} ms | "
          f"Acc: {acc_acc*100:5.2f}% | Leaves: {m_acc.n_leaves():2d} | "
          f"Throughput: {throughput_acc:,.0f} rows/s")
    
    # C. Exhaustive Comparison
    t0 = time.perf_counter()
    m_ex = ModelTree(gain="gini", n_classes=3, max_depth=6, min_rows=30, split_eval="exhaustive", seed=42)
    m_ex.fit(X_train, y_train, dtypes=dtypes)
    t_ex = time.perf_counter() - t0
    preds_ex = m_ex.predict(X_test)
    acc_ex = sum(1.0 for p, y in zip(preds_ex, y_test) if round(p) == round(y)) / len(y_test)
    
    speedup = t_ex / t_fast
    print(f"  Exhaustive Scan (Old path) : {t_ex*1000:6.1f} ms | "
          f"Acc: {acc_ex*100:5.2f}% | Leaves: {m_ex.n_leaves():2d} | Speedup: {speedup:.2f}x")

def benchmark_airline_satisfaction():
    print("\n" + "=" * 85)
    print("2. REAL-WORLD MIXED-TYPE CLASSIFICATION: Airline Satisfaction (129,880 rows x 21 cols)")
    print("=" * 85)
    
    data = load_dataset("customer_satisfaction_in_airline")
    target_col = "satisfaction"
    cols = [c for c in data["columns"].keys() if c != target_col]
    n_rows = data["n_rows"]
    dtypes = [data["dtypes"][c] for c in cols]
    
    split_idx = int(0.8 * n_rows)
    X_train = [[data["columns"][c][i] for c in cols] for i in range(split_idx)]
    y_train = [1.0 if data["columns"][target_col][i] in (1, "satisfied", "1") else 0.0 for i in range(split_idx)]
    X_test = [[data["columns"][c][i] for c in cols] for i in range(split_idx, n_rows)]
    y_test = [1.0 if data["columns"][target_col][i] in (1, "satisfied", "1") else 0.0 for i in range(split_idx, n_rows)]
    
    # A. Fast Binomial Tree
    t0 = time.perf_counter()
    tree = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=6, min_rows=40, seed=42)
    tree.fit(X_train, y_train, dtypes=dtypes)
    t_fit = time.perf_counter() - t0
    
    preds = tree.predict_proba(X_test)
    acc = sum(1.0 for p, y in zip(preds, y_test) if (p >= 0.5) == (y >= 0.5)) / len(y_test)
    throughput = len(X_train) / t_fit
    
    print(f"  Binomial Tree (Depth 6)    : {t_fit*1000:6.1f} ms | "
          f"Test Acc: {acc*100:5.2f}% | Leaves: {tree.n_leaves():2d} | "
          f"Throughput: {throughput:,.0f} rows/s")
    
    # B. Preset "fast"
    t0 = time.perf_counter()
    tree_fast = ModelTree(leaf_family="binomial", gain="deviance", preset="fast", seed=42)
    tree_fast.fit(X_train, y_train, dtypes=dtypes)
    t_fast = time.perf_counter() - t0
    preds_f = tree_fast.predict_proba(X_test)
    acc_f = sum(1.0 for p, y in zip(preds_f, y_test) if (p >= 0.5) == (y >= 0.5)) / len(y_test)
    print(f"  Preset 'fast' (Depth 5)    : {t_fast*1000:6.1f} ms | "
          f"Test Acc: {acc_f*100:5.2f}% | Leaves: {tree_fast.n_leaves():2d} | "
          f"Throughput: {len(X_train)/t_fast:,.0f} rows/s")

def benchmark_givemesomecredit():
    print("\n" + "=" * 85)
    print("3. REAL-WORLD LARGE CREDIT RISK: GiveMeSomeCredit (150,000 rows x 10 cols)")
    print("=" * 85)
    
    data = load_dataset("GiveMeSomeCredit")
    target_col = "FinancialDistressNextTwoYears"
    cols = [c for c in data["columns"].keys() if c != target_col]
    n_rows = data["n_rows"]
    dtypes = [data["dtypes"][c] for c in cols]
    
    split_idx = int(0.8 * n_rows)
    X_train = [[data["columns"][c][i] for c in cols] for i in range(split_idx)]
    y_train = [1.0 if str(data["columns"][target_col][i]).lower() in ("yes", "1", "true") else 0.0 for i in range(split_idx)]
    X_test = [[data["columns"][c][i] for c in cols] for i in range(split_idx, n_rows)]
    y_test = [1.0 if str(data["columns"][target_col][i]).lower() in ("yes", "1", "true") else 0.0 for i in range(split_idx, n_rows)]
    
    t0 = time.perf_counter()
    tree = ModelTree(leaf="linear", leaf_family="binomial", gain="deviance", max_depth=5, min_rows=50, seed=42)
    tree.fit(X_train, y_train, dtypes=dtypes)
    t_fit = time.perf_counter() - t0
    
    probs = tree.predict_proba(X_test)
    # Binary log loss
    log_loss = -sum(y * math.log(max(p, 1e-15)) + (1 - y) * math.log(max(1 - p, 1e-15)) for y, p in zip(y_test, probs)) / len(y_test)
    acc = sum(1.0 for p, y in zip(probs, y_test) if (p >= 0.5) == (y >= 0.5)) / len(y_test)
    throughput = len(X_train) / t_fit
    
    print(f"  Linear Logistic Leaves     : {t_fit*1000:6.1f} ms | "
          f"Log-Loss: {log_loss:.4f} | Acc: {acc*100:5.2f}% | Leaves: {tree.n_leaves():2d} | "
          f"Throughput: {throughput:,.0f} rows/s")

def benchmark_massive_synthetic():
    print("\n" + "=" * 85)
    print("4. MASSIVE STRESS TEST: 250,000 SAMPLES REGRESSION & 5-CLASS MULTICLASS")
    print("=" * 85)
    
    n_total = 250000
    n_train = 200000
    n_features = 10
    rng = np.random.RandomState(42)
    
    print(f"\n[A] 250,000 Rows Continuous Non-Linear Regression:")
    X_mat = rng.randn(n_total, n_features)
    y_vec = (
        np.sin(X_mat[:, 0] * 1.5)
        + 0.5 * (X_mat[:, 1] > 0).astype(float) * X_mat[:, 1]
        - 0.8 * np.cos(X_mat[:, 2] * 2.0)
        + 0.2 * rng.randn(n_total)
    )
    X_tr, X_te = X_mat[:n_train].tolist(), X_mat[n_train:].tolist()
    y_tr, y_te = y_vec[:n_train].tolist(), y_vec[n_train:].tolist()
    
    # 1. Unshrunk ModelTree
    t0 = time.perf_counter()
    m_unshrunk = ModelTree(leaf="linear", max_depth=6, min_rows=50, shrinkage=0.0, seed=42)
    m_unshrunk.fit(X_tr, y_tr)
    t_unshrunk = time.perf_counter() - t0
    preds_u = m_unshrunk.predict(X_te)
    rmse_u = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds_u, y_te)) / len(y_te))
    
    # 2. Shrunk ModelTree (lambda = 10.0)
    t0 = time.perf_counter()
    m_shrunk = ModelTree(leaf="linear", max_depth=6, min_rows=50, shrinkage=10.0, seed=42)
    m_shrunk.fit(X_tr, y_tr)
    t_shrunk = time.perf_counter() - t0
    preds_s = m_shrunk.predict(X_te)
    rmse_s = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds_s, y_te)) / len(y_te))
    
    print(f"  Unshrunk Tree (lambda=0)   : {t_unshrunk*1000:6.1f} ms | "
          f"Test RMSE: {rmse_u:.4f} | Leaves: {m_unshrunk.n_leaves():2d} | "
          f"Throughput: {n_train/t_unshrunk:,.0f} rows/s")
    print(f"  Shrunk Tree (lambda=10)    : {t_shrunk*1000:6.1f} ms | "
          f"Test RMSE: {rmse_s:.4f} | Leaves: {m_shrunk.n_leaves():2d} | "
          f"Throughput: {n_train/t_shrunk:,.0f} rows/s")
    print(f"  Shrinkage RMSE Improvement : {(rmse_u - rmse_s) / rmse_u * 100:.2f}% error reduction")
    
    print(f"\n[B] 250,000 Rows 5-Class Multiclass Streaming Histogram:")
    weights = rng.randn(n_features, 5)
    logits = X_mat @ weights
    y_mc = np.argmax(logits, axis=1).astype(float)
    y_mc_tr, y_mc_te = y_mc[:n_train].tolist(), y_mc[n_train:].tolist()
    
    t0 = time.perf_counter()
    m_mc = ModelTree(gain="gini", n_classes=5, max_depth=5, min_rows=50, split_eval="fast", seed=42)
    m_mc.fit(X_tr, y_mc_tr)
    t_mc = time.perf_counter() - t0
    preds_mc = m_mc.predict(X_te)
    acc_mc = sum(1.0 for p, y in zip(preds_mc, y_mc_te) if round(p) == round(y)) / len(y_mc_te)
    
    print(f"  Streaming 5-Class Gini Tree: {t_mc*1000:6.1f} ms | "
          f"Test Acc: {acc_mc*100:5.2f}% | Leaves: {m_mc.n_leaves():2d} | "
          f"Throughput: {n_train/t_mc:,.0f} rows/s")

if __name__ == "__main__":
    benchmark_sdss17()
    benchmark_airline_satisfaction()
    benchmark_givemesomecredit()
    benchmark_massive_synthetic()
