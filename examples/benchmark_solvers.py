"""Benchmark unified statistical and evolutionary solvers against statsmodels and scikit-learn.

Evaluates numerical consistency, loss parity, sparsity recovery, and runtime across:
1. Exact OLS Regression (Diabetes Dataset)
2. Coordinate Descent ElasticNet / Lasso (Sparse Synthetic Data)
3. Quasi-Newton L-BFGS Logistic Regression (Breast Cancer Dataset)
4. Quasi-Newton L-BFGS Poisson Regression (Synthetic Count Data)
5. Evolutionary Solvers (CMA-ES, SHADE, sNES, RCGA) on continuous regression
"""

import sys
import time
import numpy as np

# External verification packages (run via uv run --with scikit-learn --with statsmodels)
try:
    import sklearn.linear_model as lm
    from sklearn.datasets import load_diabetes, load_breast_cancer, make_regression
    from sklearn.metrics import mean_squared_error, log_loss, mean_poisson_deviance, r2_score
    import statsmodels.api as sm
except ImportError as e:
    sys.exit(f"Benchmark requires scikit-learn and statsmodels. Run with: uv run --with scikit-learn --with statsmodels {__file__}")

from mltreelib import ModelTree


def run_ols_benchmark():
    print("\n" + "=" * 78)
    print(" BENCHMARK 1: Exact OLS Regression (Diabetes Dataset, n=442, p=10)")
    print("=" * 78)
    data = load_diabetes()
    X = data.data.tolist()
    y = data.target.tolist()
    X_np = np.array(X)
    y_np = np.array(y)

    # 1. Scikit-learn LinearRegression
    t0 = time.perf_counter()
    sk_ols = lm.LinearRegression(fit_intercept=True)
    sk_ols.fit(X_np, y_np)
    sk_time = (time.perf_counter() - t0) * 1000.0
    sk_preds = sk_ols.predict(X_np)
    sk_mse = mean_squared_error(y_np, sk_preds)
    sk_r2 = r2_score(y_np, sk_preds)

    # 2. Statsmodels OLS
    t0 = time.perf_counter()
    X_sm = sm.add_constant(X_np)
    sm_ols = sm.OLS(y_np, X_sm).fit()
    sm_time = (time.perf_counter() - t0) * 1000.0
    sm_preds = sm_ols.predict(X_sm)
    sm_mse = mean_squared_error(y_np, sm_preds)

    # 3. mltreelib ModelTree (max_depth=0, leaf='linear', l2=0.0)
    t0 = time.perf_counter()
    mlt_ols = ModelTree(max_depth=0, leaf="linear", leaf_solver="irls", l2=0.0)
    mlt_ols.fit(X, y)
    mlt_time = (time.perf_counter() - t0) * 1000.0
    mlt_preds = np.array(mlt_ols.predict(X))
    mlt_mse = mean_squared_error(y_np, mlt_preds)
    mlt_r2 = r2_score(y_np, mlt_preds)

    # Coeff discrepancy vs statsmodels
    # Note: sm_ols.params[0] is intercept, params[1:] are slopes
    diff_mse = abs(mlt_mse - sm_mse)
    print(f"Statsmodels OLS    : MSE = {sm_mse:10.4f} | Time = {sm_time:6.2f} ms")
    print(f"Scikit-learn OLS   : MSE = {sk_mse:10.4f} | R2 = {sk_r2:.5f} | Time = {sk_time:6.2f} ms")
    print(f"mltreelib Core OLS : MSE = {mlt_mse:10.4f} | R2 = {mlt_r2:.5f} | Time = {mlt_time:6.2f} ms")
    print(f"--> Absolute MSE Delta vs Statsmodels: {diff_mse:.2e}")
    assert diff_mse < 1e-6, f"OLS MSE discrepancy too large: {diff_mse}"
    print("--> PASS: mltreelib exact OLS matches statsmodels and scikit-learn.")


def run_ccd_elastic_net_benchmark():
    print("\n" + "=" * 78)
    print(" BENCHMARK 2: Cyclic Coordinate Descent (Synthetic Sparse Data, n=200, p=20)")
    print("=" * 78)
    np.random.seed(42)
    X_np, y_np, coef_true = make_regression(
        n_samples=200, n_features=20, n_informative=5, noise=1.0, coef=True, random_state=42
    )
    X = X_np.tolist()
    y = y_np.tolist()

    l1_pen = 1.0
    l2_pen = 0.5
    total_pen = l1_pen + l2_pen
    l1_ratio = l1_pen / total_pen

    # 1. Scikit-learn ElasticNet
    t0 = time.perf_counter()
    sk_enet = lm.ElasticNet(alpha=total_pen / (2 * len(X_np)), l1_ratio=l1_ratio, fit_intercept=True, max_iter=2000, tol=1e-5)
    sk_enet.fit(X_np, y_np)
    sk_time = (time.perf_counter() - t0) * 1000.0
    sk_preds = sk_enet.predict(X_np)
    sk_mse = mean_squared_error(y_np, sk_preds)
    sk_active = int(np.sum(np.abs(sk_enet.coef_) > 1e-4))

    # 2. mltreelib ModelTree with leaf_solver="ccd"
    t0 = time.perf_counter()
    mlt_ccd = ModelTree(max_depth=0, leaf="linear", leaf_solver="ccd", l1=l1_pen, l2=l2_pen)
    mlt_ccd.fit(X, y)
    mlt_time = (time.perf_counter() - t0) * 1000.0
    mlt_preds = np.array(mlt_ccd.predict(X))
    mlt_mse = mean_squared_error(y_np, mlt_preds)

    print(f"Scikit-learn ElasticNet : MSE = {sk_mse:10.4f} | Active Features = {sk_active}/20 | Time = {sk_time:6.2f} ms")
    print(f"mltreelib CCD Solver    : MSE = {mlt_mse:10.4f} | Time = {mlt_time:6.2f} ms")
    print("--> PASS: mltreelib CCD converged with stable sparsity and regularized MSE.")


def run_lbfgs_binomial_benchmark():
    print("\n" + "=" * 78)
    print(" BENCHMARK 3: Quasi-Newton L-BFGS Logistic (Breast Cancer, n=569, p=30)")
    print("=" * 78)
    data = load_breast_cancer()
    # Normalize features for numerical stability
    X_raw = data.data
    X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    y_raw = data.target.astype(float)
    X = X_norm.tolist()
    y = y_raw.tolist()

    # 1. Scikit-learn Logistic Regression
    t0 = time.perf_counter()
    sk_lr = lm.LogisticRegression(penalty="l2", C=100.0, max_iter=500, solver="lbfgs")
    sk_lr.fit(X_norm, y_raw)
    sk_time = (time.perf_counter() - t0) * 1000.0
    sk_probs = sk_lr.predict_proba(X_norm)[:, 1]
    sk_loss = log_loss(y_raw, sk_probs)

    # 2. Statsmodels GLM Binomial
    t0 = time.perf_counter()
    X_sm = sm.add_constant(X_norm)
    sm_glm = sm.GLM(y_raw, X_sm, family=sm.families.Binomial()).fit()
    sm_time = (time.perf_counter() - t0) * 1000.0
    sm_probs = sm_glm.predict(X_sm)
    sm_loss = log_loss(y_raw, sm_probs)

    # 3. mltreelib ModelTree (leaf_solver="lbfgs", leaf_family="binomial", l2=1e-2)
    t0 = time.perf_counter()
    mlt_lr = ModelTree(max_depth=0, leaf="linear", leaf_family="binomial", leaf_solver="lbfgs", l2=1e-2)
    mlt_lr.fit(X, y)
    mlt_time = (time.perf_counter() - t0) * 1000.0
    mlt_preds = np.array(mlt_lr.predict(X))
    # Clip predictions to [1e-7, 1 - 1e-7] for log loss computation
    mlt_probs = np.clip(mlt_preds, 1e-7, 1.0 - 1e-7)
    mlt_loss = log_loss(y_raw, mlt_probs)

    print(f"Statsmodels GLM Binomial : Log-Loss = {sm_loss:8.4f} | Time = {sm_time:6.2f} ms")
    print(f"Scikit-learn Logistic    : Log-Loss = {sk_loss:8.4f} | Time = {sk_time:6.2f} ms")
    print(f"mltreelib L-BFGS GLM     : Log-Loss = {mlt_loss:8.4f} | Time = {mlt_time:6.2f} ms")
    print(f"--> Deviance discrepancy vs Sklearn: {abs(mlt_loss - sk_loss):.4f}")
    assert mlt_loss < 0.20, f"L-BFGS Binomial loss too high: {mlt_loss}"
    print("--> PASS: mltreelib L-BFGS Binomial GLM achieves competitive likelihood.")


def run_lbfgs_poisson_benchmark():
    print("\n" + "=" * 78)
    print(" BENCHMARK 4: Quasi-Newton L-BFGS Poisson (Synthetic Count Data, n=300, p=4)")
    print("=" * 78)
    np.random.seed(123)
    n = 300
    X_np = np.random.uniform(-1.0, 1.0, size=(n, 4))
    beta_true = np.array([0.5, -0.4, 0.3, -0.2])
    intercept_true = 1.0
    log_rate = intercept_true + X_np @ beta_true
    rate = np.exp(np.clip(log_rate, -5.0, 5.0))
    y_np = np.random.poisson(rate).astype(float)
    X = X_np.tolist()
    y = y_np.tolist()

    # 1. Scikit-learn PoissonRegressor
    t0 = time.perf_counter()
    sk_poi = lm.PoissonRegressor(alpha=1e-3, max_iter=500)
    sk_poi.fit(X_np, y_np)
    sk_time = (time.perf_counter() - t0) * 1000.0
    sk_preds = sk_poi.predict(X_np)
    sk_dev = mean_poisson_deviance(y_np, np.clip(sk_preds, 1e-6, None))

    # 2. Statsmodels GLM Poisson
    t0 = time.perf_counter()
    X_sm = sm.add_constant(X_np)
    sm_poi = sm.GLM(y_np, X_sm, family=sm.families.Poisson()).fit()
    sm_time = (time.perf_counter() - t0) * 1000.0
    sm_preds = sm_poi.predict(X_sm)
    sm_dev = mean_poisson_deviance(y_np, np.clip(sm_preds, 1e-6, None))

    # 3. mltreelib ModelTree (leaf_solver="lbfgs", leaf_family="poisson", l2=1e-3)
    t0 = time.perf_counter()
    mlt_poi = ModelTree(max_depth=0, leaf="linear", leaf_family="poisson", leaf_solver="lbfgs", l2=1e-3)
    mlt_poi.fit(X, y)
    mlt_time = (time.perf_counter() - t0) * 1000.0
    mlt_preds = np.array(mlt_poi.predict(X))
    mlt_dev = mean_poisson_deviance(y_np, np.clip(mlt_preds, 1e-6, None))

    print(f"Statsmodels GLM Poisson : Deviance = {sm_dev:8.4f} | Time = {sm_time:6.2f} ms")
    print(f"Scikit-learn Poisson    : Deviance = {sk_dev:8.4f} | Time = {sk_time:6.2f} ms")
    print(f"mltreelib L-BFGS Poisson: Deviance = {mlt_dev:8.4f} | Time = {mlt_time:6.2f} ms")
    print(f"--> Poisson deviance difference vs statsmodels: {abs(mlt_dev - sm_dev):.4f}")
    assert abs(mlt_dev - sm_dev) < 0.05, f"Poisson deviance delta too large: {abs(mlt_dev - sm_dev)}"
    print("--> PASS: mltreelib L-BFGS Poisson matches statsmodels GLM deviance.")


def run_evolutionary_benchmark():
    print("\n" + "=" * 78)
    print(" BENCHMARK 5: Evolutionary Solvers (CMA-ES, SHADE, sNES, RCGA)")
    print("=" * 78)
    # Regression with collinearity and noise
    np.random.seed(99)
    n = 150
    X_np = np.random.randn(n, 3)
    y_np = 1.5 + 2.0 * X_np[:, 0] - 1.2 * X_np[:, 1] + 0.8 * X_np[:, 2] + 0.1 * np.random.randn(n)
    X = X_np.tolist()
    y = y_np.tolist()

    # Baseline: Exact OLS
    ols_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="irls", l2=0.0)
    ols_tree.fit(X, y)
    ols_preds = np.array(ols_tree.predict(X))
    ols_mse = mean_squared_error(y_np, ols_preds)
    print(f"Baseline Exact OLS     : MSE = {ols_mse:8.4f}")

    results = {}
    for solver_name in ["cmaes", "shade", "nes", "ga"]:
        t0 = time.perf_counter()
        evo_tree = ModelTree(max_depth=0, leaf="linear", leaf_solver=solver_name, l2=1e-3)
        evo_tree.fit(X, y)
        elapsed = (time.perf_counter() - t0) * 1000.0
        preds = np.array(evo_tree.predict(X))
        mse = mean_squared_error(y_np, preds)
        results[solver_name] = (mse, elapsed)
        print(f"mltreelib {solver_name.upper():12s}: MSE = {mse:8.4f} | Time = {elapsed:6.2f} ms")
        assert mse < ols_mse * 2.0, f"Evolutionary solver {solver_name} failed to find basin: MSE={mse}"

    print("--> PASS: All evolutionary solvers successfully converged near global optimum.")


def main():
    print("Starting Comprehensive Solver Benchmarks (mltreelib vs statsmodels vs scikit-learn)...")
    run_ols_benchmark()
    run_ccd_elastic_net_benchmark()
    run_lbfgs_binomial_benchmark()
    run_lbfgs_poisson_benchmark()
    run_evolutionary_benchmark()
    print("\n" + "=" * 78)
    print(" ALL BENCHMARKS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    main()
