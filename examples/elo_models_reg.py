"""Regression model registry for the Elo arena.

Exposes make_models(meta) returning a dict of model_name -> fit_predict callable.
Every model receives raw features (numbers as floats/None, categoricals as strings).
- ModelTree: consumes raw rows with native dtypes
- LightGBM: consumes pandas category dtype (train vocabulary, no one-hot)
- CatBoost/CTBoost/ChimeraBoost: plain frames with cat_features indices
- EBM: category dtype frames
- sklearn: driver-side ordinal encoding (fit on train only)
- statsmodels / MARS / trust-free: numeric-only
- modeltrees: formula-based model trees
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def make_models(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    import numpy as np
    import pandas as pd
    from mltreelib import ModelTree

    roles = list(meta["columns"].values()) if (meta and "columns" in meta) else None
    
    def get_cat_indices(p: int) -> List[int]:
        if roles:
            return [j for j, r in enumerate(roles) if r == "categorical"]
        return []

    def plain_frames(xtr: List[List[Any]], xte: List[List[Any]], cat_cols: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p = len(xtr[0])
        cat_set = set(cat_cols)
        data_tr = {f"x{j}": [("" if row[j] is None else str(row[j])) if j in cat_set else (float("nan") if row[j] is None else float(row[j])) for row in xtr] for j in range(p)}
        data_te = {f"x{j}": [("" if row[j] is None else str(row[j])) if j in cat_set else (float("nan") if row[j] is None else float(row[j])) for row in xte] for j in range(p)}
        return pd.DataFrame(data_tr), pd.DataFrame(data_te)

    def category_frames(xtr: List[List[Any]], xte: List[List[Any]], cat_cols: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_tr, df_te = plain_frames(xtr, xte, cat_cols)
        for j in cat_cols:
            col = f"x{j}"
            df_tr[col] = df_tr[col].astype("category")
            df_te[col] = pd.Categorical(df_te[col], categories=df_tr[col].cat.categories)
        return df_tr, df_te

    def encoded_mats(xtr: List[List[Any]], xte: List[List[Any]], cat_cols: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.preprocessing import OrdinalEncoder
        p = len(xtr[0])
        cat_set = set(cat_cols)
        enc = None
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit([[str(row[j]) for j in cat_cols] for row in xtr])

        def to_mat(rows):
            mat = np.empty((len(rows), p), dtype=np.float64)
            for i, r in enumerate(rows):
                for j in range(p):
                    if j not in cat_set:
                        mat[i, j] = np.nan if r[j] is None else float(r[j])
            if enc is not None:
                codes = enc.transform([[str(row[j]) for j in cat_cols] for row in rows])
                for idx, j in enumerate(cat_cols):
                    mat[:, j] = codes[:, idx]
            return mat

        return to_mat(xtr), to_mat(xte)

    def numeric_only(xtr: List[List[Any]], xte: List[List[Any]], cat_cols: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if cat_cols:
            raise TypeError(f"Model has no native categorical handling (got {len(cat_cols)} categorical columns)")
        mat_tr = np.asarray([[np.nan if v is None else float(v) for v in r] for r in xtr], dtype=np.float64)
        mat_te = np.asarray([[np.nan if v is None else float(v) for v in r] for r in xte], dtype=np.float64)
        return mat_tr, mat_te

    ms = {}

    def attempt(name: str, builder: Callable):
        try:
            ms[name] = builder()
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            def f(*_a, **_k):
                raise RuntimeError(err_msg)
            ms[name] = f

    # 1. ModelTree variants (mltreelib)
    def make_mt(leaf: str, leaf_cols: str, min_relevance: Optional[float] = None):
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            dtypes = roles if roles and len(roles) == p else [("categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric") for j in range(p)]
            kw = {"leaf": leaf, "leaf_cols": leaf_cols}
            if min_relevance is not None:
                kw["min_relevance"] = min_relevance
            m = ModelTree(**kw)
            t0 = time.perf_counter()
            m.fit(xtr, [float(y) for y in ytr], dtypes=dtypes)
            pred = m.predict(xte)
            return pred, time.perf_counter() - t0
        return run

    ms["ModelTree(all)"] = make_mt("linear", "all")
    ms["ModelTree(path)"] = make_mt("linear", "path")
    ms["ModelTree(mean)"] = make_mt("mean", "all")
    ms["ModelTree(winnow)"] = make_mt("linear", "all", min_relevance=0.05)

    # 2. LightGBM
    def lgbm():
        import lightgbm as lgb
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, verbosity=-1)
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64))
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 3. CatBoost
    def catb():
        from catboost import CatBoostRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = CatBoostRegressor(iterations=200, verbose=0, allow_writing_files=False, cat_features=cat_cols)
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64))
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 4. CTBoost
    def ctb():
        from ctboost import CTBoostRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = CTBoostRegressor()
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64))
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 5. ChimeraBoost
    def chimera():
        from chimeraboost import ChimeraBoostRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = ChimeraBoostRegressor()
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64), cat_features=cat_cols)
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 6. EBM (interpret)
    def ebm():
        from interpret.glassbox import ExplainableBoostingRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = ExplainableBoostingRegressor()
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64))
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 7. Forust (Rust)
    def forust_model():
        from forust import GradientBooster
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            cols = [f"x{j}" for j in range(p)]
            t0 = time.perf_counter()
            m = GradientBooster(objective_type="SquaredLoss")
            m.fit(pd.DataFrame(xa, columns=cols), pd.Series(ytr))
            return m.predict(pd.DataFrame(xe, columns=cols)), time.perf_counter() - t0
        return run

    # 8. Perpetual (Rust)
    def perpetual_model():
        from perpetual import PerpetualBooster
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = PerpetualBooster(objective="SquaredLoss", budget=1.0)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 9. sklearn Tree
    def skl_tree():
        from sklearn.tree import DecisionTreeRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = DecisionTreeRegressor(random_state=0)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 10. sklearn RF
    def skl_rf():
        from sklearn.ensemble import RandomForestRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 11. ExtraTrees
    def extra_trees():
        from sklearn.ensemble import ExtraTreesRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 11b. HistGradientBoosting (native categorical support)
    def hist_gbm():
        from sklearn.ensemble import HistGradientBoostingRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = HistGradientBoostingRegressor(categorical_features=cat_cols, random_state=42)
            m.fit(df_tr, np.asarray(ytr, dtype=np.float64))
            return m.predict(df_te), time.perf_counter() - t0
        return run

    # 11c. sklearn GBM (standard gradient boosting)
    def skl_gbm():
        from sklearn.ensemble import GradientBoostingRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = GradientBoostingRegressor(random_state=42)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 12. sklearn Linear
    def skl_linear():
        from sklearn.linear_model import LinearRegression
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = LinearRegression()
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 13. KNN
    def knn():
        from sklearn.neighbors import KNeighborsRegressor
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            t0 = time.perf_counter()
            m = KNeighborsRegressor(n_neighbors=5)
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 14. MARS (mars-earth)
    def mars():
        import pymars as earth
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = earth.Earth()
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 15. trust-free
    def trustfree():
        from trust import TRUST
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = numeric_only(xtr, xte, cat_cols)
            t0 = time.perf_counter()
            m = TRUST()
            m.fit(xa, np.asarray(ytr, dtype=np.float64))
            return m.predict(xe), time.perf_counter() - t0
        return run

    # 16. statsmodels OLS
    def sm_ols():
        import statsmodels.api as sm
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = numeric_only(xtr, xte, cat_cols)
            xa = sm.add_constant(xa, has_constant="add")
            xe = sm.add_constant(xe, has_constant="add")
            t0 = time.perf_counter()
            fit = sm.OLS(np.asarray(ytr, dtype=np.float64), xa).fit()
            return fit.predict(xe), time.perf_counter() - t0
        return run

    # 17. statsmodels RLM (robust linear model)
    def sm_rlm():
        import statsmodels.api as sm
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = numeric_only(xtr, xte, cat_cols)
            xa = sm.add_constant(xa, has_constant="add")
            xe = sm.add_constant(xe, has_constant="add")
            t0 = time.perf_counter()
            fit = sm.RLM(np.asarray(ytr, dtype=np.float64), xa, M=sm.robust.norms.HuberT()).fit()
            return fit.predict(xe), time.perf_counter() - t0
        return run

    # 18. statsmodels QuantReg (median quantile regression)
    def sm_qreg():
        import statsmodels.api as sm
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = numeric_only(xtr, xte, cat_cols)
            xa = sm.add_constant(xa, has_constant="add")
            xe = sm.add_constant(xe, has_constant="add")
            t0 = time.perf_counter()
            fit = sm.QuantReg(np.asarray(ytr, dtype=np.float64), xa).fit(q=0.5)
            return fit.predict(xe), time.perf_counter() - t0
        return run

    # 19. modeltrees lmtree
    def mtrees():
        import sys
        sys.path.insert(0, "references/repos/modeltrees/src")
        from modeltrees import lmtree as mt_lmtree
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            cols = [f"x{j}" for j in range(p)]
            f = "y ~ " + " + ".join(cols) + " | " + " + ".join(cols)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            df_tr["y"] = np.asarray(ytr, dtype=float)
            t0 = time.perf_counter()
            m = mt_lmtree(f, data=df_tr)
            return np.asarray(m.predict(newdata=df_te), dtype=float), time.perf_counter() - t0
        return run

    attempt("LightGBM(200)", lgbm)
    attempt("CatBoost(200)", catb)
    attempt("CTBoost", ctb)
    attempt("ChimeraBoost", chimera)
    attempt("EBM", ebm)
    attempt("forust (Rust)", forust_model)
    attempt("perpetual (Rust)", perpetual_model)
    attempt("sklearn tree", skl_tree)
    attempt("sklearn RF", skl_rf)
    attempt("ExtraTrees", extra_trees)
    attempt("HistGradientBoosting", hist_gbm)
    attempt("sklearn GBM", skl_gbm)
    attempt("sklearn Linear", skl_linear)
    attempt("KNN", knn)
    attempt("MARS (mars-earth)", mars)
    attempt("trust-free", trustfree)
    attempt("statsmodels OLS", sm_ols)
    attempt("statsmodels RLM", sm_rlm)
    attempt("statsmodels QuantReg", sm_qreg)
    attempt("modeltrees lmtree", mtrees)

    return ms
