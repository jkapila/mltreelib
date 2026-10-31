"""Classification model registry for the Elo arena.

Exposes make_models(meta) returning a dict of model_name -> fit_predict callable.
Every model receives raw features (numbers as floats/None, categoricals as strings).
Returns probability rows aligned with meta['classes'] order.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def make_models(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    import numpy as np
    import pandas as pd
    classes = list(meta["classes"]) if (meta and "classes" in meta) else ["0", "1"]
    roles = list(meta["columns"].values()) if (meta and "columns" in meta) else None

    def get_cat_indices(p: int) -> List[int]:
        if roles:
            return [j for j, r in enumerate(roles) if r == "categorical"]
        return []

    def y_indices(y_raw: List[Any]) -> np.ndarray:
        lut = {str(c): i for i, c in enumerate(classes)}
        return np.asarray([lut[str(v)] for v in y_raw], dtype=np.int64)

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

    def align_proba(m, xte) -> List[List[float]]:
        prob = m.predict_proba(xte)
        cols = {int(lab): j for j, lab in enumerate(m.classes_)}
        return [[float(prob[i][cols[k]]) for k in range(len(classes))] for i in range(len(prob))]

    ms = {}

    def attempt(name: str, builder: Callable):
        try:
            ms[name] = builder()
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            def f(*_a, **_k):
                raise RuntimeError(err_msg)
            ms[name] = f

    # Note: ModelTree is deliberately omitted for classification while classification-specific
    # behaviors are being built in Rust core (will be attached to the Elo board later).

    # 1. LightGBM
    def lgbm():
        import lightgbm as lgb
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbosity=-1)
            m.fit(df_tr, yi)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 3. CatBoost
    def catb():
        from catboost import CatBoostClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = CatBoostClassifier(iterations=200, verbose=0, allow_writing_files=False, cat_features=cat_cols)
            m.fit(df_tr, yi)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 4. CTBoost
    def ctb():
        from ctboost import CTBoostClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = CTBoostClassifier(cat_features=cat_cols)
            m.fit(df_tr, yi)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 5. ChimeraBoost
    def chimera():
        from chimeraboost import ChimeraBoostClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = ChimeraBoostClassifier()
            m.fit(df_tr, yi, cat_features=cat_cols)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 6. EBM (interpret)
    def ebm():
        from interpret.glassbox import ExplainableBoostingClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = ExplainableBoostingClassifier()
            m.fit(df_tr, yi)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 7. Forust (Rust)
    def forust_model():
        from forust import GradientBooster
        def run(xtr, ytr, xte):
            if len(classes) != 2:
                raise RuntimeError("forust is binary-only")
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = (y_indices(ytr) == 1).astype(np.float64)
            t0 = time.perf_counter()
            m = GradientBooster(objective_type="LogLoss")
            m.fit(xa, yi)
            pred = np.asarray(m.predict(xe), dtype=np.float64)
            p1 = 1.0 / (1.0 + np.exp(-pred))
            proba = [[1.0 - v, v] for v in p1]
            return proba, time.perf_counter() - t0
        return run

    # 8. Perpetual (Rust)
    def perpetual_model():
        from perpetual import PerpetualClassifier
        def run(xtr, ytr, xte):
            if len(classes) != 2:
                raise RuntimeError("perpetual is binary-only")
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = PerpetualClassifier(objective="LogLoss", budget=1.0)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 9. sklearn Tree
    def skl_tree():
        from sklearn.tree import DecisionTreeClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = DecisionTreeClassifier(random_state=0)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 10. sklearn RF
    def skl_rf():
        from sklearn.ensemble import RandomForestClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 11. ExtraTrees
    def extra_trees():
        from sklearn.ensemble import ExtraTreesClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 11b. HistGradientBoosting (native categorical support)
    def hist_gbm():
        from sklearn.ensemble import HistGradientBoostingClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            df_tr, df_te = category_frames(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = HistGradientBoostingClassifier(categorical_features=cat_cols, random_state=42)
            m.fit(df_tr, yi)
            return align_proba(m, df_te), time.perf_counter() - t0
        return run

    # 11c. sklearn GBM (standard gradient boosting)
    def skl_gbm():
        from sklearn.ensemble import GradientBoostingClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = GradientBoostingClassifier(random_state=42)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 12. sklearn LogReg
    def skl_logreg():
        from sklearn.linear_model import LogisticRegression
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = LogisticRegression(max_iter=500)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 13. KNN
    def knn():
        from sklearn.neighbors import KNeighborsClassifier
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            means = np.nanmean(xa, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            for j in range(p):
                xa[np.isnan(xa[:, j]), j] = means[j]
                xe[np.isnan(xe[:, j]), j] = means[j]
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = KNeighborsClassifier(n_neighbors=5)
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 14. MARS (mars-earth)
    def mars():
        import pymars as earth
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = encoded_mats(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            t0 = time.perf_counter()
            m = earth.EarthClassifier()
            m.fit(xa, yi)
            return align_proba(m, xe), time.perf_counter() - t0
        return run

    # 15. statsmodels MNLogit
    def sm_mnlogit():
        import statsmodels.api as sm
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            xa, xe = numeric_only(xtr, xte, cat_cols)
            yi = y_indices(ytr)
            xa = sm.add_constant(xa, has_constant="add")
            xe = sm.add_constant(xe, has_constant="add")
            t0 = time.perf_counter()
            fit = sm.MNLogit(yi, xa).fit(disp=False)
            return [list(map(float, row)) for row in fit.predict(xe)], time.perf_counter() - t0
        return run

    # 16. modeltrees glmtree
    def mt_glmtree():
        import sys
        sys.path.insert(0, "references/repos/modeltrees/src")
        from modeltrees import glmtree
        def run(xtr, ytr, xte):
            if len(classes) != 2:
                raise RuntimeError("modeltrees glmtree is binary-only")
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            cols = [f"x{j}" for j in range(p)]
            f = "y ~ " + " + ".join(cols) + " | " + " + ".join(cols)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            df_tr["y"] = list(ytr)
            t0 = time.perf_counter()
            m = glmtree(f, data=df_tr, family="binomial")
            resp = m.predict(newdata=df_te, type="response", unseen="null")
            lv = [str(s) for s in (m.response_levels or ())]
            if len(lv) != 2:
                raise RuntimeError(f"glmtree response_levels={m.response_levels!r}")
            rows = []
            for p_val in resp:
                p_val = float(p_val)
                row = [0.0] * len(classes)
                row[classes.index(lv[0])] = 1.0 - p_val
                row[classes.index(lv[1])] = p_val
                rows.append(row)
            return rows, time.perf_counter() - t0
        return run

    # 17. modeltrees multinomtree
    def mt_multinomtree():
        import sys
        sys.path.insert(0, "references/repos/modeltrees/src")
        from modeltrees import multinomtree
        def run(xtr, ytr, xte):
            p = len(xtr[0])
            cat_cols = get_cat_indices(p)
            cols = [f"x{j}" for j in range(p)]
            f = "y ~ " + " + ".join(cols) + " | " + " + ".join(cols)
            df_tr, df_te = plain_frames(xtr, xte, cat_cols)
            df_tr["y"] = list(ytr)
            t0 = time.perf_counter()
            m = multinomtree(f, data=df_tr)
            prob = m.predict(newdata=df_te, type="prob", unseen="null")
            cols_of = {str(lv): j for j, lv in enumerate(m.response_levels or ())}
            rows = [[float(prob[i][cols_of[c]]) for c in classes] for i in range(len(prob))]
            return rows, time.perf_counter() - t0
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
    attempt("sklearn LogReg", skl_logreg)
    attempt("KNN", knn)
    attempt("MARS (mars-earth)", mars)
    attempt("statsmodels MNLogit", sm_mnlogit)
    attempt("modeltrees glmtree", mt_glmtree)
    attempt("modeltrees multinomtree", mt_multinomtree)

    return ms
