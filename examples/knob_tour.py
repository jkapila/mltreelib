"""Key-knob tour — every growth knob with real, deterministic output.

Run:  uv run --with pandas --with pyarrow python examples/knob_tour.py

The knobs covered: leaf kinds (mean/median/quantile/huber/linear), the
criterion family (regression + classification), categorical search, the
split-precision grid, per-node column subsets, cost pruning, the Chow
significance gate (with its training-log notes), and monotone constraints.
Defaults are unchanged everywhere: each block flips exactly one knob.
"""

from mltreelib import ModelTree

# ---------------------------------------------------------------- helpers
# splitmix64: deterministic pseudo-noise, no RNG dependency
def noise(i):
    z = (i + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z % 1000) / 1000.0 - 0.5


# skewed target: 200 rows near 0, one outlier at 100
skew_x = [[i / 200.0] for i in range(201)]
skew_y = [100.0 if i == 200 else 2.0 + 0.01 * noise(i) for i in range(201)]

# planted regime: two means at x=0.5, tiny noise
reg_x = [[i / 200.0] for i in range(200)]
reg_y = [(5.0 if i / 200.0 >= 0.5 else 0.0) + 0.02 * noise(i) for i in range(200)]

# classification: two blobs at x<0.5 / x>=0.5
cls_y = [1 if i / 200.0 >= 0.5 else 0 for i in range(200)]

# categorical: level codes 0..4 with different means
cat_x = [[float(i % 5), (i % 20) / 20.0] for i in range(400)]
cat_y = [float(i % 5) + 0.1 * noise(i) for i in range(400)]

print("== leaf kinds (skewed target with one outlier) ==")
for leaf, kw in [
    ("mean", {}),
    ("median", {}),
    ("huber", {}),
    ("quantile", {"tau": 0.9}),
    ("linear", {"l2": 0.1}),
]:
    m = ModelTree(leaf=leaf, max_depth=2, min_rows=10, **kw)
    m.fit(skew_x, skew_y, dtypes=["numeric"])
    print(f"  {leaf:<9} leaves={m.n_leaves():<3} pred(x=0)={m.predict([[0.0]])[0]:8.4f}")

m = ModelTree(leaf="mean", max_depth=0, min_rows=10, leaf_digest=True)
m.fit(skew_x, skew_y, dtypes=["numeric"])
q = [m.quantile([[0.0]], t)[0] for t in (0.1, 0.5, 0.9)]
print(f"  digest    q10={q[0]:.3f} q50={q[1]:.3f} q90={q[2]:.3f} (any tau post-fit)")

print("== criterion family (planted regime at x=0.5) ==")
for gain in ["sse", "friedman_mse", "absolute_error", "huber"]:
    m = ModelTree(gain=gain, max_depth=3, min_rows=10, leaf="mean")
    m.fit(reg_x, reg_y, dtypes=["numeric"])
    print(f"  {gain:<14} leaves={m.n_leaves():<3} pred(0.1)={m.predict([[0.1]])[0]:.3f} "
          f"pred(0.9)={m.predict([[0.9]])[0]:.3f}")

print("== classification criteria (0/1 labels, same break) ==")
for gain in ["entropy", "gini", "gain_ratio", "twoing", "sdr"]:
    m = ModelTree(gain=gain, max_depth=3, min_rows=10, leaf="mean")
    m.fit(reg_x, cls_y, dtypes=["numeric"])
    acc = (sum(1 for p in m.predict(reg_x) if (p >= 0.5) == bool(y))
           for y in [1 if i / 200.0 >= 0.5 else 0 for i in range(200)])
    print(f"  {gain:<10} leaves={m.n_leaves():<3} train_acc={sum(acc) / 200:.3f}")

print("== categorical levels-as-is (level codes 0..4) ==")
for order, search in [("frequency_desc", "cumulative"), ("target_mean", "single_vs_rest")]:
    m = ModelTree(max_depth=3, min_rows=10, leaf="mean",
                  cat_order=order, cat_search=search)
    m.fit(cat_x, cat_y, dtypes=["categorical", "numeric"])
    print(f"  {order}/{search}: leaves={m.n_leaves():<3} splits={[(s['col'], round(s['threshold'], 2)) for s in m.splits()[:3]]}")

print("== split_precision grid (exact scan vs 0.1 grid) ==")
for prec in [None, 0.1]:
    m = ModelTree(max_depth=3, min_rows=10, leaf="mean", split_precision=prec)
    m.fit(reg_x, reg_y, dtypes=["numeric"])
    thr = [round(s['threshold'], 3) for s in m.splits()]
    print(f"  precision={str(prec):<5} thresholds={thr}")

print("== per-node column subsets (col_sample + seed) ==")
wide_x = [[noise(i * 3 + j) for j in range(10)] for i in range(300)]
wide_y = [2.0 * wide_x[i][3] + 0.1 * noise(i) for i in range(300)]
for frac, seed in [(None, 42), (0.3, 42), (0.3, 42), (0.3, 7)]:
    m = ModelTree(col_sample=frac, seed=seed, max_depth=3, min_rows=10)
    m.fit(wide_x, wide_y, dtypes=["numeric"] * 10)
    print(f"  col_sample={str(frac):<5} seed={seed} leaves={m.n_leaves():<3} "
          f"top_importance={m.feature_importance()[:1]}")

print("== cost pruning (prune=none|bic|mdl) ==")
for prune in ["none", "bic", "mdl"]:
    m = ModelTree(prune=prune, max_depth=4, min_rows=10, leaf="mean")
    m.fit(reg_x, reg_y, dtypes=["numeric"])
    print(f"  prune={prune:<4} leaves={m.n_leaves()}")

print("== Chow significance gate (split_test='chow') ==")
noise_x = [[(i % 20) / 20.0] for i in range(400)]
noise_y = [noise(i) for i in range(400)]
m = ModelTree(leaf="mean", max_depth=3, min_rows=10,
              split_test="chow", split_alpha=1e-6, verbose=2)
m.fit(noise_x, noise_y, dtypes=["numeric"])
print(f"  pure noise     -> leaves={m.n_leaves()} (gate must stop)")
print("   ", m.training_log()[-1])
m = ModelTree(leaf="mean", max_depth=3, min_rows=10,
              split_test="chow", split_alpha=0.05, bonferroni=True, verbose=2)
m.fit(reg_x, reg_y, dtypes=["numeric"])
print(f"  planted regime -> leaves={m.n_leaves()} (gate must pass)")
for line in m.training_log():
    print("   ", line)

print("== monotone constraint (y increases with x) ==")
for mono in [[], [1]]:
    m = ModelTree(monotone=mono, max_depth=3, min_rows=10, leaf="mean")
    m.fit(reg_x, reg_y, dtypes=["numeric"])
    p = [m.predict([[v]])[0] for v in (0.2, 0.5, 0.8)]
    print(f"  monotone={str(mono):<4} preds={p}")

print("== split_cols (split-search column subset; leaves stay full) ==")
for sc in [None, [1]]:
    m = ModelTree(split_cols=sc, max_depth=3, min_rows=10, l2=0.0)
    m.fit(cat_x, cat_y, dtypes=["numeric", "numeric"])
    cols = sorted({s['col'] for s in m.splits()})
    print(f"  split_cols={str(sc):<6} split_columns={cols} leaves={m.n_leaves()}")

print("== honesty (structure/estimation split; grf-style fairness) ==")
for hon, frac in [(False, 0.5), (True, 0.5), (True, 0.2)]:
    m = ModelTree(honesty=hon, honesty_fraction=frac, max_depth=3, min_rows=5)
    m.fit(reg_x, reg_y, dtypes=["numeric"])
    p = [round(m.predict([[v]])[0], 3) for v in (0.1, 0.9)]
    print(f"  honesty={hon!s:<5} frac={frac}: leaves={m.n_leaves()} preds={p}")

print("== NA surrogate splits (maxsurrogate; rpart lineage) ==")
sur_x = [[(None if i % 20 == 0 else i / 300.0), i / 300.0 + 0.2 * noise(i)]
         for i in range(300)]
sur_y = [5.0 if i / 300.0 >= 0.5 else 0.0 for i in range(300)]
for ms in [0, 1]:
    m = ModelTree(maxsurrogate=ms, max_depth=1, min_rows=10, leaf="mean")
    m.fit(sur_x, sur_y, dtypes=["numeric", "numeric"])
    p = [round(m.predict([[None, v]])[0], 2) for v in (0.1, 0.9)]
    print(f"  maxsurrogate={ms}: pred(x0=None, x1=0.1/0.9) = {p}")
