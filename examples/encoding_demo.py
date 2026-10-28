"""mltreelib categorical-encoding demo.

Every encoder in the package exercised on a real dataset (default: churn —
5,000 rows, 51 states, mixed cardinalities, binary churn target), with the
family split made explicit: unsupervised, supervised-both, classification,
regression. Figures render when matplotlib and seaborn are available (dev
extras — the library itself does not depend on them).

Usage:
    uv run python examples/encoding_demo.py               # default: churn
    uv run python examples/encoding_demo.py bank-marketing
"""

from __future__ import annotations

import sys
from pathlib import Path

from mltreelib import (
    BaseNEncoder,
    CatBoostEncoder,
    CountEncoder,
    EmbeddingEncoder,
    HashingEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimatorEncoder,
    OneHotEncoder,
    QuantileEncoder,
    SumEncoder,
    TargetEncoder,
    TfidfEncoder,
    WoEEncoder,
    list_datasets,
    load_dataset,
    summarize,
)

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def top_bottom(pairs, n=5):
    pairs = sorted(pairs.items(), key=lambda kv: kv[1])
    return pairs[:n], pairs[-n:][::-1]


def figures(leak=None, woe=None, out_dir: Path | None = None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\n(figures skipped: install with `uv run --with seaborn --with matplotlib`)")
        return
    sns.set_theme(style="whitegrid")
    out_dir = out_dir or (Path(__file__).resolve().parent / "out")
    out_dir.mkdir(exist_ok=True)

    if leak:
        plain, ordered = leak
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(plain, ".", alpha=0.5, label="TargetEncoder (leaky fitted mean)")
        ax.plot(ordered, ".", alpha=0.5, label="CatBoostEncoder (ordered TS)")
        ax.set_xlabel("row")
        ax.set_ylabel("encoded churn")
        ax.set_title("Same levels, different values: ordered statistics hide the row's own target")
        ax.legend()
        fig.tight_layout()
        path = out_dir / "encoding_leakage.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"figure: {path}")

    if woe:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        names, vals = zip(*sorted(woe.items(), key=lambda kv: kv[1]))
        sns.barplot(x=list(vals), y=list(names), ax=ax, color="steelblue")
        ax.set_xlabel("weight of evidence for churn (class 1)")
        ax.set_title("WoE by level: positive = over-represented among churners")
        fig.tight_layout()
        path = out_dir / "encoding_woe.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"figure: {path}")


def main(argv: list[str]) -> None:
    name = argv[0] if argv else "churn"
    if name not in set(list_datasets()):
        print(f"unknown or unavailable dataset '{name}'; run the data demo first "
              f"or see datasets/MANIFEST.md")
        return

    ds = load_dataset(name)
    cats = [c for c, t in ds["dtypes"].items() if t == "categorical"]
    nums = [c for c, t in ds["dtypes"].items() if t == "numeric"]
    target = "CustomerChurned" if "CustomerChurned" in cats else cats[-1]
    feats = [c for c in cats if c != target]
    y_raw = ds["columns"][target]
    levels = sorted({v for v in y_raw if v is not None})
    y = [None if v is None else 1.0 if v == levels[-1] else 0.0 for v in y_raw]

    section(f"1. DATASET: {name} — target '{target}' -> 1.0 = '{levels[-1]}'")
    print(f"rows x cols     : {ds['n_rows']} x {ds['n_cols']}")
    print(f"categorical feats: {feats}")
    print(f"numeric feats    : {nums[:6]}{' …' if len(nums) > 6 else ''}")
    churn = summarize(y)
    print(f"base rate       : {churn['mean']:.4f} ({int(churn['sum'])} of {churn['count']})")

    state = feats[0]
    sv = ds["columns"][state]
    n_levels = len({v for v in sv if v is not None})
    print(f"primary feature : '{state}' with {n_levels} levels")

    section("2. UNSUPERVISED ENCODERS (task: both)")
    ohe = OneHotEncoder()
    ohe.fit(sv)
    print(f"OneHot     : {len(ohe.levels())} indicator columns; row 0 -> "
          f"{ {lv: ohe.transform(sv[:1])[lv][0] for lv in list(ohe.levels())[:3]} }")
    he = HashingEncoder(16)
    cols = he.transform(sv)
    print(f"Hashing    : {len(cols)} buckets; row 0 bucket = "
          f"{[k for k, v in cols.items() if v[0] == 1.0]}")
    bn = BaseNEncoder(2)
    bn.fit(sv)
    digits = bn.transform(sv)
    print(f"BaseN      : {len(digits)} binary digits for {n_levels} levels")
    ce = CountEncoder(False)
    ce.fit(sv)
    counts = ce.transform(sv)
    print(f"Count      : row 0 level occurs {fmt(counts[0])} times; "
          f"unseen -> {fmt(ce.transform(['___never___'])[0])}")
    sm = SumEncoder()
    sm.fit(sv)
    row0 = sm.transform(sv[:1])[0]
    print(f"Sum        : {len(row0)} contrast columns; row 0 -> "
          f"{[fmt(x) for x in row0[:5]]}{' …' if len(row0) > 5 else ''}")

    section("3. SUPERVISED ENCODERS (task: both) on " + state)
    te = TargetEncoder(10.0)
    te.fit(sv, y)
    loo = LeaveOneOutEncoder()
    loo_rows = loo.fit_transform(sv, y)
    cb = CatBoostEncoder(1.0, 4, 42)
    cb_rows = cb.fit_transform(sv, y)
    plain = te.transform(sv)
    per_level = {}
    for s_, p in zip(sv, plain):
        if s_ is not None:
            per_level[s_] = p
    low, high = top_bottom(per_level)
    print("TargetEncoder (smoothing=10) churn estimate, lowest levels : "
          + ", ".join(f"{k}={fmt(v)}" for k, v in low))
    print("TargetEncoder churn estimate, highest levels: "
          + ", ".join(f"{k}={fmt(v)}" for k, v in high))
    print(f"LeaveOneOut fit_transform (leakage-aware) rows 0..4: {[fmt(v) for v in loo_rows[:5]]}")
    print(f"CatBoost ordered TS      rows 0..4           : {[fmt(v) for v in cb_rows[:5]]}")
    a_state = next(s for s in sv if s is not None)
    same = {round(v, 6) for s_, v in zip(sv, cb_rows) if s_ == a_state and v is not None}
    print(f"ordered TS values for level '{a_state}': {len(same)} distinct per-row values "
          f"(plain TargetEncoder gives exactly 1 — that difference is the leakage guard)")
    me = MEstimatorEncoder(20.0)
    me.fit(sv, y)
    print(f"M-estimate (m=20) row 0: {fmt(me.transform(sv[:1])[0])} (shrunk harder toward {fmt(te.prior())})")

    section("4. CLASSIFICATION FAMILY: class-space vectors")
    tf = TfidfEncoder(True)
    tf.fit(sv, y, 2)
    vecs = tf.transform(sv[:1] + [None])
    print(f"Tfidf('{state}', row 0, L2-normalized): {[fmt(x) for x in vecs[0]]}")
    print(f"Tfidf missing row -> {vecs[1]}")
    woe = WoEEncoder(1.0)
    woe.fit(sv, y, 2)
    woe_rows = woe.transform(sv)
    per_state_woe = {}
    for s_, v in zip(sv, woe_rows):
        if s_ is not None and v is not None:
            per_state_woe[s_] = v[1]
    low, high = top_bottom(per_state_woe)
    print("WoE for churn, protective (negative) states: "
          + ", ".join(f"{k}={fmt(v)}" for k, v in low))
    print("WoE for churn, risky (positive) states     : "
          + ", ".join(f"{k}={fmt(v)}" for k, v in high))

    emb = EmbeddingEncoder(8, 0.2, 60, 7, "classification")
    emb.fit(sv, y, 2)
    preds = emb.predict(sv)
    hits = sum(1 for p, t in zip(preds, y) if p is not None and t is not None and p == t)
    observed = sum(1 for t in y if t is not None)
    print(f"Embedding (dim=8) churn accuracy on train: {hits / observed:.3f} "
          f"(base rate {churn['mean']:.3f})")
    print(f"Embedding for '{a_state}' -> {len(emb.transform([a_state])[0])}-d vector")

    section("5. REGRESSION FAMILY (task: regression) on account_length")
    y_num = ds["columns"][nums[0]]
    key = feats[1] if len(feats) > 1 else state
    kv = ds["columns"][key]
    qe = QuantileEncoder(0.5, 2.0)
    qe.fit(kv, y_num)
    print(f"Quantile (median, smoothing=2) by '{key}', row 0: {fmt(qe.transform(kv[:1])[0])}")
    reg = EmbeddingEncoder(4, 0.05, 80, 11, "regression")
    reg.fit(kv, y_num)
    rpreds = reg.predict(kv[:5])
    print(f"Embedding regression on '{nums[0]}', rows 0..4: {[fmt(v) for v in rpreds]}")
    obs = [v for v in y_num if v is not None]
    print(f"targets all non-negative -> ReLU output head "
          f"(min target {fmt(min(obs))}); negative targets would switch to linear")

    print("\nNA note: missing values pass through every encoder as missing; "
          "unseen levels use each encoder's documented fallback.")
    figures(leak=(plain, cb_rows), woe=per_state_woe)
    section("DONE — every encoder exercised on real data")


if __name__ == "__main__":
    main(sys.argv[1:])
