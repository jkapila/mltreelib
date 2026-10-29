"""mltreelib data & feature layer demo.

One script, real data, all the details: ingest and missingness, category
encoding, grouped/trended/windowed/decomposed features, the generator's ground
truth, and the drop-and-warn NA behavior. Figures are rendered when matplotlib
and seaborn are available (they are dev-only extras — the library itself does
not depend on them).

Usage:
    uv run python examples/data_layer_demo.py                       # default: churn
    uv run python examples/data_layer_demo.py diamonds wine_quality
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

from mltreelib import (
    DatasetBuilder,
    GroupedScaler,
    IndexMapper,
    calendar_features,
    decomposed_transform,
    generate,
    grouped_transform,
    lag_by_days,
    list_datasets,
    load_dataset,
    summarize,
    trended_transform,
    windowed_transform,
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


# ---------------------------------------------------------------- figures ---
def figures(variance=None, panel=None, out_dir: Path | None = None):
    """Render demo figures via mltreelib.plots (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(figures skipped: install with `uv run --with matplotlib`)")
        return
    from mltreelib import plots as mplots

    out_dir = out_dir or (Path(__file__).resolve().parent / "out")
    out_dir.mkdir(exist_ok=True)

    if variance:
        shares = {k: variance[k] for k in
                  ("numeric", "categorical", "hierarchy", "interaction", "noise")}
        ax = mplots.plot_variance_shares(shares)
        ax.figure.tight_layout()
        path = out_dir / "variance_shares.png"
        ax.figure.savefig(path, dpi=120)
        plt.close(ax.figure)
        print(f"figure: {path}")

    if panel:
        dates, values, trend, detrended, deseason, residual = panel
        axes = mplots.plot_decomposition(values, trend, detrended, deseason,
                                         residual, dates=dates)
        axes[0][0].figure.suptitle("Additive decomposition (statsmodels convention)")
        axes[0][0].figure.tight_layout()
        path = out_dir / "decomposition.png"
        axes[0][0].figure.savefig(path, dpi=120)
        plt.close(axes[0][0].figure)
        print(f"figure: {path}")


# ------------------------------------------------------- 1. real dataset ---
def dataset_report(name: str) -> None:
    section(f"1. REAL DATASET: {name}")
    ds = load_dataset(name)
    print(f"rows x cols        : {ds['n_rows']} x {ds['n_cols']}")
    by_type: dict[str, list[str]] = {}
    for col, t in ds["dtypes"].items():
        by_type.setdefault(t, []).append(col)
    for t, cols in sorted(by_type.items()):
        more = f" …(+{len(cols) - 8})" if len(cols) > 8 else ""
        print(f"{t + ' columns':17s}: {len(cols)} -> {', '.join(cols[:8])}{more}")
    na = {k: v for k, v in ds["na_counts"].items() if v}
    print(f"missing (na_counts): {na if na else 'none'}")

    numerics = by_type.get("numeric", [])
    categoricals = by_type.get("categorical", [])
    if numerics:
        print("\nnumeric summaries (count, n_missing, mean, std, min, q1, median, q3, max):")
        for col in numerics[:4]:
            s = summarize(ds["columns"][col])
            stats = " ".join(f"{s[k]:>10.4g}" for k in ("mean", "std", "min", "q1", "median", "q3", "max"))
            print(f"  {col[:28]:28s} {s['count']:6d} {s['n_missing']:5d}  {stats}")

    # ingest through the data layer with category inference
    b = DatasetBuilder(n_category=0.05, na_policy="allow")
    for col, values in ds["columns"].items():
        if ds["dtypes"][col] == "categorical":
            b.add_categorical(col, values)
        else:
            b.add_numeric(col, values)
    built = b.build()
    inferred = sorted(set(built["dtypes"]) - set(ds["columns"]) | {c for c, t in built["dtypes"].items() if t == "categorical" and ds["dtypes"].get(c) == "numeric"})
    print(f"\nDatasetBuilder(n_category=5% of rows): integral numerics promoted to categorical: {inferred or 'none'}")
    print(f"na_counts after ingest: { {k: v for k, v in built['na_counts'].items() if v} or 'none'}")

    if categoricals:
        col = categoricals[0]
        m = IndexMapper()
        m.fit(col, ds["columns"][col])
        names = m.get_all_names(col)
        sample = ", ".join(f"{n}->{m.get_index(col, n)}" for n in names[:5])
        print(f"\nIndexMapper[{col}]: {len(names)} levels; sample: {sample}")

    if categoricals and numerics:
        key, tgt = categoricals[0], numerics[0]
        gmean = grouped_transform(ds["columns"][key], ds["columns"][tgt], kind="mean")
        zs = GroupedScaler()
        zs.fit(ds["columns"][key], ds["columns"][tgt], "zscore")
        z = zs.transform(ds["columns"][key], ds["columns"][tgt])
        back = zs.inverse_transform(ds["columns"][key], z)
        print(f"\ngroup mean of {tgt} by {key} (first 5): {[fmt(v) for v in gmean[:5]]}")
        print(f"group zscore of {tgt} by {key} (first 5): {[fmt(v) for v in z[:5]]}")
        print(f"zscore -> inverse roundtrip ok       : {all((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9) for a, b in zip(back, ds['columns'][tgt]))}")


# ------------------------------------------------------ 2. generator ---
def generator_report() -> dict:
    section("2. SYNTHETIC GENERATOR WITH GROUND TRUTH")
    g = generate(
        n_rows=20_000, n_numeric=4, cardinalities=[8, 64], hierarchy_levels=2,
        branching=[4, 12], group_effect_strength=0.6, hierarchy_effect_strength=0.4,
        interaction_strength=0.2, noise=0.2, seed=7,
    )
    v = g["truth"]["variance"]
    print(f"rows: {len(g['target'])}, columns: {len(g['columns'])}")
    print("realized variance share of each component:")
    for k in ("numeric", "categorical", "hierarchy", "interaction", "noise"):
        print(f"  {k:13s} {v[k]:.3f}")
    print(f"  {'(total var)':13s} {v['total']:.3f}")
    cat0 = g["truth"]["categorical_effects"]["cat_0"][:4]
    print(f"cat_0 true level effects (first 4): {[round(x, 3) for x in cat0]}")
    return v


# ---------------------------------------------------- 3. time features ---
def time_report():
    section("3. TIME FEATURES ON A SYNTHETIC PANEL (NA handling included)")
    keys, dates, vals = [], [], []
    for k in ("store_a", "store_b"):
        for i in range(21):
            keys.append(k)
            dates.append((date(2026, 1, 5) + timedelta(days=i)).isoformat())
            base = 10.0 + 0.3 * i + (2.0 if i % 7 < 5 else -5.0)
            vals.append(base + (1.0 if k == "store_b" else 0.0))

    cal = calendar_features(dates[:7])
    print(f"dates           : {dates[0]} .. {dates[-1]} (2 stores x 21 days)")
    for f in ("weekday", "weekend", "season"):
        print(f"  {f:14s}: {[fmt(x) for x in cal[f][:7]]}")

    print(f"\nlag 7 days (first 8)          : {[fmt(v) for v in lag_by_days(keys, dates, vals, 7)[:8]]}")
    print(f"year_month mean (first 5)     : {[fmt(v) for v in trended_transform(keys, dates, vals)[:5]]}")
    w = windowed_transform(keys, dates, vals, window_days=7, kind="sum")
    print(f"7-day trailing sum (first 10) : {[fmt(v) for v in w[:10]]}")

    d = decomposed_transform(keys, dates, vals, season_granularity="weekday", trend_window=7)
    print("\ndecomposition (store_a, rows 5..12):")
    print(f"  {'value':>7} {'detrend':>9} {'deseason':>9} {'residual':>9}")
    for i in range(5, 13):
        print(f"  {vals[i]:7.2f} {fmt(d['detrended'][i]):>9} {fmt(d['deseasonalized'][i]):>9} {fmt(d['residual'][i]):>9}")

    print("\nNA handling (drop + warn):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = lag_by_days(keys[:3] + [None], dates[:3] + ["2026-01-30"], vals[:3] + [1.0], 7)
    for w_ in caught:
        print(f"  warning: {w_.message}")
    print(f"  output : {out}  (incomplete rows -> missing)")

    trend = [vals[i] - (d["detrended"][i] or vals[i]) for i in range(21)]
    return (dates[:21], vals[:21], trend, d["detrended"][:21], d["deseasonalized"][:21], d["residual"][:21])


def main(argv: list[str]) -> None:
    names = argv or ["churn"]
    available = set(list_datasets())
    for name in names:
        if name not in available:
            print(f"unknown dataset '{name}'; available: {sorted(available)[:12]} …")
            continue
        dataset_report(name)
    variance = generator_report()
    panel = time_report()
    figures(variance=variance, panel=panel)
    section("DONE — details printed above cover ingest, missingness, encodings, "
            "grouped/trended/windowed/decomposed features and ground truth")


if __name__ == "__main__":
    main(sys.argv[1:])
