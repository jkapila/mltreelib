"""Local speed run — the fastest ModelTree configurations, wall-clock.

Run:  uv run python examples/speed_run.py

Fits each configuration on every prepared benchmark split (no Elo
machinery, plain timing) and reports fit/predict wall time, RMSE and leaf
count. The speed levers, fastest first: leaf="mean" (rpart-style O(1)
running-sum scan), a coarse `split_precision` grid (fewer candidates),
`leaf_digest=False` (lean leaves). The default linear-leaf model rides
along for reference.
"""

import csv
import glob
import time

from mltreelib import ModelTree

CONFIGS = {
    "mean  grid=0.01 lean": dict(
        leaf="mean", max_depth=4, split_precision=0.01, leaf_digest=False
    ),
    "mean  grid=0.001 lean": dict(
        leaf="mean", max_depth=4, split_precision=0.001, leaf_digest=False
    ),
    "linear grid=0.001": dict(max_depth=4),
    "linear exact": dict(max_depth=4, split_precision=None),
}


def load(ds):
    """Prepared split -> typed rows. Categorical strings enter as LEVELS
    through the library's own IndexMapper (identity mapping, exact
    inverse) -- the tree never sees an encoded value: a distinct value IS
    a level and the split search partitions level sets."""
    from mltreelib import IndexMapper

    def read(part):
        with open(f"benchmarks/elo_data/{ds}_{part}.csv") as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1:]

    cols, tr = read("train")
    _, te = read("test")
    ycol = cols.index("y")
    xi = [j for j in range(len(cols)) if j != ycol]
    kinds = []
    mappers = []
    for j in xi:
        vals = [r[j] for r in tr]
        is_num = all(v == "" or _isfloat(v) for v in vals)
        if is_num:
            kinds.append("numeric")
            mappers.append(None)
        else:
            # levels-as-is: IndexMapper codes are identity, not order
            im = IndexMapper()
            im.fit(cols[j], [v if v != "" else None for v in vals])
            kinds.append("categorical")
            mappers.append(im)

    def conv(rows):
        out = []
        for r in rows:
            row = []
            for j, im in zip(xi, mappers):
                v = r[j]
                if im is None:
                    row.append(None if v == "" else float(v))
                else:
                    code = im.transform(cols[xi.index(j)] if False else "c", [v if v != "" else None])[0]
                    row.append(code)
            out.append(row)
        return out

    def yv(rows):
        return [float(r[ycol]) for r in rows]

    return conv(tr), yv(tr), conv(te), yv(te), kinds


def _isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    ds_names = sorted(
        p.split("/")[-1][: -len("_train.csv")]
        for p in glob.glob("benchmarks/elo_data/*_train.csv")
    )
    print(f"{'dataset':<22}{'config':<24}{'fit s':>8}{'pred s':>8}{'rmse':>10}{'leaves':>7}")
    totals = {c: 0.0 for c in CONFIGS}
    for ds in ds_names:
        xtr, ytr, xte, yte, kinds = load(ds)
        for name, kw in CONFIGS.items():
            m = ModelTree(min_rows=10, **kw)
            t0 = time.perf_counter()
            m.fit(xtr, ytr, dtypes=kinds)
            t1 = time.perf_counter()
            pred = m.predict(xte)
            t2 = time.perf_counter()
            rmse = (sum((p - y) ** 2 for p, y in zip(pred, yte)) / len(yte)) ** 0.5
            totals[name] += t1 - t0
            print(
                f"{ds:<22}{name:<24}{t1 - t0:>8.3f}{t2 - t1:>8.4f}"
                f"{rmse:>10.4f}{m.n_leaves():>7}"
            )
    print("\ntotal fit seconds:")
    for name, s in sorted(totals.items(), key=lambda kv: kv[1]):
        print(f"  {name:<24}{s:>8.2f}")


if __name__ == "__main__":
    main()
