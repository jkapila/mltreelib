"""Reproduce the held-out case studies in the practical guides and blog."""
import argparse
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mltreelib import ClusterModel, MetricsReport, ModelTree, plots


matplotlib.rcParams["svg.hashsalt"] = "mltreelib-real-tree-studies"

DATA = Path("datasets")
OUT = Path("docs/assets/guides")


def load(name, openml_id):
    path = DATA / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Save OpenML {openml_id}'s Parquet file to {path}")
    return pd.read_parquet(path)


def rows(frame, features):
    return [[None if pd.isna(value) else value for value in row]
            for row in frame[features].itertuples(index=False, name=None)]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def diagram(model, name):
    if dot := shutil.which("dot"):
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run([dot, "-Tsvg", "-o", str(OUT / f"{name}.svg")],
                       input=model.to_dot(), text=True, check=True)


def wine():
    df = load("wine_quality", 46964).sample(frac=1, random_state=2026)
    features = ["residual_sugar", "volatile_acidity", "alcohol", "chlorides"]
    train, refit, review = df.iloc[:2500], df.iloc[2500:5000], df.iloc[5000:].copy()
    center, scale = train[features].mean(), train[features].std()

    def design(frame):
        return ((frame[features] - center) / scale).to_numpy().tolist()

    def fit(frame):
        tree = ClusterModel(kind="pct", max_depth=2, min_rows=150)
        tree.fit(design(frame), col_names=features)
        return tree

    tree = fit(train)
    review["segment"] = tree.predict(design(review))
    other = fit(refit).predict(design(review))
    counts = pd.crosstab(review["segment"], other)
    # Jaccard agreement on co-assigned pairs excludes pairs separated by both fits.
    def pairs(n):
        return (n * (n - 1) // 2).sum()

    shared = pairs(counts.to_numpy().ravel())
    union = pairs(counts.sum(axis=1)) + pairs(counts.sum(axis=0)) - shared
    print("Wine pair Jaccard:", shared / union if union else float("nan"))
    print(review.groupby("segment")[features].agg(["size", "median"]).round(3).to_string())
    print("Wine rules:", tree.rules())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for segment, group in review.groupby("segment"):
        axes[0].scatter(group.residual_sugar, group.volatile_acidity, s=12, alpha=.45,
                        label=f"Leaf {segment} ({len(group)})")
    axes[0].set(xlabel="Residual sugar (g/L)", ylabel="Volatile acidity (g/L)")
    axes[0].legend()
    profiles = (review.groupby("segment")[features].median() - center) / scale
    profiles.T.plot.barh(ax=axes[1])
    axes[1].axvline(0, color="grey", linewidth=.8)
    axes[1].set(xlabel="Held-out median in training SD units")
    axes[1].legend(title="Leaf")
    save(fig, "real-wine-clustering")


def binary_study(dataset, openml_id, target, positive, name, features=None):
    df = load(dataset, openml_id).sample(frac=1, random_state=2026).iloc[:12000]
    train, validation, test = df.iloc[:8000], df.iloc[8000:10000], df.iloc[10000:]
    if features is None:
        features = [column for column in df if column != target]
    xtr, xval, xte = (rows(part, features) for part in (train, validation, test))
    ytr, yval, yte = [(part[target] == positive).astype(int).tolist()
                     for part in (train, validation, test)]
    model = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance",
                      max_depth=3, min_rows=100, prune="none")
    model.fit(xtr, ytr, col_names=features, dtypes=["numeric"] * len(features))
    val_scores = model.predict_proba(xval)
    scores = model.predict_proba(xte)

    def cost(threshold):
        return sum(5 if y and p < threshold else 1 if not y and p >= threshold else 0
                   for y, p in zip(yval, val_scores))

    threshold = min([.02, .05, .10, .20, .40, 1.01], key=cost)
    decisions = [int(value >= threshold) for value in scores]
    prevalence = mean(ytr)
    rep = MetricsReport("classification", y_true=yte, y_pred=decisions, proba=scores)
    base_rep = MetricsReport("classification", y_true=yte, proba=[prevalence] * len(yte))
    print(name, "positives", sum(ytr), sum(yval), sum(yte), "threshold", threshold)
    print("missing", train[features].isna().sum().nlargest(4).to_dict())
    print("test confusion", rep.tables["confusion"])
    print("AP, prevalence", rep["average_precision"], mean(yte))
    print("Brier, baseline", rep["brier"], base_rep["brier"])
    print("example split", model.splits()[:1], "leaf counts", [leaf["n"] for leaf in model.leaves()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plots.plot_pr(yte, scores, ax=axes[0])
    axes[0].axhline(mean(yte), linestyle="--", color="gray", label="Test prevalence")
    axes[0].set_title("Precision vs. recall")
    axes[0].legend()
    if name == "real-aps":
        missing = test[features].isna().any(axis=1).to_numpy()
        subset = pd.DataFrame({"missing": missing, "event": yte, "score": scores})
        print(subset.groupby("missing").agg(rows=("event", "size"), positives=("event", "sum"),
                                               observed=("event", "mean"), predicted=("score", "mean")))
        subset.groupby("missing")[["event", "score"]].mean().rename(
            index={False: "Complete", True: "Missing"},
            columns={"event": "Observed rate", "score": "Mean score"},
        ).plot.bar(ax=axes[1], rot=0)
        axes[1].set(xlabel="Test sensor status", ylabel="Rate", title="Missingness slice")
    else:
        plots.plot_calibration(yte, scores, n_bins=8, ax=axes[1])
        axes[1].set_title("Held-out probability diagnostic")
    save(fig, name)
    if name == "real-credit":
        diagram(model, "real-credit-tree")


def continuous_study(dataset, openml_id, target, configurations, name, features=None):
    df = load(dataset, openml_id).sample(frac=1, random_state=2026)
    if features is None:
        features = df.columns.drop(target).tolist()
    n_train = min(6000, int(.8 * len(df)))
    train, test = df.iloc[:n_train], df.iloc[n_train:n_train + 2000]
    xtr, xte = rows(train, features), rows(test, features)
    ytr, yte = train[target].tolist(), test[target].tolist()
    dtypes = ["numeric" if pd.api.types.is_numeric_dtype(train[c]) else "categorical"
              for c in features]
    results = {}
    structures = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, options in configurations.items():
        model = ModelTree(max_depth=2, min_rows=40, prune="none", **options)
        model.fit(xtr, ytr, col_names=features, dtypes=dtypes)
        predicted = model.predict(xte)
        results[label] = MetricsReport("regression", y_true=yte, y_pred=predicted)["rmse"]
        structures[label] = model.splits()[:1], model.leaves()[0]
        axes[1].scatter(yte, np.asarray(predicted) - yte, s=10, alpha=.35, label=label,
                        rasterized=True)
    base = MetricsReport("regression", y_true=yte, y_pred=[mean(ytr)] * len(yte))
    print(name, len(train), len(test), "RMSE", results, "mean baseline", base["rmse"])
    for label, (split, leaf) in structures.items():
        print(label, "example split", split, "first leaf support", leaf["n"],
              "columns", leaf.get("cols"), "coefficients", leaf.get("coef"))
    axes[0].bar(list(results), list(results.values()), color=["#226b8a", "#c86729"])
    axes[0].set(ylabel="Test RMSE", title="Held-out comparison")
    axes[1].axhline(0, color="gray")
    axes[1].set(xlabel=f"Observed {target}", ylabel="Prediction minus observed",
                title="Test errors by outcome")
    axes[1].legend()
    save(fig, name)
    if name == "real-concrete-leaves":
        diagram(model, "real-concrete-tree")


def credit():
    binary_study("GiveMeSomeCredit", 46929, "FinancialDistressNextTwoYears", "Yes", "real-credit")


def aps():
    binary_study("APSFailure", 46908, "AirPressureSystemFailure", "pos", "real-aps",
                 ["aa_000", "ab_000", "ac_000", "ad_000", "ae_000", "af_000",
                  "ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005"])


def diamonds():
    continuous_study("diamonds", 46923, "price",
                     {"Mean leaf": {"leaf": "mean"}, "Linear leaf": {"leaf": "linear"}},
                     "real-diamonds-leaves")


def concrete():
    continuous_study("concrete_compressive_strength", 46917, "ConcreteCompressiveStrength",
                     {"Mean leaf": {"leaf": "mean"}, "Linear leaf": {"leaf": "linear"}},
                     "real-concrete-leaves")


def housing():
    continuous_study("miami_housing", 46942, "SALE_PRC",
                     {"Squared error": {"gain": "sse", "leaf": "mean"},
                      "Tweedie": {"gain": "tweedie", "tweedie_p": 1.5, "leaf": "mean"}},
                     "real-housing-splits",
                     ["LND_SQFOOT", "TOT_LVG_AREA", "age", "OCEAN_DIST", "structure_quality"])


if __name__ == "__main__":
    studies = {fn.__name__: fn for fn in (wine, credit, aps, diamonds, concrete, housing)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", choices=list(studies))
    for name in parser.parse_args().studies:
        studies[name]()
