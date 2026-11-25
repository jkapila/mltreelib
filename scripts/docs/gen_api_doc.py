"""Generate the per-class API reference from the built module's docstrings.

    uv run python scripts/docs/gen_api_doc.py

Writes one page per class group under docs/reference/api/. Signatures and
docstrings come from the built module. Regenerate the reference after API changes.
"""

import importlib
import inspect
from pathlib import Path

import mltreelib
import mltreelib.data as data_helpers

OUT = Path("docs/reference/api")

# Public names of mltreelib.data, listed on the Data page next to the top-level ones.
DATA_HELPER_CLASSES = [
    n for n in data_helpers.__all__ if inspect.isclass(getattr(data_helpers, n))
]
DATA_HELPER_FUNCTIONS = [
    n for n in data_helpers.__all__ if inspect.isfunction(getattr(data_helpers, n))
]
# Extra modules whose public names are merged into a page (top-level names win).
EXTRA_MODULES = {"data.md": ["mltreelib.data"]}

ENCODERS = [
    "BaseNEncoder",
    "CatBoostEncoder",
    "CountEncoder",
    "EmbeddingEncoder",
    "HashingEncoder",
    "JamesSteinEncoder",
    "LeaveOneOutEncoder",
    "MEstimatorEncoder",
    "OneHotEncoder",
    "QuantileEncoder",
    "SumEncoder",
    "TargetEncoder",
    "TfidfEncoder",
    "WoEEncoder",
]

PAGES = [
    (
        "data.md",
        "Data",
        ["DatasetBuilder"] + DATA_HELPER_CLASSES,
        ["generate", "load_dataset", "list_datasets", "summarize", "winnow"]
        + DATA_HELPER_FUNCTIONS,
        (
            "Typed ingest, the synthetic generator, dataset helpers, column "
            "screening, and the `mltreelib.data` helpers for profiling, splitting, "
            "association, binning and scaling. Guides: [Data](../data.md), "
            "[Prepare data](../../guides/data.md)."
        ),
    ),
    (
        "model-tree.md",
        "ModelTree",
        ["ModelTree", "GrowthControl", "LeafControl"],
        [],
        (
            "The model tree: typed fit, split-search knobs, quantiles, training "
            "log. Guide: [Model](../model.md)."
        ),
    ),
    (
        "metrics.md",
        "Metrics",
        ["Metrics"],
        [],
        (
            "Complete evaluation reports at once. Guides: [Metrics](../metrics.md), "
            "[Evaluation reports](../reports.md)."
        ),
    ),
    (
        "encoders.md",
        "Encoders & scalers",
        ["IndexMapper", "VariableScaler", "GroupedScaler"] + ENCODERS,
        [],
        (
            "Pipeline tools for feeding downstream algorithms (sklearn and "
            "friends) — never invoked by the tree. Guide: [Data](../data.md)."
        ),
    ),
    (
        "features.md",
        "Feature functions",
        [],
        [
            "grouped_scale",
            "grouped_transform",
            "trended_transform",
            "windowed_transform",
            "lag_by_days",
            "decomposed_transform",
            "calendar_features",
        ],
        (
            "Grouped/trended/windowed transforms, lags and calendar parts. Guide: "
            "[Data processing](../../examples/data-processing.md)."
        ),
    ),
    (
        "cluster.md",
        "Clustering trees",
        [
            "ClusterModel",
            "ClusterResult",
            "ClusterTree",
            "TrieKNN",
            "PredictiveClusteringTree",
            "CLTree",
            "HDBSCAN",
            "UnsupervisedRandomForest",
        ],
        [],
        (
            "Unified clustering models and mixed-type search: ClusterModel, ClusterTree, "
            "TrieKNN, PCT (CLUS), CLTree, HDBSCAN Condensed Tree, and Unsupervised Random Forest (URF)."
        ),
    ),
    (
        "anomaly.md",
        "Anomaly & density trees",
        [
            "AnomalyModel",
            "AnomalyResult",
            "AnomalyTree",
            "GritBot",
            "DensityEstimationTree",
            "ExtendedIsolationTree",
            "HalfSpaceTree",
            "RandomCutTree",
            "ActiveAnomalyTree",
            "ContextAnomalyTree",
        ],
        [],
        (
            "Unified anomaly detection: AnomalyModel (tree-based and PCA/NMF reconstruction errors), "
            "density estimation, streaming mass scoring, CoDisp, and contextual surprise."
        ),
    ),
    (
        "components.md",
        "Component & Factor Analysis",
        [
            "ComponentModel",
            "ComponentResult",
        ],
        [],
        (
            "Consolidated component analysis: PCA with R-aligned factor rotations (Varimax, Quartimax, Promax), "
            "Truncated SVD, Whitening (ZCA/PCA), FastICA, and Non-negative Matrix Factorization (MU, ALS, HALS with L1/L2 penalties)."
        ),
    ),
    (
        "rules.md",
        "Rule-Set Models",
        [
            "RuleModel",
            "RuleResult",
            "RuleSet",
            "Rule",
        ],
        [],
        (
            "Quinlan C4.5 / C5.0 rule extraction, condition interval simplification, and pessimistic error rate pruning."
        ),
    ),
    (
        "report.md",
        "Model Reports & Diagnostics",
        [
            "ModelReport",
        ],
        [],
        (
            "Unified 4-section ModelReport representation: Objective, Learner, Controller, Diagnostics, and optional Inference."
        ),
    ),
]

# Pages sourced from a submodule instead of the top-level package.
SUBMODULE_PAGES = [
    (
        "solvers.md",
        "Solvers & optimization",
        "mltreelib.solvers",
        ["FitResult", "OptimizationResult"],
        ["fit", "optimize"],
        (
            "Standalone weighted fitting and the generic bounded optimizer. "
            "Guide: [Generic optimizer](../../examples/generic-optimizer.md)."
        ),
    ),
]

HEADER = "<!-- generated by scripts/docs/gen_api_doc.py from the built module's docstrings - do not edit -->\n"


def sig(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(...)"


def emit_class(buf, name, cls):
    buf.append(f"## {name}\n\n")
    buf.append(f"```python\nclass {name}{sig(cls)}\n```\n\n")
    doc = inspect.getdoc(cls)
    if doc:
        buf.append(doc + "\n\n")
    for mname, m in inspect.getmembers(
        cls, lambda o: callable(o) and not isinstance(o, type)
    ):
        if mname.startswith("_"):
            continue
        buf.append(f"### {name}.{mname}\n\n")
        buf.append(f"```python\n{mname}{sig(m)}\n```\n\n")
        mdoc = inspect.getdoc(m)
        if mdoc:
            buf.append(mdoc + "\n\n")

    for pname, prop in inspect.getmembers(cls, lambda o: isinstance(o, property)):
        if pname.startswith("_"):
            continue
        buf.append(f"### {name}.{pname}\n\n")
        buf.append(f"```python\n@{pname}\n```\n\n")
        pdoc = inspect.getdoc(prop)
        if pdoc:
            buf.append(pdoc + "\n\n")


def emit_fn(buf, name, fn):
    buf.append(f"## {name}\n\n")
    buf.append(f"```python\n{name}{sig(fn)}\n```\n\n")
    doc = inspect.getdoc(fn)
    if doc:
        buf.append(doc + "\n\n")


CORE_GENERIC = Path("crates/mltreelib-core/src/solver/generic")

RUST_PAGE = "rust-generic.md"

RUST_BLURB = (
    "The Rust core owns the shared optimization seam. Python callers reach it "
    "through `mltreelib.solvers.optimize`. Guide: "
    "[Generic optimizer](../../examples/generic-optimizer.md)."
)


def clean_doc(lines):
    """Join doc-comment lines into paragraphs."""
    paragraphs = []
    current = []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def method_entries(lines, header_index, type_name, prefix):
    """Collect documented methods inside one trait or impl block."""
    entries = []
    doc = []
    depth = 1
    index = header_index + 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("///"):
            doc.append(stripped[3:].strip())
            index += 1
            continue
        indent = len(line) - len(line.lstrip())
        if depth == 1 and indent == 4 and stripped.startswith(prefix):
            signature = stripped
            while not (signature.rstrip().endswith(";") or signature.rstrip().endswith("{")):
                index += 1
                if index >= len(lines):
                    break
                signature += " " + lines[index].strip()
            name = (
                signature.split("(")[0]
                .replace("pub fn ", "")
                .replace("fn ", "")
                .strip()
            )
            entries.append((f"{type_name}.{name}", signature, clean_doc(doc)))
            doc = []
            index += 1
            continue
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break
        if stripped and not stripped.startswith("//"):
            doc = []
        index += 1
    return entries


def rust_items(path):
    """Extract public items with their doc comments from one Rust source file."""
    lines = path.read_text().splitlines()
    entries = []
    doc = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("///"):
            doc.append(stripped[3:].strip())
            index += 1
            continue
        if stripped.startswith("#["):
            index += 1
            continue
        if line.startswith(("pub ", "impl ")):
            header = stripped
            while not header.rstrip().endswith(("{", ";", "}")):
                index += 1
                if index >= len(lines):
                    break
                header += " " + lines[index].strip()
            parts = header.split()
            if parts[0] == "impl":
                if " for " in header:
                    type_name = (
                        header.split(" for ")[1].split("<")[0].rstrip("{").rstrip(":").strip()
                    )
                else:
                    type_name = parts[1].split("<")[0].rstrip(":").strip()
                for name, signature, method_doc in method_entries(
                    lines, index, type_name, "pub fn "
                ):
                    entries.append((name, signature, method_doc))
                doc = []
                index += 1
                continue
            kind = parts[1]
            if kind in {"use", "mod"}:
                doc = []
                index += 1
                continue
            name = (
                parts[2]
                .split("(")[0]
                .split("<")[0]
                .rstrip("{")
                .rstrip(":;")
                .strip()
            )
            entries.append((name, header, clean_doc(doc)))
            if kind == "trait":
                for method_name, signature, method_doc in method_entries(
                    lines, index, name, "fn "
                ):
                    entries.append((method_name, signature, method_doc))
            doc = []
            index += 1
            continue
        if stripped and not stripped.startswith("//"):
            doc = []
        index += 1
    return entries


def write_rust_page():
    """Write the Rust seam reference page from the core source."""
    buf = [
        "# API reference — Rust generic seam\n",
        "\n",
        HEADER,
        "\n",
        RUST_BLURB,
        "\n\n",
        (
            "Reach the module through `mltreelib_core::generic` or "
            "`mltreelib_core::solver::generic`. The short names `Bounds`, `Budget`, "
            "`Constraint`, `ConstraintKind`, `Objective`, `ObjectiveError`, "
            "`OptimizationRequest`, `OptimizationResult`, `OptimizerKind`, "
            "`SmoothObjective`, and `SolveStatus` are re-exported from "
            "`mltreelib_core::solver`.\n\n"
        ),
    ]
    for source in sorted(CORE_GENERIC.glob("*.rs")):
        entries = rust_items(source)
        if not entries:
            continue
        buf.append(f"## solver/generic/{source.name}\n\n")
        for name, signature, doc in entries:
            buf.append(f"### {name}\n\n")
            buf.append(f"```rust\n{signature}\n```\n\n")
            if doc:
                buf.append(doc + "\n\n")
    (OUT / RUST_PAGE).write_text("".join(buf).rstrip() + "\n")
    print(f"wrote {OUT / RUST_PAGE}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    def module_members(module):
        classes = {
            n: c
            for n, c in inspect.getmembers(module, inspect.isclass)
            if not n.startswith("_")
        }
        fns = {
            n: f
            for n, f in inspect.getmembers(
                module, lambda o: callable(o) and not isinstance(o, type)
            )
            if not n.startswith("_")
        }
        return classes, fns

    def members(modules):
        classes, fns = {}, {}
        for module in modules:
            module_classes, module_fns = module_members(module)
            for n, c in module_classes.items():
                classes.setdefault(n, c)
            for n, f in module_fns.items():
                fns.setdefault(n, f)
        return classes, fns

    pages = [
        (fname, title, mltreelib, cl, fn, blurb)
        for fname, title, cl, fn, blurb in PAGES
    ]
    pages += [
        (fname, title, importlib.import_module(path), cl, fn, blurb)
        for fname, title, path, cl, fn, blurb in SUBMODULE_PAGES
    ]

    index = [
        "# API reference — class index\n",
        "\n",
        HEADER,
        "\n",
        "Every class and function with its exact signature and docstring —\n",
        "regenerated from the built module, so this never drifts from the\n",
        "code. Narrative guides with verified examples:\n",
        "[Data](../data.md) · [Model](../model.md) · [Metrics](../metrics.md) ·\n",
        "[Reports](../reports.md) · [Visualization](../../guides/visualization.md).\n",
        "\n",
        "| page | classes | functions |\n|---|---|---|\n",
    ]
    for fname, title, _module, cl, fn, _blurb in pages:
        index.append(
            f"| [{title}]({fname[:-3]}/) | {', '.join(cl) if cl else '—'} "
            f"| {', '.join(fn) if fn else '—'} |\n"
        )
    index.append(
        f"| [Rust generic seam]({RUST_PAGE[:-3]}/) | Rust core seam | — |\n"
    )
    (OUT / "index.md").write_text("".join(index))

    for fname, title, module, cl, fn, blurb in pages:
        extra = [importlib.import_module(p) for p in EXTRA_MODULES.get(fname, [])]
        classes, fns = members([module] + extra)
        buf = [f"# API reference — {title}\n", "\n", HEADER, "\n", blurb, "\n\n"]
        for name in fn:
            if name in fns:
                emit_fn(buf, name, fns[name])
        for name in cl:
            if name in classes:
                emit_class(buf, name, classes[name])
        (OUT / fname).write_text("".join(buf).rstrip() + "\n")
        print(f"wrote {OUT / fname} ({len(buf)} blocks)")
    print(f"wrote {OUT / 'index.md'}")
    write_rust_page()


if __name__ == "__main__":
    main()
