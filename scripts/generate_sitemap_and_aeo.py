"""Generate standard-compliant sitemaps (single sitemap.xml),

llms.txt, and llms-full.txt for mltreelib documentation.
Aligned with modern search engine (Google Search Console) and AEO/GEO crawler standards.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_URL = "https://jkapila.github.io/mltreelib"


def get_git_timestamp(file_path: Path) -> str:
    """Get ISO 8601 UTC timestamp from git commit history or file mtime."""
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        timestamp = res.stdout.strip()
        if timestamp:
            dt = datetime.datetime.fromisoformat(timestamp).astimezone(datetime.timezone.utc)
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # Fallback to file mtime
    try:
        mtime = os.path.getmtime(file_path)
        dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        now = datetime.datetime.now(datetime.timezone.utc)
        return now.strftime("%Y-%m-%d")


def determine_priority_and_freq(rel_url: str) -> tuple[str, str]:
    """Assign changefreq and priority based on document taxonomy."""
    if rel_url in ("", "/"):
        return "1.0", "weekly"
    if rel_url.startswith(("/quickstart", "/installation")):
        return "0.9", "monthly"
    if rel_url.startswith(("/guides", "/blog")):
        return "0.8", "weekly"
    if rel_url.startswith("/benchmarks"):
        return "0.8", "weekly"
    if rel_url.startswith("/reference"):
        return "0.7", "monthly"
    return "0.6", "monthly"


def generate_sitemaps(site_dir: Path, docs_dir: Path) -> list[dict[str, str]]:
    """Scan site directory and generate sitemap.xml."""
    pages: list[dict[str, str]] = []

    # Map of relative url -> source markdown file
    source_map: dict[str, Path] = {}
    for md_file in docs_dir.rglob("*.md"):
        rel = md_file.relative_to(docs_dir)
        if rel.name == "index.md":
            rel_url = f"/{rel.parent.as_posix()}/" if rel.parent.as_posix() != "." else "/"
        else:
            rel_url = f"/{rel.with_suffix('').as_posix()}/"
        source_map[rel_url] = md_file

    # Collect from site_dir
    html_files = sorted(site_dir.rglob("index.html"))
    for html_file in html_files:
        rel = html_file.relative_to(site_dir)
        if rel.as_posix() == "index.html":
            rel_url = "/"
        else:
            rel_url = f"/{rel.parent.as_posix()}/"

        full_url = f"{BASE_URL}{rel_url}" if rel_url != "/" else f"{BASE_URL}/"

        # Determine timestamp
        src_file = source_map.get(rel_url, html_file)
        lastmod = get_git_timestamp(src_file)
        priority, changefreq = determine_priority_and_freq(rel_url)

        pages.append(
            {
                "loc": full_url,
                "lastmod": lastmod,
                "changefreq": changefreq,
                "priority": priority,
            }
        )

    # Sort so root is first, then alphabetical
    pages.sort(key=lambda p: (0 if p["loc"] == f"{BASE_URL}/" else 1, p["loc"]))

    # Build sitemap.xml content
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    urlset_attrs = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    sitemap_lines = [f"{xml_header}<urlset {urlset_attrs}>"]
    for p in pages:
        sitemap_lines.append(
            f"  <url>\n"
            f"    <loc>{p['loc']}</loc>\n"
            f"    <lastmod>{p['lastmod']}</lastmod>\n"
            f"    <changefreq>{p['changefreq']}</changefreq>\n"
            f"    <priority>{p['priority']}</priority>\n"
            f"  </url>"
        )
    sitemap_lines.append("</urlset>\n")
    sitemap_content = "\n".join(sitemap_lines)

    # Write files to site_dir and docs_dir
    for target_dir in [site_dir, docs_dir]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "sitemap.xml").write_text(sitemap_content, encoding="utf-8")

    print(f"[OK] Generated sitemaps with {len(pages)} URLs in {site_dir} and {docs_dir}.")
    return pages


def generate_llms_txt(site_dir: Path, docs_dir: Path) -> None:
    """Generate llms.txt matching the llmstxt.org specification."""
    llms_content = f"""# mltreelib

> High-performance tree-based machine learning with a pure Rust core (`mltreelib-core`) and thin Python/R interfaces. Specializes in continuous leaf model trees, piecewise linear trees, honest causal uplift trees, multi-factor rotations, anomaly detection, and clustering with zero scikit-learn runtime dependency.

## Core Documentation

- [Quickstart]({BASE_URL}/quickstart/): Fast installation, basic model tree fitting, and prediction workflows.
- [Installation]({BASE_URL}/installation/): Installing via pip/uv with prebuilt wheels or building from source with Cargo.
- [Tree Architecture & Growth Knobs]({BASE_URL}/concepts/tree-architecture-and-knobs/): Two-control decoupling, monotone constraints, and honesty fractions.
- [Data Ingestion & Routing]({BASE_URL}/concepts/data-matrix-and-routing/): Polars/Arrow ingestion, zero-imputation missing routing, and categorical encoders.
- [Metrics, Diagnostics & Calibration]({BASE_URL}/guides/metrics-plots/): Gini, ROC/PR curves, Brier calibration, and Population Stability Index (PSI).

## Regression & Continuous Modeling

- [Comprehensive Regression Showcase]({BASE_URL}/benchmarks/regression-methods/): 4-study empirical benchmark across physical, heavy-tailed, high-volume, and bounded tabular datasets.
- [Piecewise Linear Trees & Cubist Rules]({BASE_URL}/concepts/piecewise-linear-and-cubist/): Quinlan M5 model trees, committee rule sets, and k-NN local smoothing.
- [Distributional & Censored Trees]({BASE_URL}/concepts/distributional-leaves/): Tweedie compound Poisson, Tobit censored splits, and heteroskedastic volatility trees.

## Causal & Unsupervised Learning

- [Causal Uplift Modeling]({BASE_URL}/guides/uplift/): Honest causal trees, Athey-Imbens splitting, and Qini evaluation curves.
- [Clustering & Anomaly Trees]({BASE_URL}/concepts/anomaly-trees/): AnomalyModel (EIF, DET, HST, RCT) and ClusterModel (CLTree, HDBSCAN, PCT, URF).
- [PCA, Factor Analysis & Rotations]({BASE_URL}/concepts/components-and-rotations/): ComponentModel decomposing continuous features with Varimax and Promax rotations.

## Optimization & Solvers

- [Solvers & Hyperparameter Tuning]({BASE_URL}/concepts/solvers-and-optimizers/): 14 standalone optimization solvers, black-box optimizers, CMA-ES, Nelder-Mead, L-BFGS, and AskTell interface.
- [Model Inspection & Rules]({BASE_URL}/guides/model-inspection-and-rules/): Quinlan rule extraction, DOT graph visualization, and explainable decision stumps.

## API Reference

- [API Reference Overview]({BASE_URL}/reference/api/): Full API documentation for all classes, methods, and functions.
- [ModelTree]({BASE_URL}/reference/api/model-tree/): Core tree estimator configuration and methods.
- [AnomalyModel]({BASE_URL}/reference/api/anomaly/): Unsupervised anomaly detection facade.
- [ClusterModel]({BASE_URL}/reference/api/cluster/): Unsupervised clustering facade.
- [ComponentModel]({BASE_URL}/reference/api/components/): Continuous decomposition and factor rotation facade.
- [Solvers]({BASE_URL}/reference/api/solvers/): Standalone optimization algorithms.

## How it works

- [Guides overview]({BASE_URL}/guides/): task guides, how each mechanism works, and worked examples.
- [TabArena Datasets]({BASE_URL}/benchmarks/datasets/): 51 tabular benchmark datasets from OpenML suite 457.
- [Elo Ratings]({BASE_URL}/benchmarks/elo/): Head-to-head Elo leaderboards against LightGBM, CatBoost, and XGBoost.
"""
    for target_dir in [site_dir, docs_dir]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "llms.txt").write_text(llms_content, encoding="utf-8")
    print(f"[OK] Generated llms.txt in {site_dir} and {docs_dir}.")


def generate_llms_full_txt(site_dir: Path, docs_dir: Path) -> None:
    """Generate llms-full.txt combining overview and core flagship documentation."""
    chunks = [
        f"# mltreelib — Complete LLM Context & Reference Guide\n\n"
        f"Site: {BASE_URL}/\n"
        f"Repository: https://github.com/jkapila/mltreelib\n\n"
        f"mltreelib is a tree-based machine learning library with a Rust core (`mltreelib-core`) "
        f"and thin Python/R frontends. Zero scikit-learn dependency.\n\n"
    ]

    key_files = [
        docs_dir / "quickstart.md",
        docs_dir / "installation.md",
        docs_dir / "benchmarks/regression-methods.md",
        docs_dir / "concepts/piecewise-linear-and-cubist.md",
        docs_dir / "concepts/distributional-leaves.md",
        docs_dir / "guides/uplift.md",
        docs_dir / "concepts/anomaly-trees.md",
        docs_dir / "concepts/clustering-trees.md",
        docs_dir / "concepts/tree-architecture-and-knobs.md",
        docs_dir / "concepts/solvers-and-optimizers.md",
        docs_dir / "concepts/components-and-rotations.md",
    ]

    for f in key_files:
        if f.exists():
            chunks.append(f"\n\n---\n\n## Document: {f.name}\n\n")
            chunks.append(f.read_text(encoding="utf-8"))

    full_content = "".join(chunks)
    for target_dir in [site_dir, docs_dir]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "llms-full.txt").write_text(full_content, encoding="utf-8")
    print(f"[OK] Generated llms-full.txt ({len(full_content)} chars) in {site_dir} and {docs_dir}.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    docs_dir = root / "docs"

    generate_llms_txt(site_dir, docs_dir)
    generate_llms_full_txt(site_dir, docs_dir)
    generate_sitemaps(site_dir, docs_dir)


if __name__ == "__main__":
    main()
