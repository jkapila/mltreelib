"""Elo board -> docs: interactive tables with Excel-style frozen key columns
and highlighted column-level winners.

Generates:
1. docs/benchmarks/elo_overall.md   (Overall Arena: Combined Regression + Classification)
2. docs/benchmarks/elo_board.md     (Regression Arena: test RMSE)
3. docs/benchmarks/elo_cla_board.md (Classification Arena: Accuracy & LogLoss, Binary vs Multiclass)
"""

import csv
import itertools
import json
import logging
import statistics
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("benchmarks/elo_run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("elo.table")

SCORES_DIR = Path("docs/benchmarks/scores/v0.4.11")
SCORES_REG = SCORES_DIR / "elo_scores.csv"
SCORES_CLA = SCORES_DIR / "elo_cla_scores.csv"
CLA_TASKS = SCORES_DIR / "classification_tasks.json"

DOC_OVERALL = Path("docs/benchmarks/elo_overall.md")
DOC_REG = Path("docs/benchmarks/elo_board.md")
DOC_CLA = Path("docs/benchmarks/elo_cla_board.md")


def read_scores(path: Path, metric: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return [
            row
            for row in csv.DictReader(source)
            if row.get("model") and row["model"] != "model" and row.get(metric)
        ]


def compute_elo_ratings(
    datasets: list[str],
    models: list[str],
    data_by: dict[tuple[str, str], float],
    lower_is_better: bool = True,
    k_factor: float = 32.0,
    start_rating: float = 1000.0,
) -> dict[str, float]:
    """Compute round-robin Elo ratings given a dictionary of (dataset, model) -> score."""
    ratings = {m: start_rating for m in models}
    for ds in datasets:
        here = {m: data_by[(ds, m)] for m in models if (ds, m) in data_by}
        for a, b in itertools.combinations(sorted(here), 2):
            va, vb = here[a], here[b]
            if lower_is_better:
                sa = 1.0 if va < vb else (0.0 if va > vb else 0.5)
            else:
                sa = 1.0 if va > vb else (0.0 if va < vb else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * ((1.0 - sa) - (1.0 - ea))
    return ratings


def build_interactive_table(
    table_id: str,
    datasets: list[str],
    models: list[str],
    ratings: dict[str, float],
    fit_txt: dict[str, str],
    raw_fits: dict[str, list[float]],
    data_by: dict[tuple[str, str], float],
    metric_label: str,
    lower_is_better: bool = True,
    fmt_fn=lambda v: f"{v:.3f}",
) -> str:
    """Build a self-contained sticky interactive table with column-level winner highlighting."""
    order = sorted(models, key=lambda m: -ratings.get(m, 0.0))
    max_elo = max(ratings.values()) if ratings else 0.0

    fastest_model = (
        min(models, key=lambda m: statistics.mean(raw_fits.get(m, [999999.0])))
        if models
        else None
    )

    best_per_ds = {}
    for ds in datasets:
        vals = [
            data_by[(ds, m)]
            for m in models
            if (ds, m) in data_by and data_by[(ds, m)] is not None
        ]
        if vals:
            best_per_ds[ds] = min(vals) if lower_is_better else max(vals)

    shorts = {}
    for d in datasets:
        s = d.removeprefix("ta_")
        shorts[d] = s
    dupes = {s for s in shorts.values() if list(shorts.values()).count(s) > 1}
    shorts = {d: (d if s in dupes else s) for d, s in shorts.items()}

    th = "".join(
        f"<th data-t='{i + 3}'>{shorts[d]}</th>" for i, d in enumerate(datasets)
    )
    body = []

    for m in order:
        n = sum(
            1 for d in datasets if (d, m) in data_by and data_by[(d, m)] is not None
        )
        cells = []
        for ds in datasets:
            v = data_by.get((ds, m))
            if v is None:
                cells.append("<td class='n'>·</td>")
            else:
                is_win = (
                    abs(v - best_per_ds.get(ds, 999999.0)) < 1e-9
                    if lower_is_better
                    else abs(v - best_per_ds.get(ds, -999999.0)) < 1e-9
                )
                val_str = fmt_fn(v)
                if is_win:
                    cells.append(
                        f"<td class='n win' title='Best {metric_label} on {ds}'><b>{val_str}</b></td>"
                    )
                else:
                    cells.append(f"<td class='n'>{val_str}</td>")

        elo_cls = "k n elo-top" if abs(ratings[m] - max_elo) < 1e-6 else "k n"
        fit_cls = "k n fit-fast" if m == fastest_model else "k n"
        m_fit = fit_txt.get(m, "·")

        body.append(
            f"<tr><td class='{elo_cls}'><b>{ratings[m]:.1f}</b></td>"
            f"<td class='{fit_cls}'>{m_fit}</td>"
            f"<td class='k n m-name'><b>{m}</b> <span class='c'>{n}/{len(datasets)}</span></td>"
            + "".join(cells)
            + "</tr>"
        )

    return f"""
<div id="{table_id}" class="elo-interactive-wrap">
<style>
#{table_id} {{ overflow-x: auto; margin: 1.2em 0; border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0); border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
#{table_id} table {{ border-collapse: separate; border-spacing: 0; font-size: 0.78rem; white-space: nowrap; font-family: var(--md-code-font-family, monospace); width: 100%; }}
#{table_id} th, #{table_id} td {{ padding: 5px 11px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #e6e6e6); text-align: right; background: var(--md-default-bg-color, #fff); box-sizing: border-box; }}
#{table_id} th {{ cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 4; font-weight: 600; background: var(--md-default-fg-color--lightest, #f8f9fa); border-bottom: 2px solid var(--md-default-fg-color--light, #ccc); }}
#{table_id} th:hover {{ color: var(--md-accent-fg-color, #0288d1); background: #eef7fc; }}
#{table_id} td.n {{ text-align: right; }}
#{table_id} tr:hover td {{ background: var(--md-default-fg-color--lightest, #f4f6f8); }}

/* Sticky Left Key Columns */
#{table_id} th.k1, #{table_id} td:nth-child(1) {{ position: sticky; left: 0; z-index: 2; width: 85px; min-width: 85px; }}
#{table_id} th.k2, #{table_id} td:nth-child(2) {{ position: sticky; left: 85px; z-index: 2; width: 145px; min-width: 145px; }}
#{table_id} th.k3, #{table_id} td:nth-child(3) {{ position: sticky; left: 230px; z-index: 2; width: 220px; min-width: 220px; text-align: left; }}
#{table_id} th.k1, #{table_id} th.k2, #{table_id} th.k3 {{ z-index: 5; background: var(--md-default-fg-color--lightest, #f0f2f5); }}

/* Column Winner Highlighting */
#{table_id} td.win {{ background: #e8f5e9 !important; color: #2e7d32 !important; font-weight: 700; }}
#{table_id} td.elo-top {{ color: #e65100; font-weight: 700; }}
#{table_id} td.fit-fast {{ color: #0277bd; font-weight: 700; }}
#{table_id} td.m-name {{ text-align: left; }}
#{table_id} .c {{ color: var(--md-default-fg-color--light, #888); font-size: 0.70rem; }}
</style>
<table>
<thead><tr>
<th class='k1 k' data-t='0'>ELO</th>
<th class='k2 k' data-t='1'>FIT TIME (±sd)</th>
<th class='k3 k' data-t='2'>MODEL</th>{th}
</tr></thead>
<tbody>{"".join(body)}</tbody>
</table>
<script>
(function () {{
  var t = document.querySelector('#{table_id} table');
  if (!t) return;
  var dir = {{}};
  t.querySelectorAll('th').forEach(function (th) {{
    th.addEventListener('click', function () {{
      var i = +th.dataset.t, d = dir[i] = -(dir[i] || 1);
      var rows = Array.prototype.slice.call(t.tBodies[0].rows);
      rows.sort(function (a, b) {{
        var x = a.cells[i].textContent.trim(), y = b.cells[i].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        return d * (isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny);
      }});
      rows.forEach(function (r) {{ t.tBodies[0].appendChild(r); }});
    }});
  }});
}})();
</script>
</div>
"""


def generate_regression():
    if not SCORES_REG.exists():
        return
    rows = read_scores(SCORES_REG, "rmse")
    by = {(r["dataset"], r["model"]): float(r["rmse"]) for r in rows}
    datasets = sorted({d for (d, _) in by})
    models = sorted({m for (_, m) in by})
    ratings = compute_elo_ratings(datasets, models, by, lower_is_better=True)

    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    fit_txt = {}
    for m, xs in fits.items():
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        fit_txt[m] = f"{statistics.mean(xs):.2f}±{sd:.2f}s"

    html = build_interactive_table(
        table_id="elo-reg-interactive",
        datasets=datasets,
        models=models,
        ratings=ratings,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by,
        metric_label="RMSE",
        lower_is_better=True,
        fmt_fn=lambda v: f"{v:.3f}",
    )

    header = f"""# Regression Elo Arena

*Across **{len(datasets)} benchmark datasets** evaluated on **test RMSE** (round-robin pairwise, K=32).*  
*Click any column header to sort. Sticky headers and frozen key columns on the left.*

[Overall Arena](elo_overall.md) · [Classification Arena](elo_cla_board.md) · [Run notes and score files](elo.md)

---
"""
    DOC_REG.write_text(header + html)
    log.info(f"wrote {DOC_REG} ({len(datasets)} datasets x {len(models)} models)")


def generate_classification():
    if not SCORES_CLA.exists():
        return
    rows = read_scores(SCORES_CLA, "accuracy")

    by_acc = {
        (r["dataset"], r["model"]): float(r["accuracy"])
        for r in rows
        if r.get("accuracy")
    }
    by_logloss = {
        (r["dataset"], r["model"]): float(r["logloss"])
        for r in rows
        if r.get("logloss") and float(r["logloss"]) == float(r["logloss"])
    }

    all_datasets = sorted({d for (d, _) in by_acc})
    all_models = sorted({m for (_, m) in by_acc})

    task_types = json.loads(CLA_TASKS.read_text())
    if set(task_types) != set(all_datasets) or not set(task_types.values()) <= {
        "binary", "multiclass"
    }:
        raise ValueError("Classification task types must match the score snapshot")
    binary_ds = [ds for ds in all_datasets if task_types[ds] == "binary"]
    multiclass_ds = [ds for ds in all_datasets if task_types[ds] == "multiclass"]

    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    fit_txt = {}
    for m, xs in fits.items():
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        fit_txt[m] = f"{statistics.mean(xs):.2f}±{sd:.2f}s"

    # 1. Overall Accuracy Elo
    ratings_acc = compute_elo_ratings(all_datasets, all_models, by_acc, lower_is_better=False)
    html_acc = build_interactive_table(
        table_id="elo-cla-acc",
        datasets=all_datasets,
        models=all_models,
        ratings=ratings_acc,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by_acc,
        metric_label="Accuracy",
        lower_is_better=False,
        fmt_fn=lambda v: f"{v:.4f}",
    )

    # 2. Overall LogLoss Elo
    models_ll = sorted({m for (_, m) in by_logloss})
    ratings_ll = compute_elo_ratings(all_datasets, models_ll, by_logloss, lower_is_better=True)
    html_ll = build_interactive_table(
        table_id="elo-cla-logloss",
        datasets=all_datasets,
        models=models_ll,
        ratings=ratings_ll,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by_logloss,
        metric_label="LogLoss",
        lower_is_better=True,
        fmt_fn=lambda v: f"{v:.3f}",
    )

    # 3. Binary Only Accuracy Elo
    ratings_bin = compute_elo_ratings(binary_ds, all_models, by_acc, lower_is_better=False)
    html_bin = build_interactive_table(
        table_id="elo-cla-binary",
        datasets=binary_ds,
        models=all_models,
        ratings=ratings_bin,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by_acc,
        metric_label="Accuracy",
        lower_is_better=False,
        fmt_fn=lambda v: f"{v:.4f}",
    )

    # 4. Multiclass Only Accuracy Elo
    models_mc = sorted({m for (d, m) in by_acc if d in multiclass_ds})
    ratings_mc = compute_elo_ratings(multiclass_ds, models_mc, by_acc, lower_is_better=False)
    html_mc = build_interactive_table(
        table_id="elo-cla-multiclass",
        datasets=multiclass_ds,
        models=models_mc,
        ratings=ratings_mc,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by_acc,
        metric_label="Accuracy",
        lower_is_better=False,
        fmt_fn=lambda v: f"{v:.4f}",
    )

    doc_text = f"""# Classification Elo Arena

*Across **{len(all_datasets)} classification datasets** ({len(binary_ds)} binary, {len(multiclass_ds)} multiclass).*

*Click any column header to sort. Frozen key columns (ELO, Fit Time, Model) remain sticky.*

**Navigation:**  
- [1. All Datasets (Test Accuracy)](#1-all-datasets-test-accuracy-elo)  
- [2. Binary Datasets ({len(binary_ds)})](#2-binary-classification-datasets-{len(binary_ds)})  
- [3. Multiclass Datasets ({len(multiclass_ds)})](#3-multiclass-classification-datasets-{len(multiclass_ds)})  
- [4. All Datasets (Test LogLoss)](#4-all-datasets-test-logloss-elo)  
- [Overall Arena](elo_overall.md) · [Regression Arena](elo_board.md) · [Run notes and score files](elo.md)

---

## 1. All Datasets (Test Accuracy Elo)
*Evaluates raw classification accuracy across all {len(all_datasets)} datasets.*

{html_acc}

---

## 2. Binary Classification Datasets ({len(binary_ds)})
*Evaluates models strictly on binary classification problems.*

{html_bin}

---

## 3. Multiclass Classification Datasets ({len(multiclass_ds)})
*Evaluates multiclass classification accuracy.*

{html_mc}

---

## 4. All Datasets (Test LogLoss Elo)
*Evaluates cross-entropy / probability calibration across all {len(all_datasets)} datasets.*

{html_ll}
"""
    DOC_CLA.write_text(doc_text)
    log.info(f"wrote {DOC_CLA} (38 datasets across 4 views)")


def generate_overall():
    if not SCORES_REG.exists() or not SCORES_CLA.exists():
        return

    rows_reg = read_scores(SCORES_REG, "rmse")
    rows_cla = read_scores(SCORES_CLA, "accuracy")

    by_reg = {(r["dataset"], r["model"]): float(r["rmse"]) for r in rows_reg}
    by_cla = {(r["dataset"], r["model"]): float(r["accuracy"]) for r in rows_cla}

    dss_reg = sorted({d for (d, _) in by_reg})
    dss_cla = sorted({d for (d, _) in by_cla})

    all_models = sorted({m for (_, m) in by_reg} | {m for (_, m) in by_cla})
    total_datasets = len(dss_reg) + len(dss_cla)

    # Combined round-robin Elo
    ratings = {m: 1000.0 for m in all_models}
    for ds in dss_reg:
        here = {m: by_reg[(ds, m)] for m in all_models if (ds, m) in by_reg}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (sa - ea)
            ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))

    for ds in dss_cla:
        here = {m: by_cla[(ds, m)] for m in all_models if (ds, m) in by_cla}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] > here[b] else (0.0 if here[a] < here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (sa - ea)
            ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))

    # Reg Elo only
    ratings_reg = compute_elo_ratings(dss_reg, sorted({m for (_, m) in by_reg}), by_reg, lower_is_better=True)
    # Cla Accuracy Elo only
    ratings_cla = compute_elo_ratings(dss_cla, sorted({m for (_, m) in by_cla}), by_cla, lower_is_better=False)

    # Collect fits and metadata
    fits = {}
    langs = {}
    cats = {}
    for r in rows_reg + rows_cla:
        m = r["model"]
        fits.setdefault(m, []).append(float(r["fit_s"]))
        if r.get("lang"):
            langs[m] = r["lang"]
        if r.get("cat_handling"):
            cats[m] = r["cat_handling"]

    order = sorted(all_models, key=lambda m: -ratings[m])
    max_elo = max(ratings.values())

    rows_html = []
    for rank, m in enumerate(order, 1):
        n_reg = sum(1 for d in dss_reg if (d, m) in by_reg)
        n_cla = sum(1 for d in dss_cla if (d, m) in by_cla)
        tot_ds = n_reg + n_cla
        r_tot = ratings[m]
        r_reg_str = f"{ratings_reg[m]:.1f}" if m in ratings_reg else "·"
        r_cla_str = f"{ratings_cla[m]:.1f}" if m in ratings_cla else "·"

        xs = fits.get(m, [0.0])
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        fit_str = f"{statistics.mean(xs):.2f}±{sd:.2f}s"

        elo_cls = "k n elo-top" if abs(r_tot - max_elo) < 1e-6 else "k n"
        lang_str = langs.get(m, "·")
        cat_str = cats.get(m, "·")

        rows_html.append(
            f"<tr>"
            f"<td class='n' style='text-align:center;'>{rank}</td>"
            f"<td class='{elo_cls}'><b>{r_tot:.1f}</b></td>"
            f"<td class='k n m-name'><b>{m}</b></td>"
            f"<td class='n'>{r_reg_str}</td>"
            f"<td class='n'>{r_cla_str}</td>"
            f"<td class='n'>{tot_ds}/{total_datasets}</td>"
            f"<td class='n'>{fit_str}</td>"
            f"<td class='n' style='text-align:center;'>{lang_str}</td>"
            f"<td class='n' style='text-align:center;'>{cat_str}</td>"
            f"</tr>"
        )

    table_id = "elo-overall-table"
    table_html = f"""
<div id="{table_id}" class="elo-interactive-wrap">
<style>
#{table_id} {{ overflow-x: auto; margin: 1.2em 0; border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0); border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
#{table_id} table {{ border-collapse: separate; border-spacing: 0; font-size: 0.82rem; white-space: nowrap; font-family: var(--md-code-font-family, monospace); width: 100%; }}
#{table_id} th, #{table_id} td {{ padding: 7px 12px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #e6e6e6); text-align: right; background: var(--md-default-bg-color, #fff); box-sizing: border-box; }}
#{table_id} th {{ cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 4; font-weight: 600; background: var(--md-default-fg-color--lightest, #f8f9fa); border-bottom: 2px solid var(--md-default-fg-color--light, #ccc); }}
#{table_id} th:hover {{ color: var(--md-accent-fg-color, #0288d1); background: #eef7fc; }}
#{table_id} td.n {{ text-align: right; }}
#{table_id} td.m-name {{ text-align: left; }}
#{table_id} tr:hover td {{ background: var(--md-default-fg-color--lightest, #f4f6f8); }}
#{table_id} td.elo-top {{ color: #e65100; font-weight: 700; }}
</style>
<table>
<thead><tr>
<th data-t='0'>RANK</th>
<th data-t='1'>OVERALL ELO</th>
<th data-t='2' style='text-align:left;'>MODEL</th>
<th data-t='3'>REG ELO</th>
<th data-t='4'>CLA ELO</th>
<th data-t='5'>DATASETS</th>
<th data-t='6'>FIT TIME (±sd)</th>
<th data-t='7'>LANG</th>
<th data-t='8'>CAT HANDLING</th>
</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
<script>
(function () {{
  var t = document.querySelector('#{table_id} table');
  if (!t) return;
  var dir = {{}};
  t.querySelectorAll('th').forEach(function (th) {{
    th.addEventListener('click', function () {{
      var i = +th.dataset.t, d = dir[i] = -(dir[i] || 1);
      var rows = Array.prototype.slice.call(t.tBodies[0].rows);
      rows.sort(function (a, b) {{
        var x = a.cells[i].textContent.trim(), y = b.cells[i].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        return d * (isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny);
      }});
      rows.forEach(function (r) {{ t.tBodies[0].appendChild(r); }});
    }});
  }});
}})();
</script>
</div>
"""

    doc_text = f"""# Overall Benchmark Arena

The local arena combines **{total_datasets} datasets** across **{len(all_models)} entrants**. It uses test RMSE for {len(dss_reg)} regression tasks and test accuracy for {len(dss_cla)} classification tasks.

[Run notes and score files](elo.md) · [Regression Arena](elo_board.md) · [Classification Arena](elo_cla_board.md) · [Dataset sources](datasets.md)

## Overall leaderboard

Select a column heading to sort.

{table_html}
"""
    DOC_OVERALL.write_text(doc_text)
    log.info(f"wrote {DOC_OVERALL} ({len(all_models)} models across {total_datasets} datasets)")


def main():
    generate_overall()
    generate_regression()
    generate_classification()


if __name__ == "__main__":
    main()
