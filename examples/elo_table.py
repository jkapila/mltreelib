"""Elo board -> docs: interactive tables with Excel-style frozen key columns
and highlighted column-level winners.

Generates:
1. docs/benchmarks/elo_board.md     (Regression: test RMSE)
2. docs/benchmarks/elo_cla_board.md (Classification: test LogLoss & Accuracy)

Features:
- Frozen key columns on the left (ELO, FIT TIME, MODEL) via sticky positioning.
- Interactive click-to-sort on every column header.
- Column-level winner highlighting:
  * Green badge: best metric per dataset column.
  * Blue badge: fastest fit time.
  * Orange badge: highest Elo rating.
"""

import csv
import itertools
from pathlib import Path
import statistics
import logging

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

SCORES_REG = Path("benchmarks/elo_scores.csv")
SCORES_CLA = Path("benchmarks/elo_cla_scores.csv")
DOC_REG = Path("docs/benchmarks/elo_board.md")
DOC_CLA = Path("docs/benchmarks/elo_cla_board.md")


def build_interactive_table(
    table_id: str,
    datasets: list,
    models: list,
    ratings: dict,
    fit_txt: dict,
    raw_fits: dict,
    data_by: dict,
    metric_label: str,
    lower_is_better: bool = True,
    fmt_fn=lambda v: f"{v:.3f}",
) -> str:
    """Build a self-contained sticky interactive table with column-level winner highlighting."""
    order = sorted(models, key=lambda m: -ratings[m])
    max_elo = max(ratings.values()) if ratings else 0.0

    # Find fastest model
    fastest_model = min(models, key=lambda m: statistics.mean(raw_fits.get(m, [999999.0]))) if models else None

    # Find best score per dataset
    best_per_ds = {}
    for ds in datasets:
        vals = [data_by[(ds, m)] for m in models if (ds, m) in data_by and data_by[(ds, m)] is not None]
        if vals:
            best_per_ds[ds] = min(vals) if lower_is_better else max(vals)

    shorts = {}
    for d in datasets:
        s = d[3:] if d.startswith("ta_") else d
        shorts[d] = s
    dupes = {s for s in shorts.values() if list(shorts.values()).count(s) > 1}
    shorts = {d: (d if s in dupes else s) for d, s in shorts.items()}

    th = "".join(f"<th data-t='{i + 3}'>{shorts[d]}</th>" for i, d in enumerate(datasets))
    body = []

    for m in order:
        n = sum(1 for d in datasets if (d, m) in data_by and data_by[(d, m)] is not None)
        cells = []
        for ds in datasets:
            v = data_by.get((ds, m))
            if v is None:
                cells.append("<td class='n'>·</td>")
            else:
                is_win = (v == best_per_ds.get(ds))
                val_str = fmt_fn(v)
                if is_win:
                    cells.append(f"<td class='n win' title='Best {metric_label} on {ds}'><b>{val_str}</b></td>")
                else:
                    cells.append(f"<td class='n'>{val_str}</td>")

        elo_cls = "k n elo-top" if ratings[m] == max_elo else "k n"
        fit_cls = "k n fit-fast" if m == fastest_model else "k n"

        body.append(
            f"<tr><td class='{elo_cls}'><b>{ratings[m]:.1f}</b></td>"
            f"<td class='{fit_cls}'>{fit_txt[m]}</td>"
            f"<td class='k n m-name'><b>{m}</b> <span class='c'>{n}/{len(datasets)}</span></td>"
            + "".join(cells) + "</tr>"
        )

    html = f"""
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
<tbody>{''.join(body)}</tbody>
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
    return html


def generate_regression():
    if not SCORES_REG.exists():
        return
    rows = [r for r in csv.DictReader(open(SCORES_REG)) if r.get("model") and r["model"] != "model" and r.get("rmse")]
    by = {(r["dataset"], r["model"]): float(r["rmse"]) for r in rows}
    datasets = sorted({d for (d, _) in by})
    models = sorted({m for (_, m) in by})

    ratings = {m: 1000.0 for m in models}
    for ds in datasets:
        here = {m: by[(ds, m)] for m in models if (ds, m) in by}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (sa - ea)
            ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))

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

    DOC_REG.write_text(
        "<!-- generated by examples/elo_table.py - do not edit by hand -->\n\n" + html
    )
    log.info(f"wrote {DOC_REG} ({len(datasets)} datasets x {len(models)} models)")


def generate_classification():
    if not SCORES_CLA.exists():
        return
    rows = [r for r in csv.DictReader(open(SCORES_CLA)) if r.get("model") and r["model"] != "model" and r.get("logloss")]
    by = {(r["dataset"], r["model"]): float(r["logloss"]) for r in rows if r["logloss"] != "" and float(r["logloss"]) == float(r["logloss"])}
    datasets = sorted({d for (d, _) in by})
    models = sorted({m for (_, m) in by})

    ratings = {m: 1000.0 for m in models}
    for ds in datasets:
        here = {m: by[(ds, m)] for m in models if (ds, m) in by}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (sa - ea)
            ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))

    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    fit_txt = {}
    for m, xs in fits.items():
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        fit_txt[m] = f"{statistics.mean(xs):.2f}±{sd:.2f}s"

    html = build_interactive_table(
        table_id="elo-cla-interactive",
        datasets=datasets,
        models=models,
        ratings=ratings,
        fit_txt=fit_txt,
        raw_fits=fits,
        data_by=by,
        metric_label="LogLoss",
        lower_is_better=True,
        fmt_fn=lambda v: f"{v:.3f}",
    )

    DOC_CLA.write_text(
        "<!-- generated by examples/elo_table.py - do not edit by hand -->\n\n" + html
    )
    log.info(f"wrote {DOC_CLA} ({len(datasets)} datasets x {len(models)} models)")


def main():
    generate_regression()
    generate_classification()


if __name__ == "__main__":
    main()

