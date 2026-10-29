"""Elo board -> docs: ONE interactive table (Excel-style frozen key columns).

Reads benchmarks/elo_scores.csv, recomputes Elo (same protocol as
elo_report.py) and writes docs/benchmarks/elo_board.md containing a single
self-contained interactive HTML table:

- frozen key columns on the left (ELO, FIT mean+/-sd, MODEL) via sticky
  positioning - the dataset columns scroll horizontally under them, like
  freeze panes in Excel;
- click any column header to sort;
- per-dataset cells are test RMSE, best per column bolded.

    uv run python examples/elo_table.py
"""

import csv
import itertools
import statistics
from pathlib import Path

OUT = Path("benchmarks/elo_scores.csv")
DOC = Path("docs/benchmarks/elo_board.md")


def main():
    rows = [r for r in csv.DictReader(open(OUT)) if r["model"] != "model"]
    by = {(r["dataset"], r["model"]): (float(r["rmse"]), float(r["mae"])) for r in rows}
    datasets = sorted({d for (d, _) in by})
    models = sorted({m for (_, m) in by})

    ratings = {m: 1000.0 for m in models}
    for ds in datasets:
        here = {m: v[0] for (d, m), v in by.items() if d == ds}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (sa - ea)
            ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))
    order = sorted(models, key=lambda m: -ratings[m])

    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    fit_txt = {}
    for m, xs in fits.items():
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        fit_txt[m] = f"{statistics.mean(xs):.2f}±{sd:.2f}s"

    shorts = {}
    for d in datasets:
        s = d[3:] if d.startswith("ta_") else d
        shorts[d] = s
    dupes = {s for s in shorts.values() if list(shorts.values()).count(s) > 1}
    shorts = {d: (d if s in dupes else s) for d, s in shorts.items()}
    best = {ds: min(v[0] for (d, _), v in by.items() if d == ds) for ds in datasets}

    # header: 3 frozen key columns, then datasets
    th = "".join(f"<th data-t='{i + 3}'>{shorts[d]}</th>" for i, d in enumerate(datasets))
    body = []
    for m in order:
        n = sum(1 for (d, mm) in by if mm == m)
        cells = []
        for ds in datasets:
            v = by.get((ds, m))
            if v is None:
                cells.append("<td class=n>·</td>")
            else:
                bold = v[0] == best[ds]
                cells.append(f"<td class=n><b>{v[0]:.2f}</b></td>" if bold
                             else f"<td class=n>{v[0]:.2f}</td>")
        body.append(
            f"<tr><td class='k n'>{ratings[m]:.0f}</td>"
            f"<td class='k n'>{fit_txt[m]}</td>"
            f"<td class='k n'><b>{m}</b> <span class=c>{n}/{len(datasets)}</span></td>"
            + "".join(cells) + "</tr>"
        )

    html = f"""
<div id="elo-interactive">
<style>
#elo-interactive {{ overflow-x: auto; }}
#elo-interactive table {{ border-collapse: collapse; font-size: 0.75rem; white-space: nowrap; font-family: var(--md-text-font-family, monospace); }}
#elo-interactive th, #elo-interactive td {{ padding: 2px 10px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #ddd); text-align: right; background: var(--md-default-bg-color, #fff); }}
#elo-interactive th {{ cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 2; }}
#elo-interactive th:hover {{ color: var(--md-accent-fg-color, #06c); }}
#elo-interactive td.n {{ text-align: right; }}
#elo-interactive tr:hover td {{ background: var(--md-default-fg-color--lightest, #f5f5f5); }}
/* freeze panes: the three key columns stay put while datasets scroll */
#elo-interactive th.k1, #elo-interactive td:nth-child(1) {{ position: sticky; left: 0; z-index: 1; }}
#elo-interactive th.k2, #elo-interactive td:nth-child(2) {{ position: sticky; left: 56px; z-index: 1; }}
#elo-interactive th.k3, #elo-interactive td:nth-child(3) {{ position: sticky; left: 148px; z-index: 1; }}
#elo-interactive th.k1, #elo-interactive th.k2, #elo-interactive th.k3 {{ z-index: 3; }}
#elo-interactive .c {{ color: var(--md-default-fg-color--light, #999); font-size: 0.68rem; }}
</style>
<table>
<thead><tr>
<th class='k1 k' data-t='-1'>ELO</th>
<th class='k2 k' data-t='-2'>FIT (mean±sd)</th>
<th class='k3 k' data-t='-3'>MODEL</th>{th}
</tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<script>
(function () {{
  var t = document.querySelector('#elo-interactive table');
  if (!t) return;
  var dir = {{}};
  t.querySelectorAll('th').forEach(function (th) {{
    th.addEventListener('click', function () {{
      var i = +th.dataset.t, d = dir[i] = -(dir[i] || 1);
      var rows = Array.prototype.slice.call(t.tBodies[0].rows);
      rows.sort(function (a, b) {{
        var x = a.cells[i + 3].textContent.trim(), y = b.cells[i + 3].textContent.trim();
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

    DOC.write_text(
        "<!-- generated by examples/elo_table.py - do not edit by hand -->\n\n"
        + html
    )
    print(f"wrote {DOC} ({len(datasets)} datasets x {len(models)} models, frozen key columns)")


if __name__ == "__main__":
    main()
