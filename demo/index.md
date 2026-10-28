# Demo walkthrough

Run it yourself:

```
uv run python examples/data_layer_demo.py              # default: churn
uv run python examples/data_layer_demo.py diamonds wine_quality
```

One script, real data, every detail printed: dataset shape and column types, missing counts, numeric summaries, category inference (`n_category`), `IndexMapper` levels, grouped means and group-zscores (with inverse roundtrip check), the generator's realized variance shares, calendar/lag/trended/windowed features, the additive decomposition table, and the drop-and-warn NA behavior.

## The generator's ground truth

Where the signal lives in a generated dataset — a knob-controlled answer key for structure-recovery experiments:

## Additive decomposition

Series, trend (centered moving average), detrended, deseasonalized and residual components — residuals sit at machine zero on a clean synthetic panel:

Figures are produced by the demo when `matplotlib` and `seaborn` are installed (they are dev extras — the library itself does not depend on them).

> [!NOTE]
>
> **Missing data in the demo**
>
> The demo deliberately feeds an incomplete row to the time features and shows the `UserWarning` and the missing outputs — that behavior is covered by tests, not just prose.
