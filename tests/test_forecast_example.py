"""Check that the forecasting example uses only prior-day outcomes."""

from pathlib import Path
from runpy import run_path

past_sales_mean = run_path(str(Path(__file__).parents[1] / "examples/guides/timeseries_forecasting.py"))["past_sales_mean"]


def test_forecast_window_excludes_current_day():
    keys = ["store"] * 3
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]

    assert past_sales_mean(keys, dates, [1.0, 2.0, 3.0]) == [None, 1.0, 1.5]
    assert past_sales_mean(keys, dates, [1.0, 2.0, 3.0]) == past_sales_mean(
        keys, dates, [1.0, 2.0, 300.0]
    )
