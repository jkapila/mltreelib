import sys
import time
from pathlib import Path

# Add examples/ to sys.path
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from elo_budget import run_with_budget
from elo_common import ScoreSheet, calculate_elo


def test_run_with_budget_success():
    cmd = [sys.executable, "-c", "import time; time.sleep(0.05); print('RESULT ok')"]
    out, code, timed_out = run_with_budget(cmd, budget_s=5.0)
    assert not timed_out
    assert code == 0
    assert "RESULT ok" in out


def test_run_with_budget_timeout_kill():
    # Process sleeps for 10s with budget of 1s; should be killed via SIGKILL
    t0 = time.monotonic()
    cmd = [sys.executable, "-c", "import time; time.sleep(10.0); print('UNREACHABLE')"]
    out, code, timed_out = run_with_budget(cmd, budget_s=1.0)
    dt = time.monotonic() - t0
    assert timed_out
    assert dt < 4.0  # Process killed and reaped promptly
    assert "UNREACHABLE" not in out


def test_scoresheet_upsert_and_done_pairs(tmp_path):
    csv_file = tmp_path / "test_scores.csv"
    columns = ["dataset", "model", "rmse", "mae", "fit_s"]
    sheet = ScoreSheet(csv_file, columns)

    assert sheet.done_pairs() == set()

    sheet.upsert("ds1", "m1", rmse=1.23, mae=0.99, fit_s=0.5)
    assert ("ds1", "m1") in sheet.done_pairs()

    # Upsert overwrite
    sheet.upsert("ds1", "m1", rmse=1.11, mae=0.88, fit_s=0.4)
    assert ("ds1", "m1") in sheet.done_pairs()
    
    # Verify file content
    with open(csv_file, "r") as fh:
        lines = [line.strip() for line in fh.readlines() if line.strip()]
    assert len(lines) == 2  # header + 1 row
    assert "1.1100" in lines[1]


def test_calculate_elo_ratings():
    rows = [
        {"dataset": "ds1", "model": "better", "rmse": 1.0},
        {"dataset": "ds1", "model": "worse", "rmse": 2.0},
        {"dataset": "ds2", "model": "better", "rmse": 1.5},
        {"dataset": "ds2", "model": "worse", "rmse": 2.5},
    ]
    ratings = calculate_elo(rows, metric_col="rmse", lower_is_better=True)
    assert ratings["better"] > 1000.0
    assert ratings["worse"] < 1000.0
    assert ratings["better"] + ratings["worse"] == 2000.0  # zero-sum Elo updates
