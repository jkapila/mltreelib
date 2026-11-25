"""Hard per-call budget for benchmark fits: one secondary process per
(algorithm x dataset), own session, process-group kill on timeout.

Codex-reviewed design (see AGENTS.md - codex is a standing buddy):
- workers launch with [sys.executable, ...] under start_new_session=True, so
  the worker's pid == its process-group id and every descendant inherits it;
- one time.monotonic() deadline covers launch + load + fit + predict + output;
- on timeout: SIGKILL to the group (main + secondary PIDs die together),
  tolerate ProcessLookupError, then wait() to reap (kill and reap are
  separate steps);
- SIGALRM is NOT used: handlers only run between bytecodes, so native fits
  (LightGBM/CatBoost) ignore them.
"""

import os
import signal
import subprocess
import time


def run_with_budget(argv, budget_s=300.0, env=None, cwd=None):
    """Run argv as a secondary process group with a hard wall-clock budget.

    Returns (stdout_text, returncode, timed_out). stdout is drained via
    communicate() with the remaining budget; stderr passes through.
    """
    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=None,  # inherits: diagnostics go straight to the log
        env=env,
        cwd=cwd,
        start_new_session=True,  # own session: pid == pgid, descendants join
    )
    pgid = proc.pid  # with a new session the child leads its own group
    timed_out = False
    try:
        remaining = budget_s - (time.monotonic() - start)
        out, _ = proc.communicate(timeout=max(remaining, 0.01))
    except subprocess.TimeoutExpired:
        timed_out = True
        for sig in (signal.SIGKILL,):
            try:
                os.killpg(pgid, sig)  # main + secondary PIDs together
            except ProcessLookupError:
                pass  # already dead and reaped
        try:
            out, _ = proc.communicate(timeout=30)  # reap + drain remnants
        except subprocess.TimeoutExpired:
            out = b""
            proc.kill()
            proc.wait()
    return (out or b"").decode("utf-8", "replace"), proc.returncode, timed_out
