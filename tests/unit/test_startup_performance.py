"""
Regression guard for CLI startup time.

Keeping ``deet --help`` fast requires careful attention to lazy loading.

This test guards against degradation of startup time
by spawning a cold interpreter (so it measures real import cost, which an
in-process CliRunner would not), and takes the best of several runs to limit
scheduling/cache noise. The budget can be relaxed for slow environments via the
``DEET_STARTUP_BUDGET_S`` environment variable
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTUP_BUDGET_S = float(os.environ.get("DEET_STARTUP_BUDGET_S", "0.5"))
MEASURED_RUNS = 3


def _time_help_invocation() -> float:
    """Run ``deet --help`` in a fresh interpreter and return wall-clock seconds."""
    start = time.perf_counter()
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "deet.scripts.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    # Guard against measuring a fast crash instead of a working --help.
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
    return elapsed


def test_deet_help_starts_within_budget() -> None:
    _time_help_invocation()  # warm-up: prime bytecode cache / filesystem cache
    best = min(_time_help_invocation() for _ in range(MEASURED_RUNS))
    assert best < STARTUP_BUDGET_S, (
        f"`deet --help` cold start took {best:.2f}s, over the "
        f"{STARTUP_BUDGET_S:.2f}s budget. A heavy import was probably added to the "
        f"startup path -- profile with "
        f"`python -X importtime -m deet.scripts.cli`. If the environment is simply "
        f"slow, relax the budget via the DEET_STARTUP_BUDGET_S env var."
    )
