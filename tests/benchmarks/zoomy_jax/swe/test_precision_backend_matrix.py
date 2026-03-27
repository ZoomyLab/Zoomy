import os
import subprocess
import sys

import pytest


@pytest.mark.large
@pytest.mark.benchmark
@pytest.mark.jax
def test_precision_backend_matrix_script_runs():
    env = os.environ.copy()
    env["LOGURU_LEVEL"] = "WARNING"
    cmd = [
        sys.executable,
        "-m",
        "tests.scripts.zoomy_core.swe.benchmark_precision_backend_matrix",
    ]
    # Heavy benchmark end-to-end. It is disabled by default via marker gating.
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (proc.stdout[-500:] + "\n" + proc.stderr[-500:])

