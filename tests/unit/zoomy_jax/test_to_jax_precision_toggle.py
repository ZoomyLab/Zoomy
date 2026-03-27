import os
import subprocess
import sys

import pytest


pytest.importorskip("jax")


def _run_dtype(enable_x64: str) -> str:
    env = os.environ.copy()
    env["ZOOMY_JAX_ENABLE_X64"] = enable_x64
    code = (
        "import jax.numpy as jnp; "
        "from zoomy_jax.transformation import to_jax; "
        "print(jnp.asarray(1.0).dtype)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_to_jax_respects_precision_toggle_env():
    dtype32 = _run_dtype("0")
    dtype64 = _run_dtype("1")
    assert "float32" in dtype32
    assert "float64" in dtype64

