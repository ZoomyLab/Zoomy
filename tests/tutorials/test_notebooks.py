"""Execute every ``tutorials/**/*.ipynb`` on the CPU: one test per notebook.

    pytest tests/tutorials -m tutorial -rs

A notebook whose ``metadata.zoomy`` tags ask for more than ``requires:cpu`` (``gpu``, ``mpi``,
``container``) is collected and skipped, with that reason.  The kernel runs this interpreter
with ``PYTHONNOUSERSITE=1``, JAX on the CPU and a non-interactive matplotlib; ``TUTORIAL_TIMEOUT``
(seconds, default 900) is the budget of one notebook, all cells together.  The metadata is
validated by ``docs/scripts/generate_tutorial_index.py``; a notebook with bad metadata fails here
too.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "scripts"))
import generate_tutorial_index as index  # noqa: E402

pytestmark = pytest.mark.tutorial

TIMEOUT = float(os.environ.get("TUTORIAL_TIMEOUT", "900"))
KERNEL = "zoomy-tutorial"
NOTEBOOKS = index.scan(index.SRC)


def pytest_generate_tests(metafunc):
    if "notebook" not in metafunc.fixturenames:
        return
    if not NOTEBOOKS:
        metafunc.parametrize(
            "notebook",
            [pytest.param(None, id="none", marks=pytest.mark.skip(reason="no notebook under tutorials/"))],
        )
        return
    metafunc.parametrize("notebook", [pytest.param(nb, id=nb.rel) for nb in NOTEBOOKS])


@pytest.fixture
def cpu_kernel(tmp_path, monkeypatch):
    """A kernelspec that starts this interpreter, and the environment of a CPU-only run."""
    spec = tmp_path / "kernels" / KERNEL
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": KERNEL, "language": "python",
    }))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return KERNEL


def execute(path: Path, kernel: str, budget: float) -> None:
    """Run the cells of ``path`` in order; CellExecutionError on a failing cell, TimeoutError
    when the cells together need more than ``budget`` seconds."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, kernel_name=kernel, timeout=math.ceil(budget), startup_timeout=120,
                            resources={"metadata": {"path": str(path.parent)}})
    deadline = time.monotonic() + budget
    client.reset_execution_trackers()
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{path.name}: {budget:.0f} s used up before cell {i}")
            client.timeout = math.ceil(left)
            client.execute_cell(cell, i)


def test_tutorial_notebook(notebook, cpu_kernel):
    assert not notebook.errors, "\n".join(notebook.errors)
    if not notebook.runs_on_cpu:
        pytest.skip(f"requires {', '.join(notebook.requires)}: not run on the CPU runner "
                    "(tests/tutorials/README.md)")
    execute(index.SRC / notebook.rel, cpu_kernel, TIMEOUT)
