"""Timing of the symbolic solvers (JAX, foam, AMReX, NumPy): each test marches one
representative case ``ZOOMY_TIMING_REPS`` (3) times, checks the final state,
and appends a record to timings.jsonl (README.md).

    pytest tests/symsolver -m benchmark            # on an otherwise idle machine
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_cases  # noqa: E402
import timing_paths  # noqa: E402
import timing_records  # noqa: E402
import timing_run  # noqa: E402

pytestmark = pytest.mark.benchmark


def _gpu():
    if os.environ.get("ZOOMY_TIMING_GPU", "1") == "0":
        return False
    return bool(timing_records.metadata()["n_gpus"])


def _skip_unavailable(backend, config):
    if backend in ("foam", "amrex") and shutil.which("apptainer") is None:
        pytest.skip("no apptainer (OpenFOAM / AMReX containers)")
    if config.startswith("gpu") and not _gpu():
        pytest.skip("no GPU (nvidia-smi), or ZOOMY_TIMING_GPU=0")
    if backend == "jax" and config.startswith("gpu") and shutil.which("micromamba") is None:
        pytest.skip("JAX gpu1 needs micromamba and the zoomy-cuda env")


@pytest.mark.parametrize("case,backend,config", timing_cases.params(),
                         ids=lambda v: str(v))
def test_timing(case, backend, config):
    _skip_unavailable(backend, config)
    tc = timing_cases.CASES[case]
    reps = int(os.environ.get("ZOOMY_TIMING_REPS", "3"))
    work = timing_paths.work_dir() / case / backend / config
    try:
        m = timing_run.measure(tc, backend, config, reps, work)
    except Exception as e:
        timing_records.append(timing_records.failure(case, backend, config,
                                                     f"error: {type(e).__name__}: {str(e)[-300:]}"))
        raise
    problems, diff = [], None
    if not np.isfinite(m.Q).all():
        problems.append("non-finite final state")
    if abs(m.t - tc.t_end) > 1e-8 * max(1.0, tc.t_end):
        problems.append(f"final time {m.t!r} != t_end {tc.t_end!r}")
    if m.steps_run is not None and m.steps_run != tc.steps:
        problems.append(f"{m.steps_run} loop iterations, expected {tc.steps}")
    if config.startswith("gpu") and backend == "jax" and m.device not in ("gpu", "cuda"):
        problems.append(f"JAX ran on {m.device!r}, not a GPU")
    if backend == "jax" and config == "serial":
        if not problems:
            timing_run.store_reference(tc, m)
    elif tc.agree.get(f"{backend}:{config}") and not problems:
        diff = timing_run.agreement(tc.agree[f"{backend}:{config}"], m.Q, m.names,
                                    timing_run.reference(tc))
        if not diff["ok"]:
            problems.append(f"{diff['metric']} vs JAX serial {diff['value']:.3e}, need "
                            f"{diff['tol']} ({diff['source']})")
    timing_records.append(timing_records.record(
        case, backend, config, cells=m.cells, steps=tc.steps, dt=tc.dt, march=m.march, setup=m.setup,
        compile=m.compile, device=m.device, detail=m.detail, diff=diff,
        status="ok" if not problems else "FAIL: " + "; ".join(problems)))
    assert not problems, "; ".join(problems)
