"""Shared fixtures for the zoomy_server test suite.

Key tricks:
  * The real jobs module uses a ProcessPoolExecutor so solves run in
    subprocesses. That makes test adapters (defined inside a test module)
    unreachable — subprocesses can't import them. We swap in a
    ThreadPoolExecutor for the test session so the fake adapter runs
    in-process.
  * A tiny MockSolverAdapter writes a real (tiny) HDF5 file so download
    endpoints can return a non-empty payload.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from zoomy_server import jobs as jobs_module
from zoomy_server import routes as routes_module
from zoomy_server.adapter import SolverAdapter


class MockSolverAdapter(SolverAdapter):
    """Synchronous, deterministic adapter used by the test suite.

    ``solve`` writes a minimal HDF5 file to ``output_dir`` and sleeps a
    little so tests can observe the running → complete transition.
    """

    tag = "mock"
    solve_sleep_s = 0.05  # short enough for fast tests, long enough to observe

    def solve(self, case_dir, output_dir, on_progress):
        time.sleep(self.solve_sleep_s)
        on_progress(0, 0.0, 0.01)
        # Write a real (minimal) HDF5 so /results/hdf5 can serve it.
        path = os.path.join(output_dir, "simulation.h5")
        try:
            import h5py
            with h5py.File(path, "w") as f:
                g = f.create_group("mesh")
                g.attrs["dim"] = 1
                g.create_dataset("vertices", data=[[0.0], [1.0]])
                g.create_dataset("cells", data=[[0, 1]])
                fg = f.create_group("fields")
                snap = fg.create_group("iteration_0")
                snap.create_dataset("time", data=0.0)
                snap.create_dataset("Q", data=[[1.0]])
        except ImportError:
            # Fall back to a tiny binary placeholder — the download test
            # only asserts non-empty payload.
            with open(path, "wb") as f:
                f.write(b"\x89HDF\r\n\x1a\n")  # HDF5 magic header
        on_progress(1, 0.01, 0.01)


@pytest.fixture(scope="session", autouse=True)
def _use_thread_executor():
    """Replace the module-level ProcessPoolExecutor with a ThreadPool for
    the whole test session, so tests can use adapter classes defined in
    test modules."""
    original = jobs_module.EXECUTOR
    jobs_module.EXECUTOR = ThreadPoolExecutor(max_workers=2)
    try:
        yield
    finally:
        jobs_module.EXECUTOR.shutdown(wait=False, cancel_futures=True)
        jobs_module.EXECUTOR = original


@pytest.fixture
def clean_jobs():
    """Ensure a clean JOBS dict per test (tests can submit new ones)."""
    jobs_module.JOBS.clear()
    yield
    jobs_module.JOBS.clear()


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    """Redirect JOBS_DIR into a tmp_path so test jobs don't touch the
    shared /tmp location."""
    d = tmp_path / "zoomy_jobs"
    d.mkdir()
    monkeypatch.setattr(jobs_module, "JOBS_DIR", str(d))
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


@pytest.fixture
def case_dir(tmp_path):
    """A minimal case directory with the files the adapter base class
    expects (settings.json). The MockSolverAdapter ignores the content
    so a stub is enough."""
    d = tmp_path / "case"
    d.mkdir()
    (d / "settings.json").write_text(json.dumps({}))
    return str(d)


@pytest.fixture
def client(clean_jobs, jobs_dir):
    """TestClient with the mock adapter installed on the routes."""
    adapter = MockSolverAdapter()
    routes_module.set_adapter(adapter)
    from zoomy_server.server import ZoomyServer
    server = ZoomyServer(adapter)
    return TestClient(server.app)
