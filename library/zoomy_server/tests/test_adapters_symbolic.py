"""Each solver adapter runs one smoke case in-process on its symbolic solver: a 1-D SWE dam
break between walls (40 cells, t 0.2), from settings (no run.py) and, for NumPy, from the
generated run cell (`zoomy_prepost.case._run_code`) as run.py.  Pass: `simulation.h5` with the
final snapshot at `time_end`, finite, and the wall-bounded mass kept to round-off."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL = '''import numpy as np
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models import SWE
from zoomy_core.systemmodel import SystemModel

model = SystemModel.from_model(SWE(
    dimension=1, parameters={"g": 9.81, "n_m": 0.0},
    boundary_conditions=BC.BoundaryConditions(
        [BC.FromModel(tag="left", definition="wall"),
         BC.FromModel(tag="right", definition="wall")]),
    initial_conditions=IC.RP(
        high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),
        low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),
        jump_position_x=5.0)))
display(model.describe())
'''
MESH = '''from zoomy_core.mesh import LSQMesh

mesh = LSQMesh.create_1d((0.0, 10.0), 40)
'''
SETTINGS = {"time_end": 0.2, "cfl": 0.9, "output_snapshots": 4,
            "domain": [0.0, 10.0], "n_cells": 40}


def _case(tmp_path, backend, run_cell=False):
    d = tmp_path / "case"
    d.mkdir()
    (d / "model.py").write_text(MODEL)
    (d / "settings.json").write_text(json.dumps(dict(SETTINGS, backend=backend)))
    if run_cell:
        from zoomy_prepost.case import _run_code
        (d / "mesh.py").write_text(MESH)
        (d / "run.py").write_text(_run_code())
    out = tmp_path / "out"
    out.mkdir()
    return str(d), str(out)


def _check(out):
    import h5py
    with h5py.File(os.path.join(out, "simulation.h5"), "r") as f:
        keys = sorted(f["fields"], key=lambda k: int(k.split("_")[-1]))
        first, last = f["fields"][keys[0]], f["fields"][keys[-1]]
        Q0, Q, t = np.asarray(first["Q"]), np.asarray(last["Q"]), float(np.asarray(last["time"]))
    assert t == pytest.approx(SETTINGS["time_end"], abs=1e-12)
    assert np.all(np.isfinite(Q))
    assert abs(Q[1].sum() - Q0[1].sum()) <= 1e-12 * Q0[1].sum()
    assert np.abs(Q - Q0).max() > 1e-3            # it marched
    return Q


def _progress():
    calls = []
    return calls, lambda *a: calls.append(a)


def _cxx_available(module):
    """The C++ backend runs here: inside its image, or apptainer and the image on the host."""
    try:
        mod = __import__(module, fromlist=["x"])
    except ImportError:
        return False
    if module == "zoomy_amrex.symsolver":
        return mod.in_image() or (shutil.which("apptainer") and mod.SIF.exists())
    from zoomy_foam import _pipeline
    return _pipeline.in_image() or (shutil.which("apptainer") and _pipeline.SIF.exists())


def test_numpy_settings(tmp_path):
    from zoomy_server.adapters.numpy import NumpyAdapter
    case, out = _case(tmp_path, "numpy")
    calls, cb = _progress()
    NumpyAdapter().solve(case, out, cb)
    _check(out)
    assert calls[-1][1] == pytest.approx(SETTINGS["time_end"])


def test_numpy_run_cell(tmp_path):
    from zoomy_server.adapters.numpy import NumpyAdapter
    case, out = _case(tmp_path, "numpy", run_cell=True)
    _, cb = _progress()
    NumpyAdapter().solve(case, out, cb)
    _check(out)


def test_jax_settings(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("zoomy_jax")
    from zoomy_server.adapters.jax import JaxAdapter
    from zoomy_server.adapters.numpy import NumpyAdapter
    case, out = _case(tmp_path, "jax")
    _, cb = _progress()
    JaxAdapter().solve(case, out, cb)
    Q = _check(out)
    (tmp_path / "np").mkdir()
    case2, out2 = _case(tmp_path / "np", "numpy")
    NumpyAdapter().solve(case2, out2, cb)
    assert np.abs(Q - _check(out2)).max() <= 1e-12          # JAX = NumPy (round-off)


@pytest.mark.parametrize("module,adapter", [
    ("zoomy_amrex.symsolver", "zoomy_server.adapters.amrex.AmrexAdapter"),
    ("zoomy_foam.symsolver", "zoomy_server.adapters.foam.FoamAdapter")])
def test_cxx_settings(tmp_path, module, adapter):
    if not _cxx_available(module):
        pytest.skip(f"{module}: neither inside its image nor apptainer + image here")
    import importlib
    from zoomy_server.adapters.numpy import NumpyAdapter
    mod_path, cls = adapter.rsplit(".", 1)
    case, out = _case(tmp_path, "amrex" if "amrex" in module else "foam")
    _, cb = _progress()
    getattr(importlib.import_module(mod_path), cls)().solve(case, out, cb)
    Q = _check(out)
    (tmp_path / "np").mkdir()
    case2, out2 = _case(tmp_path / "np", "numpy")
    NumpyAdapter().solve(case2, out2, cb)
    assert np.abs(Q - _check(out2)).max() <= 1e-12          # C++ = NumPy (round-off, order 1)


@pytest.mark.parametrize("module,adapter", [
    ("zoomy_amrex.symsolver", "zoomy_server.adapters.amrex.AmrexAdapter"),
    ("zoomy_foam.symsolver", "zoomy_server.adapters.foam.FoamAdapter")])
def test_cxx_run_cell(tmp_path, module, adapter):
    """The generated run cell's C++ branch (settings["backend"] amrex / OpenFOAM) as run.py."""
    if not _cxx_available(module):
        pytest.skip(f"{module}: neither inside its image nor apptainer + image here")
    import importlib
    mod_path, cls = adapter.rsplit(".", 1)
    case, out = _case(tmp_path, "amrex" if "amrex" in module else "OpenFOAM", run_cell=True)
    _, cb = _progress()
    getattr(importlib.import_module(mod_path), cls)().solve(case, out, cb)
    if not os.path.exists(os.path.join(out, "simulation.h5")):      # foam: run in place
        shutil.copy(os.path.join(case, "simulation.h5"), out)
    _check(out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
