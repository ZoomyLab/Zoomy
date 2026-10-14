"""The NumPy backend distributed (`comm=`, `numpy_backend.parallel`): `mpiexec -n 3` runs of
`_numpy_mpi_run.py` against serial runs on rank 0, and the serial backend without mpi4py.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.solver, pytest.mark.small]
HERE = Path(__file__).resolve().parent


def _mpirun(what, n=3):
    """The JSON result of `_numpy_mpi_run.py what` on `n` ranks."""
    pytest.importorskip("mpi4py")
    exe = shutil.which("mpiexec", path=str(Path(sys.executable).parent)) or shutil.which("mpiexec")
    if exe is None:
        pytest.skip("no mpiexec")
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run([exe, "-n", str(n), sys.executable, str(HERE / "_numpy_mpi_run.py"),
                          what], capture_output=True, text=True, timeout=600, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_distributed_hyperbolic_equals_serial():
    """O2 minmod + MOOD in walls and O2 periodic in x on 3 ranks = serial (round-off), the mass
    integral a collective = serial, halo copies = their owners, a NaN on one rank is the global
    min on every rank (MPI.MIN alone drops it)."""
    r = _mpirun("hyperbolic")
    for case in ("walls_mood", "periodic"):
        assert r[case]["rel"] <= 1e-13, r
        assert r[case]["t"][0] == r[case]["t"][1] == 1.0
        assert r[case]["mass"] <= 1e-14
    assert r["halo"] and r["nan_min"] and r["any"], r


def test_distributed_linsolve_equals_serial():
    """`linsolve` of a halo-coupled two-row operator on 3 ranks: PETSc and the distributed
    FGMRES = serial GMRES (tolerance 1e-13)."""
    pytest.importorskip("petsc4py")
    r = _mpirun("linsolve")
    assert r["petsc"] <= 1e-11 and r["krylov"] <= 1e-11, r


def test_serial_backend_without_mpi4py():
    """The serial backend imports and marches with mpi4py and petsc4py unavailable (the browser):
    they are imported only by a distributed run."""
    code = (
        "import sys; sys.modules['mpi4py'] = None; sys.modules['petsc4py'] = None\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "from test_numpy_backend import bump\n"
        "from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy\n"
        "mesh, nsm = bump(8)\n"
        "s = SymbolicHyperbolicSolverNumpy(time_end=0.2, mood=False)\n"
        "Q, A = s.setup(mesh, nsm)\n"
        "t, Q, A = s.run(Q, A)\n"
        "assert t == 0.2 and 'mpi4py' not in [m for m in sys.modules if sys.modules[m]]\n"
        "print('ok')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600, env=env)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-3000:]
