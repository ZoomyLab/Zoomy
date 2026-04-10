import numpy as np
import pytest

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.numpy
def test_imex_numpy_jv_backends_run_and_match():
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80, lsq_degree=2)
    n = mesh.n_inner_cells

    model_a = ClassicalGreenNaghdi1D()
    solver_a = IMEXSourceSolver(time_end=0.06, compute_dt=timestepping.adaptive(CFL=0.5))
    object.__setattr__(solver_a, "source_mode", "auto")
    object.__setattr__(solver_a, "jv_backend", "analytic")
    object.__setattr__(solver_a, "implicit_maxiter", 6)
    object.__setattr__(solver_a, "gmres_maxiter", 30)
    Qa, _ = solver_a.solve(mesh, model_a, write_output=False)

    model_b = ClassicalGreenNaghdi1D()
    solver_b = IMEXSourceSolver(time_end=0.06, compute_dt=timestepping.adaptive(CFL=0.5))
    object.__setattr__(solver_b, "source_mode", "auto")
    object.__setattr__(solver_b, "jv_backend", "fd")
    object.__setattr__(solver_b, "fd_eps", 1e-7)
    object.__setattr__(solver_b, "implicit_maxiter", 6)
    object.__setattr__(solver_b, "gmres_maxiter", 30)
    Qb, _ = solver_b.solve(mesh, model_b, write_output=False)

    Qa = np.asarray(Qa[:, :n], dtype=float)
    Qb = np.asarray(Qb[:, :n], dtype=float)
    diff = Qa - Qb
    l2 = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))

    assert np.isfinite(Qa).all()
    assert np.isfinite(Qb).all()
    assert l2 < 1e-8
    assert linf < 1e-7

