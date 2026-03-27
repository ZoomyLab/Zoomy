import numpy as np
import pytest

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from tests.common.baseline_store import assert_or_create_array_baseline
from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver


pytest.importorskip("jax")
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax


@pytest.mark.small
@pytest.mark.regression
@pytest.mark.numpy
@pytest.mark.jax
def test_imex_numpy_vs_jax_gn_small_regression():
    mesh_np = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80, lsq_degree=2)
    model_np = ClassicalGreenNaghdi1D()
    solver_np = IMEXSourceSolver(time_end=0.06, compute_dt=timestepping.adaptive(CFL=0.5))
    object.__setattr__(solver_np, "source_mode", "auto")
    object.__setattr__(solver_np, "jv_backend", "analytic")
    object.__setattr__(solver_np, "implicit_maxiter", 6)
    object.__setattr__(solver_np, "gmres_maxiter", 30)
    Q_np, _ = solver_np.solve(mesh_np, model_np, write_output=False)
    n = mesh_np.n_inner_cells
    Q_ref = np.asarray(Q_np[:, :n], dtype=float)

    mesh_jax = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80, lsq_degree=2)
    model_jax = ClassicalGreenNaghdi1D()
    solver_jax = IMEXSourceSolverJax(time_end=0.06, compute_dt=timestepping.adaptive(CFL=0.5))
    object.__setattr__(solver_jax, "source_mode", "auto")
    object.__setattr__(solver_jax, "jv_backend", "ad")
    object.__setattr__(solver_jax, "implicit_maxiter", 6)
    object.__setattr__(solver_jax, "gmres_maxiter", 30)
    Q_jax, _ = solver_jax.solve(mesh_jax, model_jax, write_output=False)
    Q_jax_np = np.asarray(Q_jax[:, :n], dtype=float)

    diff = Q_jax_np - Q_ref
    l2 = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    assert np.isfinite(l2)
    assert np.isfinite(linf)
    assert l2 < 1e-8
    assert linf < 1e-7

    assert_or_create_array_baseline(
        "imex_numpy_vs_jax_gn_small",
        {
            "Q_numpy": Q_ref,
            "Q_jax": Q_jax_np,
            "l2": np.asarray([l2]),
            "linf": np.asarray([linf]),
        },
        rtol=1e-8,
        atol=1e-10,
    )

