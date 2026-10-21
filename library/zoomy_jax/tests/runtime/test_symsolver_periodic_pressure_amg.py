"""JAX Chorin on a periodic VAM(1) channel: the assembled pressure solvers (pyamg, hypre, the
CPU default 'auto') equal the matrix-free GMRES.  Their colour assembly takes its sparsity from
`operator_neighbors` (the LSQ stencil and the periodic partners); with the stencil alone the
cross-seam entries were missing and every assembled solve failed its accept gate."""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_core.symsolver.strategies.chorin import predictor_nsm
from zoomy_core.systemmodel.system_model import _DT_SYMBOL, SystemModel

pytestmark = [pytest.mark.small, pytest.mark.jax]


def _run(solver):
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left")]
    model = VAM(level=1, dimension=2, boundary_conditions=BC.BoundaryConditions(bcs),
                closures=[Newtonian(), NavierSlip(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    s = SymbolicChorinSolverJax(time_end=0.05, dt=0.01, mood=False, cfl_check_every=0,
                                pressure_linear_solver=solver)
    Q, A = s.setup(LSQMesh.create_1d(domain=(0, 2), n_inner_cells=16),
                   predictor_nsm(r.SM_pred, r.SM_press, r.SM_corr), r.SM_press, r.SM_corr)
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = np.asarray(Q).copy()
    Q[names.index("h")] = 0.5 + 0.05 * np.sin(np.pi * x)
    Q[names.index("q_0")] = 0.1 * Q[names.index("h")]
    _, Q, _ = s.run(Q, s.initial_aux(Q, A))
    return np.asarray(Q)


def test_periodic_assembled_pressure_equals_gmres():
    pytest.importorskip("pyamg")
    ref = _run(LinearSolver.gmres)
    for solver in (LinearSolver.pyamg, LinearSolver.auto):
        assert np.abs(_run(solver) - ref).max() <= 1e-9
