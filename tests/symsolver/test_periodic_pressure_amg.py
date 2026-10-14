"""The AMG pressure solve on a periodic channel (NumPy backend; the JAX runtime assembles the same
pattern): the colour-assembled matrix must carry the cross-seam couplings.

The derivative aux of the pressure at a seam cell samples the partner cell across the seam (the
seam face's ghost), which the LSQ stencil (`lsq_neighbors`) does not list; the assembly pattern
lacked those entries, so the AMG-preconditioned GMRES solved a different matrix and the true
residual stayed O(1) (0.78 on the calibration case's periodic VAM(3) channel, 2.2 here).
`operator_neighbors` adds the periodic partners."""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.symsolver.numpy_backend import SymbolicChorinSolverNumpy
from zoomy_core.symsolver.numpy_backend.assembly import operator_neighbors
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_core.symsolver.strategies.chorin import predictor_nsm
from zoomy_core.systemmodel.system_model import _DT_SYMBOL, SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]
pyamg = pytest.importorskip("pyamg")


def _split():
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left")]
    model = VAM(level=1, dimension=2, boundary_conditions=BC.BoundaryConditions(bcs),
                closures=[Newtonian(), NavierSlip(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    return predictor_nsm(r.SM_pred, r.SM_press, r.SM_corr), r.SM_press, r.SM_corr


def _run(solver):
    s = SymbolicChorinSolverNumpy(time_end=0.05, dt=0.01, mood=False, cfl_check_every=0,
                                  pressure_linear_solver=solver)
    Q, A = s.setup(LSQMesh.create_1d(domain=(0, 2), n_inner_cells=16), *_split())
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = np.asarray(Q)
    Q[names.index("h")] = 0.5 + 0.05 * np.sin(np.pi * x)
    Q[names.index("q_0")] = 0.1 * Q[names.index("h")]
    _, Q, _ = s.run(Q, s.initial_aux(Q, A))
    return np.asarray(Q), names


def test_operator_neighbors_hold_the_periodic_partner():
    pred, _, _ = _split()
    mesh = ensure_lsq_mesh(LSQMesh.create_1d(domain=(0, 2), n_inner_cells=16), pred)
    mesh.resolve_periodic_bcs(pred._bc_source)
    nb = operator_neighbors(mesh)
    assert 15 in nb[0] and 0 in nb[15]
    assert 15 not in np.asarray(mesh.lsq_neighbors)[0]     # the stencil alone lacks it


def test_periodic_pyamg_equals_gmres():
    """The default pressure solver ('auto' = PyAMG when installed) on a periodic VAM(1)
    channel: converges and equals the matrix-free GMRES run (both at the 1e-10 pressure
    tolerance).  Before: 'relative residual 2.168e+00 > accept gate'."""
    a, names = _run(LinearSolver.auto)
    b, _ = _run(LinearSolver.gmres)
    assert np.abs(a[names.index("P_0")]).max() > 1e-2           # a live pressure
    assert np.abs(a - b).max() <= 1e-9
