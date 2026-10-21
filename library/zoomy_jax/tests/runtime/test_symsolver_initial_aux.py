"""`SymbolicHyperbolicSolverJax.initial_aux` takes array-likes: NumPy arrays (and lists) are
converted, so the aux equals the one of the JAX arrays.  Before, a NumPy `Q` reached the
vmapped BC kernels and raised `TracerArrayConversionError` (the boundary ghosts of the
derivative aux index it with a traced face)."""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.small, pytest.mark.jax]


def test_initial_aux_accepts_numpy():
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    sm = SystemModel.from_model(SME(
        level=1, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BC.BoundaryConditions([BC.FromModel(tag=t, definition="wall")
                                                   for t in ("left", "right")])))
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0.0, 0.5 + 0.05 * np.exp(-(x[0] - 5.0) ** 2), 0.1, 0.0]))
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2))
    assert any(e["kind"] == "derivative" for e in nsm.aux_derivatives)  # the ghost path runs
    s = SymbolicHyperbolicSolverJax(time_end=0.02, dt=0.005, mood=False)
    Q, A = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20), nsm)
    ref = np.asarray(s.initial_aux(Q, A))
    Qn, An = np.asarray(Q), np.asarray(A)
    assert np.array_equal(np.asarray(s.initial_aux(Qn, An)), ref)
    assert np.array_equal(np.asarray(s.initial_aux(Qn.tolist(), An.tolist(),
                                                   np.asarray(s.parameters))), ref)
    t, Q1, _ = s.run(Qn, s.initial_aux(Qn, An))
    assert float(t) == pytest.approx(0.02) and np.isfinite(np.asarray(Q1)).all()
