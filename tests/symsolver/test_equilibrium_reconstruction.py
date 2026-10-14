"""A model's `equilibrium_reconstruction` the symbolic solvers do not honour raises at setup
(before 2026-10-14 it was silently ignored): moving-equilibrium well-balancing (`bernoulli`,
`projected_bernoulli`) is not implemented (ANALYSIS 58); `audusse` needs the hydrostatic
reconstruction of a Positive* Riemann class."""
import pytest

import zoomy_core.numerics as N
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _setup(reconstruction, riemann=None):
    sm = SystemModel.from_model(SME(
        level=1, dimension=2, equilibrium_reconstruction=reconstruction,
        closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")])))
    assert sm.equilibrium_reconstruction == reconstruction
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=None if riemann is None else getattr(N, riemann))
    SymbolicHyperbolicSolverNumpy(time_end=0.01, dt=0.01).setup(
        LSQMesh.create_1d(domain=(0, 1), n_inner_cells=8), nsm)


@pytest.mark.parametrize("reconstruction", ["bernoulli", "projected_bernoulli"])
def test_moving_equilibrium_not_implemented(reconstruction):
    with pytest.raises(NotImplementedError, match=rf"equilibrium_reconstruction='{reconstruction}'"):
        _setup(reconstruction)


def test_audusse_needs_a_hydrostatic_reconstruction():
    with pytest.raises(ValueError, match="riemann=NonconservativeRusanov"):
        _setup("audusse", "NonconservativeRusanov")
    _setup("audusse")                      # PositiveNonconservativeRusanov (the default)
    _setup("none", "NonconservativeRusanov")
