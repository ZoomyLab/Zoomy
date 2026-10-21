"""The JAX symbolic solver refuses a derivative aux it does not compute (it would read 0):
an `unknown` target such as `h*zeta`, which before 2026-10-14 JAX read as 0 silently."""
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.small, pytest.mark.jax]


def test_unknown_target_raises_at_setup():
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")])))
    sm.initial_conditions = IC.Constant(constants=lambda n: [0.0, 1.0, 0.0][:n])
    nsm = NumericalSystemModel.from_system_model(sm)
    nsm.aux_registry = list(nsm.aux_registry or []) + [dict(
        kind="derivative", name="h*zeta_x", row=0, atom=None, aux_symbol=None,
        target_name="h*zeta", multi_index=(1,), target_kind="unknown")]
    s = SymbolicHyperbolicSolverJax(time_end=0.01, dt=0.01)
    with pytest.raises(ValueError, match=r"jax: the aux 'h\*zeta_x' .*target_kind 'unknown'"):
        s.setup(LSQMesh.create_1d(domain=(0, 1), n_inner_cells=8), nsm)
