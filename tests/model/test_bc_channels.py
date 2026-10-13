"""The three BC channels (`face`, `ghost`, `normal_gradient`) and their overrides: a BC given as
`face=` / `ghost=` / `normal_gradient=` supplies that channel on the slots of its `on`, the rest
keep the host BC's own value.
"""
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.model, pytest.mark.small]


@pytest.fixture(scope="module")
def sm():
    walls = BC.BoundaryConditions([BC.Wall(tag=t) for t in ("left", "right")])
    return SystemModel.from_model(SME(level=0, dimension=2, boundary_conditions=walls,
                                      closures=[Newtonian(), ManningFriction(), StressFree()]))


def channels(sm, bc):
    """(face, ghost, normal gradient) of `bc`, bound to `sm` as attach does."""
    def prepare(b):
        if hasattr(b, "resolve"):
            b.resolve(sm)
        b.bind_on([str(v) for v in sm.state])
        for inner in b.channels().values():
            prepare(inner)
    prepare(bc)
    args = (sm.time, sm.position, sp.Symbol("distance"), sm.variables, sm.aux_variables,
            sm.parameters, sm.normal)
    return (list(bc.compute_boundary_condition(*args)),
            list(bc.compute_ghost_state(*args)), list(bc.compute_boundary_gradient(*args)))


def test_normal_gradient_channel(sm):
    wall_face, wall_ghost, wall_grad = channels(sm, BC.Wall(tag="left"))
    face, ghost, grad = channels(sm, BC.Wall(tag="left", normal_gradient=BC.Flux(on="h",
                                                                                 gradient=2.0)))
    h = [str(v) for v in sm.state].index("h")
    assert face == wall_face and ghost == wall_ghost
    assert float(grad[h]) == 2.0 and all(g == w for i, (g, w) in enumerate(zip(grad, wall_grad)) if i != h)


def test_ghost_channel_keeps_the_face(sm):
    wall_face, _, _ = channels(sm, BC.Wall(tag="left"))
    face, ghost, _ = channels(sm, BC.Wall(tag="left", ghost=BC.Dirichlet(on="momentum", value=0)))
    q = [str(v) for v in sm.state].index("q_0")
    assert face == wall_face
    assert sp.simplify(ghost[q] + sm.variables[q]) == 0       # Dirichlet 0: ghost = -q
