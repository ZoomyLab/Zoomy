"""The JAX march raises instead of ending silently (ANALYSIS 49, item 1): a step that does not
advance the time stops the `lax.while_loop` at the last state and `check.march` (a host
callback) raises; `run` re-raises it as `FloatingPointError`.  Forward AD through a good march
is unaffected.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, PositiveNonconservativeHLL
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

pytestmark = [pytest.mark.small, pytest.mark.jax]


def _dambreak(n=20):
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 1.0 + (x[0] < 0.5), 0.0]))
    return (NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeHLL),
            LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n))


@pytest.mark.parametrize("dt", [None, 0.01])
def test_non_finite_state_raises(dt):
    nsm, mesh = _dambreak()
    s = SymbolicHyperbolicSolverJax(time_end=0.05, cfl=0.4, dt=dt)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    assert float(t) == pytest.approx(0.05) and np.isfinite(np.asarray(Q)).all()
    Qn = jnp.asarray(Q0).at[1, 7].set(jnp.nan)
    with pytest.raises(FloatingPointError, match="non-finite state" if dt else
                       r"stopped at t = 0 .*initial state"):
        s.run(Qn, A)
    Qb = jnp.asarray(Q0).at[2, 7].set(1e300)
    with pytest.raises(FloatingPointError, match="zoomy march"):
        s.run(Qb, A)
    # forward AD through the guarded loop: d h_end / d h0 is finite
    _, tan = jax.jvp(lambda q: s.run(q, A)[1], (jnp.asarray(Q0),),
                     (jnp.zeros_like(Q0).at[1].set(1.0),))
    assert np.isfinite(np.asarray(tan)).all()
