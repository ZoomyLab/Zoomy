"""`run` traced first inside a caller's `jax.jit`, then called again (eagerly and jitted): the
shape-keyed program cache of `hoisted_jit` must not keep the caller's tracers."""
import numpy as np
import jax
import jax.numpy as jnp

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax


def test_run_inside_outer_jit_then_again():
    jax.config.update("jax_enable_x64", True)
    bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
    model = SME(level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=bcs)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 1.0 + 0.2 * np.exp(-(x[0] - 5) ** 2), 1.0]))
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))
    s = SymbolicHyperbolicSolverJax(time_end=0.05, dt=1e-2, mood=False)
    Q0, A0 = s.setup(LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20), nsm)
    p = s.parameters

    def f(p):
        return s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)[1]

    jitted = jax.jit(lambda p: jnp.sum(f(p)))(p)
    eager = jnp.sum(f(p))
    again = jax.jit(lambda p: jnp.sum(f(p)))(p)
    assert np.isfinite(float(jitted))
    assert float(jitted) == float(eager) == float(again)
