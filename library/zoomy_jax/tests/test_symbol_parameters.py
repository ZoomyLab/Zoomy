"""Parameters by symbol on the JAX solver: partial mappings, the Parameters pytree, RunResult."""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.symsolver.solver import AdMode, SymbolicHyperbolicSolverJax


@pytest.fixture(scope="module")
def case():
    Q = SME.state(level=0, dimension=2)
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, parameters={"n": 0.03}, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=[BC.Dirichlet("left", on=Q.q_0, value=0.5),
                             BC.Dirichlet("right", on=Q.h, value=0.6 - Q.b)]))
    s = SymbolicHyperbolicSolverJax(time_end=2.0, dt=0.02, mood=False, ad_mode=AdMode.reverse)
    Q0, A0 = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20),
                     NumericalSystemModel.from_system_model(sm))
    x = jnp.asarray(np.linspace(0.25, 9.75, 20))
    Qi = jnp.stack([0.01 * x, 0.5 + 0 * x, 0.5 + 0 * x])
    return s, Qi, A0, sm


def test_grad_by_symbol(case):
    s, Qi, A0, sm = case
    n = sm.parameters.n
    J = lambda Q: jnp.sum(Q[1] ** 2)                                    # noqa: E731
    J_of_n = lambda v: J(s.run(Qi, s.initial_aux(Qi, A0), parameters={n: v}).Q)   # noqa: E731
    g = jax.grad(J_of_n)(s.parameters[n])
    eps = 1e-6
    fd = (J_of_n(s.parameters[n] + eps) - J_of_n(s.parameters[n] - eps)) / (2 * eps)
    assert abs(g - fd) < 1e-6 * abs(fd)
    G = jax.grad(lambda p: J(s.run(Qi, s.initial_aux(Qi, A0), parameters=p).Q))(s.parameters)
    assert abs(G[n] - g) < 1e-10 * abs(g)


def test_old_vector_forms(case):
    s, Qi, A0, sm = case
    p = s.parameters
    i = p.names().index("n")
    a = s.run(Qi, s.initial_aux(Qi, A0), parameters=p.at[i].set(0.04))[1]     # old style
    b = s.run(Qi, s.initial_aux(Qi, A0), parameters={sm.parameters.n: 0.04}).Q
    np.testing.assert_allclose(a, b)


def test_boundary_parameter_sensitivity():
    import sympy as sp
    Q = SME.state(level=0, dimension=2)
    H = sp.Symbol("h_tail", positive=True)
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, parameters={"n": 0.03, "h_tail": 0.6},
        closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=[BC.Dirichlet("left", on=Q.q_0, value=0.5),
                             BC.Dirichlet("right", on=Q.h, value=H - Q.b)]))
    s = SymbolicHyperbolicSolverJax(time_end=2.0, dt=0.02, mood=False, ad_mode=AdMode.reverse)
    Q0, A0 = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20),
                     NumericalSystemModel.from_system_model(sm))
    x = jnp.linspace(0.25, 9.75, 20)
    Qi = jnp.stack([0.01 * x, 0.5 + 0 * x, 0.5 + 0 * x])
    P = sm.parameters
    f = lambda v: jnp.sum(s.run(Qi, s.initial_aux(Qi, A0), parameters={P.h_tail: v}).Q[1])  # noqa: E731
    g = jax.grad(f)(s.parameters[P.h_tail])
    fd = (f(0.6 + 1e-6) - f(0.6 - 1e-6)) / 2e-6
    assert abs(g - fd) < 1e-6 * abs(fd)
