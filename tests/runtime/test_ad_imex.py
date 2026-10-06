"""Automatic differentiation through the IMEX solver (implicit source).

1-D SME(level=0) with [Newtonian(), ManningFriction(), StressFree()],
Manning n = 0.03, NSM ``regularize_fractional_powers(1e-6)`` (the friction
carries ``hinv**(7/3)``).  The IC is at rest on the right half (u = 0), so the
friction derivative is taken at u = 0 too.  Order 1, adaptive CFL 0.9, a short
run.  The implicit source is solved cell-wise (``local``) and by Newton-GMRES
(``global``); both are ``lax.custom_root`` roots, differentiated by the
implicit function theorem.

(a) forward jvp w.r.t. n and g == central FD;
(b) reverse grad (checkpointed loop) == jvp;
(c) <grad_Q0 J, v> == jvp(J, Q0; v), Qaux rebuilt from Q0.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.operations import regularize_fractional_powers
from zoomy_core.systemmodel.system_model import SystemModel

import models
from conftest import CFL, _adaptive

pytest.importorskip("equinox")

N_CELLS = 24
T_END = 0.04
CHECKPOINTS = 3


def partly_at_rest_ic(x):
    X = float(x[0])
    h = 1.0 + 0.2 * np.exp(-((X - 0.4) / 0.1) ** 2)
    q = 0.3 * np.sin(np.pi * X / 0.5) ** 2 if X < 0.5 else 0.0
    return np.array([0.0, h, q])


@pytest.fixture(scope="module")
def nsm():
    sm = SystemModel.from_model(models.swe(dimension=2, bc="extrapolation"))
    sm.initial_conditions = IC.UserFunction(function=partly_at_rest_ic)
    sm.parameter_values.n = 0.03
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1),
        extra_operations=[regularize_fractional_powers(1e-6)])


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


@pytest.mark.small
@pytest.mark.jax
@pytest.mark.parametrize("source_mode", ["local", "global"])
def test_ad_imex(nsm, source_mode):
    from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=N_CELLS)
    solver = IMEXSourceSolverJax(
        time_end=T_END, compute_dt=_adaptive(CFL, mesh),
        march_log_interval=0, ad_checkpoints=CHECKPOINTS,
        source_mode=source_mode, gmres_tol=1e-12)
    Q0, _ = solver.setup_simulation(mesh, nsm)
    names = list(nsm.parameter_values.as_dict())
    i_n, i_g = names.index("n"), names.index("g")
    iq = models.state_index(nsm, "q_0")
    assert float(Q0[iq, -1]) == 0.0

    def J(p, q0):
        Q, _ = solver.run_simulation(q0, write_output=False, parameters=p)
        return jnp.sum(Q[iq] ** 2)

    p0 = solver.parameters
    assert float(p0[i_n]) == pytest.approx(0.03)

    # (a) forward vs central FD, in n and g.
    solver.ad_mode = "forward"
    d_fwd = {}
    for i in (i_n, i_g):
        e = jnp.zeros_like(p0).at[i].set(1.0)
        _, d = jax.jvp(lambda p: J(p, Q0), (p0,), (e,))
        eps = 1e-6 * max(abs(float(p0[i])), 1.0)
        fd = (J(p0 + eps * e, Q0) - J(p0 - eps * e, Q0)) / (2 * eps)
        assert np.isfinite(float(d)) and float(d) != 0.0
        assert _rel(d, fd) < 1e-6, (names[i], float(d), float(fd))
        d_fwd[i] = d
    v = jax.random.normal(jax.random.PRNGKey(1), Q0.shape, dtype=Q0.dtype)
    _, dJ_v = jax.jvp(lambda q: J(p0, q), (Q0,), (v,))

    # (b) + (c) reverse: checkpointed loop.
    solver.ad_mode = "reverse"
    g_p, g_Q0 = jax.grad(J, argnums=(0, 1))(p0, Q0)
    for i in (i_n, i_g):
        assert _rel(g_p[i], d_fwd[i]) < 1e-9, (names[i], float(g_p[i]),
                                               float(d_fwd[i]))
    assert _rel(jnp.vdot(g_Q0, v), dJ_v) < 1e-9
