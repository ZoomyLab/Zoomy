"""Automatic differentiation through the adaptive explicit time loop.

1-D SME(level=0), smooth wet IC over a bed bump with NONZERO velocity
everywhere (the friction source carries sqrt(sum q^2), whose derivative is
0*inf at u = 0), order 2 + venkatakrishnan (minmod is piecewise smooth: AD and FD disagree),
adaptive CFL 0.9 to a short t_end (~25 steps).  The time grid is NOT frozen:
the derivative includes d(dt)/d(input).

(a) forward jvp w.r.t. g == central FD;
(b) reverse grad (checkpointed loop, few checkpoints => recomputation) == jvp;
(c) <grad_Q0 J, v> == jvp(J, Q0; v), Qaux rebuilt from Q0 (Qaux=None);
(d) switching ad_mode on one solver re-traces and both modes still work;
(e) bernoulli moving-equilibrium WB: forward jvp w.r.t. g == central FD
    (g enters the reconstruction; a fwd-vs-rev check cannot see a baked g).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
from conftest import CFL, _adaptive

pytest.importorskip("equinox")

N_CELLS = 40
T_END = 0.15
CHECKPOINTS = 4


def smooth_bump_ic(x):
    X = float(x[0])
    b = 0.1 * np.exp(-((X - 0.5) / 0.1) ** 2)
    h = 1.0 + 0.1 * np.sin(2.0 * np.pi * X) - b
    q = 0.5 + 0.1 * np.cos(2.0 * np.pi * X)
    return np.array([b, h, q])


@pytest.fixture(scope="module")
def setup():
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    sm = SystemModel.from_model(models.swe(dimension=2, bc="extrapolation"))
    sm.initial_conditions = IC.UserFunction(function=smooth_bump_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2,
                                              limiter="venkatakrishnan"))
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=N_CELLS)
    solver = HyperbolicSolver(time_end=T_END, compute_dt=_adaptive(CFL, mesh),
                              march_log_interval=0, ad_checkpoints=CHECKPOINTS)
    Q0, _ = solver.setup_simulation(mesh, nsm)
    names = list(nsm.parameter_values.as_dict())
    ig = names.index("g")
    ih = models.state_index(nsm, "h")
    assert float(solver.parameters[ig]) == pytest.approx(9.81)
    return solver, Q0, ig, ih


def _J_of(solver, ih):
    def J(p, Q0):
        Q, _ = solver.run_simulation(Q0, write_output=False, parameters=p)
        return jnp.sum(Q[ih] ** 2)
    return J


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


@pytest.mark.small
@pytest.mark.jax
def test_ad_forward_reverse_and_field(setup):
    solver, Q0, ig, ih = setup
    J = _J_of(solver, ih)
    p0 = solver.parameters
    e_g = jnp.zeros_like(p0).at[ig].set(1.0)

    # (a) forward: jvp in g vs central FD (the adaptive dt moves with g).
    solver.ad_mode = "forward"
    J0, dJ_fwd = jax.jvp(lambda p: J(p, Q0), (p0,), (e_g,))
    eps = 1e-5
    fd = (J(p0 + eps * e_g, Q0) - J(p0 - eps * e_g, Q0)) / (2 * eps)
    assert np.isfinite(float(dJ_fwd)) and float(dJ_fwd) != 0.0
    assert _rel(dJ_fwd, fd) < 1e-5, (float(dJ_fwd), float(fd))

    # Field tangent for (c), still in forward mode.
    v = jax.random.normal(jax.random.PRNGKey(0), Q0.shape, dtype=Q0.dtype)
    _, dJ_v = jax.jvp(lambda q: J(p0, q), (Q0,), (v,))

    # (b) + (c) reverse: checkpointed loop, CHECKPOINTS << n_steps.
    solver.ad_mode = "reverse"
    J0_rev, (g_p, g_Q0) = jax.value_and_grad(J, argnums=(0, 1))(p0, Q0)
    assert _rel(J0_rev, J0) < 1e-12
    assert _rel(g_p[ig], dJ_fwd) < 1e-8, (float(g_p[ig]), float(dJ_fwd))
    dot = jnp.vdot(g_Q0, v)
    assert _rel(dot, dJ_v) < 1e-8, (float(dot), float(dJ_v))


@pytest.mark.small
@pytest.mark.jax
def test_ad_mode_switch_retraces(setup):
    solver, Q0, ig, ih = setup
    J = _J_of(solver, ih)
    p0 = solver.parameters
    e_g = jnp.zeros_like(p0).at[ig].set(1.0)

    solver.ad_mode = "reverse"
    g_rev = jax.grad(J)(p0, Q0)[ig]
    assert solver._time_loop_cache[0][-2:] == ("reverse", CHECKPOINTS)

    solver.ad_mode = "forward"
    _, d_fwd = jax.jvp(lambda p: J(p, Q0), (p0,), (e_g,))
    assert solver._time_loop_cache[0][-2:] == ("forward", CHECKPOINTS)
    assert _rel(g_rev, d_fwd) < 1e-8

    # forward mode has no transpose; reverse mode has no jvp
    with pytest.raises(Exception):
        jax.grad(J)(p0, Q0)


def _bernoulli_solver(g):
    """Fresh bernoulli-WB solver with ``g`` set in the MODEL, so every
    setup-time consumer of ``g`` sees it too (the FD ground truth)."""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    from cases import bcs_for
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=bcs_for("extrapolation", 2),
                equilibrium_reconstruction="bernoulli")
    sm = SystemModel.from_model(model)
    assert sm.resolve_equilibrium_kernel() == "bernoulli_closed_form"
    sm.parameter_values.g = g
    sm.initial_conditions = IC.UserFunction(function=smooth_bump_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=N_CELLS)
    solver = HyperbolicSolver(time_end=T_END, compute_dt=_adaptive(CFL, mesh),
                              march_log_interval=0)
    Q0, _ = solver.setup_simulation(mesh, nsm)
    ig = list(nsm.parameter_values.as_dict()).index("g")
    assert float(solver.parameters[ig]) == g
    return solver, Q0, ig, models.state_index(nsm, "h")


@pytest.mark.small
@pytest.mark.jax
def test_ad_bernoulli_wb_dg_matches_fd():
    # FD from solvers BUILT at g +- eps: an FD through the same parameters
    # vector would share any baked g with the jvp and agree with it anyway.
    g0, eps = 9.81, 1e-5
    solver, Q0, ig, ih = _bernoulli_solver(g0)
    p0 = solver.parameters
    e_g = jnp.zeros_like(p0).at[ig].set(1.0)
    _, dJ = jax.jvp(lambda p: _J_of(solver, ih)(p, Q0), (p0,), (e_g,))

    def J_at(g):
        s, q0, _, _ = _bernoulli_solver(g)
        return _J_of(s, ih)(s.parameters, q0)

    fd = (J_at(g0 + eps) - J_at(g0 - eps)) / (2 * eps)
    assert _rel(dJ, fd) < 1e-5, (float(dJ), float(fd))
