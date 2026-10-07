"""Observers through the adaptive time loop (``run_observed``).

Same 1-D SME(level=0) setup as test_ad.py: smooth wet IC over a bed bump,
nonzero velocity, order 2 + venkatakrishnan, adaptive CFL 0.9, ~25 steps; the
time grid moves with the input.

(a) run_observed's final state == run_simulation's, bitwise;
(b) running max of h over all cells and steps: forward jvp in g == central
    FD, reverse grad (checkpointed loop) == jvp;
(c) gauge time series of q (linear interpolation in time): reverse
    <grad_p sum(w * series), e_g> == forward jvp, a dot-product test.
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
    ig = list(nsm.parameter_values.as_dict()).index("g")
    p0 = solver.parameters
    e_g = jnp.zeros_like(p0).at[ig].set(1.0)
    return (solver, Q0, p0, e_g, models.state_index(nsm, "h"),
            models.state_index(nsm, "q_0"))


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


@pytest.mark.small
@pytest.mark.jax
def test_observer_running_max(setup):
    from zoomy_jax.fvm.observers import running_max
    solver, Q0, p0, e_g, ih, _ = setup
    obs = running_max(ih)

    # (a) the observer does not touch the march
    solver.ad_mode = "forward"
    Q_ref, Qaux_ref = solver.run_simulation(Q0, write_output=False)
    Q, Qaux, hmax = solver.run_observed(Q0, obs)
    assert np.array_equal(np.asarray(Q), np.asarray(Q_ref))
    assert np.array_equal(np.asarray(Qaux), np.asarray(Qaux_ref))
    assert float(hmax) >= float(jnp.max(Q0[ih]))

    # (b) forward vs central FD, then reverse vs forward
    def J(p):
        return solver.run_observed(Q0, obs, parameters=p)[2]

    _, d_fwd = jax.jvp(J, (p0,), (e_g,))
    eps = 1e-5
    fd = (J(p0 + eps * e_g) - J(p0 - eps * e_g)) / (2 * eps)
    assert np.isfinite(float(d_fwd)) and float(d_fwd) != 0.0
    assert _rel(d_fwd, fd) < 1e-5, (float(d_fwd), float(fd))

    solver.ad_mode = "reverse"
    g_rev = jax.grad(J)(p0)
    solver.ad_mode = "forward"
    assert _rel(jnp.vdot(g_rev, e_g), d_fwd) < 1e-8, (
        float(jnp.vdot(g_rev, e_g)), float(d_fwd))


@pytest.mark.small
@pytest.mark.jax
def test_observer_gauge_series(setup):
    from zoomy_jax.fvm.observers import gauge_series
    solver, Q0, p0, e_g, _, iq = setup
    times = np.linspace(0.0, T_END, 7)
    cells = [5, 20, 33]
    obs = gauge_series(iq, cells, times)
    w = jax.random.normal(jax.random.PRNGKey(2), (times.size, len(cells)),
                          dtype=Q0.dtype)

    solver.ad_mode = "forward"
    Q, _, acc = solver.run_observed(Q0, obs)
    series = np.asarray(acc.values)
    assert series.shape == (times.size, len(cells)) and np.isfinite(series).all()
    assert np.array_equal(series[0], np.asarray(Q0[iq, cells]))
    np.testing.assert_allclose(series[-1], np.asarray(Q[iq, cells]),
                               rtol=1e-13)

    def J(p):
        return jnp.sum(w * solver.run_observed(Q0, obs, parameters=p)[2].values)

    _, d_fwd = jax.jvp(J, (p0,), (e_g,))
    assert np.isfinite(float(d_fwd)) and float(d_fwd) != 0.0
    solver.ad_mode = "reverse"
    g_rev = jax.grad(J)(p0)
    solver.ad_mode = "forward"
    assert _rel(jnp.vdot(g_rev, e_g), d_fwd) < 1e-8, (
        float(jnp.vdot(g_rev, e_g)), float(d_fwd))
