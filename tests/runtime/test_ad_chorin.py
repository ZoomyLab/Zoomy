"""The Chorin split VAM solver on the base time loop, with AD.

VAM(level=1, dimension=2) Escalante bump (``models.vam``), 30 cells, order 1,
``time_order`` 1, pressure tolerance ``VAM_PRESSURE_TOL``.  Short runs: 4 steps
of the constant step ``DT = 2**-5`` (a power of two, so the time grid is exact
and ``run_simulation`` takes exactly 4 steps), or the adaptive Chorin CFL
(``cfl=CFL``) to ``T_END``.

(a) ``run_simulation`` with ``timestepping.constant(DT)`` == ``run_jit_steps``
    at the same ``DT`` and step count;
(b) the adaptive Chorin law == ``timestepping.adaptive`` with the mesh
    dimension and degree 0;
(c) adaptive dt: forward jvp in g == central FD; reverse grad
    (``ad_mode="reverse"``) == forward jvp, in g and in a Q0 direction;
(d) ``time0``: two chunks [0, 2 DT] + [2 DT, 4 DT] == one run [0, 4 DT], and
    the second chunk takes 2 steps ending at 4 DT.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import SystemModel

import models
from cases import (ESC_DOMAIN, VAM_PRESSURE_TOL, bump_ic, chorin_split_for,
                   set_state_width)
from conftest import CFL
from zoomy_jax.fvm.observers import Observer

pytest.importorskip("equinox")

N_CELLS = 30
DT = 2.0 ** -5
N_STEPS = 4
T_END = N_STEPS * DT
CHECKPOINTS = 3


@pytest.fixture(scope="module")
def setup():
    from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
    model = models.vam(level=1, dimension=2, bc="bump")
    triple = chorin_split_for(model, SystemModel.from_model(model))
    set_state_width(triple[0])
    mesh = LSQMesh.create_1d(domain=ESC_DOMAIN, n_inner_cells=N_CELLS)
    solver = ChorinSplitVAMSolverJax(
        *triple, time_end=T_END, compute_dt=timestepping.constant(DT),
        march_log_interval=0, ad_checkpoints=CHECKPOINTS,
        pressure_tol=VAM_PRESSURE_TOL)
    Q0 = np.array(solver.setup_simulation(mesh))
    xc = np.asarray(solver._rt_mesh.cell_centers)[:, :solver.nc]
    for j in range(solver.nc):
        Q0[:, j] = bump_ic(xc[:, j])
    names = list(solver.nsm.parameter_values.as_dict())
    ih = models.state_index(solver.nsm, "h")
    return solver, jnp.asarray(Q0), names.index("g"), ih


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


@pytest.mark.small
@pytest.mark.jax
def test_run_simulation_matches_run_jit_steps(setup):
    solver, Q0, _, _ = setup
    Q, Qaux = solver.run_simulation(Q0, write_output=False)
    Qj, Qauxj, _, _, t = solver.run_jit_steps(DT, N_STEPS, Q0)
    assert float(t) == T_END
    assert np.isfinite(np.asarray(Q)).all()
    np.testing.assert_allclose(np.asarray(Q), np.asarray(Qj), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(Qaux), np.asarray(Qauxj),
                               rtol=0, atol=1e-13)


@pytest.mark.small
@pytest.mark.jax
def test_adaptive_law_matches_core_timestepping(setup):
    solver, Q0, _, _ = setup
    Qaux0 = solver._qaux_from_q(Q0, solver.parameters)
    core = timestepping.adaptive(CFL=CFL, dimension=1, degree=0,
                                 dt_max=solver.nsm.dt_max)
    ref = core(Q0, Qaux0, solver.parameters, solver._rt_inradius_face,
               solver._rt_eigenvalue_op)
    solver.cfl = CFL
    try:
        dt = solver.compute_timestep(Q0, Qaux0)
    finally:
        solver.cfl = None
    assert float(dt) == float(ref)


@pytest.mark.small
@pytest.mark.jax
def test_ad_forward_reverse_fd(setup):
    solver, Q0, ig, ih = setup

    def J(p, q0):
        Q, _ = solver.run_simulation(q0, write_output=False, parameters=p)
        return jnp.sum(Q[ih] ** 2)

    p0 = solver.parameters
    e_g = jnp.zeros_like(p0).at[ig].set(1.0)
    v = jax.random.normal(jax.random.PRNGKey(1), Q0.shape, dtype=Q0.dtype)
    solver.cfl = CFL
    try:
        solver.ad_mode = "forward"
        _, d_g = jax.jvp(lambda p: J(p, Q0), (p0,), (e_g,))
        _, d_v = jax.jvp(lambda q: J(p0, q), (Q0,), (v,))
        eps = 1e-4 * float(p0[ig])
        fd = (J(p0 + eps * e_g, Q0) - J(p0 - eps * e_g, Q0)) / (2 * eps)
        assert np.isfinite(float(d_g)) and float(d_g) != 0.0
        assert _rel(d_g, fd) < 1e-7, (float(d_g), float(fd))

        solver.ad_mode = "reverse"
        g_p, g_Q0 = jax.grad(J, argnums=(0, 1))(p0, Q0)
    finally:
        solver.ad_mode = "forward"
        solver.cfl = None
    assert np.isfinite(np.asarray(g_Q0)).all()
    assert _rel(g_p[ig], d_g) < 1e-10, (float(g_p[ig]), float(d_g))
    assert _rel(jnp.vdot(g_Q0, v), d_v) < 1e-10


@pytest.mark.small
@pytest.mark.jax
def test_time0_continues_a_chunked_run(setup):
    solver, Q0, _, _ = setup
    Q_ref, _ = solver.run_simulation(Q0, write_output=False)

    steps = Observer(
        init=lambda Q, Qaux, p: (jnp.zeros(()), jnp.zeros(())),
        update=lambda acc, time, dt, Q, Qaux, p: (acc[0] + 1, time))
    try:
        solver.time_end = 2 * DT
        Q1, Qaux1 = solver.run_simulation(Q0, write_output=False)
        solver.time_end = T_END
        Q2, _, (n, t) = solver.run_observed(Q1, steps, Qaux=Qaux1,
                                            time0=2 * DT)
    finally:
        solver.time_end = T_END
    assert int(n) == N_STEPS - 2 and float(t) == T_END
    np.testing.assert_allclose(np.asarray(Q2), np.asarray(Q_ref),
                               rtol=0, atol=1e-13)
