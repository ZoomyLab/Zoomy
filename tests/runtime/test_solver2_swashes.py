"""solver2 (the v6 unified march) on the two SWASHES dam breaks.

Runs the NEW march side-by-side with the production ``solver_jax`` — nothing
is deleted, and this file asserts only on PHYSICS (finite, ``h >= 0``,
``dt > 0`` every step, mass conserved, momentum non-zero), never on exit
status.

Model: DERIVED ``SME(level=0)`` from the derivation cache, capless
(``update_variables is None``).  CFL 0.9 — the 1-D user law, never reduced.
"""
from __future__ import annotations

import numpy as np
import pytest
from zoomy_core.mesh import LSQMesh

from conftest import CFL
from cases import ritter_ic, stoker_ic
from models import swe, state_index
from zoomy_core.numerics import Limiter

DOMAIN, N_CELLS, T_END = (0.0, 10.0), 100, 1.0


def _nsm(ic, order):
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel

    sm = SystemModel.from_model(swe(2, "swashes"))
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    assert nsm.update_variables is None, (
        "the derived SWE must be CAP-FREE — a wet/dry momentum cap here "
        "silently zeroes the SWASHES depths (cid=54)")
    return nsm


def _march(nsm, order, recon="conservative"):
    """Drive the solver2 blocks step by step so dt and the state can be
    asserted on EVERY step (a whole-run call could only check the end)."""
    from zoomy_jax.fvm.solver2 import MarchSolver, blocks as B, describe_nsm

    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    kw = ({} if recon == "conservative" else
          dict(reconstruction_variables=recon, free_surface_h_index=1,
               free_surface_b_index=0))
    solver = MarchSolver(time_end=T_END, mood_redo=(order >= 2), **kw)
    S = solver.setup_march(mesh, nsm, CFL=CFL)
    print(describe_nsm(nsm))          # user law: print the NSM matrices first

    nc = mesh.n_inner_cells
    h = state_index(nsm, "h")
    p = solver._rt_parameters
    dts = []
    while float(S.time) < T_END - 1e-14:
        lo, hi, dt = solver.step_head(S, p, T_END - S.time, None)
        dt = B.assert_dt_admissible(dt, lo, hi, solver.MeshRT.inradius_f,
                                    time=S.time, iteration=S.iteration)
        assert dt > 0.0 and np.isfinite(dt), (
            f"dt = {dt} at t = {float(S.time)} — a non-positive dt is a "
            f"FAILURE, not a stopping condition")
        S, _ = solver.hyperbolic_step(S, dt, p)
        Qi = np.asarray(S.Q)[:, :nc]
        assert np.isfinite(Qi).all(), f"non-finite at t = {float(S.time)}"
        assert Qi[h].min() >= 0.0, (
            f"h = {Qi[h].min():.3e} < 0 at t = {float(S.time)} — REPORTED, "
            f"never clipped (user law: nothing floors h)")
        dts.append(dt)
    return np.asarray(S.Q)[:, :nc], np.asarray(mesh.cell_volumes[:nc]), dts


@pytest.mark.jax
@pytest.mark.parametrize("order", [1, 2])
def test_solver2_stoker_wet(order):
    nsm = _nsm(stoker_ic, order)
    Q, dx, dts = _march(nsm, order)
    h, q = state_index(nsm, "h"), state_index(nsm, "q_0")
    assert Q[h].min() > 0.0, "the wet dam break must stay wet everywhere"
    assert np.abs(Q[q]).max() > 0.0, "momentum is zero — the cap bug is back"
    # flat bed, open ends, no wave has reached the boundary at t = 1 s
    mass = float(np.sum(Q[h] * dx))
    assert abs(mass - 0.03) < 1e-12, f"mass drift {mass - 0.03:.3e}"


@pytest.mark.jax
@pytest.mark.parametrize("order,recon", [(1, "conservative"), (2, "eta")])
def test_solver2_ritter_dry(order, recon):
    """The dry dam break.  At order 2 this needs the wet/dry-aware
    (Audusse/KP ``eta``) reconstruction: with the plain conservative one the
    dry front drives ``|lambda| = |q|/max(1e-14, h)`` to ~1e9 and dt collapses
    to ~1e-11 — a REPORTED finding at the law CFL, never a CFL reduction and
    never an h floor.  See the thesis SWASHES case, which omits order 2 on
    ``ritter_dry`` for the same reason."""
    nsm = _nsm(ritter_ic, order)
    Q, dx, dts = _march(nsm, order, recon=recon)
    h, q = state_index(nsm, "h"), state_index(nsm, "q_0")
    assert Q[h].min() >= 0.0
    assert Q[h].min() == 0.0, "the dry side must stay EXACTLY dry — no floor"
    assert np.abs(Q[q]).max() > 0.0
    mass = float(np.sum(Q[h] * dx))
    assert abs(mass - 0.025) < 1e-12, f"mass drift {mass - 0.025:.3e}"


@pytest.mark.jax
def test_reduce_dt_rejects_combined_window_and_tend():
    """D7 / amendment 7: ``dt_window`` REPLACES the ``t_end`` clamp; the
    min-combined form is the measured preCICE abort, so it must not even be
    expressible."""
    import jax.numpy as jnp
    from zoomy_jax.fvm.solver2 import reduce_dt

    lo, hi = jnp.array([-1.0]), jnp.array([1.0])
    r = jnp.array([0.5])
    with pytest.raises(ValueError, match="XOR"):
        reduce_dt(lo, hi, r, None, CFL=0.9, dimension=1,
                  t_remaining=1.0, dt_window=0.1)


@pytest.mark.jax
def test_assert_dt_admissible_is_fatal():
    """v6 march-honesty guard: dt <= 0 aborts and names the offending face."""
    import jax.numpy as jnp
    from zoomy_jax.fvm.solver2 import assert_dt_admissible

    lo = jnp.array([-1.0, -1e30])
    hi = jnp.array([1.0, 1e30])
    r = jnp.array([0.5, 0.5])
    with pytest.raises(FloatingPointError, match="f = 1"):
        assert_dt_admissible(0.0, lo, hi, r, time=0.0, iteration=0)
