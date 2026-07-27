"""SWASHES subcritical bump — the SMOOTH, wet/wet second-order gate.

The other two SWASHES cases cannot carry a second-order floor and are pinned
instead: ``stoker_wet`` has a shock (measured order-2 rate 1.009, the
shock-pollution cap) and ``ritter_dry`` has a dry front.  This case — SWASHES
3.1.1, subcritical flow over a bump — is smooth and wet everywhere, so it is
the one entitled to a real floor.

THE FLOOR IS > 2, NOT 1.9 (user, 2026-07-22): "1.9 is NOT second order."
``ORDER_FLOOR[2] = 1.9`` is used by the smooth-problem tests elsewhere; it is
deliberately NOT used here.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL, fit_order, march

#: Strictly greater than 2 — a "second-order" scheme that measures 1.9 is not
#: second order.  Asserted on the SMOOTH case only.
SMOOTH_ORDER_FLOOR = 2.0


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_swashes_bump_sub_is_second_order(overwrite):
    """Three-grid COV of h against the SWASHES steady analytic; rate > 2.

    THE LIMITER IS ``venkatakrishnan``, not ``minmod`` (cid 154).  minmod
    clips the smooth bump EXTREMUM to first order, which set up a shared
    order-2 limit cycle on it: the state never reached the 1e-6 steadiness
    gate on either backend (numpy 9.29e-4, jax 1.06e-3, t_end-independent
    even probed to t=1000 s), so no rate could be fitted.  The smooth,
    differentiable ``venkatakrishnan`` limiter does not clip the extremum,
    breaks the limit cycle, and the march settles below the steadiness gate
    and delivers clean second order.  ``BC.Characteristic`` (the subcritical
    pair: discharge imposed upstream, depth downstream) is unchanged — a full
    far-field target is the wrong physics for a subcritical bump.
    """
    Ns, errs, resids = [100, 200, 400], [], []
    t0 = time.perf_counter()
    for n in Ns:
        model = models.swe(dimension=2, bc="bump_sub")
        sm = SystemModel.from_model(model)
        sm.initial_conditions = IC.UserFunction(function=bump_sub_ic)
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(
                order=2, limiter="venkatakrishnan"))
        if n == Ns[0]:
            print(describe(nsm))
        set_state_width(nsm)
        mesh = LSQMesh.create_1d(domain=BUMP_SUB_DOMAIN, n_inner_cells=n)
        Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=BUMP_SUB_T_END)

        # STEADINESS GATE — TEMPORAL, not spatial.  The analytic table is the
        # t -> inf limit, so a march that has not settled would be compared
        # against the wrong state and would still yield a plausible rate.
        #
        # The first cut of this gate measured the SPATIAL spread of q and
        # required it below 1e-3.  That was wrong: at steady state the discrete
        # q is constant only to within DISCRETIZATION error, so that gate was
        # thresholding the very quantity this test measures.  It read 5.63e-03
        # at t = 200 s and 5.72e-03 at t = 600 s — flat under a 3x longer
        # march, which is exactly what "already steady" looks like.
        #
        # Steadiness is a property of TIME: march a little further and see
        # whether the state moves.
        Q2, _ = march(nsm, mesh, cfl=CFL, t_end=BUMP_SUB_T_END + BUMP_SUB_T_SETTLE)
        scale = max(float(np.max(np.abs(np.asarray(Q[1, :n], float)))), 1e-30)
        resid = float(np.max(np.abs(np.asarray(Q2[:, :n], float)
                                    - np.asarray(Q[:, :n], float))) / scale)
        resids.append(resid)
        assert np.all(np.isfinite(Q)), f"bump_sub went non-finite at n={n}"
        assert resid < 1e-6, (
            f"n={n}: not steady — the state still moves by {resid:.3e} "
            f"(relative) over a further {BUMP_SUB_T_SETTLE} s. The analytic "
            f"table is the steady limit, so the comparison would be against "
            f"the wrong state. Raise BUMP_SUB_T_END.")
        errs.append(l1_vs_analytic(Q, mesh, "bump_sub", t=BUMP_SUB_T_END))
    elapsed = time.perf_counter() - t0

    rate = fit_order(Ns, errs)
    print(f"bump_sub order 2: L1 {errs}, steady residuals {resids}, "
          f"rate {rate:.3f}")
    refs.dump("swashes_bump_sub_o2", N=np.array(Ns), l1=np.array(errs),
              rate=np.array([rate]), h=np.asarray(Q[1], float))
    assert rate > SMOOTH_ORDER_FLOOR, (
        f"bump_sub is SMOOTH and wet/wet, so it must be genuinely second "
        f"order: measured {rate:.3f}, required > {SMOOTH_ORDER_FLOOR}. "
        f"L1 = {errs} on N = {Ns}.")
    refs.check("swashes_bump_sub_o2", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(Ns), l1=np.array(errs), rate=np.array([rate]))
    refs.check_time("swashes_bump_sub_o2", elapsed, overwrite)
