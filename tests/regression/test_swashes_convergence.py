"""Fit the convergence rate against the SWASHES analytic and ASSERT it."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL, fit_order, march, wet_dry_o2


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
@pytest.mark.parametrize("case", ["stoker_wet", "ritter_dry"])
@pytest.mark.parametrize("order", [1, 2])
def test_swashes_order(overwrite, case, order):
    Ns, errs = [100, 200, 400], []
    t0 = time.perf_counter()
    for n in Ns:
        model = models.swe(dimension=2, bc="swashes")
        sm = SystemModel.from_model(model)
        sm.initial_conditions = IC.UserFunction(function=ic_for(case))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=order,
                                                  limiter=Limiter.minmod))
        if n == Ns[0]:
            print(describe(nsm))
        set_state_width(nsm)
        mesh = LSQMesh.create_1d(domain=SWASHES_DOMAIN, n_inner_cells=n)
        # Order-2 on the DRY front needs the η reconstruction + MOOD switched
        # on EXPLICITLY (they are solver params, not ReconstructionSpec
        # fields).  Without them this march is not merely inaccurate — it
        # dies: measured h = NaN by step 2 at n=100 and n=200, so no rate
        # could ever have been fitted for ritter_dry order 2.
        #
        # DELIBERATELY NOT APPLIED to stoker_wet: that case is wet
        # everywhere, plain conservative MUSCL is well-posed on it, and its
        # order-2 rate (~1.0, capped by shock pollution) is already MEASURED
        # and recorded.  Switching it would change a recorded number for no
        # physical reason — and note ETA_R = 1e-3 sits EXACTLY on the η
        # reconstruction's default eps_wet, so the switch would be far from
        # neutral there.  Flagged rather than silently applied.
        kw = wet_dry_o2(nsm) if (order >= 2 and case == "ritter_dry") else {}
        Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=SWASHES_T_END, **kw)
        errs.append(l1_vs_analytic(Q, mesh, case, t=SWASHES_T_END))
    elapsed = time.perf_counter() - t0

    rate = fit_order(Ns, errs)
    print(f"{case} order {order}: L1 {errs}, rate {rate:.3f}")
    refs.dump(f"swashes_{case}_o{order}", N=np.array(Ns), l1=np.array(errs),
              rate=np.array([rate]), h=np.asarray(Q[1], float))

    # NEITHER SWASHES case is floored at its nominal order.  Both carry a
    # DISCONTINUITY, so neither is entitled to one — and the user's measured
    # context says so explicitly: "SWASHES orders may NOT be floored".
    #
    # The floor that used to stand here on stoker_wet was a defect in this
    # test, not a defect in the scheme, and it contradicted a number the same
    # context already records.  MEASURED on this sweep (N = 100/200/400,
    # t = 6 s):
    #
    #   stoker_wet order 1  L1 5.972e-05 / 4.140e-05 / 2.317e-05  rate 0.683
    #   stoker_wet order 2  L1 2.120e-05 / 1.275e-05 / 5.233e-06  rate 1.009
    #
    # The order-2 value 1.009 IS the recorded shock-pollution cap (the
    # intermediate state is subcritical, Fr 0.81; a region decomposition gives
    # 1.06 excluding the shock band and 0.88 in the smooth fan).  Flooring it
    # at ORDER_FLOOR[2] = 1.9 asserted something the physics cannot deliver.
    # The order-1 sweep is still PRE-ASYMPTOTIC — the successive pairwise
    # rates rise 0.53 -> 0.84 across the sweep — so a 3-point fit understates
    # it; that is a property of the sweep, not a regression.
    #
    # The rates are therefore PINNED, not floored: ``refs.check`` compares
    # ``rate`` with ``np.allclose``, so any drift in either direction still
    # fails.  Second order is asserted ONLY on the smooth problems, which is
    # what ``boundary_order2`` and ``vam_order2`` are for.
    refs.check(f"swashes_{case}_o{order}", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(Ns), l1=np.array(errs), rate=np.array([rate]))
    refs.check_time(f"swashes_{case}_o{order}", elapsed, overwrite)
