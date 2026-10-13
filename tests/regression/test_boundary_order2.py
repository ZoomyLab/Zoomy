"""Order 2 must hold IN THE BOUNDARY CELLS, not only in the interior.

The probe is the RETRIEVED acoustic standing wave in a CLOSED BOX (Wall at
both ends, exact closed-form solution) — see ``cases.ACOUSTIC_*`` for the
provenance and for why it replaced the constructed smooth-Dirichlet case.
Two properties make it the right boundary probe:

* the error is measured against a CLOSED FORM, not against a finer run of the
  same code — so a rate below 2 is a statement about the scheme, not about a
  self-reference sharing the scheme's own defect;
* the boundary condition under test IS the Wall, and the wave reflects off
  both ends during the period marched here.

The windowed decomposition (full / interior / left strip / right strip) is
kept: the defect this test exists to catch showed up as an ASYMMETRY between
the two strips, which a full-domain norm dilutes below visibility.
"""
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
from conftest import CFL, ORDER_FLOOR, fit_order, march


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_second_order_at_boundaries(overwrite):
    Ns = [20, 40, 80, 160, 320, 640]
    t_end = acoustic_t_end()
    err = {"full": [], "interior": [], "left": [], "right": []}
    t0 = time.perf_counter()
    for n in Ns:
        model = models.swe(dimension=2, bc="wall")
        sm = SystemModel.from_model(model)
        sm.initial_conditions = IC.UserFunction(function=acoustic_ic)
        sm.aux_initial_conditions = IC.Constant(
            constants=lambda k: np.zeros(k))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
        if n == Ns[0]:
            print(describe(nsm))
        set_state_width(nsm)
        mesh = LSQMesh.create_1d(domain=ACOUSTIC_DOMAIN, n_inner_cells=n)
        Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=t_end)
        for w, e in acoustic_l2_by_window(Q, mesh).items():
            err[w].append(e)
    elapsed = time.perf_counter() - t0

    rates = {w: fit_order(Ns, e) for w, e in err.items()}
    print("boundary-decomposed rates: "
          + ", ".join(f"{w} {r:.3f}" for w, r in rates.items()))
    for w, e in err.items():
        pair = ["  --"] + [f"{np.log2(e[i - 1] / e[i]):5.2f}"
                           for i in range(1, len(e))]
        print(f"    {w:9s} L2       " + "  ".join(f"{v:.4e}" for v in e))
        print(f"    {w:9s} pairwise " + " ".join(pair))
    # Dumped BEFORE the floor assert.  This IS a smooth problem, so the floor
    # below is legitimate and must stay — and a bare ``AssertionError: rate
    # 0.736`` cannot tell a reader WHERE it failed; these vectors can.
    refs.dump("boundary_order2", N=np.array(Ns),
              **{f"l2_{w}": np.array(e) for w, e in err.items()},
              **{f"rate_{w}": np.array([r]) for w, r in rates.items()},
              h=np.asarray(Q[1], float))
    for w, r in rates.items():
        assert r > ORDER_FLOOR[2], \
            f"order-2 rate in the {w} window is {r:.3f} — not clean there"
    refs.check("boundary_order2", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(Ns),
               **{f"l2_{w}": np.array(e) for w, e in err.items()},
               **{f"rate_{w}": np.array([r]) for w, r in rates.items()})
    refs.check_time("boundary_order2", elapsed, overwrite)
