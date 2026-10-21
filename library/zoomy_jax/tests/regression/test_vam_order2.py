"""VAM order 2 by THREE-GRID RICHARDSON — no analytic solution needed."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL, ORDER_FLOOR, restrict


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_vam_second_order(overwrite):
    """    p = log2( ||u_N - u_2N|| / ||u_2N - u_4N|| )

    With u_h = u_exact + C h^p + ..., the exact solution CANCELS in the
    differences.  (Using the finest grid as a reference was rejected: only a
    factor 2 apart, its own error biases the rate downward.)

    PROVES the order, NOT the consistency: a wrong scheme can self-converge
    to a wrong limit.  VAM correctness lives in the bump-vs-experiment test
    and the core VAM->SME O(mu) smooth-limit test.

    THE ORDER KNOBS MUST BE PASSED EXPLICITLY.  ``ReconstructionSpec`` defaults
    to ``order=1`` and ``ChorinSplitVAMSolverJax.time_order`` defaults to 1, so
    a call that passes neither runs a FULLY FIRST-ORDER scheme.  This test did
    exactly that and asserted second order against it — measured, at the
    defaults: aggregate -1.456 with the pressure rows at -1.61/-1.86.  Passing
    ``order=2`` alone moves the SAME run to aggregate +1.183 with the pressure
    rows at +0.93/+1.00.  The pressure divergence was never the elliptic
    solver; it was the reconstruction order.
    """
    Ns, T_END = [64, 128, 256], 0.05          # N, 2N, 4N — a couple of steps
    model = models.vam(level=1, dimension=2, bc="periodic")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    print(describe(triple[0]))
    set_state_width(triple[0])

    sols, auxs = {}, {}
    t0 = time.perf_counter()
    for n in Ns:
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
        sols[n], auxs[n] = chorin_march(triple, mesh, cfl=CFL,
                                        ic=smooth_vam_ic, t_end=T_END,
                                        pressure_tol=VAM_PRESSURE_TOL,
                                        reconstruction=ReconstructionSpec(order=2),
                                        time_order=2)
    elapsed = time.perf_counter() - t0

    D1 = np.abs(restrict(sols[Ns[1]]) - sols[Ns[0]])
    D2 = np.abs(restrict(sols[Ns[2]]) - sols[Ns[1]])
    d1, d2 = D1.mean(), D2.mean()
    rate = float(np.log2(d1 / d2))

    # PER-ROW breakdown, always printed.  The whole-state mean is dominated by
    # whichever row is worst, so a bare aggregate rate cannot say WHICH part of
    # the scheme failed to converge — and here the answer is the whole finding:
    # the conservative rows converge while the pressure modes diverge, because
    # the unpreconditioned elliptic solve stagnates under refinement (see
    # cases.VAM_PRESSURE_TOL).
    print("VAM Richardson, per state row:")
    for i, name in enumerate(str(s) for s in triple[0].state):
        a, b = D1[i].mean(), D2[i].mean()
        r = np.log2(a / b) if a > 0 and b > 0 else float("nan")
        print(f"    {name:6s} |u_N-u_2N| {a:.3e}  |u_2N-u_4N| {b:.3e}  "
              f"rate {r:+.3f}")
    print(f"VAM Richardson: |u_N-u_2N| {d1:.3e}, |u_2N-u_4N| {d2:.3e}, "
          f"observed order {rate:.3f}")
    # Dumped BEFORE the floor assert (see refs.dump): the per-row rates ARE
    # the finding here, and the assert below destroys them.
    rows = [str(s) for s in triple[0].state]
    refs.dump("vam_order2", N=np.array(Ns), diffs=np.array([d1, d2]),
              rate=np.array([rate]), rows=np.array(rows),
              d1_rows=np.array([D1[i].mean() for i in range(len(rows))]),
              d2_rows=np.array([D2[i].mean() for i in range(len(rows))]),
              Q=sols[Ns[2]])
    assert rate > ORDER_FLOOR[2], (
        f"VAM observed order {rate:.3f} — not 2nd. See the per-row breakdown "
        f"above: if the P_* rows carry negative rates while h/q/r are "
        f"positive, this is the stagnating unpreconditioned pressure solve "
        f"(cases.VAM_PRESSURE_TOL), not a defect in the spatial scheme.")
    refs.check("vam_order2", overwrite, Q=sols[Ns[2]], Qaux=auxs[Ns[2]],
               N=np.array(Ns), diffs=np.array([d1, d2]),
               rate=np.array([rate]))
    refs.check_time("vam_order2", elapsed, overwrite)
