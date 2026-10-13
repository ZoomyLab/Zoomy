"""MMS friction convergence (jax): the source must integrate at the flux's order.

Method of Manufactured Solutions for SME(0)+[Newtonian, ManningFriction,
StressFree] in a fully-owned subcritical regime (Fr in [0.33, 0.50], everywhere
wet, monotone smootherstep W* that is C2-flat at both ends so the Dirichlet
boundary values are exact and inject no boundary error). ``S(W*)`` carries the
ACTIVE ManningFriction source, so this pins the friction-SOURCE treatment.

Regression guard for coordd cid 151: the stock jax solver used to integrate the
source by a SEPARATE 1st-order split (SSP-RK2 flux, then explicit-Euler source),
which caps every source-carrying row at O1 at steady state. Folding the source
into the RK2 RHS at the flux's order (mirroring solver_numpy) recovers clean O2.
If the split is ever reintroduced, EOC_q collapses from ~2.4-2.8 back to ~1.0 and
this test fails. Companion full case: thesis/cases/verification/1d_friction.
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.analysis.manufactured import (
    exact_cell_field, install_manufactured_source)
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    ManningFriction, Newtonian, StressFree)
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from conftest import CFL, march

L, H0, A, Q0, AQ = 10.0, 1.0, 0.5, 1.566, 0.4     # smootherstep amplitudes
G, N_MANNING = 9.81, 0.03                          # friction ACTIVE
TEND = 40.0                                        # >> numerical relaxation (steady by t~20)
GRIDS = [24, 48]


def _W_star(x):
    """Monotone smootherstep S = 6t^5 - 15t^4 + 10t^3 (C2-flat at both ends ->
    exact Dirichlet values, no boundary pollution); flat bed."""
    t = x / L
    S = 6 * t**5 - 15 * t**4 + 10 * t**3
    return {"b": sp.Integer(0), "h": H0 + A * S, "q_0": sp.Float(Q0) + AQ * S}


def _run(n):
    model = SME(level=0, dimension=2,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                parameters={"n": N_MANNING, "g": G})
    sm = SystemModel.from_model(model)
    x = sm.space[0]
    W = _W_star(x)
    install_manufactured_source(sm, W)             # W* is now an exact steady state
    idx = {str(s): i for i, s in enumerate(sm.state)}
    bl = [BC.Dirichlet(tag=side, on=nm,
                       value=float(sp.lambdify(x, W[nm], "numpy")(xb)))
          for side, xb in (("left", 0.0), ("right", L))
          for nm in ("b", "h", "q_0")]
    sm.attach_boundary_conditions(BC.BoundaryConditions(bl))
    fns = [sp.lambdify([x], W[str(s)], "numpy") for s in sm.state]
    sm.initial_conditions = IC.UserFunction(
        function=lambda xx: np.array([float(f(xx[0])) for f in fns]))
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    mesh = LSQMesh.create_1d(domain=(0.0, L), n_inner_cells=n)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.venkatakrishnan))
    Q, _ = march(nsm, mesh, cfl=CFL, t_end=TEND,
                 free_surface_h_index=idx["h"], free_surface_b_index=idx["b"],
                 free_surface_momentum_indices=[idx["q_0"]],
                 reconstruction_variables="eta")
    ni = mesh.n_inner_cells
    Wc = exact_cell_field(sm, W, mesh.cell_centers[:, :ni])
    return np.asarray(Q[:, :ni]), Wc


def _interior_l1(Q, Wc, ni, frac=0.1):
    m = int(frac * ni)
    return (float(np.mean(np.abs(Q[1, m:ni - m] - Wc[1, m:ni - m]))),
            float(np.mean(np.abs(Q[2, m:ni - m] - Wc[2, m:ni - m]))))


@pytest.mark.small
@pytest.mark.jax
def test_mms_friction_source_reaches_o2():
    """Source-carrying rows reach the flux's design order (cid 151).

    With the source folded into the RK2 RHS both h and q converge at 2nd order;
    the pre-fix explicit-Euler source split read O1 (EOC ~1.0) on q."""
    eh, eq = [], []
    for n in GRIDS:
        Q, Wc = _run(n)
        l1h, l1q = _interior_l1(Q, Wc, Q.shape[1])
        eh.append(l1h)
        eq.append(l1q)
    eoc_h = float(np.log2(eh[0] / eh[1]))
    eoc_q = float(np.log2(eq[0] / eq[1]))
    print(f"MMS friction EOC: h={eoc_h:.2f} q={eoc_q:.2f} "
          f"(L1_h={eh}, L1_q={eq})")
    # Clean O2 on BOTH rows.  1.9 is the suite's order-2 floor (conftest
    # ORDER_FLOOR[2]); the pre-fix split reads ~1.0 on q, so a regression
    # trips this comfortably.
    assert eoc_h >= 1.9, f"h not 2nd order: EOC_h={eoc_h:.2f}"
    assert eoc_q >= 1.9, f"q not 2nd order (source split regressed?): EOC_q={eoc_q:.2f}"
