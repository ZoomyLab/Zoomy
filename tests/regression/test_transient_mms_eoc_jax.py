"""TRANSIENT MMS: the order test a STEADY MMS structurally cannot do.

A steady MMS installs a source making ``W*(x)`` an exact equilibrium, so what it
measures is how well the scheme HOLDS that equilibrium.  Once the well-balancing
is good it stops measuring the discretisation at all — ``wb_comparison`` case B
reads EOC 3.01/2.81/2.42 at order 1 and 3.01/2.81/2.41 at order **2**, i.e. it
cannot separate the orders (cid 211).

Here ``W*`` depends on TIME, so ``manufactured_source`` carries its ``∂_t W*``
term and nothing is an equilibrium.  The measured order is then a statement
about the SSP-RK time integrator, the flux, and how the source is coupled into
the RK stages — which is exactly what cid 151 was about (a separately-split
explicit-Euler source collapses the momentum rows to O1 while the flux stays
O2, invisible to a steady test).

Friction is LINEAR (``NavierSlip``, ``tau_b = lambda_s u``) so the relaxation
time ``1/lambda_s`` is a knob rather than something that drifts with the
solution the way Manning's ``q|q|/h^{7/3}`` does.  The case is everywhere wet
(h ~ 1.0 +/- 0.15), so no wet/dry treatment is involved.

Measured with the case's own grids (see thesis/cases/verification/transient_mms):
order 1 reaches 0.98 and order 2 reaches 2.00 by 640 cells; on 40 cells they
read 0.96 / 1.92, which is pre-asymptotic rather than a defect.  unlimited,
venkatakrishnan and minmod agree on this smooth field.
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.analysis.manufactured import (
    exact_cell_field, install_manufactured_source)
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from conftest import march

L, H0, Q0, AH, AQ = 10.0, 1.0, 1.0, 0.15, 0.20
OMEGA, G, CFL, TEND = 2.0 * np.pi / 5.0, 9.81, 0.4, 1.0
GRIDS = [80, 160]              # small tier: two grids, one pairwise rate
LAMBDA_S = 1.0                 # linear friction ON


def _W(sm, x, t):
    bump = sp.sin(sp.pi * x / L) ** 2      # vanishes to 2nd order at both ends
    return {"b": sp.Integer(0),
            "h": H0 + AH * bump * sp.cos(OMEGA * t),
            "q_0": Q0 + AQ * bump * sp.sin(OMEGA * t)}


def _l1_h(order, n):
    sm = SystemModel.from_model(SME(
        level=0, dimension=2,
        closures=[Newtonian(), NavierSlip(), StressFree()],
        parameters={"g": G, "lambda_s": LAMBDA_S}))
    x, t = sm.space[0], sm.time
    W = _W(sm, x, t)
    install_manufactured_source(sm, W)          # carries d_t W*
    names = [str(s) for s in sm.state]
    sm.attach_boundary_conditions(BC.BoundaryConditions([
        BC.Dirichlet(tag=side, on=nm,
                     value=float(sp.lambdify((x, t), W[nm], "numpy")(xb, 0.0)))
        for side, xb in (("left", 0.0), ("right", L)) for nm in names]))
    f0 = [sp.lambdify((x, t), W[s], "numpy") for s in names]
    sm.initial_conditions = IC.UserFunction(
        function=lambda xx: np.array([float(f(xx[0], 0.0)) for f in f0]))
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    mesh = LSQMesh.create_1d(domain=(0.0, L), n_inner_cells=n)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    idx = {s: i for i, s in enumerate(names)}
    Q, _ = march(nsm, mesh, cfl=CFL, t_end=TEND,
                 free_surface_h_index=idx["h"],
                 free_surface_b_index=idx["b"],
                 free_surface_momentum_indices=[idx[s] for s in names
                                                if s.startswith("q_")],
                 reconstruction_variables="eta")
    ni = mesh.n_inner_cells
    # the reference MOVES: evaluate W* at the final time
    Wc = exact_cell_field(sm, {k: v.subs(t, TEND) for k, v in W.items()},
                          mesh.cell_centers[:, :ni])
    Qn = np.asarray(Q)[:, :ni]
    m = int(0.1 * ni)
    return float(np.mean(np.abs(Qn[idx["h"], m:ni - m] - Wc[idx["h"], m:ni - m])))


@pytest.mark.small
@pytest.mark.jax
@pytest.mark.parametrize("order,floor", [(1, 0.90), (2, 1.85)])
def test_transient_mms_reaches_design_order(order, floor):
    """The UNSTEADY manufactured solution converges at the design order.

    Floors are below the design order on purpose: these are the small-tier
    grids (80/160), and the case itself shows the rate still climbing with
    refinement (order 2: 1.99 at 160 -> 2.00 at 640).  The floors are set to
    catch an ORDER COLLAPSE — the cid-151 failure, where a split source drags a
    formally-O2 scheme to ~1 — not to pin a digit."""
    e = [_l1_h(order, n) for n in GRIDS]
    rate = float(np.log2(e[0] / e[1]))
    assert rate >= floor, (
        f"transient MMS order {order}: EOC_h = {rate:.2f} < {floor} "
        f"(L1 = {e[0]:.3e} -> {e[1]:.3e}).  An unsteady manufactured solution "
        "measures the TIME integrator and the source coupling; a collapse here "
        "is the cid-151 signature (source split out of the RK stages) and a "
        "steady MMS cannot see it.")


@pytest.mark.small
@pytest.mark.jax
def test_transient_source_actually_carries_the_time_derivative():
    """Non-vacuity: the manufactured source MUST depend on time.

    Without the ``∂_t W*`` term the case would silently degenerate into a
    steady MMS and the test above would still pass while measuring the wrong
    thing."""
    from zoomy_core.analysis.manufactured import manufactured_source
    sm = SystemModel.from_model(SME(
        level=0, dimension=2,
        closures=[Newtonian(), NavierSlip(), StressFree()],
        parameters={"g": G, "lambda_s": LAMBDA_S}))
    x, t = sm.space[0], sm.time
    S_t = manufactured_source(sm, _W(sm, x, t))
    assert any(sp.sympify(e).has(t) for e in sp.flatten(S_t)), (
        "the manufactured source has no time dependence — d_t W* is missing "
        "and this is a steady MMS wearing a transient's name")
    # ...and a STEADY W* must still produce a time-free source, so the term is
    # genuinely opt-in and every existing steady case is unaffected.
    S_s = manufactured_source(sm, {"b": sp.Integer(0),
                                   "h": H0 + AH * sp.sin(x),
                                   "q_0": sp.Float(Q0)})
    assert not any(sp.sympify(e).has(t) for e in sp.flatten(S_s)), (
        "a steady W* produced a time-dependent source — the d_t term is not "
        "vanishing as it must")
