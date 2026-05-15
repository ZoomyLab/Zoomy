"""VAM(1,2,2) bump test on :class:`ChorinSplitVAMSolver`.

Counterpart to :mod:`test_vam_topography_dae` — same physical setup
(cosine bump bottom, VAM(1,2,2) chain DAE), but driven by the Chorin
projection split solver instead of the monolithic IMEX-DAE.  The
order-2 dynamic case is the Chorin solver's reason for existence —
the monolithic Newton stalls at cond ~1e7; Chorin's explicit
predictor + linear elliptic solve avoids that entirely.

Scope of this initial test suite — verify the pipeline runs and
basic invariants hold:

* lake-at-rest holds (h doesn't drift, velocities stay zero) — to
  the spatial scheme's accuracy.  The current predictor is a crude
  central-flux Euler/SSP-RK2 (well-balancing requires η = h+b
  reconstruction + MUSCL — layered on top later), so the tolerance
  here is loose.  Later refinement tightens it.
* dynamic bump perturbation propagates with bounded mass loss + finite
  output.  Demonstrates the three-substep pipeline drives non-trivial
  state evolution under topography.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.splitter import split_for_pressure
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver


H_REST = 1.0
B_AMPL = 0.2
B_HALF = 0.5
B_CENTRE = 2.0
DOMAIN_LEN = 4.0


def _bump_b(x):
    """Cosine bump on |x - B_CENTRE| < B_HALF; zero outside."""
    inside = np.abs(x - B_CENTRE) < B_HALF
    b = np.zeros_like(x)
    arg = np.pi * (x[inside] - B_CENTRE) / B_HALF
    b[inside] = B_AMPL * 0.5 * (1.0 + np.cos(arg))
    return b


def _find_b_row(sm):
    """Locate the aux row holding the static topography ``b``."""
    for entry in sm.aux_registry:
        if entry["kind"] == "function" and entry["name"] == "b":
            return entry["row"]
    raise AssertionError("aux_registry has no function-aux row 'b'")


@pytest.fixture(scope="module")
def split_122():
    """Chain DAE → conservative form → split for pressure.

    The chain DAE's primitive-form mass matrix has state-dependent
    entries on the higher-order momentum rows (e.g. row 1 is
    ``[U_0, h, 0, …]`` ⇒ residual ``∂_t(h·U_0)``).  Going to
    conservative state ``q_k = h·U_k/c_k`` (with ``c_k = 2k+1``)
    cleans the j=0 rows to ``M=I``; HyperbolicSolver then correctly
    integrates ``∂_t q = -∂_x F + S`` on those rows without the
    missing 1/h factor (the "cheating" we'd otherwise have).

    The j=1 rows retain residual state-dependent off-diagonals
    ``(-q_U0 + q_U1)/h`` in the ∂_t h column — these are zero at
    lake-at-rest and small under modest dynamics, but a complete
    cleanup needs a chain-derivation-level rewrite (push the cross-
    coupling into the NCP via the continuity equation).
    """
    m = VAMModelGalerkin(level=1)
    sm = SystemModel.from_model(m)
    h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state
    q_U0, q_U1, q_W0, q_W1 = sp.symbols(
        "q_U0 q_U1 q_W0 q_W1", real=True,
    )
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, P_0, P_1],
        transform={U_0: q_U0 / h,     U_1: 3 * q_U1 / h,
                   W_0: q_W0 / h,     W_1: 3 * q_W1 / h},
    )
    dt = sp.Symbol("dt", positive=True)   # Python-safe Symbol name
    return split_for_pressure(sm, [P_0, P_1], dt)


@pytest.fixture
def mesh_1d():
    return BaseMesh.create_1d(domain=(0.0, DOMAIN_LEN), n_inner_cells=32)


def _setup_lake_at_rest_with_bump(solver, mesh, *, perturb=False):
    """Setup + inject cosine-bump topography + lake-at-rest IC.

    With ``perturb=True``, adds a small cosine perturbation to h on
    top of lake-at-rest to drive dynamics.
    """
    Q0 = solver.setup_simulation(mesh)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]

    # Inject static topography onto every sub-system's Qaux pool.
    # Each sub-system has its own aux row indexing; ``set_function_aux``
    # locates ``b`` on each registry independently.
    b_vals = _bump_b(x)
    solver.set_function_aux("b", b_vals)
    # Refresh derivative aux (b_x, b_xx, h_x, ...) in every pool.
    solver.update_aux_variables()

    # Lake-at-rest: h = H_REST - b, all velocities zero, pressure
    # determined by the algebraic constraints (we leave it at zero for
    # the first step; a single pressure solve at t=0 would project it
    # to the manifold).
    Q0[:] = 0.0
    Q0[0, :] = H_REST - b_vals
    if perturb:
        Q0[0, :] += 0.01 * np.cos(2 * np.pi * x / DOMAIN_LEN)
    solver._sim_Q = Q0.copy()
    solver.update_aux_variables()
    return Q0


def test_chorin_solver_flat_bottom_lake_at_rest(split_122, mesh_1d):
    """Lake-at-rest on a *flat* bottom is preserved to machine
    precision.  No well-balancing trick needed when b=0 everywhere
    because the Rusanov flux of ``(1/2)·g·h²`` is exact on uniform h.
    Confirms the Chorin pipeline (predictor + pressure + corrector)
    does *nothing* when there are no dynamics — the predictor leaves
    Q untouched, the pressure data forcing is exactly zero, GMRES
    short-circuits, and the corrector applies a zero update."""
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        time_end=0.05,
        reconstruction_order=1,
    )
    Q0 = solver.setup_simulation(mesh_1d)
    nc = solver.nc
    # b = 0 everywhere (no bump); h = H_REST uniform; velocities zero.
    solver.set_function_aux("b", np.zeros(nc))
    solver.update_aux_variables()
    Q0[:] = 0.0
    Q0[0, :] = H_REST
    solver._sim_Q = Q0.copy()
    solver.update_aux_variables()

    dt = 0.005
    for _ in range(10):
        solver.step(dt)
    Q = solver._sim_Q

    assert np.all(np.isfinite(Q))
    drift_h = float(np.max(np.abs(Q[0, :] - Q0[0, :])))
    assert drift_h < 1e-12, (
        f"flat-bottom lake-at-rest drift {drift_h:.3e} should be at "
        f"machine precision"
    )
    # Velocity / pressure stay exactly zero.
    assert np.max(np.abs(Q[1:5, :])) < 1e-12
    assert np.max(np.abs(Q[5:7, :])) < 1e-10


@pytest.mark.xfail(
    reason="Rusanov flux at order 1 is not well-balanced for shallow-"
           "water-like systems on a bump bottom — drift in the "
           "predictor injects a forcing term R(P=0) that GMRES "
           "correctly responds to, amplifying drift through the "
           "Chorin coupling.  Fix is the η = h+b SurfaceReconstruction "
           "trick (prior art in DAESolver); deferred per agreed plan.",
    strict=False,
)
def test_chorin_solver_lake_at_rest_on_bump_xfail(split_122, mesh_1d):
    """Lake-at-rest preservation on a cosine bump — the well-balancing
    target.  Currently xfails because the Rusanov flux is not WB."""
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        time_end=0.05,
        reconstruction_order=1,
    )
    Q0 = _setup_lake_at_rest_with_bump(solver, mesh_1d, perturb=False)
    dt = 0.005
    for _ in range(10):
        solver.step(dt)
    Q = solver._sim_Q
    assert np.all(np.isfinite(Q))
    drift_h = float(np.max(np.abs(Q[0, :] - Q0[0, :])))
    assert drift_h < 1e-12, f"lake-at-rest drift {drift_h:.3e}"


def test_chorin_solver_propagates_bump_perturbation(split_122, mesh_1d):
    """Chorin solver propagates a small η perturbation over a cosine
    bump.  Smoke test: state evolves, mass loss is bounded, output is
    finite.  Order-of-accuracy verification is the next refinement
    (well-balanced MUSCL predictor)."""
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        time_end=0.05,
        reconstruction_order=1,
    )
    Q0 = _setup_lake_at_rest_with_bump(solver, mesh_1d, perturb=True)

    dx = float(solver._sim_mesh.cell_volumes[0])
    mass_0 = Q0[0, :].sum() * dx

    dt = 0.005
    n_steps = int(0.05 / dt)
    for _ in range(n_steps):
        solver.step(dt)
    Q = solver._sim_Q

    assert np.all(np.isfinite(Q)), "Chorin step produced non-finite Q"
    mass_T = Q[0, :].sum() * dx
    # Crude central-flux is conservative on interior faces only; some
    # boundary outflow on a closed domain is expected.  Bound it
    # loosely for the smoke check.
    assert abs(mass_T - mass_0) / abs(mass_0) < 5e-2, (
        f"Mass loss {abs(mass_T - mass_0) / abs(mass_0):.3e} > 5 %"
    )
    # Perturbation actually moved.
    assert np.max(np.abs(Q[0, :] - Q0[0, :])) > 1e-5
