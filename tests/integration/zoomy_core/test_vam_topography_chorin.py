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
    m = VAMModelGalerkin(level=1)
    sm = SystemModel.from_model(m)
    name_to_sym = {str(s): s for s in sm.state}
    dt = sp.Symbol(r"\Delta t", positive=True)
    return split_for_pressure(
        sm, [name_to_sym["P_0"], name_to_sym["P_1"]], dt,
    )


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


def test_chorin_solver_runs_on_bump_topography(split_122, mesh_1d):
    """Chorin split solver advances a lake-at-rest IC over a cosine
    bump for multiple steps without crashing or producing NaN.

    Tolerances are loose — the current predictor is unlimited central
    flux (not well-balanced); lake-at-rest will drift by O(dx²) per
    step.  This pins the pipeline, not the spatial accuracy."""
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        time_end=0.1,
        reconstruction_order=1,
    )
    Q0 = _setup_lake_at_rest_with_bump(solver, mesh_1d, perturb=False)

    dt = 0.005
    n_steps = int(0.05 / dt)
    for _ in range(n_steps):
        solver.step(dt)
    Q = solver._sim_Q

    assert Q.shape == Q0.shape
    assert np.all(np.isfinite(Q)), "Chorin step produced non-finite Q"
    # Lake-at-rest drift bounded — Rusanov flux at order 1 is *not*
    # well-balanced for shallow-water-like systems without an
    # η = h+b surface reconstruction (that's the next refinement
    # layer; see DAESolver's _surface_recon for the prior art).  At
    # the current spatial scheme drift is O(b_x²)·dt·n_steps; we
    # bound it loosely as the pipeline smoke test.
    drift_h = float(np.max(np.abs(Q[0, :] - Q0[0, :])))
    assert drift_h < 0.1, f"lake-at-rest drift {drift_h:.3e} too large"


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
