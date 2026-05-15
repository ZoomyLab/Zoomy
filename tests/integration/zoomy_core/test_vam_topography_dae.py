"""VAM(1,2,2) over a fixed cosine-bump bottom — TDD oracle.

The Chorin split target.  Three regimes:

* ``test_dae_lake_at_rest_order1_well_balanced``  — order 1, lake-at-rest
  IC over a non-trivial bump.  DAESolver docstring claims well-balancing
  to ~1e-14 via the η = h+b SurfaceReconstruction; we pin that.
* ``test_dae_bump_perturbation_propagates_order1`` — order 1, η = H + δ·cos
  perturbation over the bump propagates with bounded mass loss and
  constraint residual at Newton tol.
* ``test_dae_lake_at_rest_order2_xfail``           — order 2 lake-at-rest
  is the Chorin-split target.  Expected to stall the monolithic
  Newton (cond ~1e7); xfail until the split solver lands.

Topography ``b(x)`` is a cosine bump on a 1-D domain, injected through
the ``aux_registry`` function-row mechanism (no Model subclassing
needed).  ``b_x`` is filled by the registry-driven LSQ derivative in
``Solver.update_qaux``.
"""
from __future__ import annotations

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.fvm.solver_dae_numpy import DAESolver
import zoomy_core.fvm.timestepping as ts


H_REST = 1.0          # rest water depth (η = h + b = H_REST everywhere)
B_AMPL = 0.2          # bump amplitude
B_HALF = 0.5          # bump half-width
B_CENTRE = 2.0        # bump centre on the 1-D domain
DOMAIN_LEN = 4.0


def _bump_b(x):
    """Cosine bump on |x - B_CENTRE| < B_HALF; zero outside."""
    inside = np.abs(x - B_CENTRE) < B_HALF
    b = np.zeros_like(x)
    arg = np.pi * (x[inside] - B_CENTRE) / B_HALF
    b[inside] = B_AMPL * 0.5 * (1.0 + np.cos(arg))
    return b


def _b_row(sm):
    """Find the aux row holding the topography function-aux ``b``."""
    for entry in sm.aux_registry:
        if entry["kind"] == "function" and entry["name"] == "b":
            return entry["row"]
    raise AssertionError("aux_registry has no function-aux row 'b'")


def _setup_bumpbottom_lake_at_rest(solver, mesh, model):
    """Run setup, then inject ``b(x)`` and adjust ``h`` so that
    η = h + b = H_REST.  Returns the projected ``Q0``."""
    Q0 = solver.setup_simulation(mesh, model)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    b = _bump_b(x)

    # Inject the static topography into the function-aux row.
    b_row = _b_row(solver.sm)
    solver._sim_Qaux[b_row, :] = b
    # Re-run update_qaux once so derivative-aux rows (b_x, h_x, ...)
    # reflect the new b field.
    solver._sim_Qaux = solver.update_qaux(
        Q0, solver._sim_Qaux, Q0, solver._sim_Qaux,
        solver._sim_mesh, solver.sm, solver._sim_parameters,
        0.0, 0.0,
    )

    # Lake-at-rest: h = H_REST - b, all velocities zero, pressure modes
    # determined by the algebraic constraints (project_to_manifold).
    Q0[:] = 0.0
    Q0[0, :] = H_REST - b
    Q0 = solver.project_to_manifold(Q0)
    solver._sim_Q = Q0.copy()
    solver._sim_time = 0.0
    return Q0


@pytest.fixture
def vam_model():
    return VAMModelGalerkin(level=1)


@pytest.fixture
def mesh_1d():
    return BaseMesh.create_1d(domain=(0.0, DOMAIN_LEN), n_inner_cells=32)


def test_dae_lake_at_rest_order1_well_balanced(mesh_1d, vam_model):
    """Order-1 DAESolver preserves lake-at-rest over a cosine-bump
    bottom to machine precision (well-balanced via η = h+b
    SurfaceReconstruction + cell-interior non-conservative integral)."""
    solver = DAESolver(
        time_end=0.5,
        method="ars232",
        compute_dt=ts.constant(dt=0.05),
        reconstruction_order=1,
    )
    Q0 = _setup_bumpbottom_lake_at_rest(solver, mesh_1d, vam_model)
    dt = 0.05
    for _ in range(int(solver.time_end / dt)):
        solver.step(dt)
        solver._sim_time += dt
    Q = solver._sim_Q

    # h should not have drifted at all (lake at rest is a steady state
    # of the well-balanced order-1 scheme).
    drift_h = np.max(np.abs(Q[0, :] - Q0[0, :]))
    assert drift_h < 1e-12, f"lake-at-rest h drift = {drift_h:.3e}"
    # Velocities stay zero.
    assert np.max(np.abs(Q[1:5, :])) < 1e-12


def test_dae_bump_perturbation_propagates_order1(mesh_1d, vam_model):
    """Order-1 DAESolver propagates a small η perturbation over a
    cosine-bump bottom with bounded mass loss; constraint rows stay
    at Newton tolerance throughout."""
    solver = DAESolver(
        time_end=0.2,
        method="ars232",
        compute_dt=ts.constant(dt=0.025),
        reconstruction_order=1,
    )
    Q0 = _setup_bumpbottom_lake_at_rest(solver, mesh_1d, vam_model)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    # Add a small η perturbation on top of lake-at-rest.
    Q0[0, :] += 0.01 * np.cos(2 * np.pi * x / DOMAIN_LEN)
    Q0 = solver.project_to_manifold(Q0)
    solver._sim_Q = Q0.copy()

    dx = float(solver._sim_mesh.cell_volumes[0])
    mass_0 = Q0[0, :].sum() * dx
    dt = 0.025
    for _ in range(int(solver.time_end / dt)):
        solver.step(dt)
        solver._sim_time += dt
    Q = solver._sim_Q

    mass_T = Q[0, :].sum() * dx
    assert abs(mass_T - mass_0) / abs(mass_0) < 5e-3

    Y = Q.T.ravel()
    fT = solver.f_I(solver._sim_time, Y).reshape(nc, solver.n_state).T
    assert np.max(np.abs(fT[solver.alg_idx, :])) < 1e-7

    # Wave actually moved: h-state deviation from IC is non-trivial.
    assert np.max(np.abs(Q[0, :] - Q0[0, :])) > 1e-5


def test_dae_lake_at_rest_order2_well_balanced(mesh_1d, vam_model):
    """Order-2 lake-at-rest over a bump.  No dynamics → Newton residual
    is identically zero (well-balancing is exact via the η = h+b
    SurfaceReconstruction + cell-interior non-conservative integral),
    so the Newton conditioning issue cannot manifest here.  This is the
    cheap regression guard on the order-2 spatial scheme."""
    solver = DAESolver(
        time_end=0.2,
        method="ars232",
        compute_dt=ts.constant(dt=0.025),
        reconstruction_order=2,
    )
    Q0 = _setup_bumpbottom_lake_at_rest(solver, mesh_1d, vam_model)
    dt = 0.025
    for _ in range(int(solver.time_end / dt)):
        solver.step(dt)
        solver._sim_time += dt
    Q = solver._sim_Q
    drift_h = np.max(np.abs(Q[0, :] - Q0[0, :]))
    assert drift_h < 1e-12, f"order-2 lake-at-rest drift = {drift_h:.3e}"


@pytest.mark.xfail(
    reason="Order-2 monolithic DAE stage Newton stalls at cond ~1e7 "
           "once dynamics fire (well-balancing zero RHS no longer "
           "applies).  Chorin-split solver is the supported home for "
           "order-2 dynamics — this test is the TDD oracle for that "
           "work.",
    strict=False,
)
def test_dae_bump_perturbation_order2_xfail(mesh_1d, vam_model):
    """Order-2 dynamic propagation over a bump.  Spatial scheme is
    correct; time integration is the unfixed piece — the monolithic
    IMEX-ARK stage Newton's finite-difference Jacobian is unreliable
    at the algebraic-row conditioning, Newton degrades to linear
    convergence and does not reach ``newton_tol``."""
    solver = DAESolver(
        time_end=0.1,
        method="ars232",
        compute_dt=ts.constant(dt=0.02),
        reconstruction_order=2,
    )
    Q0 = _setup_bumpbottom_lake_at_rest(solver, mesh_1d, vam_model)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    Q0[0, :] += 0.01 * np.cos(2 * np.pi * x / DOMAIN_LEN)
    Q0 = solver.project_to_manifold(Q0)
    solver._sim_Q = Q0.copy()

    dx = float(solver._sim_mesh.cell_volumes[0])
    mass_0 = Q0[0, :].sum() * dx
    dt = 0.02
    for _ in range(int(solver.time_end / dt)):
        solver.step(dt)
        solver._sim_time += dt
    Q = solver._sim_Q
    mass_T = Q[0, :].sum() * dx
    assert abs(mass_T - mass_0) / abs(mass_0) < 5e-3
    Y = Q.T.ravel()
    fT = solver.f_I(solver._sim_time, Y).reshape(nc, solver.n_state).T
    assert np.max(np.abs(fT[solver.alg_idx, :])) < 1e-7
    assert np.max(np.abs(Q[0, :] - Q0[0, :])) > 1e-5
