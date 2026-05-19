"""``DAESolver`` end-to-end on VAM(1,2,2).

Validates that the proper Solver-class DAE driver (sibling of
``HyperbolicSolver`` and ``IMEXSolver``) integrates the chain DAE
correctly: mass conserved exactly, algebraic constraints enforced to
machine precision, wave propagates.

The auto-scan in ``SystemModel.from_model`` routes topography ``b``
and every Derivative atom into ``Qaux``; the default
``Solver.update_qaux`` fills the derivative rows via LSQ.  Function-
aux entries (``b``) stay zero unless the user overrides
``update_qaux`` — i.e. the default is flat bottom.
"""
from __future__ import annotations

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.fvm.solver_dae_numpy import DAESolver
import zoomy_core.fvm.timestepping as ts


@pytest.fixture
def vam_model():
    return VAMModelGalerkin(level=1)


@pytest.fixture
def mesh_1d():
    return BaseMesh.create_1d(domain=(0.0, 4.0), n_inner_cells=16)


def test_dae_solver_setup(mesh_1d, vam_model):
    """``DAESolver.setup_simulation`` builds the SystemModel,
    auto-scans aux atoms into ``aux_state``, and detects the DAE
    partition."""
    solver = DAESolver(time_end=0.1, method="ars343",
                       compute_dt=ts.constant(dt=0.05))
    Q = solver.setup_simulation(mesh_1d, vam_model)

    assert Q.shape == (8, mesh_1d.n_inner_cells)
    # 6 evolution (mass + 2 xmom + 2 zmom + bathymetry) + 2 algebraic.
    assert solver.evol_idx.tolist() == [0, 1, 2, 3, 4, 5]
    assert solver.alg_idx.tolist() == [6, 7]
    # ``b`` is now a STATE; auto-scan still picks up derivatives such
    # as ``b_x`` (target_kind="state", state_index pointing at b).
    aux_names = {str(s) for s in solver.sm.aux_state}
    assert "b" not in aux_names, "b should be a state, not aux"
    assert "b_x" in aux_names
    assert "h_x" in aux_names


def test_dae_solver_residual_at_rest(mesh_1d, vam_model):
    """Rest state (h=H, all velocities zero, flat bottom — function-
    aux ``b`` defaults to zero) has zero implicit RHS."""
    solver = DAESolver(time_end=0.1, method="ars343",
                       compute_dt=ts.constant(dt=0.05))
    solver.setup_simulation(mesh_1d, vam_model)
    nc = solver.nc
    Q = np.zeros((solver.n_state, nc)); Q[0, :] = 1.0
    Y = Q.T.ravel()
    f = solver.f_I(0.0, Y).reshape(nc, solver.n_state).T
    assert np.max(np.abs(f)) < 1e-12, (
        f"|f_I(rest)|_inf = {np.max(np.abs(f)):.3e}"
    )


def test_dae_solver_cosine_bump(mesh_1d, vam_model):
    """Cosine-bump initial condition propagates with ARS343.

    Mass loss is bounded by 0.1 % (Extrapolation BC allows boundary
    flux); constraint residual at Newton tol; h responds; U_0 builds
    from zero.
    """
    solver = DAESolver(time_end=0.1, method="ars343",
                       compute_dt=ts.constant(dt=0.05))
    Q0 = solver.setup_simulation(mesh_1d, vam_model)

    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    H = 1.0
    Q0[:] = 0.0
    Q0[0, :] = H + 0.05 * np.cos(2 * np.pi * x / 4.0)
    Q0 = solver.project_to_manifold(Q0)

    dx = float(solver._sim_mesh.cell_volumes[0])
    mass_0 = Q0[0, :].sum() * dx

    solver._sim_Q = Q0.copy()
    solver._sim_time = 0.0
    dt = 0.05
    for _ in range(int(solver.time_end / dt)):
        solver.step(dt)
        solver._sim_time += dt
    Q = solver._sim_Q

    mass_T = Q[0, :].sum() * dx
    assert abs(mass_T - mass_0) / abs(mass_0) < 1e-3, (
        f"Mass change {abs(mass_T - mass_0):.3e} > 0.1% of {mass_0}"
    )

    Y = Q.T.ravel()
    fT = solver.f_I(solver._sim_time, Y).reshape(nc, solver.n_state).T
    assert np.max(np.abs(fT[solver.alg_idx, :])) < 1e-8

    assert np.max(np.abs(Q[0, :] - Q0[0, :])) > 1e-5
    assert np.max(np.abs(Q[1, :])) > 1e-4


def test_dae_solver_solve_convenience(mesh_1d, vam_model):
    """One-step ARS232 — basic plumbing check via ``solve``-style use."""
    solver = DAESolver(time_end=0.05, method="ars232",
                       compute_dt=ts.constant(dt=0.05))
    Q0 = solver.setup_simulation(mesh_1d, vam_model)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    Q0[:] = 0.0
    Q0[0, :] = 1.0 + 0.02 * np.cos(2 * np.pi * x / 4.0)
    Q0 = solver.project_to_manifold(Q0)
    solver._sim_Q = Q0.copy()
    solver._sim_time = 0.0
    solver.step(0.05)
    Q = solver._sim_Q
    assert Q.shape == Q0.shape
    dx = float(solver._sim_mesh.cell_volumes[0])
    m0 = Q0[0, :].sum() * dx
    mT = Q[0, :].sum() * dx
    assert abs(mT - m0) / abs(m0) < 1e-4
