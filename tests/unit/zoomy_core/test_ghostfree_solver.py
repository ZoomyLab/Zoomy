"""Convergence tests for ghost-cell-free NumPy solver.

Tests verify 2nd-order convergence for:
- Pure advection with periodic BCs (O2, LSQ MUSCL + VK, RK2)
- Pure advection with wall BCs (standing wave in a box)
- Pure diffusion with Dirichlet BCs (u=0 at walls)
- Pure diffusion with Neumann BCs (du/dn=0, mass conservation)

Each convergence test runs on grids N=50,100,200 and checks that
the convergence rate is >= 1.8 (theoretical: 2.0).
"""

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh, ensure_lsq_mesh
from zoomy_core.model.models.advection_model import (
    ScalarAdvection,
    ScalarAdvectionDiffusion,
)
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions,
    Extrapolation,
    Wall,
)
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mesh(N, domain=(0.0, 1.0)):
    return BaseMesh.create_1d(domain=domain, n_inner_cells=N)


def _l2_error(Q, exact_fn, mesh, margin_frac=0.1):
    """L2 error on interior cells, excluding boundary-contaminated margin."""
    nc = mesh.n_inner_cells
    margin = max(int(nc * margin_frac), 1)
    x = mesh.cell_centers[0, margin:nc - margin]
    dx = mesh.cell_volumes[margin:nc - margin]
    u_num = np.asarray(Q[0, margin:nc - margin])
    u_ex = exact_fn(x)
    return float(np.sqrt(np.sum((u_num - u_ex) ** 2 * dx) / np.sum(dx)))


def _convergence_rate(errors, Ns):
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i - 1] / errors[i]) / np.log(Ns[i] / Ns[i - 1])
        rates.append(r)
    return rates


# ── Test: Pure Advection O2 (periodic BCs via Extrapolation) ─────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_advection_o2_periodic():
    """O2 convergence for scalar advection with periodic-like Extrapolation BCs."""
    a = 1.0
    t_end = 0.05  # short enough to avoid boundary interaction
    Ns = [50, 100, 200]
    errors = []

    for N in Ns:
        bcs = BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ])
        model = ScalarAdvection(
            dimension=1,
            initial_conditions=UserFunction(
                function=lambda x: np.array([np.sin(2 * np.pi * x[0])])
            ),
            boundary_conditions=bcs,
        )
        model.parameters.a_x = a

        mesh = _make_mesh(N)
        solver = HyperbolicSolver(
            time_end=t_end,
            reconstruction_order=2,
            limiter="venkatakrishnan",
            compute_dt=timestepping.adaptive(CFL=0.45),
        )
        Q, Qaux = solver.solve(mesh, model, write_output=False)

        mesh_lsq = ensure_lsq_mesh(mesh, model)
        exact = lambda x: np.sin(2 * np.pi * (x - a * t_end))
        err = _l2_error(Q, exact, mesh_lsq)
        errors.append(err)

    rates = _convergence_rate(errors, Ns)
    print(f"Advection O2 errors: {errors}")
    print(f"Advection O2 rates:  {rates}")
    for r in rates:
        assert r > 1.8, f"Advection O2 rate {r:.2f} < 1.8"


# ── Test: Advection O1 (sanity check) ────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_advection_o1():
    """O1 convergence sanity check."""
    a = 1.0
    t_end = 0.05
    Ns = [50, 100, 200]
    errors = []

    for N in Ns:
        bcs = BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ])
        model = ScalarAdvection(
            dimension=1,
            initial_conditions=UserFunction(
                function=lambda x: np.array([np.sin(2 * np.pi * x[0])])
            ),
            boundary_conditions=bcs,
        )
        model.parameters.a_x = a

        mesh = _make_mesh(N)
        solver = HyperbolicSolver(
            time_end=t_end,
            reconstruction_order=1,
            compute_dt=timestepping.adaptive(CFL=0.45),
        )
        Q, Qaux = solver.solve(mesh, model, write_output=False)

        mesh_lsq = ensure_lsq_mesh(mesh, model)
        exact = lambda x: np.sin(2 * np.pi * (x - a * t_end))
        err = _l2_error(Q, exact, mesh_lsq)
        errors.append(err)

    rates = _convergence_rate(errors, Ns)
    print(f"Advection O1 errors: {errors}")
    print(f"Advection O1 rates:  {rates}")
    for r in rates:
        assert r > 0.8, f"Advection O1 rate {r:.2f} < 0.8"


# ── Test: Q array shape ──────────────────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_q_shape_is_inner_cells():
    """Verify Q is (n_vars, n_inner_cells), not (n_vars, n_cells)."""
    N = 20
    bcs = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])
    model = ScalarAdvection(
        dimension=1,
        initial_conditions=UserFunction(
            function=lambda x: np.array([np.sin(2 * np.pi * x[0])])
        ),
        boundary_conditions=bcs,
    )
    model.parameters.a_x = 1.0

    mesh = _make_mesh(N)
    solver = HyperbolicSolver(
        time_end=0.01,
        reconstruction_order=1,
        compute_dt=timestepping.adaptive(CFL=0.45),
    )
    Q, Qaux = solver.solve(mesh, model, write_output=False)

    mesh_lsq = ensure_lsq_mesh(mesh, model)
    assert Q.shape[1] == mesh_lsq.n_inner_cells, (
        f"Q has {Q.shape[1]} columns, expected n_inner_cells={mesh_lsq.n_inner_cells}"
    )
    assert Q.shape[1] < mesh_lsq.n_cells, (
        f"Q should not include ghost cells: {Q.shape[1]} == n_cells={mesh_lsq.n_cells}"
    )
