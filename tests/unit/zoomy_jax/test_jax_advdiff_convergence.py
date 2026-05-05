"""Convergence tests for JAX solvers: advection, diffusion, and advection-diffusion.

Tests verify 2nd-order convergence for:
- Pure advection (HyperbolicSolver, O2, MUSCL+VK, RK2)
- Advection-diffusion IMEX (IMEXSourceSolverJax, O2, MUSCL+VK, RK2 + CN diffusion)

Each test runs on grids N=100, 200, 400 and checks that the convergence
rate is >= 1.8 (theoretical: 2.0).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from zoomy_core.mesh import BaseMesh, ensure_lsq_mesh
from zoomy_core.model.models.advection_model import ScalarAdvection, ScalarAdvectionDiffusion
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
import zoomy_core.fvm.timestepping as timestepping

from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mesh(N, domain=(0.0, 1.0)):
    """Create a 1D mesh with N inner cells."""
    return BaseMesh.create_1d(domain=domain, n_inner_cells=N)


def _smooth_ic(x):
    """Smooth sine initial condition for convergence tests."""
    return np.array([np.sin(2 * np.pi * x[0])])


def _exact_advection(x, t, a=1.0):
    """Exact solution for pure advection: u(x,t) = sin(2*pi*(x - a*t))."""
    return np.sin(2 * np.pi * (x - a * t))


def _exact_advdiff(x, t, a=1.0, nu=1e-3):
    """Exact solution for advection-diffusion with periodic BCs.

    u(x,t) = exp(-4*pi^2*nu*t) * sin(2*pi*(x - a*t))
    """
    return np.exp(-4 * np.pi**2 * nu * t) * np.sin(2 * np.pi * (x - a * t))


def _l2_error(Q, exact, mesh):
    """L2 error norm over inner cells."""
    nc = mesh.n_inner_cells
    x = mesh.cell_centers[0, :nc]
    dx = mesh.cell_volumes[:nc]
    u_num = np.asarray(Q[0, :nc])
    u_ex = exact(x)
    return float(np.sqrt(np.sum((u_num - u_ex)**2 * dx) / np.sum(dx)))


def _convergence_rate(errors, Ns):
    """Compute convergence rate from errors at different resolutions."""
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(Ns[i] / Ns[i-1])
        rates.append(r)
    return rates


# ── Test: Pure Advection O2 ─────────────────────────────────────────────────

@pytest.mark.jax
def test_advection_o2_convergence():
    """Verify 2nd-order convergence for pure advection with JAX HyperbolicSolver."""
    a = 1.0
    t_end = 0.1
    Ns = [100, 200, 400]
    errors = []

    for N in Ns:
        bcs = BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ])
        model = ScalarAdvection(
            dimension=1,
            initial_conditions=UserFunction(function=_smooth_ic),
            boundary_conditions=bcs,
        )
        model.parameters.update(dict(zip(model.parameters.keys(), [a])))

        mesh = _make_mesh(N)
        solver = HyperbolicSolver(
            time_end=t_end,
            reconstruction_order=2,
            limiter="venkatakrishnan",
            compute_dt=timestepping.adaptive(CFL=0.45),
        )

        Q, Qaux = solver.solve(mesh, model, write_output=False)

        exact = lambda x: _exact_advection(x, t_end, a=a)
        mesh_lsq = ensure_lsq_mesh(mesh, model)
        err = _l2_error(Q, exact, mesh_lsq)
        errors.append(err)

    rates = _convergence_rate(errors, Ns)
    print(f"Advection O2 errors: {errors}")
    print(f"Advection O2 rates:  {rates}")

    for r in rates:
        assert r > 1.8, f"Advection O2 convergence rate {r:.2f} < 1.8"


# ── Test: Advection-Diffusion IMEX O2 ───────────────────────────────────────

@pytest.mark.jax
def test_advdiff_imex_o2_convergence():
    """Verify 2nd-order convergence for advection-diffusion with JAX IMEX solver.

    Uses MUSCL+VK reconstruction, RK2 for explicit advection stage,
    and Crank-Nicolson for implicit diffusion.
    """
    a = 1.0
    nu = 1e-2  # relatively large nu so diffusion effect is clear
    t_end = 0.05
    Ns = [100, 200, 400]
    errors = []

    for N in Ns:
        bcs = BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ])
        model = ScalarAdvectionDiffusion(
            dimension=1,
            nu=nu,
            initial_conditions=UserFunction(function=_smooth_ic),
            boundary_conditions=bcs,
        )
        model.parameters.update(dict(zip(model.parameters.keys(), [a, nu])))

        mesh = _make_mesh(N)
        solver = IMEXSourceSolverJax(
            time_end=t_end,
            reconstruction_order=2,
            limiter="venkatakrishnan",
            compute_dt=timestepping.adaptive(CFL=0.45, nu=nu),
        )

        Q, Qaux = solver.solve(mesh, model, write_output=False)

        exact = lambda x, _t=t_end, _a=a, _nu=nu: _exact_advdiff(x, _t, a=_a, nu=_nu)
        mesh_lsq = ensure_lsq_mesh(mesh, model)
        err = _l2_error(Q, exact, mesh_lsq)
        errors.append(err)

    rates = _convergence_rate(errors, Ns)
    print(f"AdvDiff IMEX O2 errors: {errors}")
    print(f"AdvDiff IMEX O2 rates:  {rates}")

    for r in rates:
        assert r > 1.8, f"AdvDiff IMEX O2 convergence rate {r:.2f} < 1.8"


# ── Test: Pure Diffusion IMEX O2 ────────────────────────────────────────────

@pytest.mark.jax
def test_diffusion_imex_o2_convergence():
    """Verify 2nd-order convergence for pure diffusion (zero advection).

    Uses Crank-Nicolson implicit diffusion with zero advection velocity.
    """
    nu = 1e-2
    t_end = 0.05
    Ns = [50, 100, 200]
    errors = []

    for N in Ns:
        bcs = BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ])
        model = ScalarAdvectionDiffusion(
            dimension=1,
            nu=nu,
            initial_conditions=UserFunction(function=_smooth_ic),
            boundary_conditions=bcs,
        )
        # Zero advection velocity, only diffusion
        model.parameters.update(dict(zip(model.parameters.keys(), [0.0, nu])))

        mesh = _make_mesh(N)
        solver = IMEXSourceSolverJax(
            time_end=t_end,
            reconstruction_order=2,
            limiter="venkatakrishnan",
            compute_dt=timestepping.adaptive(CFL=0.45, nu=nu),
        )

        Q, Qaux = solver.solve(mesh, model, write_output=False)

        exact = lambda x, _t=t_end, _nu=nu: _exact_advdiff(x, _t, a=0.0, nu=_nu)
        mesh_lsq = ensure_lsq_mesh(mesh, model)
        err = _l2_error(Q, exact, mesh_lsq)
        errors.append(err)

    rates = _convergence_rate(errors, Ns)
    print(f"Diffusion O2 errors: {errors}")
    print(f"Diffusion O2 rates:  {rates}")

    for r in rates:
        assert r > 1.8, f"Diffusion O2 convergence rate {r:.2f} < 1.8"


if __name__ == "__main__":
    print("=" * 60)
    print("JAX Advection O2 Convergence")
    print("=" * 60)
    test_advection_o2_convergence()

    print()
    print("=" * 60)
    print("JAX Advection-Diffusion IMEX O2 Convergence")
    print("=" * 60)
    test_advdiff_imex_o2_convergence()

    print()
    print("=" * 60)
    print("JAX Pure Diffusion IMEX O2 Convergence")
    print("=" * 60)
    test_diffusion_imex_o2_convergence()

    print()
    print("All convergence tests passed!")
