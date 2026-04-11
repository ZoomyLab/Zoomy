"""
Wall BC convergence: SWE wave reflection off a wall.

Setup: 1D SWE, small Gaussian perturbation on flat water (h=1),
propagates right, reflects off solid wall at x=1, returns to center.
The initial perturbation is symmetric, so after the round trip the
surface profile should match the initial condition (reversed direction).

Measures L2 error of h after round trip vs initial h.
"""

import os
import sys
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from zoomy_core.mesh import BaseMesh, ensure_lsq_mesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models.sme_model import SMEInviscid


def gaussian_perturbation(x, x0=0.5, sigma=0.05, amplitude=0.01):
    """Small surface perturbation on h=1."""
    return 1.0 + amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def run_swe_bounce_back(reconstruction_order, N, domain=(0, 1)):
    """SWE wave: Gaussian perturbation at x=0.5, wall at x=1.

    Sound speed c = sqrt(g*h) ≈ 3.13 for h=1, g=9.81.
    Round trip: 0.5/c * 2 ≈ 0.32s. Use t=0.35s to capture full return.
    """
    g = 9.81
    h0 = 1.0
    c = np.sqrt(g * h0)
    t_end = 2.0 * 0.5 / c  # round trip: center → wall → center

    mesh = BaseMesh.create_1d(domain=domain, n_inner_cells=N)

    model = SMEInviscid(level=0)
    # parameters: g, eps, ez, rho, lamda, nu
    pv = np.array(model.parameter_values, dtype=float)
    pv[list(model.parameters.keys()).index("g")] = g
    model.parameter_values = pv

    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            0.0,  # b = 0 (flat bed)
            gaussian_perturbation(x[0]),  # h = 1 + small bump
            0.0,  # hu = 0 (at rest)
        ])
    )
    model.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Wall(tag="right", momentum_field_indices=[[2]]),
    ])

    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.4),
        reconstruction_order=reconstruction_order,
    )
    Q, _ = solver.solve(mesh, model, write_output=False)

    lsq = ensure_lsq_mesh(mesh, model)
    nc = lsq.n_inner_cells
    xc = lsq.cell_centers[0, :nc]

    h_initial = gaussian_perturbation(xc)
    h_final = Q[1, :nc]  # h is variable index 1

    l2 = np.sqrt(np.sum((h_final - h_initial) ** 2) / N)
    return l2


def run_swe_interior(reconstruction_order, N):
    """Control: same SWE wave but large domain, no wall interaction."""
    g = 9.81
    h0 = 1.0
    c = np.sqrt(g * h0)
    t_end = 0.5 / c  # one-way trip: center → right (no wall hit)

    mesh = BaseMesh.create_1d(domain=(0, 2), n_inner_cells=N)
    model = SWEModel()
    model.parameter_values = np.array([g, 1e-6])

    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            0.0,
            gaussian_perturbation(x[0]),
            0.0,
        ])
    )
    model.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.4),
        reconstruction_order=reconstruction_order,
    )
    Q, _ = solver.solve(mesh, model, write_output=False)

    lsq = ensure_lsq_mesh(mesh, model)
    nc = lsq.n_inner_cells
    xc = lsq.cell_centers[0, :nc]

    # Exact: perturbation splits into left+right traveling waves, each half amplitude
    # At t = 0.5/c the right-traveling wave is at x=1.0
    # For small perturbations, h(x,t) ≈ 1 + A/2 * [G(x-ct) + G(x+ct)]
    h_exact = 1.0 + 0.005 * (
        np.exp(-((xc - 0.5 - c * t_end) ** 2) / (2 * 0.05**2))
        + np.exp(-((xc - 0.5 + c * t_end) ** 2) / (2 * 0.05**2))
    )
    l2 = np.sqrt(np.sum((Q[1, :nc] - h_exact) ** 2) / N)
    return l2


def print_rates(label, resolutions, errors):
    print(f"\n  {label}:")
    print(f"    {'N':>6s}  {'L2 error':>12s}  {'rate':>6s}")
    for i, (N, e) in enumerate(zip(resolutions, errors)):
        if i > 0:
            rate = np.log2(errors[i - 1] / e)
            print(f"    {N:6d}  {e:12.6e}  {rate:6.2f}")
        else:
            print(f"    {N:6d}  {e:12.6e}     --")


def main():
    resolutions = [50, 100, 200, 400]

    print("=" * 60)
    print("SWE WALL BOUNCE-BACK CONVERGENCE")
    print("Small Gaussian perturbation → wall → return to center")
    print("=" * 60)

    e_bb_o1 = [run_swe_bounce_back(1, N) for N in resolutions]
    print_rates("O1 (constant) + Wall", resolutions, e_bb_o1)

    e_bb_o2 = [run_swe_bounce_back(2, N) for N in resolutions]
    print_rates("O2 (MUSCL+VK) + Wall", resolutions, e_bb_o2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, errors in [
        ("O1 bounce", e_bb_o1), ("O2 bounce", e_bb_o2),
    ]:
        rates = [np.log2(errors[i-1]/errors[i]) for i in range(1, len(errors))]
        avg = np.mean(rates)
        print(f"  {label}: avg rate = {avg:.2f}  (finest L2 = {errors[-1]:.2e})")


if __name__ == "__main__":
    main()
