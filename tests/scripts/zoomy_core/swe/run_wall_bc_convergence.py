"""
Wall BC convergence test: verify O2 with gradient extrapolation.

Uses scalar advection with Gaussian pulse reflecting off a wall.
Compares O1 (constant reconstruction) vs O2 (MUSCL) convergence rates.
"""

import os
import sys
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models.advection_model import ScalarAdvection


def exact_solution(x, t, a=1.0, sigma=0.02):
    """Gaussian advecting right at speed a. Wraps at x=1 (periodic-like)."""
    x0 = 0.3
    return np.exp(-((x - x0 - a * t) ** 2) / (2 * sigma))


def run_advection_convergence(reconstruction_order, resolutions, t_end=0.05):
    """Run convergence study with Extrapolation BCs (no wall interaction)."""
    errors = []
    for N in resolutions:
        mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=N)
        model = ScalarAdvection(dimension=1)
        model.parameter_values = np.array([1.0])
        model.initial_conditions = IC.UserFunction(
            function=lambda x: np.array([exact_solution(x[0], 0.0)])
        )
        model.boundary_conditions = BC.BoundaryConditions([
            BC.Extrapolation(tag="left"),
            BC.Extrapolation(tag="right"),
        ])

        solver = HyperbolicSolver(
            time_end=t_end,
            compute_dt=ts.adaptive(CFL=0.4),
            reconstruction_order=reconstruction_order,
        )
        Q, _ = solver.solve(mesh, model, write_output=False)
        from zoomy_core.mesh import ensure_lsq_mesh
        lsq = ensure_lsq_mesh(mesh, model)
        nc = lsq.n_inner_cells
        xc = lsq.cell_centers[0, :nc]
        u_exact = exact_solution(xc, t_end)
        l2 = np.sqrt(np.sum((Q[0, :nc] - u_exact) ** 2) / N)
        errors.append(l2)
    return errors


def run_wall_bc_convergence(reconstruction_order, resolutions, t_end=0.02):
    """Run convergence study with wall (reflective) BC on the right.

    Advects a Gaussian toward the right wall and measures error
    before it reaches the wall — the wall BC affects the gradient
    computation at boundary cells.
    """
    errors = []
    for N in resolutions:
        mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=N)
        model = ScalarAdvection(dimension=1)
        model.parameter_values = np.array([1.0])
        model.initial_conditions = IC.UserFunction(
            function=lambda x: np.array([exact_solution(x[0], 0.0)])
        )
        model.boundary_conditions = BC.BoundaryConditions([
            BC.Extrapolation(tag="left"),
            BC.Wall(tag="right", momentum_field_indices=[[0]]),
        ])

        solver = HyperbolicSolver(
            time_end=t_end,
            compute_dt=ts.adaptive(CFL=0.4),
            reconstruction_order=reconstruction_order,
        )
        Q, _ = solver.solve(mesh, model, write_output=False)
        from zoomy_core.mesh import ensure_lsq_mesh
        lsq = ensure_lsq_mesh(mesh, model)
        nc = lsq.n_inner_cells
        xc = lsq.cell_centers[0, :nc]
        u_exact = exact_solution(xc, t_end)
        l2 = np.sqrt(np.sum((Q[0, :nc] - u_exact) ** 2) / N)
        errors.append(l2)
    return errors


def print_convergence(label, resolutions, errors):
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
    print("BOUNDARY CONVERGENCE: Extrapolation BC (interior accuracy)")
    print("=" * 60)

    e_o1 = run_advection_convergence(1, resolutions)
    print_convergence("O1 (constant)", resolutions, e_o1)

    e_o2 = run_advection_convergence(2, resolutions)
    print_convergence("O2 (MUSCL+VK)", resolutions, e_o2)

    print()
    print("=" * 60)
    print("BOUNDARY CONVERGENCE: Wall BC on right boundary")
    print("=" * 60)

    e_wall_o1 = run_wall_bc_convergence(1, resolutions)
    print_convergence("O1 (constant) + Wall", resolutions, e_wall_o1)

    e_wall_o2 = run_wall_bc_convergence(2, resolutions)
    print_convergence("O2 (MUSCL+VK) + Wall", resolutions, e_wall_o2)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, errors in [
        ("O1 extrap", e_o1), ("O2 extrap", e_o2),
        ("O1 wall", e_wall_o1), ("O2 wall", e_wall_o2),
    ]:
        rates = [np.log2(errors[i-1]/errors[i]) for i in range(1, len(errors))]
        avg = np.mean(rates)
        print(f"  {label:<12s}: avg rate = {avg:.2f}")


if __name__ == "__main__":
    main()
