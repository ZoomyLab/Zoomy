"""
Wall BC convergence: SWE wave bounce-back.

Gaussian perturbation on h=1 at center, propagates to wall, reflects, returns.
Measures L2 error of h after round trip vs initial.

Compares:
  - O1 constant reconstruction
  - O2 MUSCL + Green-Gauss gradients at boundaries
  - O2 MUSCL + LSQ gradients at boundaries
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
    return 1.0 + amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def run_bounce_back(reconstruction_order, N, gradient_method="green_gauss"):
    """SWE wave round trip: center → wall → center."""
    g = 9.81
    h0 = 1.0
    c = np.sqrt(g * h0)
    t_end = 2.0 * 0.5 / c  # round trip

    mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=N)
    model = SMEInviscid(level=0)
    pv = np.array(model.parameter_values, dtype=float)
    pv[list(model.parameters.keys()).index("g")] = g
    model.parameter_values = pv

    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            0.0,
            gaussian_perturbation(x[0]),
            0.0,
        ])
    )
    # System-aware BCs: h gets use_gradient=False, momentum gets True
    model._system.boundary_conditions.apply(BC.SystemExtrapolation(), tag="left")
    model._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    model.boundary_conditions = BC.compile_system_bcs(
        model._system.boundary_conditions,
        model._equation_variable_map,
        model.dimension,
    )

    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=reconstruction_order))
    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.3),
        gradient_method=gradient_method,
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)

    lsq = ensure_lsq_mesh(mesh, model)
    nc = lsq.n_inner_cells
    xc = lsq.cell_centers[0, :nc]

    h_initial = gaussian_perturbation(xc)
    h_final = Q[1, :nc]
    l2 = np.sqrt(np.sum((h_final - h_initial) ** 2) / N)
    return l2


def run_interior(reconstruction_order, N, gradient_method="green_gauss"):
    """Control: no wall interaction. Gaussian on large domain, short time."""
    g = 9.81
    c = np.sqrt(g)
    t_end = 0.15 / c  # short, pulse stays far from boundaries

    mesh = BaseMesh.create_1d(domain=(0, 2), n_inner_cells=N)
    model = SMEInviscid(level=0)
    pv = np.array(model.parameter_values, dtype=float)
    pv[list(model.parameters.keys()).index("g")] = g
    model.parameter_values = pv

    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            0.0,
            gaussian_perturbation(x[0], x0=1.0),
            0.0,
        ])
    )
    model.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=reconstruction_order))
    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.3),
        gradient_method=gradient_method,
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)

    lsq = ensure_lsq_mesh(mesh, model)
    nc = lsq.n_inner_cells
    xc = lsq.cell_centers[0, :nc]

    # SWE linearized: perturbation splits into two waves, each half amplitude
    h_exact = 1.0 + 0.005 * (
        np.exp(-((xc - 1.0 - c * t_end) ** 2) / (2 * 0.05**2))
        + np.exp(-((xc - 1.0 + c * t_end) ** 2) / (2 * 0.05**2))
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

    print("=" * 65)
    print("SWE WALL BOUNCE-BACK CONVERGENCE")
    print("Gaussian perturbation → wall → return to center")
    print("=" * 65)

    # Interior control (no wall)
    print("\n--- Interior only (no wall interaction) ---")
    e_int_o1 = [run_interior(1, N) for N in resolutions]
    print_rates("O1 constant", resolutions, e_int_o1)

    e_int_o2_gg = [run_interior(2, N, "green_gauss") for N in resolutions]
    print_rates("O2 MUSCL + Green-Gauss", resolutions, e_int_o2_gg)

    e_int_o2_lsq = [run_interior(2, N, "lsq") for N in resolutions]
    print_rates("O2 MUSCL + LSQ", resolutions, e_int_o2_lsq)

    # Bounce-back (wall interaction)
    print("\n--- Bounce-back (wall at x=1) ---")
    e_bb_o1 = [run_bounce_back(1, N) for N in resolutions]
    print_rates("O1 constant + Wall", resolutions, e_bb_o1)

    e_bb_o2_gg = [run_bounce_back(2, N, "green_gauss") for N in resolutions]
    print_rates("O2 MUSCL + GG + Wall", resolutions, e_bb_o2_gg)

    e_bb_o2_lsq = [run_bounce_back(2, N, "lsq") for N in resolutions]
    print_rates("O2 MUSCL + LSQ + Wall", resolutions, e_bb_o2_lsq)

    # Summary
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for label, errors in [
        ("O1 interior     ", e_int_o1),
        ("O2 GG interior  ", e_int_o2_gg),
        ("O2 LSQ interior ", e_int_o2_lsq),
        ("O1 wall bounce  ", e_bb_o1),
        ("O2 GG wall      ", e_bb_o2_gg),
        ("O2 LSQ wall     ", e_bb_o2_lsq),
    ]:
        rates = [np.log2(errors[i-1]/errors[i]) for i in range(1, len(errors))]
        avg = np.mean(rates) if rates else 0
        print(f"  {label}: avg rate = {avg:.2f}  (finest L2 = {errors[-1]:.2e})")


if __name__ == "__main__":
    main()
