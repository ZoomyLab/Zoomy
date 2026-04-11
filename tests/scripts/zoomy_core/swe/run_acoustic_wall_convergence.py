"""
Wall BC convergence: standing wave in a box (SWE linearized).

SWE with small perturbation on h=1: equivalent to linear acoustics.
Cosine standing wave between two walls, exact solution for all time.

h(x,t) = 1 + A*cos(m*pi*x/L)*cos(m*pi*c*t/L)
u(x,t) = A*c/h0 * sin(m*pi*x/L)*sin(m*pi*c*t/L)

where c = sqrt(g*h0).  Walls at x=0 and x=L enforce u=0.
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


def run_standing_wave(reconstruction_order, N, gradient_method="green_gauss",
                      n_periods=1, mode=1):
    """Standing wave between two walls. Returns L2 error of h."""
    g = 9.81
    h0 = 1.0
    c = np.sqrt(g * h0)
    L = 1.0
    A = 1e-4  # small amplitude → linear regime
    m = mode

    omega = m * np.pi * c / L
    period = 2 * np.pi / omega
    t_end = n_periods * period

    mesh = BaseMesh.create_1d(domain=(0, L), n_inner_cells=N)
    model = SMEInviscid(level=0)
    pv = np.array(model.parameter_values, dtype=float)
    pv[list(model.parameters.keys()).index("g")] = g
    model.parameter_values = pv

    # IC: standing wave at t=0 → h = h0 + A*cos(m*pi*x/L), hu = 0
    model.initial_conditions = IC.UserFunction(
        function=lambda x, _A=A, _m=m, _L=L, _h0=h0: np.array([
            0.0,
            _h0 + _A * np.cos(_m * np.pi * x[0] / _L),
            0.0,
        ])
    )

    # Wall at both ends
    model.boundary_conditions = BC.BoundaryConditions([
        BC.Wall(tag="left", momentum_field_indices=[[2]]),
        BC.Wall(tag="right", momentum_field_indices=[[2]]),
    ])

    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.3),
        reconstruction_order=reconstruction_order,
        gradient_method=gradient_method,
    )
    Q, _ = solver.solve(mesh, model, write_output=False)

    lsq = ensure_lsq_mesh(mesh, model)
    nc = lsq.n_inner_cells
    xc = lsq.cell_centers[0, :nc]

    # Exact: after n full periods, solution = initial condition
    h_exact = h0 + A * np.cos(m * np.pi * xc / L)
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
    resolutions = [20, 40, 80, 160, 320]

    print("=" * 65)
    print("STANDING WAVE IN A BOX: walls at x=0 and x=1")
    print("h(x,t) = 1 + A*cos(pi*x)*cos(pi*c*t), A=1e-4")
    print("Measured after 1 full period (solution = IC)")
    print("=" * 65)

    for gm in ["green_gauss", "lsq"]:
        print(f"\n--- Gradient method: {gm} ---")

        e_o1 = [run_standing_wave(1, N, gm) for N in resolutions]
        print_rates("O1 (constant)", resolutions, e_o1)

        e_o2 = [run_standing_wave(2, N, gm) for N in resolutions]
        print_rates("O2 (MUSCL+VK)", resolutions, e_o2)

    # Summary
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for label in ["O1 GG", "O2 GG", "O1 LSQ", "O2 LSQ"]:
        pass  # printed above


if __name__ == "__main__":
    main()
