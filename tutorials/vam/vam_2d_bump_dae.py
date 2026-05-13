"""VAM(1,2,2) 2D radial-bump test case via :class:`DAESolver`.

Uses ``VAMModelGalerkin(dimension=3)`` — the fully 2D-horizontal chain
with state ``(h, U_0, U_1, V_0, V_1, W_0, W_1, P_0, P_1)``.  The radial
Gaussian bump on ``h`` should propagate radially outward and remain
rotationally symmetric.

Output: ``vam_2d_bump_dae.h5`` + ``vam_2d_bump_dae.<i>.vtk``
(open ``vam_2d_bump_dae.vtk.series`` in Paraview).
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.fvm.solver_dae_numpy import DAESolver
import zoomy_core.fvm.timestepping as ts
from zoomy_core.misc.misc import Settings, Zstruct


# ── Configuration ────────────────────────────────────────────────────

NX, NY = 10, 10       # 9-state × 100 cells = 900 unknowns
L = 6.0               # square domain [-L/2, L/2]²
H = 1.0
AMP = 0.05
SIGMA = 1.0
G = 9.81
T_END = 0.1
OUTPUT_DIR = "outputs_vam_2d_bump_dae"
FILENAME = "vam_2d_bump_dae"


def main():
    print("=" * 70)
    print("VAM(1,2,2) 2D radial-bump DAE simulation")
    print("=" * 70)
    print(f"  domain        : [{-L/2}, {L/2}]²")
    print(f"  cells         : {NX} × {NY} = {NX*NY}")
    print(f"  H, amp, sigma : {H}, {AMP}, {SIGMA}")
    print(f"  T_end         : {T_END}")

    # 1. Model — dimension=3 → state has both U_k and V_k.
    model = VAMModelGalerkin(level=1, dimension=3,
                             eigenvalue_mode="numerical")
    model.parameters.g = G
    n_state = model.n_variables
    print(f"  state         : {list(model.variables.keys())}")

    # 2. Initial conditions — radial Gaussian bump on h.
    def ic(x):
        Q = np.zeros(n_state)
        r2 = x[0]**2 + x[1]**2
        Q[0] = H + AMP * np.exp(-r2 / SIGMA**2)
        return Q

    model.initial_conditions = UserFunction(function=ic)

    # 3. Extrapolation BC on all four sides.
    model.boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
        Extrapolation(tag="bottom"),
        Extrapolation(tag="top"),
    ])

    # 4. Mesh.
    mesh = BaseMesh.create_2d(
        domain=(-L/2, L/2, -L/2, L/2), nx=NX, ny=NY,
    )

    # 5. Solver.
    n_snapshots = 11
    settings = Settings.default()
    settings.output = Zstruct(
        directory=OUTPUT_DIR,
        filename=FILENAME,
        snapshots=n_snapshots,
        clean_directory=True,
    )

    solver = DAESolver(
        time_end=T_END,
        method="ars232",
        compute_dt=ts.adaptive(CFL=0.3),
        newton_tol=1e-8,
        settings=settings,
    )

    # 6. Run.  ``b`` is a proper aux field; default flat bottom.
    Q_final, Qaux_final = solver.solve(
        mesh, model, write_output=True,
    )

    # 7. Diagnostics.
    print(f"\n[result] final t = {solver._sim_time:.4f}")
    print(f"         h range = [{Q_final[0, :].min():.4f}, "
          f"{Q_final[0, :].max():.4f}]")
    nc = solver.nc
    dx_avg = float(np.mean(solver._sim_mesh.cell_volumes[:nc])) ** (1/solver.dim)
    cv = solver._sim_mesh.cell_volumes[:nc]
    mass0 = H * L * L
    massT = float(np.sum(Q_final[0, :] * cv))
    print(f"         mass(0) ≈ {mass0:.4f}")
    print(f"         mass(T) = {massT:.4f}  (Δ={abs(massT-mass0):.3e})")
    print(f"         max |U_0| = {np.max(np.abs(Q_final[1, :])):.4e}")

    # 8. Paraview export.
    print(f"\n[export] writing VTK time series to {OUTPUT_DIR}/...")
    solver.export_vtk(filename=FILENAME)
    print(f"[export] Open {OUTPUT_DIR}/{FILENAME}.vtk.series in Paraview.")

    return Q_final


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
