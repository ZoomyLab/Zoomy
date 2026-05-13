"""VAM(1,2,2) 1D cosine-bump test case via :class:`DAESolver`.

End-to-end demonstration:
    Model → SystemModel → numpy runtime → DAE solver → HDF5 snapshots
    → Paraview VTK series.

Wave-speed verification against Escalante 2024 eq (10):

    C²/(g·H)  =  (1 + (kH)²/12) / (1 + 5(kH)²/12 + (kH)⁴/144).

A single-mode cosine ``h(x) = H + amp·cos(2π·x/L)`` is initialised on a
periodic-like (Extrapolation BC) interval long enough that the wave
doesn't interact with the boundaries during the run.  The mode splits
into two halves travelling at ±C(k=2π/L); we measure the apparent
phase speed from the cell-by-cell shift and compare to the closed-form
dispersion expectation.

Run::

    python tutorials/vam/vam_1d_bump_dae.py

Output: ``vam_1d_bump_dae.h5`` + ``vam_1d_bump_dae.<i>.vtk`` (Paraview
time series).
"""
from __future__ import annotations

import os
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

NX = 40             # dense FD Jacobian → keep grid small for v1
L = 20.0
H = 1.0
AMP = 0.02
N_MODES = 1
G = 9.81
T_END = 1.0
OUTPUT_DIR = "outputs_vam_1d_bump_dae"
FILENAME = "vam_1d_bump_dae"


def escalante_eq10_phase_speed(k, H_=H, g_=G):
    """Closed-form phase speed from Escalante 2024 eq (10)."""
    kH = k * H_
    return np.sqrt(g_ * H_ * (1 + kH**2 / 12) / (1 + 5*kH**2/12 + kH**4/144))


def main():
    print("=" * 70)
    print("VAM(1,2,2) 1D cosine-bump DAE simulation")
    print("=" * 70)
    print(f"  domain        : [0, {L}]")
    print(f"  cells         : {NX}")
    print(f"  H             : {H}")
    print(f"  cosine amp    : {AMP}")
    print(f"  k = 2π·{N_MODES}/L = {2*np.pi*N_MODES/L:.4f}")
    print(f"  T_end         : {T_END}")

    # 1. Model — dimension=2 (1D-horizontal), Cantero-Chinchilla form.
    model = VAMModelGalerkin(level=1, dimension=2)
    model.parameters.g = G

    # 2. Initial conditions — cosine bump in h, rest in everything else.
    n_state = model.n_variables

    def ic(x):
        Q = np.zeros(n_state)
        Q[0] = H + AMP * np.cos(2 * np.pi * N_MODES * x[0] / L)
        return Q

    model.initial_conditions = UserFunction(function=ic)

    # 3. Boundary conditions — Extrapolation on both ends.
    model.boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])

    # 4. Mesh.
    mesh = BaseMesh.create_1d(domain=(0.0, L), n_inner_cells=NX)

    # 5. Solver.
    n_snapshots = 21
    settings = Settings.default()
    settings.output = Zstruct(
        directory=OUTPUT_DIR,
        filename=FILENAME,
        snapshots=n_snapshots,
        clean_directory=True,
    )

    solver = DAESolver(
        time_end=T_END,
        method="ars343",
        compute_dt=ts.adaptive(CFL=0.3),
        newton_tol=1e-9,
        settings=settings,
    )

    # 6. Run.  Bathymetry ``b(x)`` is a proper aux field — default
    # flat bottom (b ≡ 0) means ``topography`` is unset.
    Q_final, Qaux_final = solver.solve(
        mesh, model, write_output=True,
    )

    # 7. Diagnostics.
    print(f"\n[result] final t = {solver._sim_time:.4f}")
    print(f"         h range = [{Q_final[0, :].min():.4f}, "
          f"{Q_final[0, :].max():.4f}]")
    dx = float(solver._sim_mesh.cell_volumes[0])
    mass0 = H * L
    massT = Q_final[0, :].sum() * dx
    print(f"         mass(0) ≈ {mass0:.4f}")
    print(f"         mass(T) = {massT:.4f}  (Δ={abs(massT-mass0):.3e})")

    # Wave-speed verification.
    k = 2 * np.pi * N_MODES / L
    c_pred = escalante_eq10_phase_speed(k, H, G)
    # Cross-correlation phase shift: project final h onto cos(k(x-c·T))
    # and sin(k(x-c·T)), pick c that minimises sin component.
    x = solver._sim_mesh.cell_centers[0, :NX]
    h_dev = Q_final[0, :] - H
    # Standing-wave decomposition: amp_c·cos(k·x − ω·t) + amp_c·cos(k·x + ω·t)
    # = 2·amp_c·cos(ω·t)·cos(k·x).
    # So at time T, amplitude factor is cos(ω·T) = cos(k·c·T).
    cos_kx = np.cos(k * x)
    proj = np.sum(h_dev * cos_kx) * dx / (L / 2)   # ≈ amp · cos(ω·T)
    cos_omega_T = proj / AMP
    cos_omega_T = float(np.clip(cos_omega_T, -1.0, 1.0))
    omega_T_obs = np.arccos(cos_omega_T)
    c_obs = omega_T_obs / (k * T_END)
    print(f"\n[dispersion] k = 2π·{N_MODES}/L = {k:.4f},  kH = {k*H:.4f}")
    print(f"             Escalante eq (10) c = {c_pred:.6f}")
    print(f"             observed c          = {c_obs:.6f}")
    print(f"             relative error      = "
          f"{abs(c_obs - c_pred)/c_pred:.3e}")

    # 8. Paraview export.
    print(f"\n[export] writing VTK time series to {OUTPUT_DIR}/...")
    solver.export_vtk(filename=FILENAME)
    print(f"[export] Open {OUTPUT_DIR}/{FILENAME}.vtk.series in Paraview.")

    return Q_final, c_obs, c_pred


if __name__ == "__main__":
    sys.exit(0 if main()[0] is not None else 1)
