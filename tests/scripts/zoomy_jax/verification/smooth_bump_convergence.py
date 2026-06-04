"""Smooth lake-at-rest perturbation — convergence test.

Initial condition: ``h(x, 0) = h_0 + A · exp(-((x − x_0)/σ)²)`` with
``hu = hv = 0`` and ``b = 0``.  No shocks, no rarefactions — the
solution is C^∞ everywhere.  At t = 0+ the perturbation splits into a
left-going and a right-going small-amplitude gravity wave (linearised
about ``h_0``).  Reference solution: run at a very high resolution
and compare lower-resolution runs against it.

A scheme that's *truly* 2nd-order on the SystemModel + LSQ-MUSCL +
PositiveNonconservativeHLL stack should achieve L¹ rate ≈ 2 between
successive doublings of ``nx`` on this problem.  A rate of 1 means
the time stepping or the Riemann solver is the limiting order; a rate
of 0 means there's a structural bug.

Compare JAX vs NumPy backends so we can localise.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wetdry_dambreak_probe import setup_solver, build_swe_sm  # noqa: E402
from stoker_convergence import make_step, run_to_time  # noqa: E402
from zoomy_jax.mesh.mesh import convert_mesh_to_jax  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402


LX = 20.0
LY = 0.1
NY = 2
H_0 = 1.0
A = 0.01            # 1% perturbation
SIGMA = 1.0
X_0 = LX / 2
T_END = 0.5
G = 9.81


def bump_analytical(x, t):
    """Linearised wave-equation solution for the Gaussian bump IC.

    δh(x, t) = (A/2) [exp(-((x - x_0 - c t)/σ)²) + exp(-((x - x_0 + c t)/σ)²)]
    Accurate to O(A/h_0) ~ 1% (we use A = 0.01·h_0 so the linearisation
    error is small enough that this gives a clean reference for
    convergence analysis).
    """
    c = float(np.sqrt(G * H_0))
    x = np.asarray(x, dtype=float)
    return H_0 + 0.5 * A * (
        np.exp(-((x - X_0 - c * t) / SIGMA) ** 2)
        + np.exp(-((x - X_0 + c * t) / SIGMA) ** 2)
    )


def set_bump_ic(solver, Q):
    nc = int(solver._rt_mesh.n_inner_cells)
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc]
    h0 = H_0 + A * np.exp(-((xc - X_0) / SIGMA) ** 2)
    Q = Q.at[0, :nc].set(0.0)
    Q = Q.at[1, :nc].set(jnp.asarray(h0, dtype=Q.dtype))
    Q = Q.at[2, :nc].set(0.0)
    Q = Q.at[3, :nc].set(0.0)
    return Q


def run_one(nx, order, mode, cfl, time_scheme):
    mesh = LSQMesh.create_2d((0.0, LX, 0.0, LY), nx, NY)
    solver, Q, Qaux, sm = setup_solver(
        mesh, order=order, reconstruction_mode=mode, limiter="minmod")
    Q = set_bump_ic(solver, Q)
    Q, t_end, n_steps = run_to_time(
        solver, Q, Qaux, t_end=T_END, cfl=cfl, time_scheme=time_scheme)
    nc = int(solver._rt_mesh.n_inner_cells)
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc]
    h = np.asarray(Q)[1, :nc]
    return xc, h, n_steps


def main():
    nx_list = [int(x) for x in
               os.environ.get("BUMP_NX_LIST", "50,100,200,400").split(",")]
    print(f"Smooth Gaussian bump (h_0={H_0}, A={A}, σ={SIGMA}, T_END={T_END})")
    print(f"Reference: analytical linearised wave-equation solution.\n")

    schemes = {
        "O1 FE":      (1, "conservative", 0.5,  "fe"),
        "O1 RK2":     (1, "conservative", 0.5,  "ssprk2"),
        "O2 cons FE": (2, "conservative", 0.25, "fe"),
        "O2 cons RK2":(2, "conservative", 0.25, "ssprk2"),
        "O2 η-MUSCL RK2": (2, "eta",      0.25, "ssprk2"),
    }
    for label, (order, mode, cfl, ts) in schemes.items():
        print(f"=== {label} ===")
        results = []
        for nx in nx_list:
            t0 = time.perf_counter()
            xc, h, n_steps = run_one(nx, order, mode, cfl, ts)
            wall = time.perf_counter() - t0
            h_ana = bump_analytical(xc, T_END)
            err = np.abs(h - h_ana)
            L1 = float(np.mean(err) * LX)
            Linf = float(np.max(err))
            results.append((nx, L1, Linf, n_steps, wall))
        for nx, L1, Linf, n_steps, wall in results:
            print(f"  nx={nx:>5}  L1={L1:.4e}  Linf={Linf:.4e}  "
                  f"steps={n_steps:>5}  wall={wall:.1f}s")
        if len(results) >= 2:
            print("  rates: ", end="")
            for i in range(1, len(results)):
                ratio = nx_list[i] / nx_list[i - 1]
                r_L1 = np.log(results[i - 1][1] / max(results[i][1], 1e-30)) / np.log(ratio)
                print(f"L1={r_L1:+.2f}  ", end="")
            print()
        print()


if __name__ == "__main__":
    main()
