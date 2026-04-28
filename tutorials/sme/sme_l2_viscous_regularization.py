"""SME L=2 — Newtonian τ_xx viscous regularization.

Add a Newtonian shear stress τ_xx = 2μ · ∂_x u  (incompressible
in-plane) to the x-momentum equation:

   ∂_t u + u·∂_x u + w·∂_z u + g·∂_x η = 2ν · ∂_xx u

with ν = μ/ρ.  Linearised at a homogeneous flat-bottom base state
(ū_i, h̄, ∂_x h̄ = 0), the projection of ν · ∂_xx u against φ_j(z)
gives by Legendre orthogonality (∫_0^1 φ_i φ_j dξ = δ_ij/(2j+1)):

   ∫_b^η ∂_xx u_ansatz · φ_j dz  →  h̄/(2j+1) · ∂_xx δu_j

so the viscous coefficient matrix is *diagonal* in the (u_0, u_1, u_2)
sub-block, with entries h̄, h̄/3, h̄/5 (and zero on the h-row, since
continuity has no viscous term).

Plane-wave dispersion at finite k:

   ω · M_t · q̂  =  k · M_x · q̂  −  i · 2ν · k² · M_visc · q̂

so generalized eigenvalue problem  ``(k M_x − 2iν k² M_visc, M_t)``
gives ω(k).  Stability:  Im(ω) ≤ 0.

This script sweeps (ū_1, ū_2) at fixed k for several ν, and plots the
unstable region (where max Im ω > 0).  Compares against the inviscid
SME L=2.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from sme_l2_hyperbolicity_compare import (              # noqa: E402
    build_route_b, _flat_bottom, h, u_0, u_1, u_2, g,
)
from zoomy_core.analysis import linearise, extract_quasilinear_pencil


def _max_im(eigs):
    eigs = eigs[np.isfinite(eigs)]
    return float(np.max(np.imag(eigs)) if eigs.size else 0)   # signed (positive ⇒ unstable)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--U1-max", type=float, default=3.0)
    parser.add_argument("--U2-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    parser.add_argument("--k-val", type=float, default=1.0,
                        help="wavenumber at which to evaluate the dispersion")
    parser.add_argument("--nus", type=str, default="0.0,0.001,0.01,0.1",
                        help="comma-separated kinematic viscosities to test")
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=2 — Newtonian τ_xx viscous regularisation")
    print("=" * 70)

    # Build route B → Schur to route A's principal-symbol matrices.
    sys_b = _flat_bottom(build_route_b())
    base_b = {f: sp.Symbol(f.func.__name__ + "_bar", real=True)
              for f in sys_b.fields}
    lin_b = linearise(sys_b, base_b)
    M_t_b, [M_x_b], M_0_b = extract_quasilinear_pencil(lin_b)
    M_t_uu = M_t_b[:4, :4]
    M_x_uu = M_x_b[:4, :4]
    M_x_au = M_x_b[4:, :4]
    M_0_uw = M_0_b[:4, 4:]
    M_0_aw = M_0_b[4:, 4:]
    M_x_eff = M_x_uu - M_0_uw * M_0_aw.inv() * M_x_au

    # Lambdify into numpy.
    h_bar, u0_bar, u1_bar, u2_bar = (base_b[h], base_b[u_0],
                                      base_b[u_1], base_b[u_2])
    syms = [g, h_bar, u0_bar, u1_bar, u2_bar]
    f_M_t = sp.lambdify(syms, M_t_uu, modules="numpy")
    f_M_x = sp.lambdify(syms, M_x_eff, modules="numpy")

    # Viscous coefficient matrix at homogeneous base.
    # Equation order in route B's [evol] block is
    #     [xmom_j=0, xmom_j=1, xmom_j=2, KBC_top].
    # Field column order is [h, u_0, u_1, u_2].
    # The viscous term 2ν·∂_xx u in xmom_j projects to 2ν·h/(2j+1)·∂_xx δu_j,
    # which lives in (row j, col u_j).  KBC_top has no viscous term.
    # Result: super-diagonal pattern matching M_t_uu's first three rows.
    def M_visc(H_val):
        M = np.zeros((4, 4))
        M[0, 1] = H_val            # xmom_j=0 has 2ν h ∂_xx δu_0
        M[1, 2] = H_val / 3.0      # xmom_j=1 has 2ν h/3 ∂_xx δu_1
        M[2, 3] = H_val / 5.0      # xmom_j=2 has 2ν h/5 ∂_xx δu_2
        return M

    nus = [float(s.strip()) for s in args.nus.split(",")]
    print(f"\nTesting kinematic viscosities ν: {nus}")
    print(f"  fixed wavenumber k = {args.k_val}")
    print(f"  grid: {args.n_points}×{args.n_points} on "
          f"(u_1, u_2) ∈ [{-args.U1_max},{args.U1_max}]"
          f"×[{-args.U2_max},{args.U2_max}]")

    u1_grid = np.linspace(-args.U1_max, args.U1_max, args.n_points)
    u2_grid = np.linspace(-args.U2_max, args.U2_max, args.n_points)

    from scipy.linalg import eig
    Mv = M_visc(args.H)

    nh_maps = {}
    for nu in nus:
        max_im_grid = np.zeros((args.n_points, args.n_points))
        for i, U1 in enumerate(u1_grid):
            for j, U2 in enumerate(u2_grid):
                params = (args.g, args.H, args.U0, float(U1), float(U2))
                Mt = np.asarray(f_M_t(*params), dtype=complex)
                Mx = np.asarray(f_M_x(*params), dtype=complex)
                A = args.k_val * Mx - 1j * 2 * nu * args.k_val**2 * Mv
                eigs, _ = eig(A, Mt)
                eigs = eigs[np.isfinite(eigs)]
                # Im(ω) > 0 → unstable.
                max_im_grid[j, i] = float(np.max(np.imag(eigs))) if eigs.size else 0
        nh_maps[nu] = max_im_grid
        n_unstable = int(np.sum(max_im_grid > 1e-6))
        peak = float(np.max(max_im_grid))
        print(f"  ν = {nu:8.5f}: {n_unstable}/{args.n_points**2} unstable "
              f"({100*n_unstable/args.n_points**2:.2f}%); peak Im ω = {peak:.4f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        v_max = max(np.max(m) for m in nh_maps.values())
        v_min = min(np.min(m) for m in nh_maps.values())
        v_max_abs = max(abs(v_max), abs(v_min), 1e-12)
        extent = [-args.U1_max, args.U1_max, -args.U2_max, args.U2_max]
        fig, axes = plt.subplots(1, len(nus), figsize=(4.2 * len(nus), 4))
        if len(nus) == 1:
            axes = [axes]
        for ax, nu in zip(axes, nus):
            im = ax.imshow(nh_maps[nu], extent=extent, origin="lower",
                           aspect="auto", cmap="seismic",
                           vmin=-v_max_abs, vmax=v_max_abs)
            ax.set_title(f"ν = {nu}", fontsize=10)
            ax.set_xlabel(r"$\bar u_1$")
            ax.set_ylabel(r"$\bar u_2$")
            plt.colorbar(im, ax=ax, label=r"max Im $\omega$ (>0 = unstable)")
        fig.suptitle(
            f"SME L=2 + Newtonian τ_xx — max Im ω(k) at k={args.k_val}.   "
            f"H={args.H}, U_0={args.U0}, g={args.g}", y=1.02)
        fig.tight_layout()
        out_path = "tutorials/sme/sme_l2_viscous_regularization.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")

    # Find the *minimum ν* that stabilises the worst point in the
    # unstable region (where ν=0 has Im ω > 0).  A common scaling.
    inviscid = nh_maps[nus[0]] if nus[0] == 0.0 else None
    if inviscid is not None:
        peak_invisc = float(np.max(inviscid))
        if peak_invisc > 0:
            # ω_invisc + ω_visc = 0 ⇒ peak Im ω - 2ν k² · m_visc_eig = 0
            # Approximate m_visc_eig ≈ smallest eigenvalue of M_t^-1 M_visc.
            m_visc_eig_min = 1.0 / 5.0   # most "moment" mode (u_2)
            nu_crit = peak_invisc / (2 * args.k_val**2 * m_visc_eig_min)
            print(f"\nNaïve estimate of critical ν to stabilise (k={args.k_val}):")
            print(f"  ν_crit ≈ peak_inviscid / (2 k² · m_visc_min) = "
                  f"{peak_invisc:.3f} / (2·{args.k_val}²·{m_visc_eig_min:.3f}) "
                  f"= {nu_crit:.4f}")
            re_estim = args.U0 if abs(args.U0) > 1e-9 else 1.0
            print(f"  in Re-units (U·H/ν): Re_crit ≈ "
                  f"{re_estim * args.H / nu_crit:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
