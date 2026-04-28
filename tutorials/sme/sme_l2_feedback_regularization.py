"""SME L=2 — explore feedback-scaled regularization.

Observation from sme_l2_hyperbolicity_compare.py:

   route A's M_x = M_x_uu  +  feedback,
                  ↑direct   ↑Schur complement of the algebraic w-block

where feedback = − M_0_uw · M_0_aw⁻¹ · M_x_au.  Inspecting xmom_j0, h-col:

   direct[0,0]   = g·h + u_0·u_1 − u_0·u_2 − u_1²/3 + u_1·u_2 − 2·u_2²/5
   feedback[0,0] =                          + 2·u_1²/3 − u_1·u_2 + 3·u_2²/5
   TOTAL         = g·h + u_0·u_1 − u_0·u_2 + u_1²/3            +   u_2²/5
                                            └────────────┬─────────────┘
                                       Koellermeier-style destabiliser

The stabilising piece is in the *direct* block, while the *feedback*
through the algebraic w-loop carries the destabiliser.  Scaling the
feedback by ε ∈ [0, 1] interpolates:

   ε = 1: original SME (route A)
   ε = 0: only the direct M_x_uu (purely stabilising in this entry)

This script sweeps ε and plots the loss-of-hyperbolicity region in the
(u_1, u_2) plane.  Compare with Koellermeier's all-or-nothing flux
modification.

Run:
    python tutorials/sme/sme_l2_feedback_regularization.py
    python tutorials/sme/sme_l2_feedback_regularization.py --no-plot   # CI
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

from zoomy_core.symbolic import polynomial_integrate as poly_int
from zoomy_core.analysis import (
    PDESystem, linearise, extract_quasilinear_pencil,
)


# ---------------------------------------------------------------------------
# Setup (re-use route B builder via import)
# ---------------------------------------------------------------------------

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])  # tutorials/sme
from sme_l2_hyperbolicity_compare import (              # noqa: E402
    build_route_b, _flat_bottom, h, u_0, u_1, u_2, g,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--U1-max", type=float, default=3.0)
    parser.add_argument("--U2-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--epsilons", type=str, default="0.0,0.5,1.0",
                        help="comma-separated feedback-scale values to plot")
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=2 feedback-scale regularization")
    print("=" * 70)

    # Build route B linearised system, extract block decomposition.
    sys_b = _flat_bottom(build_route_b())
    base_b = {f: sp.Symbol(f.func.__name__ + "_bar", real=True)
              for f in sys_b.fields}
    lin_b = linearise(sys_b, base_b)
    M_t_b, [M_x_b], M_0_b = extract_quasilinear_pencil(lin_b)

    # Reduce to the effective (4×4) flux Jacobian: M_x_eff(ε) =
    # M_x_uu + ε·feedback.  The mass matrix is just M_t_uu (the (eu,eu)
    # block of M_t).
    M_t_uu = M_t_b[:4, :4]
    M_x_uu = M_x_b[:4, :4]
    M_x_au = M_x_b[4:, :4]
    M_0_uw = M_0_b[:4, 4:]
    M_0_aw = M_0_b[4:, 4:]
    feedback = -M_0_uw * M_0_aw.inv() * M_x_au
    eps = sp.Symbol("epsilon", real=True)
    M_x_eff = M_x_uu + eps * feedback

    # Lambdify M_x_eff(ε, h_bar, u_0_bar, u_1_bar, u_2_bar) and
    # M_t_uu(ε, …) → numpy.
    h_bar, u0_bar, u1_bar, u2_bar = (base_b[h], base_b[u_0],
                                      base_b[u_1], base_b[u_2])
    syms = [eps, g, h_bar, u0_bar, u1_bar, u2_bar]
    f_M_x = sp.lambdify(syms, M_x_eff, modules="numpy")
    f_M_t = sp.lambdify(syms, M_t_uu, modules="numpy")

    epsilons = [float(s.strip()) for s in args.epsilons.split(",")]
    print(f"\nSweeping epsilons: {epsilons}")
    print(f"  grid: {args.n_points}×{args.n_points} on "
          f"(u_1, u_2) ∈ [{-args.U1_max},{args.U1_max}]"
          f"×[{-args.U2_max},{args.U2_max}]")
    print(f"  fixed: H={args.H}, U_0={args.U0}, g={args.g}")

    u1_grid = np.linspace(-args.U1_max, args.U1_max, args.n_points)
    u2_grid = np.linspace(-args.U2_max, args.U2_max, args.n_points)

    from scipy.linalg import eig
    nh_maps = {}
    for ε in epsilons:
        max_im = np.zeros((args.n_points, args.n_points))
        for i, U1 in enumerate(u1_grid):
            for j, U2 in enumerate(u2_grid):
                A = np.asarray(f_M_x(ε, args.g, args.H, args.U0,
                                     float(U1), float(U2)), dtype=complex)
                B = np.asarray(f_M_t(ε, args.g, args.H, args.U0,
                                     float(U1), float(U2)), dtype=complex)
                eigs, _ = eig(A, B)
                eigs = eigs[np.isfinite(eigs)]
                max_im[j, i] = float(np.max(np.abs(np.imag(eigs))) if eigs.size else 0)
        nh_maps[ε] = max_im
        nh_frac = float(np.mean(max_im > 1e-6))
        print(f"  ε={ε:+.2f}: {nh_frac:.1%} of grid is non-hyperbolic")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        n = len(epsilons)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        v_max = max(np.nanmax(m) for m in nh_maps.values())
        v_max = max(v_max, 1e-12)
        extent = [-args.U1_max, args.U1_max, -args.U2_max, args.U2_max]
        for k, ε in enumerate(epsilons):
            im = axes[0, k].imshow(nh_maps[ε], extent=extent, origin="lower",
                                   aspect="auto", cmap="magma",
                                   vmin=0, vmax=v_max)
            label = (f"ε={ε:+.2f}: SME (route A)" if ε == 1.0
                     else f"ε={ε:+.2f}: direct only (no feedback)" if ε == 0.0
                     else f"ε={ε:+.2f}")
            axes[0, k].set_title(label)
            axes[0, k].set_xlabel(r"$\bar u_1$")
            axes[0, k].set_ylabel(r"$\bar u_2$")
            plt.colorbar(im, ax=axes[0, k])
        fig.suptitle(
            f"SME L=2 — feedback-scale ε regularisation. "
            f"max |Im λ| ≥ 1e-6 ⇒ non-hyperbolic.   "
            f"H={args.H}, U_0={args.U0}, g={args.g}", y=1.02)
        fig.tight_layout()
        out_path = "tutorials/sme/sme_l2_feedback_regularization.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
