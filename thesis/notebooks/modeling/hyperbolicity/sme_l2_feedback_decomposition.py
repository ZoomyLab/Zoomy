"""SME L=2 — symmetric / antisymmetric decomposition of the
feedback contribution.

Building on the block decomposition of route B's pencil:

   M_x_eff(ε) = M_x_uu  +  ε · feedback,        feedback = − M_0_uw · M_0_aw⁻¹ · M_x_au

the principal-symbol matrix is

   A(ε)        = M_t_uu⁻¹ · M_x_eff(ε)
                = A_direct  +  ε · A_feedback

where A_feedback = M_t_uu⁻¹ · feedback.  Decompose A_feedback into
symmetric + antisymmetric pieces (w.r.t. some inner product).  We
explore two inner products:

  (a) Euclidean  (just the matrix transpose).
  (b) Energy-weighted: ⟨x, y⟩_W = xᵀ W y with W = diag(g, h, h/3, h/5)
      (the natural projection-of-energy weights).  Symmetric w.r.t. W
      means ``W · A = (W · A)ᵀ``.

For each, we compute eigenvalues of:

  • A_direct        — no feedback (the ε=0 baseline; fully hyperbolic).
  • A_direct + A_feedback        — full SME (ε=1).
  • A_direct + sym-only feedback — keep only the W-symmetric part.
  • A_direct + anti-only feedback — keep only the W-antisymmetric part.

A model that's symmetrisable (∃ S>0 such that S·M_x is symmetric) has
purely real eigenvalues.  If the bad behaviour lives in the
antisymmetric part, retaining only the symmetric part of the feedback
gives a structure-preserving regularisation.

Run:
    python tutorials/sme/sme_l2_feedback_decomposition.py
    python tutorials/sme/sme_l2_feedback_decomposition.py --no-plot
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


def _split_W_sym(A, W):
    """Decompose A = A_s + A_a where A_s is W-symmetric and A_a
    W-antisymmetric: W A_s = (W A_s)ᵀ.  Returns (A_s, A_a)."""
    WA = W @ A
    WA_sym = 0.5 * (WA + WA.T)
    WA_anti = 0.5 * (WA - WA.T)
    A_s = np.linalg.solve(W, WA_sym)
    A_a = np.linalg.solve(W, WA_anti)
    return A_s, A_a


def _max_im(eigs):
    eigs = eigs[np.isfinite(eigs)]
    return float(np.max(np.abs(np.imag(eigs))) if eigs.size else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--U1-max", type=float, default=3.0)
    parser.add_argument("--U2-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=2 feedback decomposition (W-symmetric vs antisymmetric)")
    print("=" * 70)

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
    feedback = -M_0_uw * M_0_aw.inv() * M_x_au

    h_bar, u0_bar, u1_bar, u2_bar = (base_b[h], base_b[u_0],
                                      base_b[u_1], base_b[u_2])
    syms = [g, h_bar, u0_bar, u1_bar, u2_bar]
    f_M_t = sp.lambdify(syms, M_t_uu, modules="numpy")
    f_M_x_uu = sp.lambdify(syms, M_x_uu, modules="numpy")
    f_feedback = sp.lambdify(syms, feedback, modules="numpy")

    u1_grid = np.linspace(-args.U1_max, args.U1_max, args.n_points)
    u2_grid = np.linspace(-args.U2_max, args.U2_max, args.n_points)

    cases = ["direct only (ε=0)",
             "direct + full feedback (ε=1, route A)",
             "direct + Eucl-sym(feedback) only",
             "direct + Eucl-antisym(feedback) only",
             "direct + W-sym(feedback) only",
             "direct + W-antisym(feedback) only"]
    nh_maps = {label: np.zeros((args.n_points, args.n_points)) for label in cases}

    for i, U1 in enumerate(u1_grid):
        for j, U2 in enumerate(u2_grid):
            params = (args.g, args.H, args.U0, float(U1), float(U2))
            Mt = np.asarray(f_M_t(*params), dtype=float)
            Mxu = np.asarray(f_M_x_uu(*params), dtype=float)
            F = np.asarray(f_feedback(*params), dtype=float)

            # Principal-symbol matrices: A = Mt^{-1} (M_x_uu + α·feedback')
            # where feedback' is full / sym / antisym.
            try:
                Mt_inv = np.linalg.inv(Mt)
            except np.linalg.LinAlgError:
                continue
            A_direct = Mt_inv @ Mxu
            A_full = A_direct + Mt_inv @ F

            # Euclidean sym/antisym of the principal-symbol contribution
            # of the feedback alone:
            A_fb = Mt_inv @ F
            A_fb_sym_eucl  = 0.5 * (A_fb + A_fb.T)
            A_fb_anti_eucl = 0.5 * (A_fb - A_fb.T)

            # Energy-weighted W = diag(g, h, h/3, h/5):
            W = np.diag([args.g, args.H, args.H / 3.0, args.H / 5.0])
            A_fb_sym_W, A_fb_anti_W = _split_W_sym(A_fb, W)

            cases_mats = {
                cases[0]: A_direct,
                cases[1]: A_full,
                cases[2]: A_direct + A_fb_sym_eucl,
                cases[3]: A_direct + A_fb_anti_eucl,
                cases[4]: A_direct + A_fb_sym_W,
                cases[5]: A_direct + A_fb_anti_W,
            }
            for label, A in cases_mats.items():
                eigs = np.linalg.eigvals(A)
                nh_maps[label][j, i] = _max_im(eigs)

    print("\nNon-hyperbolic fraction of the (u_1, u_2) grid (max |Im λ| > 1e-6):\n")
    for label in cases:
        frac = float(np.mean(nh_maps[label] > 1e-6))
        peak = float(np.max(nh_maps[label]))
        print(f"  {label:48s}  {frac:7.2%}   peak |Im λ| = {peak:.3f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        v_max = max(np.max(m) for m in nh_maps.values())
        v_max = max(v_max, 1e-12)
        extent = [-args.U1_max, args.U1_max, -args.U2_max, args.U2_max]

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, label in zip(axes.flat, cases):
            im = ax.imshow(nh_maps[label], extent=extent, origin="lower",
                           aspect="auto", cmap="magma", vmin=0, vmax=v_max)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel(r"$\bar u_1$")
            ax.set_ylabel(r"$\bar u_2$")
            plt.colorbar(im, ax=ax)
        fig.suptitle(
            f"SME L=2 feedback decomposition.   "
            f"max |Im λ| ≥ 1e-6 ⇒ non-hyperbolic.   "
            f"H={args.H}, U_0={args.U0}, g={args.g}", y=1.00)
        fig.tight_layout()
        out_path = "tutorials/sme/sme_l2_feedback_decomposition.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
