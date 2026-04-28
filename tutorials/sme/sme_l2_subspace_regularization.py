"""SME L=2 — minimum-norm Hermitian projection on the (u_1, u_2) sub-block.

The eigenmode analysis from sme_l2_regularization_findings.md shows
that the SME L=2 loss-of-hyperbolicity is carried by an eigenvector
that lives essentially in the (u_1, u_2) 2-D subspace (h ~ 4 %,
u_0 ~ 1 %).  This script tests three targeted regularisations:

  (1) symmetrise A's 2×2 sub-block ``A[2:4, 2:4]`` w.r.t. Euclidean
      inner product;
  (2) symmetrise A's 2×2 sub-block w.r.t. energy-weighted inner
      product W = diag(g, h, h/3, h/5);
  (3) zero the off-diagonal cross-couplings between the (h, u_0)
      shallow-water block and the (u_1, u_2) moment block.

The principal-symbol matrix ``A := M_t_uu⁻¹ · M_x_eff`` is split
block-wise as

      [ A_11  A_12 ]    A_11 = (h, u_0) ↔ (h, u_0)
   A =[            ]
      [ A_21  A_22 ]    A_22 = (u_1, u_2) ↔ (u_1, u_2)

(1) replaces A_22 with (A_22 + A_22ᵀ)/2.  (2) replaces A_22 with the
W-symmetric part: W·A_22 → ½(W·A_22 + (W·A_22)ᵀ).  (3) zeros A_12 and
A_21.

We compare each with the direct-only baseline (which is fully
hyperbolic but discards the feedback entirely).
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
    return float(np.max(np.abs(np.imag(eigs))) if eigs.size else 0)


def _herm_subblock_eucl(A):
    """Symmetrise A[2:4, 2:4] in Euclidean inner product."""
    A = A.copy()
    A[2:4, 2:4] = 0.5 * (A[2:4, 2:4] + A[2:4, 2:4].T)
    return A


def _herm_subblock_W(A, W22):
    """Symmetrise A[2:4, 2:4] in W-inner product on the moment subspace.
    W22 is the 2x2 weight (e.g., diag(h/3, h/5))."""
    A = A.copy()
    A22 = A[2:4, 2:4]
    WA22 = W22 @ A22
    WA22_sym = 0.5 * (WA22 + WA22.T)
    A22_new = np.linalg.solve(W22, WA22_sym)
    A[2:4, 2:4] = A22_new
    return A


def _zero_cross_coupling(A):
    """Zero the (1,2)-block coupling: A[0:2, 2:4] = 0 and A[2:4, 0:2] = 0."""
    A = A.copy()
    A[0:2, 2:4] = 0
    A[2:4, 0:2] = 0
    return A


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
    print("SME L=2 — targeted (u_1, u_2) sub-block regularisations")
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
    M_x_eff = M_x_uu + feedback

    h_bar, u0_bar, u1_bar, u2_bar = (base_b[h], base_b[u_0],
                                      base_b[u_1], base_b[u_2])
    syms = [g, h_bar, u0_bar, u1_bar, u2_bar]
    f_M_t = sp.lambdify(syms, M_t_uu, modules="numpy")
    f_M_x_eff = sp.lambdify(syms, M_x_eff, modules="numpy")
    f_M_x_uu = sp.lambdify(syms, M_x_uu, modules="numpy")

    u1_grid = np.linspace(-args.U1_max, args.U1_max, args.n_points)
    u2_grid = np.linspace(-args.U2_max, args.U2_max, args.n_points)

    cases = ["full SME (route A)",
             "direct only (no feedback)",
             "Eucl-Hermitise A[2:4, 2:4]",
             "W-Hermitise A[2:4, 2:4]   (W22 = diag(h/3, h/5))",
             "zero the (h,u_0) ↔ (u_1,u_2) cross-coupling"]
    nh_maps = {label: np.zeros((args.n_points, args.n_points)) for label in cases}

    W22 = np.diag([args.H / 3.0, args.H / 5.0])
    for i, U1 in enumerate(u1_grid):
        for j, U2 in enumerate(u2_grid):
            params = (args.g, args.H, args.U0, float(U1), float(U2))
            Mt = np.asarray(f_M_t(*params), dtype=float)
            Mxe = np.asarray(f_M_x_eff(*params), dtype=float)
            Mxu = np.asarray(f_M_x_uu(*params), dtype=float)
            try:
                Mt_inv = np.linalg.inv(Mt)
            except np.linalg.LinAlgError:
                continue
            A_full = Mt_inv @ Mxe
            A_direct = Mt_inv @ Mxu
            A_eucl = _herm_subblock_eucl(A_full)
            A_W = _herm_subblock_W(A_full, W22)
            A_no_cross = _zero_cross_coupling(A_full)

            mats = {
                cases[0]: A_full,
                cases[1]: A_direct,
                cases[2]: A_eucl,
                cases[3]: A_W,
                cases[4]: A_no_cross,
            }
            for label, A in mats.items():
                nh_maps[label][j, i] = _max_im(np.linalg.eigvals(A))

    print("\nNon-hyperbolic fraction of (u_1, u_2) grid (max |Im λ| > 1e-6):\n")
    for label in cases:
        frac = float(np.mean(nh_maps[label] > 1e-6))
        peak = float(np.max(nh_maps[label]))
        print(f"  {label:55s}  {frac:7.2%}   peak |Im λ| = {peak:.3f}")

    # Quantify "how different from route A": Frobenius distance per sample.
    print("\nMean Frobenius distance ||A_reg − A_full||_F over the grid:")
    avg_dist = {}
    for i, U1 in enumerate(u1_grid):
        for j, U2 in enumerate(u2_grid):
            params = (args.g, args.H, args.U0, float(U1), float(U2))
            Mt = np.asarray(f_M_t(*params), dtype=float)
            Mxe = np.asarray(f_M_x_eff(*params), dtype=float)
            Mxu = np.asarray(f_M_x_uu(*params), dtype=float)
            try:
                Mt_inv = np.linalg.inv(Mt)
            except np.linalg.LinAlgError:
                continue
            A_full = Mt_inv @ Mxe
            for label, fn in [(cases[1], lambda A: Mt_inv @ Mxu),
                              (cases[2], _herm_subblock_eucl),
                              (cases[3], lambda A: _herm_subblock_W(A, W22)),
                              (cases[4], _zero_cross_coupling)]:
                d = np.linalg.norm(fn(A_full) - A_full, "fro")
                avg_dist.setdefault(label, []).append(d)
    for label in cases[1:]:
        d = np.mean(avg_dist[label])
        print(f"  {label:55s}  ⟨||·||_F⟩ = {d:.3f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        v_max = max(np.max(m) for m in nh_maps.values())
        v_max = max(v_max, 1e-12)
        extent = [-args.U1_max, args.U1_max, -args.U2_max, args.U2_max]
        fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4))
        for ax, label in zip(axes, cases):
            im = ax.imshow(nh_maps[label], extent=extent, origin="lower",
                           aspect="auto", cmap="magma", vmin=0, vmax=v_max)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel(r"$\bar u_1$")
            ax.set_ylabel(r"$\bar u_2$")
            plt.colorbar(im, ax=ax)
        fig.suptitle(
            f"SME L=2 targeted sub-block regularisations.   "
            f"H={args.H}, U_0={args.U0}, g={args.g}", y=1.00)
        fig.tight_layout()
        out_path = "tutorials/sme/sme_l2_subspace_regularization.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
