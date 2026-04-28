"""SME L=3 — does the single-entry regularization pattern from L=2 hold?

At L=2, zeroing A[1, 2] (= 2ū_1/3 at base) suffices for full
hyperbolicity, with Frobenius distance ~1.0.  This script runs the
analogous single-entry search at L=3 and reports which entries fully
regularize the system over a (ū_2, ū_3) grid.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from sme_l3_mode_pattern import (              # noqa: E402
    build_route_a_L3, _flat_bottom, h, u_0, u_1, u_2, u_3, g, u_funcs
)
from zoomy_core.analysis import linearise, extract_quasilinear_pencil


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--U-max", type=float, default=2.5)
    p.add_argument("--n-points", type=int, default=15)
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--H", type=float, default=1.0)
    p.add_argument("--U0", type=float, default=0.0)
    args = p.parse_args()

    print("=" * 70)
    print("SME L=3 — single-entry regularization grid-search")
    print("=" * 70)
    sa = _flat_bottom(build_route_a_L3())
    base = {f: sp.Symbol(f.func.__name__ + "_bar", real=True) for f in sa.fields}
    lin = linearise(sa, base)
    M_t, [M_x], _ = extract_quasilinear_pencil(lin)

    syms = [g, base[h]] + [base[u_funcs[i]] for i in range(4)]
    f_M_t = sp.lambdify(syms, M_t, "numpy")
    f_M_x = sp.lambdify(syms, M_x, "numpy")

    # Sweep (ū_2, ū_3) at ū_0 = U0, ū_1 = 0
    u_grid = np.linspace(-args.U_max, args.U_max, args.n_points)
    n_pts = args.n_points

    # Pre-compute A_full at each grid point.
    A_grids = np.zeros((n_pts, n_pts, 5, 5), dtype=complex)
    for i, U2 in enumerate(u_grid):
        for j, U3 in enumerate(u_grid):
            params = (args.g, args.H, args.U0, 0.0, float(U2), float(U3))
            Mt = np.asarray(f_M_t(*params), dtype=float)
            Mx = np.asarray(f_M_x(*params), dtype=float)
            try:
                A_grids[i, j] = np.linalg.solve(Mt, Mx)
            except np.linalg.LinAlgError:
                A_grids[i, j] = np.nan

    print(f"\n{n_pts*n_pts} grid points; field order: [h, u_0, u_1, u_2, u_3].")
    print("\nSingle-entry zeroings (off-diagonal only).  "
          "Reporting (entry, fraction non-hyp, mean ||·||_F):\n")

    field_names = ["h", "u_0", "u_1", "u_2", "u_3"]
    results = []
    for r in range(5):
        for c in range(5):
            if r == c:
                continue
            nh_frac = 0
            fro_sum = 0.0
            n_valid = 0
            for i in range(n_pts):
                for j in range(n_pts):
                    A = A_grids[i, j].copy()
                    if np.any(np.isnan(A)):
                        continue
                    A_orig = A.copy()
                    A[r, c] = 0
                    eigs = np.linalg.eigvals(A)
                    nh = float(np.max(np.abs(np.imag(eigs))))
                    nh_frac += int(nh > 1e-6)
                    fro_sum += np.linalg.norm(A - A_orig, "fro")
                    n_valid += 1
            results.append((r, c, nh_frac / n_valid, fro_sum / n_valid))

    # Sort: prefer 0% non-hyp, then smallest Frobenius.
    results.sort(key=lambda x: (x[2], x[3]))
    print(f"{'(r, c) → entry':30s}  non-hyp   ⟨‖·‖_F⟩")
    print("-" * 60)
    for r, c, nh, fro in results[:15]:
        label = f"A[{r}, {c}] = 0  ({field_names[r]}-eq <- ∂_x{field_names[c]})"
        print(f"{label:30s}   {nh:6.2%}    {fro:.3f}")

    # The original (unregularized) baseline
    nh0 = 0; n_valid0 = 0
    for i in range(n_pts):
        for j in range(n_pts):
            A = A_grids[i, j]
            if np.any(np.isnan(A)):
                continue
            eigs = np.linalg.eigvals(A)
            nh0 += int(np.max(np.abs(np.imag(eigs))) > 1e-6)
            n_valid0 += 1
    print(f"\nOriginal SME L=3:  {nh0/n_valid0:.2%} non-hyperbolic on this grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
