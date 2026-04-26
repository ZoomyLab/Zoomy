"""Sample-based hyperbolicity test for VAM at arbitrary (M, N).

Uses the **same** analysis pipeline as ``sme_hyperbolicity.py`` —
the only difference is the model builder and the resulting
generalised-eigenvalue pencil shape.  In particular, the VAM pencil
``M_t`` is **singular** (constraints contribute zero rows in the
∂_t coefficients), so we rely on ``scipy.linalg.eig(A, B)`` which
returns infinite eigenvalues for the constraint modes.  The
``drop_infinite=True`` flag in
``zoomy_core.analysis.is_hyperbolic_at`` filters those out before
checking real-ness.

The hyperbolic-mode count for VAM (M, N) at finite k is the number
of ω solutions to ``det M(ω, k) = 0`` — for (1, 2) the dispersion
script gave 3 solutions (so a pair of propagating modes + one
trivial); we expect the high-k principal symbol to give the same
number of finite generalised eigenvalues.

Demonstrates: the unified analysis library handles VAM (DAE
structure, constraint rows) and SME (pure hyperbolic) with the
identical entry points.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from escalante2024_dispersion import build_vam_pde_system        # noqa: E402
from escalante2024_generic import t, x, b, g                     # noqa: E402

from zoomy_core.analysis import (                                 # noqa: E402
    linearise,
    extract_quasilinear_pencil,
    sample_hyperbolicity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--U-range", type=float, default=1.0)
    parser.add_argument("--P-range", type=float, default=1.0)
    parser.add_argument("--H-min", type=float, default=0.1)
    parser.add_argument("--H-max", type=float, default=2.0)
    parser.add_argument("--g", type=float, default=9.81)
    args = parser.parse_args()

    print(f"=== VAM hyperbolicity sampling — (M, N) = ({args.M}, {args.N}) ===\n")

    sys_vam, u_coeffs, w_coeffs, p_coeffs = build_vam_pde_system(args.M, args.N)

    # b → constant for the hyperbolicity test (rest state).  Use H so
    # ∂_x b = 0 trivially; the value cancels.
    H_const = sp.Symbol("H_const", positive=True)
    sys_vam = sys_vam.with_substitutions({b: -H_const})

    # Symbolic constant base state (one symbol per field).
    base = {}
    consts = {}
    for f in sys_vam.fields:
        head = f.func.__name__ if hasattr(f.func, "__name__") else str(f.func)
        sym = sp.Symbol(f"{head}_bar", real=True)
        base[f] = sym
        consts[head] = sym
    sys_lin = linearise(sys_vam, base)

    M_t, M_xa, M_0 = extract_quasilinear_pencil(sys_lin)
    M_x = M_xa[0]
    print(f"Built VAM(M={args.M}, N={args.N}) PDESystem: "
          f"{sys_vam.n_equations()} equations × {sys_vam.n_fields()} fields")
    print(f"Pencil (M_x, M_t):                         {M_x.shape}")
    print(f"M_t rank (≤ n_eq): {M_t.rank()} / {M_t.shape[0]}  "
          f"(rank-deficiency = #constraints)\n")

    # Substitute g.
    g_val = sp.Float(args.g)
    M_x = M_x.subs(g, g_val)
    M_t = M_t.subs(g, g_val)

    # Sampling ranges.
    h_const = consts["h"]
    parameter_ranges = {h_const: (args.H_min, args.H_max)}
    for name, sym in consts.items():
        if name == "h":
            continue
        rng = (-args.U_range, args.U_range) if name.startswith(("u", "w")) else (-args.P_range, args.P_range)
        parameter_ranges[sym] = rng

    rng = np.random.default_rng(args.seed)
    report = sample_hyperbolicity(
        M_x, M_t, parameter_ranges,
        n_samples=args.n_samples,
        rng=rng,
        constraint_filter=lambda s: s[h_const] > 0,
        drop_infinite=True,
    )
    print(report.summary())

    if report.samples:
        all_eigs = np.concatenate([s.eigenvalues for s in report.samples])
        finite_real = all_eigs[np.isfinite(all_eigs)
                               & (np.abs(np.imag(all_eigs)) < 1e-9)].real
        if len(finite_real) > 0:
            print(f"\nReal-eigenvalue range:  [{finite_real.min():.3f}, "
                  f"{finite_real.max():.3f}]")
            print(f"Average #finite eigenvalues per sample: "
                  f"{len(finite_real) / len(report.samples):.2f}")


if __name__ == "__main__":
    main()
