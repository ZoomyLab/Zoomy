"""Sample-based hyperbolicity test for SME at arbitrary level.

Uses the new ``zoomy_core.analysis`` library:

  1. Build SME(level=L) PDESystem.
  2. Linearise around a symbolic constant base state
     (H, U_0, …, U_L).
  3. Extract the pencil (M_t, M_x).
  4. Sample numerical (H, U_0, …, U_L) over user-specified ranges,
     evaluate generalised eigenvalues of (M_x, M_t), and report
     the fraction of states for which all eigenvalues are real.

Hyperbolicity for SME:

  Level 0 (= SWE): always hyperbolic for h > 0.  Eigenvalues
                   u_0 ± sqrt(g H).
  Level 1:         eigenvalues u_0, u_0 ± sqrt(g H + u_1²).  Always
                   real for h > 0 (Kowalski-Torrilhon 2019).
  Level 2:         non-trivially non-hyperbolic in some regions
                   (K&T 2019 discuss this); the sampler should
                   find them.

Ground rule: nothing model-specific lives in the analysis library;
this script is the only SME-specific piece.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

# Re-use the SME builder so the model logic stays in one place.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sme_builder import build_sme_pde_system, t, x, g     # noqa: E402

from zoomy_core.analysis import (                            # noqa: E402
    linearise,
    extract_quasilinear_pencil,
    sample_hyperbolicity,
)


def _symbolic_constant_base_state(fields):
    """Map each field f(t, x) to a fresh real symbol named after f's head.

    Returns ``(base_state_dict, constants_list)``.
    """
    base = {}
    consts = []
    for f in fields:
        name = f.func.__name__ if hasattr(f.func, "__name__") else str(f.func)
        const_name = f"{name}_bar"
        sym = sp.Symbol(const_name, real=True)
        base[f] = sym
        consts.append(sym)
    return base, consts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", type=int, default=2,
                        help="SME Galerkin level (default 2)")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--U-range", type=float, default=2.0,
                        help="velocity moment range (-U..+U) for u_0..u_L")
    parser.add_argument("--H-min", type=float, default=0.05)
    parser.add_argument("--H-max", type=float, default=5.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--print-every", type=int, default=0,
                        help="if > 0, print eigenvalues for every Nth sample")
    args = parser.parse_args()

    L = args.level
    print(f"=== SME hyperbolicity sampling — level {L} ===\n")

    # --- 1. Build PDE system + linearise around constant base state ---
    sys_sme, h_field, u_fields = build_sme_pde_system(L)
    base, consts = _symbolic_constant_base_state(sys_sme.fields)
    sys_lin = linearise(sys_sme, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    M_x = M_xa[0]                                            # 1D — single axis

    print(f"Built SME(level={L}) PDESystem:  "
          f"{sys_sme.n_equations()} equations × {sys_sme.n_fields()} fields")
    print(f"Linearised pencil (M_x, M_t):    {M_x.shape}\n")

    # --- 2. Substitute g (it's a positive constant; keep H, U_i symbolic) ---
    g_val = sp.Float(args.g)
    M_x = M_x.subs(g, g_val)
    M_t = M_t.subs(g, g_val)

    # --- 3. Sampling ranges ---
    H_const = consts[0]                                      # base h̄
    U_consts = consts[1:]                                    # base ū_0..ū_L
    parameter_ranges = {H_const: (args.H_min, args.H_max)}
    for u_const in U_consts:
        parameter_ranges[u_const] = (-args.U_range, args.U_range)

    # --- 4. Sample ---
    rng = np.random.default_rng(args.seed)
    report = sample_hyperbolicity(
        M_x, M_t, parameter_ranges,
        n_samples=args.n_samples,
        rng=rng,
        constraint_filter=lambda s: s[H_const] > 0,
    )
    print(report.summary())
    print()

    # Per-mode statistics: distribution of eigenvalues over samples.
    all_eigs = np.concatenate([s.eigenvalues for s in report.samples])
    finite = all_eigs[np.isfinite(all_eigs)]
    real_finite = finite[np.abs(np.imag(finite)) < 1e-9].real
    if len(real_finite) > 0:
        print(f"Real-eigenvalue range:  [{real_finite.min():.3f}, "
              f"{real_finite.max():.3f}]")
        print(f"Real-eigenvalue median: {np.median(real_finite):.3f}")
    if args.print_every > 0:
        for i, s in enumerate(report.samples):
            if i % args.print_every == 0:
                vals = ", ".join(f"{c}={v:.3f}" for c, v in s.sample.items())
                print(f"sample[{i}] {vals}  →  eigs = {s.eigenvalues}")


if __name__ == "__main__":
    main()
