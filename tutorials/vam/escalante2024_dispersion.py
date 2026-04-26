"""VAM dispersion relation — generalised to (M, N).

Builds the full nonlinear VAM PDE system from
``escalante2024_generic.py`` (so the model logic is shared), wraps it
in a ``zoomy_core.analysis.PDESystem``, linearises around the rest
state, and solves the plane-wave dispersion ``det M(ω, k) = 0``.

For (M, N) = (1, 2) the script prints the symbolic ``C² / (g H)``
expression and verifies it matches Escalante et al. 2024 eq (8):

    C²/(gH) = (1 + (kH)²/12) / (1 + 5(kH)²/12 + (kH)⁴/144).

For (M, N) = (2, 3) (or any other combo) the expression is computed
fresh — there's no published reference to compare against.

Ground rule: the analysis library
(``zoomy_core.analysis``) knows nothing about VAM; the VAM-specific
parts live entirely in this tutorial.
"""
from __future__ import annotations

import argparse
import sys

import sympy as sp

# Re-use VAM build helpers from the generic derivation so the model
# logic stays in one place.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from escalante2024_generic import (                              # noqa: E402
    setup_ansatz,
    derive_continuity_j,
    derive_xmom_j,
    derive_zmom_j,
    kbc_bottom_solve_wN,
    t, x, xi, h, b, g,
)

from zoomy_core.analysis import (                                 # noqa: E402
    PDESystem,
    linearise,
    plane_wave_dispersion,
)


def build_vam_pde_system(M, N):
    """Construct the nonlinear VAM PDESystem at degree (M, N).

    Fields: ``[h, u_0, …, u_M, w_0, …, w_N, p_0, …, p_N]`` —
    the M + 2N + 4 unknowns in the (t, x) plane.

    Equations (M + 2N + 4 of them, all = 0):
      1                : continuity j=0
      2..M+2           : x-momentum j=0..M
      M+3..M+N+2       : z-momentum j=0..N-1
      M+N+3..M+2N+2    : continuity j=1..N (constraints)
      M+2N+3           : w_N − KBC-bottom closure
      M+2N+4           : p_N − surface BC closure
    """
    phi, u_coeffs, w_coeffs, p_coeffs, u, w, p, omega = setup_ansatz(M, N)

    fields = [h] + list(u_coeffs) + list(w_coeffs) + list(p_coeffs)

    eqs = []
    # 1: continuity j=0
    eqs.append(derive_continuity_j(0, phi, u, omega))
    # 2..M+2: x-momentum j=0..M
    for j in range(M + 1):
        eqs.append(derive_xmom_j(j, phi, u, p, omega))
    # M+3..M+N+2: z-momentum j=0..N-1
    for j in range(N):
        eqs.append(derive_zmom_j(j, phi, u, w, p, omega))
    # M+N+3..M+2N+2: continuity j=1..N (constraints, kept in raw form)
    for j in range(1, N + 1):
        eqs.append(derive_continuity_j(j, phi, u, omega))
    # M+2N+3: w_N closure (algebraic)
    w_N_rhs = kbc_bottom_solve_wN(omega, w_coeffs, N)
    eqs.append(w_coeffs[N] - w_N_rhs)
    # M+2N+4: p_N closure (surface BC: p|_{xi=1} = 0)
    p_at_1 = sp.expand(p.subs(xi, 1))
    p_N_rhs = sp.solve(p_at_1, p_coeffs[N])[0]
    eqs.append(p_coeffs[N] - p_N_rhs)

    return PDESystem(
        equations=eqs,
        fields=fields,
        time=t,
        space=[x],
        parameters={g: sp.Symbol("g", positive=True)},
    ), u_coeffs, w_coeffs, p_coeffs


def rest_base_state(fields):
    """Rest state: h = H, all velocity / pressure moments = 0.

    Returns a dict {field: base_value} for use by ``linearise``.
    """
    H = sp.Symbol("H", positive=True)
    base = {}
    for f in fields:
        head_name = f.func.__name__ if hasattr(f.func, "__name__") else str(f.func)
        if head_name == "h":
            base[f] = H
        else:
            base[f] = sp.S.Zero
    return base, H


def reference_C2_over_gH_paper(k, H):
    """Paper eq (8) of Escalante et al. 2024 — only valid at (M, N) = (1, 2):

        C² / (g H) = (1 + (kH)²/12) / (1 + 5(kH)²/12 + (kH)⁴/144).
    """
    kH = k * H
    num = 1 + kH**2 / 12
    den = 1 + 5 * kH**2 / 12 + kH**4 / 144
    return num / den


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if (M,N)=(1,2) doesn't match paper")
    args = parser.parse_args()

    M, N = args.M, args.N
    print(f"=== VAM dispersion — generic (M, N) = ({M}, {N}) ===\n")

    sys_vam, u_coeffs, w_coeffs, p_coeffs = build_vam_pde_system(M, N)
    print(f"Built nonlinear PDE system: {sys_vam}")

    # b → -H, so ∂_x b = 0 and η = h - H is the surface displacement.
    H = sp.Symbol("H", positive=True)
    b_sub = {b: -H}
    sys_vam = sys_vam.with_substitutions(b_sub)

    # Linearise around rest state.
    base, H = rest_base_state(sys_vam.fields)
    sys_lin = linearise(sys_vam, base)
    print(f"Linearised: {sys_lin.n_equations()} equations × "
          f"{sys_lin.n_fields()} fields\n")

    # Plane-wave dispersion.
    print("Solving det M(ω, k) = 0 for ω ...  (may take a few seconds)")
    k_sym = sp.Symbol("k", real=True, nonzero=True)
    omega_sym = sp.Symbol("omega", real=True)
    disp = plane_wave_dispersion(sys_lin, k=k_sym, omega=omega_sym,
                                 simplify=True, factor_in_target=True)
    print(f"  determinant has degree {sp.Poly(disp['determinant'], omega_sym).total_degree()} in ω")
    print(f"  {len(disp['solutions'])} ω solutions\n")

    # Phase velocities ω/k.
    pvs = [sp.simplify(s / k_sym) for s in disp['solutions']]
    # Drop trivial ω=0 roots (constraints → static modes); keep the
    # propagating wave branches.
    propagating = [pv for pv in pvs if sp.simplify(pv) != 0]
    # Distinct values up to sign.
    abs_pvs = sorted({sp.simplify(pv**2) for pv in propagating},
                     key=lambda e: str(e))
    print("Distinct C² (one per propagating mode pair):")
    for c2 in abs_pvs:
        c2_over_gH = sp.simplify(c2 / (sp.Symbol("g", positive=True) * H))
        print(f"  C² / (g H) = {sp.simplify(c2_over_gH)}")

    # If (M, N) = (1, 2), compare to the paper.
    if (M, N) == (1, 2):
        print()
        print("--- Paper comparison (Escalante 2024 eq 8) ---")
        ref = reference_C2_over_gH_paper(k_sym, H)
        ref_s = sp.simplify(ref)
        print(f"  reference  C²/(gH) = {ref_s}")
        match_found = False
        for c2 in abs_pvs:
            c2_over_gH = sp.simplify(c2 / (sp.Symbol("g", positive=True) * H))
            diff = sp.simplify(sp.cancel(c2_over_gH - ref_s))
            if diff == 0:
                print(f"  ✓ MATCH on C²/(gH) = {sp.simplify(c2_over_gH)}")
                match_found = True
                break
        if not match_found:
            print("  ✗ No matching mode found.  Listing all candidate C²/(gH):")
            for c2 in abs_pvs:
                c2_over_gH = sp.simplify(c2 / (sp.Symbol("g", positive=True) * H))
                print(f"     {c2_over_gH}  (diff = "
                      f"{sp.simplify(sp.cancel(c2_over_gH - ref_s))})")
            if args.strict:
                sys.exit(1)


if __name__ == "__main__":
    main()
