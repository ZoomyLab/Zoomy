"""SME dispersion relation at arbitrary level.

Sanity check for the unified analysis library on the SME family —
linearises ``build_sme_pde_system(level)`` around the rest state
(h = H, all u_i = 0) and solves det M(ω, k) = 0 for ω(k).

  Level 0 (SWE):    expected C² = g H.
  Level ≥ 1:        expected C² = g H (linearisation around rest
                    state — higher moments vanish in the linearisation
                    so the lowest mode is just SWE).

Compares each derived ``C²`` against ``gH`` and reports MATCH /
MISMATCH.  This is a thin tutorial showing how the same analysis
pipeline used for VAM applies to SME without code changes.
"""
from __future__ import annotations

import argparse
import sys

import sympy as sp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sme_builder import build_sme_pde_system, g                # noqa: E402

from zoomy_core.analysis import (                                # noqa: E402
    linearise,
    plane_wave_dispersion,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    L = args.level
    print(f"=== SME dispersion — level {L} ===\n")

    sys_sme, h, u_coeffs = build_sme_pde_system(L)
    H = sp.Symbol("H", positive=True)
    base = {h: H}
    for ui in u_coeffs:
        base[ui] = sp.S.Zero
    sys_lin = linearise(sys_sme, base)
    print(f"Linearised: {sys_lin.n_equations()} equations × "
          f"{sys_lin.n_fields()} fields\n")

    k_sym = sp.Symbol("k", real=True, nonzero=True)
    omega_sym = sp.Symbol("omega", real=True)
    print("Solving det M(ω, k) = 0 ...")
    disp = plane_wave_dispersion(sys_lin, k=k_sym, omega=omega_sym,
                                 simplify=True, factor_in_target=True)
    pvs = [sp.simplify(s / k_sym) for s in disp["solutions"]]
    propagating = [pv for pv in pvs if sp.simplify(pv) != 0]
    abs_pvs = sorted({sp.simplify(pv**2) for pv in propagating},
                     key=lambda e: str(e))

    expected = g * H
    print(f"Expected C² (rest state):  C² = g H  =  {expected}\n")
    print("Distinct C² (propagating mode pairs):")
    n_match = 0
    for c2 in abs_pvs:
        c2_simpl = sp.simplify(c2)
        diff = sp.simplify(c2_simpl - expected)
        ok = (diff == 0)
        mark = "✓ MATCH" if ok else "  "
        print(f"  {mark}  C² = {c2_simpl}")
        if ok:
            n_match += 1

    if n_match == 0:
        print("\n✗ No matching propagating mode found.")
        if args.strict:
            sys.exit(1)
    else:
        print(f"\n✓ {n_match} propagating mode(s) match C² = g H.")


if __name__ == "__main__":
    main()
