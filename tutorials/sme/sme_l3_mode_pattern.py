"""SME L=3 — does the "instability lives in the top two modes" pattern hold?

For SME L=2 the unstable eigenvector at corner spurs has |u_1|² + |u_2|²
≈ 100 % of its energy in the moment subspace; (h, u_0) barely
participate.  This script extends route A to L=3 (state h, u_0..u_3),
finds non-hyperbolic base states by sweeping (ū_2, ū_3) (the highest
two moments), and decomposes the unstable eigenvector to test whether
it's again dominated by the top two modes (u_2, u_3).

If the pattern holds, the cleanest regularisation strategy at any L
is: leave (h, u_0, …, u_{L-2}) alone, only modify the (u_{L-1}, u_L)
sub-block + its cross-coupling to the rest.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

from zoomy_core.symbolic import (
    leibniz_general, fundamental_theorem, polynomial_integrate as poly_int,
)
from zoomy_core.analysis import (
    PDESystem, linearise, extract_quasilinear_pencil,
)


# ---------------------------------------------------------------------------
# Symbols + ansatz for L=3 (M=3, four u-modes)
# ---------------------------------------------------------------------------

t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
z = sp.Symbol("z", real=True)
xi = sp.Symbol("xi", real=True)
g = sp.Symbol("g", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b

u_funcs = [sp.Function(f"u_{i}", real=True)(t, x) for i in range(4)]
u_0, u_1, u_2, u_3 = u_funcs

zeta = (z - b) / h
PHI = [
    sp.S.One,
    1 - 2 * zeta,
    1 - 6 * zeta + 6 * zeta**2,
    1 - 12 * zeta + 30 * zeta**2 - 20 * zeta**3,
]
u_ansatz = sum(u_funcs[i] * PHI[i] for i in range(4))


def project_z(integrand, j):
    integrand_xi = sp.expand((PHI[j] * integrand).xreplace({z: xi * h + b}).doit())
    return poly_int(integrand_xi * h, xi, 0, 1)


def build_route_a_L3():
    """Route A SME L=3: close w via depth-integrated continuity, project
    x-mom against φ_0..φ_3."""
    u_op = sp.Function("u", real=True)(t, x, z)
    w_op = sp.Function("w", real=True)(t, x, z)
    z_prime = sp.Symbol("z_prime", real=True)
    u_zp = u_op.xreplace({z: z_prime})
    w_zp = w_op.xreplace({z: z_prime})

    int_dx_u = leibniz_general(sp.Derivative(u_zp, x), z_prime, b, z)
    int_dz_w = fundamental_theorem(sp.Derivative(w_zp, z_prime), z_prime, b, z)
    cont_after_kbc = sp.expand(
        (int_dx_u + int_dz_w).xreplace(
            {w_op.xreplace({z: b}): u_op.xreplace({z: b}) * sp.Derivative(b, x)}
        )
    )
    w_solution = sp.solve(cont_after_kbc, w_op)[0]

    xmom = (sp.Derivative(u_op, t)
            + u_op * sp.Derivative(u_op, x)
            + w_op * sp.Derivative(u_op, z)
            + g * sp.Derivative(eta, x))
    xmom_with_w = sp.expand(xmom.xreplace({w_op: w_solution}).doit())

    u_func = u_op.func

    def _ansatz_at(e):
        z_val = e.args[2]
        return u_ansatz.xreplace({z: z_val})

    xmom_full = sp.expand(
        xmom_with_w.replace(
            lambda e: isinstance(e, sp.Function) and e.func == u_func,
            _ansatz_at,
        ).doit()
    )

    xmom_js = [sp.expand(project_z(xmom_full, j)) for j in (0, 1, 2, 3)]
    cont_h = sp.Derivative(h, t) + sp.Derivative(h * u_0, x).doit()

    eqs = [cont_h, *xmom_js]
    fields = [h, *u_funcs]
    return PDESystem(equations=eqs, fields=fields, time=t, space=[x],
                     parameters={g: g})


def _flat_bottom(system):
    repl = {sp.Derivative(b, x): 0, sp.Derivative(b, t): 0, b: 0}
    return PDESystem(
        equations=[sp.expand(e.xreplace(repl).doit()) for e in system.equations],
        fields=system.fields, time=system.time, space=system.space,
        parameters=system.parameters,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    parser.add_argument("--U-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=21)
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=3 — unstable mode decomposition (extends L=2 pattern check)")
    print("=" * 70)

    print("Building route A L=3 (slow — large symbolic expressions) …")
    sys_a = _flat_bottom(build_route_a_L3())
    print(f"  {len(sys_a.equations)} equations × {len(sys_a.fields)} fields")

    base_a = {f: sp.Symbol(f.func.__name__ + "_bar", real=True)
              for f in sys_a.fields}
    lin = linearise(sys_a, base_a)
    M_t, [M_x], _ = extract_quasilinear_pencil(lin)

    h_bar = base_a[h]
    u_bars = [base_a[u_funcs[i]] for i in range(4)]
    syms = [g, h_bar, *u_bars]
    f_M_t = sp.lambdify(syms, M_t, modules="numpy")
    f_M_x = sp.lambdify(syms, M_x, modules="numpy")

    print(f"\nSweeping ū_2, ū_3 ∈ [{-args.U_max}, {args.U_max}], "
          f"holding ū_0={args.U0}, ū_1=0, h={args.H}, g={args.g}")
    print(f"  grid: {args.n_points}×{args.n_points}\n")

    u_grid = np.linspace(-args.U_max, args.U_max, args.n_points)
    instability_records = []

    from numpy.linalg import eig
    for i, U2 in enumerate(u_grid):
        for j, U3 in enumerate(u_grid):
            params = (args.g, args.H, args.U0, 0.0, float(U2), float(U3))
            Mt = np.asarray(f_M_t(*params), dtype=complex)
            Mx = np.asarray(f_M_x(*params), dtype=complex)
            try:
                A = np.linalg.solve(Mt, Mx)
            except np.linalg.LinAlgError:
                continue
            eigs, vecs = eig(A)
            bad_idx = np.argmax(np.abs(np.imag(eigs)))
            im_max = float(np.abs(np.imag(eigs[bad_idx])))
            if im_max > 1e-6:
                v = vecs[:, bad_idx]
                v = v / np.max(np.abs(v))
                # |v_i|² fraction of mode energy.
                weights = np.abs(v) ** 2
                frac = weights / np.sum(weights)
                instability_records.append((U2, U3, im_max, frac))

    print(f"Found {len(instability_records)} non-hyperbolic samples.\n")
    if instability_records:
        # Average mode-energy distribution across all unstable samples.
        all_frac = np.array([r[3] for r in instability_records])
        mean_frac = all_frac.mean(axis=0)
        std_frac = all_frac.std(axis=0)
        labels = ["h", "u_0", "u_1", "u_2", "u_3"]
        print("Average mode-energy fraction in the unstable eigenvector "
              "(over all unstable samples):")
        for label, m, s in zip(labels, mean_frac, std_frac):
            bar = "█" * int(m * 50)
            print(f"  {label}: {m:5.1%} ± {s:5.2%}   {bar}")

        # Top-2 modes consistency.
        top2_frac = np.sum(np.sort(all_frac, axis=1)[:, -2:], axis=1).mean()
        print(f"\nTop-2 modes contribution (average):  {top2_frac:.1%}")
        # Are the top 2 always (u_2, u_3)?
        top2_indices = np.argsort(all_frac, axis=1)[:, -2:]
        same_pattern = np.sum(np.all(np.sort(top2_indices, axis=1) == [3, 4],
                                      axis=1))
        print(f"Of {len(instability_records)} unstable points, "
              f"{same_pattern} have (u_2, u_3) as the dominant two modes "
              f"({100*same_pattern/len(instability_records):.1f}%).")

        # Show a few extreme samples.
        instability_records.sort(key=lambda r: -r[2])
        print("\nTop 5 unstable samples:")
        print(f"  {'(ū_2, ū_3)':18s}  {'|Im λ|':10s}  mode-energy [h, u_0, u_1, u_2, u_3]")
        for U2, U3, im_max, frac in instability_records[:5]:
            frac_str = "[" + ", ".join(f"{f:.2f}" for f in frac) + "]"
            print(f"  ({U2:+.2f}, {U3:+.2f})    {im_max:.4f}   {frac_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
