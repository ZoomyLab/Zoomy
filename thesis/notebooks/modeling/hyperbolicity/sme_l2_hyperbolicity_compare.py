"""Compare SME L=2 hyperbolicity: route A (close w via continuity) vs
route B (keep w explicit, ``N_w = M+1 = 3``).

Question: do the two routes give the same eigenvalue traces, or does
the DAE structure of route B introduce numerical fragility?

Analytically: route A and route B are equivalent.  Route B's algebraic
constraints (cont projections + KBCs) determine the four ``w_i`` from
the four evolution variables; eliminating them reduces route B to
route A exactly.

Numerically: route A is a clean 4×4 quasilinear pencil — its
hyperbolicity is read off as eigenvalues of ``(M_x, M_t)``.  Route B
is an 8×8 PDE-DAE.  Its hyperbolicity is naturally posed as
**plane-wave dispersion**: at fixed wavenumber k, solve ``det M(ω, k)
= 0`` for ω, where ``M(ω, k) = -iω M_t + ik M_x + M_0`` (the M_0 term
encodes the algebraic w-couplings, which the principal-symbol pencil
``(M_x, M_t)`` alone cannot resolve).

This script:
  1. Builds both PDE systems in physical-z, basic primitives only.
  2. Linearises both around a homogeneous base state.
  3. Sweeps ``u_1`` (with ``u_2`` fixed), at fixed k=1, and computes
     ω(k=1) by solving the full plane-wave det polynomial for each
     base state.
  4. Plots Re/Im(ω/k) traces for both routes overlaid + the per-mode
     |ω_A − ω_B| difference.

Run:
    python tutorials/sme/sme_l2_hyperbolicity_compare.py
    python tutorials/sme/sme_l2_hyperbolicity_compare.py --no-plot   # CI
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
    PDESystem, linearise, plane_wave_matrix,
)


# ---------------------------------------------------------------------------
# Common symbols + ansatz
# ---------------------------------------------------------------------------

t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
z = sp.Symbol("z", real=True)
xi = sp.Symbol("xi", real=True)
g = sp.Symbol("g", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b

u_0 = sp.Function("u_0", real=True)(t, x)
u_1 = sp.Function("u_1", real=True)(t, x)
u_2 = sp.Function("u_2", real=True)(t, x)

# Shifted Legendre basis, paper convention φ_i(0) = 1.
zeta = (z - b) / h
PHI = [
    sp.S.One,
    1 - 2 * zeta,
    1 - 6 * zeta + 6 * zeta**2,
    1 - 12 * zeta + 30 * zeta**2 - 20 * zeta**3,
]

u_ansatz = u_0 * PHI[0] + u_1 * PHI[1] + u_2 * PHI[2]


def project_z(integrand, j):
    integrand_xi = sp.expand((PHI[j] * integrand).xreplace({z: xi * h + b}).doit())
    return poly_int(integrand_xi * h, xi, 0, 1)


# ---------------------------------------------------------------------------
# Route A — close w via depth-integrated continuity
# ---------------------------------------------------------------------------

def build_route_a():
    """Returns a PDESystem in primitive variables (h, u_0, u_1, u_2)."""
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

    xmom_js = [sp.expand(project_z(xmom_full, j)) for j in (0, 1, 2)]
    cont_h = sp.Derivative(h, t) + sp.Derivative(h * u_0, x).doit()

    eqs = [cont_h, xmom_js[0], xmom_js[1], xmom_js[2]]
    fields = [h, u_0, u_1, u_2]
    return PDESystem(equations=eqs, fields=fields, time=t, space=[x],
                     parameters={g: g})


# ---------------------------------------------------------------------------
# Route B — keep w explicit, N_w = 3
# ---------------------------------------------------------------------------

def build_route_b():
    """Returns a PDESystem in (h, u_0, u_1, u_2, w_0, w_1, w_2, w_3)."""
    w_0 = sp.Function("w_0", real=True)(t, x)
    w_1 = sp.Function("w_1", real=True)(t, x)
    w_2 = sp.Function("w_2", real=True)(t, x)
    w_3 = sp.Function("w_3", real=True)(t, x)
    w_ansatz = (w_0 * PHI[0] + w_1 * PHI[1]
                + w_2 * PHI[2] + w_3 * PHI[3])

    xmom_full = (sp.Derivative(u_ansatz, t).doit()
                 + u_ansatz * sp.Derivative(u_ansatz, x).doit()
                 + w_ansatz * sp.Derivative(u_ansatz, z).doit()
                 + g * sp.Derivative(eta, x).doit())
    cont_full = sp.Derivative(u_ansatz, x).doit() + sp.Derivative(w_ansatz, z).doit()

    xmom_js = [sp.expand(project_z(xmom_full, j)) for j in (0, 1, 2)]
    cont_js = [sp.expand(project_z(cont_full, j)) for j in (0, 1, 2)]   # j=3 trivial

    kbc_bot = sp.expand(
        w_ansatz.xreplace({z: b}) - u_ansatz.xreplace({z: b}) * sp.Derivative(b, x)
    )
    kbc_top = sp.expand(
        w_ansatz.xreplace({z: eta})
        - sp.Derivative(eta, t).doit()
        - u_ansatz.xreplace({z: eta}) * sp.Derivative(eta, x).doit()
    )

    # 8 equations, 8 fields.  Evolution-like: x-mom_j=0..2, KBC_top.
    # Algebraic: cont_j=0..2, KBC_bot.
    eqs = [
        xmom_js[0], xmom_js[1], xmom_js[2], kbc_top,
        cont_js[0], cont_js[1], cont_js[2], kbc_bot,
    ]
    fields = [h, u_0, u_1, u_2, w_0, w_1, w_2, w_3]
    return PDESystem(equations=eqs, fields=fields, time=t, space=[x],
                     parameters={g: g})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_bottom(system: PDESystem) -> PDESystem:
    repl = {sp.Derivative(b, x): 0, sp.Derivative(b, t): 0, b: 0}
    return PDESystem(
        equations=[sp.expand(e.xreplace(repl).doit()) for e in system.equations],
        fields=system.fields, time=system.time, space=system.space,
        parameters=system.parameters,
    )


def _build_planewave(system: PDESystem):
    """Linearise + plane-wave ansatz; return (M(ω, k, base...), base_dict)."""
    base = {f: sp.Symbol(f.func.__name__ + "_bar", real=True)
            for f in system.fields}
    sys_lin = linearise(system, base)
    k_sym = sp.Symbol("k", real=True)
    omega_sym = sp.Symbol("omega", real=True)
    M, _ = plane_wave_matrix(sys_lin, k=k_sym, omega=omega_sym)
    return M, base, k_sym, omega_sym


def _make_M_eval(M, omega_sym, all_param_syms):
    """Compile M(ω, params) → numpy.ndarray.  Returns a callable
    ``f(omega_val, *param_vals)`` that returns a complex (n×n) array.
    Avoids sympy.det entirely — caller takes numerical det."""
    fn = sp.lambdify([omega_sym] + list(all_param_syms), M, modules="numpy")
    def evaluator(omega_val, params):
        return np.asarray(fn(omega_val, *params), dtype=complex)
    return evaluator


def _disp_speeds(M_eval, params, k_val, *, n_evolution=4, deg_max=10):
    """Sample det(M(ω_j, params)) at ``deg_max+1`` ω values, fit a
    polynomial (the det is polynomial of degree ≤ matrix size in ω),
    take roots."""
    omegas = np.linspace(-2.0, 2.0, deg_max + 1) + 0.7j * np.arange(deg_max + 1)
    dets = np.array([np.linalg.det(M_eval(om, params)) for om in omegas])
    coeffs = np.polyfit(omegas, dets, deg_max)
    # Trim leading zeros (true polynomial degree may be < deg_max).
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-12 * max(abs(c) for c in coeffs):
        coeffs = coeffs[1:]
    if len(coeffs) <= 1:
        return np.full(n_evolution, np.nan + 1j * np.nan)
    roots = np.roots(coeffs)
    speeds = roots / k_val
    finite = speeds[np.isfinite(speeds.real) & np.isfinite(speeds.imag)]
    if len(finite) > n_evolution:
        non_zero = finite[np.abs(finite) > 1e-10]
        if len(non_zero) >= n_evolution:
            finite = non_zero[np.argsort(np.abs(non_zero))][:n_evolution]
        else:
            finite = finite[np.argsort(np.abs(finite))][:n_evolution]
    elif len(finite) < n_evolution:
        pad = np.full(n_evolution - len(finite), np.nan + 1j * np.nan)
        finite = np.concatenate([finite, pad])
    return np.sort_complex(finite)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--U1-max", type=float, default=2.0)
    parser.add_argument("--U2-max", type=float, default=1.5)
    parser.add_argument("--n-points", type=int, default=81)
    parser.add_argument("--n-points-2d", type=int, default=41)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    parser.add_argument("--U2", type=float, default=0.0)
    parser.add_argument("--k-val", type=float, default=1.0)
    parser.add_argument("--scan-2d", action="store_true",
                        help="Also produce a 2D (u_1, u_2) loss-of-hyperbolicity scan.")
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=2 hyperbolicity — route A vs route B (plane-wave at finite k)")
    print("=" * 70)

    print("Building route A …")
    sys_a = _flat_bottom(build_route_a())
    M_a, base_a, k_sym, omega_sym = _build_planewave(sys_a)
    print(f"  plane-wave matrix: {M_a.shape}")

    print("Building route B …")
    sys_b = _flat_bottom(build_route_b())
    M_b, base_b, _, _ = _build_planewave(sys_b)
    print(f"  plane-wave matrix: {M_b.shape}")

    h_a, u0_a, u1_a, u2_a = base_a[h], base_a[u_0], base_a[u_1], base_a[u_2]
    h_b, u0_b, u1_b, u2_b = base_b[h], base_b[u_0], base_b[u_1], base_b[u_2]
    w_bars = [base_b[next(f for f in sys_b.fields
                          if f.func.__name__ == f"w_{i}")] for i in range(4)]

    # Compile both plane-wave matrices to numpy (avoids slow sympy det).
    syms_a = [k_sym, g, h_a, u0_a, u1_a, u2_a]
    syms_b = [k_sym, g, h_b, u0_b, u1_b, u2_b] + w_bars
    eval_a = _make_M_eval(M_a, omega_sym, syms_a)
    eval_b = _make_M_eval(M_b, omega_sym, syms_b)
    deg_a = M_a.shape[0]
    deg_b = M_b.shape[0]

    u1_grid = np.linspace(-args.U1_max, args.U1_max, args.n_points)
    eigs_a, eigs_b = [], []

    print(f"\nSweeping {args.n_points} points across u_1 ∈ "
          f"[{-args.U1_max}, {args.U1_max}] (u_0={args.U0}, u_2={args.U2}, "
          f"k={args.k_val}, g={args.g})")

    for U1 in u1_grid:
        params_a = [args.k_val, args.g, args.H, args.U0, float(U1), args.U2]
        params_b = [args.k_val, args.g, args.H, args.U0, float(U1), args.U2,
                    0.0, 0.0, 0.0, 0.0]
        eigs_a.append(_disp_speeds(eval_a, params_a, args.k_val,
                                   n_evolution=4, deg_max=deg_a))
        eigs_b.append(_disp_speeds(eval_b, params_b, args.k_val,
                                   n_evolution=4, deg_max=deg_b))

    eigs_a = np.array(eigs_a)
    eigs_b = np.array(eigs_b)

    if eigs_a.shape == eigs_b.shape:
        max_diff = np.max(np.abs(eigs_a - eigs_b))
        print(f"\nmax |λ_A − λ_B| over sweep:  {max_diff:.3e}  "
              f"(machine eps ≈ 2.2e-16; bigger means numerical fragility)")
    else:
        print(f"\nshape mismatch: λ_A {eigs_a.shape}, λ_B {eigs_b.shape}")

    # Report a few sample states.
    print("\nSample (u_1, eigenvalues_A, eigenvalues_B):")
    for idx in (0, args.n_points // 4, args.n_points // 2,
                3 * args.n_points // 4, args.n_points - 1):
        print(f"  u_1={u1_grid[idx]:+.3f}: ")
        print(f"    A = {eigs_a[idx]}")
        print(f"    B = {eigs_b[idx]}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        ax_re, ax_im, ax_diff = axes

        for k in range(eigs_a.shape[1]):
            ax_re.plot(u1_grid, eigs_a[:, k].real, "b-", lw=1.5,
                       label="route A" if k == 0 else None)
            ax_re.plot(u1_grid, eigs_b[:, k].real, "r--", lw=1.0,
                       label="route B" if k == 0 else None)
            ax_im.plot(u1_grid, eigs_a[:, k].imag, "b-", lw=1.5)
            ax_im.plot(u1_grid, eigs_b[:, k].imag, "r--", lw=1.0)

        ax_re.set_xlabel(r"$\bar u_1$")
        ax_re.set_ylabel(r"Re $\omega/k$")
        ax_re.set_title("Re wave speed")
        ax_re.legend()
        ax_re.grid(True, alpha=0.3)

        ax_im.set_xlabel(r"$\bar u_1$")
        ax_im.set_ylabel(r"Im $\omega/k$")
        ax_im.set_title("Im wave speed — non-zero ⇒ loss of hyperbolicity")
        ax_im.grid(True, alpha=0.3)

        if eigs_a.shape == eigs_b.shape:
            for k in range(eigs_a.shape[1]):
                d = np.abs(eigs_a[:, k] - eigs_b[:, k])
                d = np.where(d < 1e-18, 1e-18, d)
                ax_diff.semilogy(u1_grid, d, lw=1.0, label=f"mode {k}")
            ax_diff.axhline(2.2e-16, color="k", ls=":", lw=0.8, label="machine ε")
            ax_diff.set_xlabel(r"$\bar u_1$")
            ax_diff.set_ylabel(r"$|\omega_A - \omega_B|$")
            ax_diff.set_title("Per-mode A↔B difference")
            ax_diff.legend(fontsize=8)
            ax_diff.grid(True, alpha=0.3, which="both")

        fig.suptitle(
            f"SME L=2 hyperbolicity — H={args.H}, U_0={args.U0}, "
            f"U_2={args.U2}, k={args.k_val}, g={args.g}", y=1.02)
        fig.tight_layout()
        out_path = "tutorials/sme/sme_l2_hyperbolicity_compare.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")

    # -----------------------------------------------------------------
    # Optional 2D loss-of-hyperbolicity scan
    # -----------------------------------------------------------------
    if args.scan_2d:
        print(f"\n2D scan: {args.n_points_2d}×{args.n_points_2d} grid over "
              f"(u_1, u_2) ∈ [{-args.U1_max},{args.U1_max}]×"
              f"[{-args.U2_max},{args.U2_max}]")
        u1_2d = np.linspace(-args.U1_max, args.U1_max, args.n_points_2d)
        u2_2d = np.linspace(-args.U2_max, args.U2_max, args.n_points_2d)
        max_im_a = np.zeros((args.n_points_2d, args.n_points_2d))
        max_im_b = np.zeros((args.n_points_2d, args.n_points_2d))
        for i, U1 in enumerate(u1_2d):
            for j, U2 in enumerate(u2_2d):
                pa = [args.k_val, args.g, args.H, args.U0, float(U1), float(U2)]
                pb = [args.k_val, args.g, args.H, args.U0, float(U1), float(U2),
                      0.0, 0.0, 0.0, 0.0]
                ea = _disp_speeds(eval_a, pa, args.k_val, n_evolution=4, deg_max=deg_a)
                eb = _disp_speeds(eval_b, pb, args.k_val, n_evolution=4, deg_max=deg_b)
                max_im_a[j, i] = np.nanmax(np.abs(np.imag(ea)))
                max_im_b[j, i] = np.nanmax(np.abs(np.imag(eb)))

        nh_a_frac = float(np.mean(max_im_a > 1e-6))
        nh_b_frac = float(np.mean(max_im_b > 1e-6))
        print(f"  Route A: {nh_a_frac:.1%} of grid is non-hyperbolic "
              f"(max |Im ω| > 1e-6)")
        print(f"  Route B: {nh_b_frac:.1%} of grid is non-hyperbolic")
        print(f"  max |max_im_A − max_im_B|: "
              f"{np.max(np.abs(max_im_a - max_im_b)):.3e}")

        if not args.no_plot:
            import matplotlib.pyplot as plt
            fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
            extent = [-args.U1_max, args.U1_max, -args.U2_max, args.U2_max]
            v_max = max(np.nanmax(max_im_a), np.nanmax(max_im_b), 1e-12)
            im0 = axes2[0].imshow(max_im_a, extent=extent, origin="lower",
                                  aspect="auto", cmap="magma", vmin=0, vmax=v_max)
            axes2[0].set_title("route A: max |Im ω/k|")
            axes2[0].set_xlabel(r"$\bar u_1$")
            axes2[0].set_ylabel(r"$\bar u_2$")
            plt.colorbar(im0, ax=axes2[0])

            im1 = axes2[1].imshow(max_im_b, extent=extent, origin="lower",
                                  aspect="auto", cmap="magma", vmin=0, vmax=v_max)
            axes2[1].set_title("route B: max |Im ω/k|")
            axes2[1].set_xlabel(r"$\bar u_1$")
            axes2[1].set_ylabel(r"$\bar u_2$")
            plt.colorbar(im1, ax=axes2[1])

            im2 = axes2[2].imshow(np.abs(max_im_a - max_im_b),
                                  extent=extent, origin="lower",
                                  aspect="auto", cmap="viridis",
                                  norm="log")
            axes2[2].set_title("|A − B| (log scale)")
            axes2[2].set_xlabel(r"$\bar u_1$")
            axes2[2].set_ylabel(r"$\bar u_2$")
            plt.colorbar(im2, ax=axes2[2])

            fig2.suptitle(
                f"SME L=2 loss-of-hyperbolicity — H={args.H}, U_0={args.U0}, "
                f"k={args.k_val}, g={args.g}", y=1.02)
            fig2.tight_layout()
            out_path_2d = "tutorials/sme/sme_l2_hyperbolicity_2d.png"
            fig2.savefig(out_path_2d, dpi=140, bbox_inches="tight")
            print(f"  2D plot saved: {out_path_2d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
