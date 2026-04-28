"""VAM derivation generalised to arbitrary (M, N) — Escalante et al. 2024.

Polynomial ansatz on σ ∈ [0, 1]:

    u(t, x, ξ) = Σ_{i=0}^M u_i(t, x) φ_i(ξ)
    w(t, x, ξ) = Σ_{i=0}^N w_i(t, x) φ_i(ξ)
    p(t, x, ξ) = Σ_{i=0}^N p_i(t, x) φ_i(ξ)

with shifted Legendre on [0, 1] (paper's sign convention φ_i(0) = 1).

Closure structure (derived purely by counting and KBC application):

    Evolution (M + N + 2 equations):
        Continuity j=0       → ∂_t h evolution
        x-momentum j=0..M    → evolution of (h u_0, …, h u_M)
        z-momentum j=0..N-1  → evolution of (h w_0, …, h w_{N-1})

    Closures (N + 2 equations):
        Continuity j=1..N    → N constraints (after eliminating ∂_t h)
        KBC at ξ=0           → algebraic w_N closure
        Surface BC p|_{ξ=1}=0 → algebraic p_N closure

Total: M + 2N + 4 conditions for the M + 2N + 4 unknowns
(h, u_0…u_M, w_0…w_N, p_0…p_N).  ✓

The case (M, N) = (1, 2) reproduces eq (4)-(5) of the paper and is
verified literally; other (M, N) just print the derived equations.
"""
from __future__ import annotations

import argparse
import sys
import sympy as sp


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

t, x, xi = sp.symbols("t x xi", real=True)
g = sp.Symbol("g", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b


# ---------------------------------------------------------------------------
# Shifted Legendre basis with paper's φ_i(0)=1 convention
# ---------------------------------------------------------------------------

def shifted_legendre_basis(n_max):
    s = sp.Symbol("s")
    return [sp.expand(sp.legendre(i, s).subs(s, 1 - 2 * xi))
            for i in range(n_max + 1)]


def integrate_polynomial(integrand, var=xi, lo=0, hi=1):
    expr = sp.expand(integrand)
    if not expr.has(var):
        return (hi - lo) * expr
    poly = sp.Poly(expr, var)
    anti = poly.integrate().as_expr()
    return sp.expand(anti.subs(var, hi) - anti.subs(var, lo))


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------

def setup_ansatz(M, N):
    nmax = max(M, N)
    phi = shifted_legendre_basis(nmax)
    u_coeffs = [sp.Function(f"u_{i}", real=True)(t, x) for i in range(M + 1)]
    w_coeffs = [sp.Function(f"w_{i}", real=True)(t, x) for i in range(N + 1)]
    p_coeffs = [sp.Function(f"p_{i}", real=True)(t, x) for i in range(N + 1)]
    u = sum((u_coeffs[i] * phi[i] for i in range(M + 1)), sp.S.Zero)
    w = sum((w_coeffs[i] * phi[i] for i in range(N + 1)), sp.S.Zero)
    p = sum((p_coeffs[i] * phi[i] for i in range(N + 1)), sp.S.Zero)
    omega = (w
             - xi * sp.Derivative(h, t)
             - u * (xi * sp.Derivative(h, x) + sp.Derivative(b, x)))
    return phi, u_coeffs, w_coeffs, p_coeffs, u, w, p, omega


# ---------------------------------------------------------------------------
# Galerkin projections (KBC-aware: ω(0) = ω(1) = 0 used to drop ωX boundary)
# ---------------------------------------------------------------------------

def derive_continuity_j(j, phi, u, omega):
    """Project ``∂_t h + ∂_x(h u) + ∂_ξ ω = 0`` against φ_j.

    IBP the ``∂_ξ ω`` term: ``∫φ_j ∂_ξ ω dξ = [φ_j ω]_0^1 − ∫φ_j' ω dξ``.
    KBCs ω(0) = ω(1) = 0 kill the boundary.
    """
    int_phi_j = integrate_polynomial(phi[j])
    int_phi_j_u = integrate_polynomial(phi[j] * u)
    dphi_j = sp.diff(phi[j], xi)
    int_dphi_omega = integrate_polynomial(dphi_j * omega)
    return (int_phi_j * sp.Derivative(h, t)
            + sp.Derivative(h * int_phi_j_u, x)
            - int_dphi_omega)


def derive_xmom_j(j, phi, u, p, omega):
    """j-th projection of x-mom (inviscid):

        ∂_t(h u) + ∂_x(h u² + h p) + g h ∂_x η
            + ∂_ξ(ω u − p ∂_x(ξh+b)) = 0
    """
    int_phi_j = integrate_polynomial(phi[j])
    int_phi_j_u = integrate_polynomial(phi[j] * u)
    int_phi_j_u2 = integrate_polynomial(phi[j] * u**2)
    int_phi_j_p = integrate_polynomial(phi[j] * p)
    p_at_0 = sp.expand(p.subs(xi, 0))
    p_at_1 = sp.expand(p.subs(xi, 1))
    phi_j_at_0 = sp.expand(phi[j].subs(xi, 0))    # = 1
    phi_j_at_1 = sp.expand(phi[j].subs(xi, 1))    # = (-1)^j
    # Boundary from ∂_ξ: ω·u parts vanish via KBCs.
    boundary = (-phi_j_at_1 * p_at_1 * sp.Derivative(eta, x)
                + phi_j_at_0 * p_at_0 * sp.Derivative(b, x))
    # Interior IBP term: −∫φ_j' (ω u − p ∂_x(ξh+b)) dξ
    dphi_j = sp.diff(phi[j], xi)
    inner = omega * u - p * (xi * sp.Derivative(h, x) + sp.Derivative(b, x))
    int_dphi_inner = integrate_polynomial(dphi_j * inner)
    return (sp.Derivative(h * int_phi_j_u, t)
            + sp.Derivative(h * int_phi_j_u2 + h * int_phi_j_p, x)
            + g * h * sp.Derivative(eta, x) * int_phi_j
            + boundary - int_dphi_inner)


def derive_zmom_j(j, phi, u, w, p, omega):
    """j-th projection of z-mom (inviscid):

        ∂_t(h w) + ∂_x(h u w) + ∂_ξ(ω w + p) = 0
    """
    int_phi_j_w = integrate_polynomial(phi[j] * w)
    int_phi_j_uw = integrate_polynomial(phi[j] * u * w)
    p_at_0 = sp.expand(p.subs(xi, 0))
    p_at_1 = sp.expand(p.subs(xi, 1))
    phi_j_at_0 = sp.expand(phi[j].subs(xi, 0))
    phi_j_at_1 = sp.expand(phi[j].subs(xi, 1))
    boundary = phi_j_at_1 * p_at_1 - phi_j_at_0 * p_at_0
    dphi_j = sp.diff(phi[j], xi)
    inner = omega * w + p
    int_dphi_inner = integrate_polynomial(dphi_j * inner)
    return (sp.Derivative(h * int_phi_j_w, t)
            + sp.Derivative(h * int_phi_j_uw, x)
            + boundary - int_dphi_inner)


def kbc_bottom_solve_wN(omega, w_coeffs, N):
    """``ω(0) = 0``, solved for ``w_N``."""
    omega_0 = sp.expand(omega.subs(xi, 0))
    omega_0 = omega_0.subs({sp.Derivative(b, t): 0})
    return sp.solve(omega_0, w_coeffs[N])[0]


def kbc_top(omega):
    """``ω(1) = 0`` — typically equivalent to continuity j=N
    (after dt_h subst and using KBC bottom).  Returned for inspection."""
    omega_1 = sp.expand(omega.subs(xi, 1))
    return omega_1.subs({sp.Derivative(b, t): 0})


# ---------------------------------------------------------------------------
# Canonicalisation helpers
# ---------------------------------------------------------------------------

def expand_derivatives(expr):
    def step(e):
        if isinstance(e, sp.Add):
            return sp.Add(*[step(a) for a in e.args])
        if isinstance(e, sp.Mul):
            return sp.Mul(*[step(a) for a in e.args])
        if isinstance(e, sp.Derivative):
            inner = step(e.expr)
            wrt_pairs = e.variable_count
            v, n = wrt_pairs[0]
            rest = wrt_pairs[1:]
            if n > 1:
                rest = ((v, n - 1),) + tuple(rest)
            if isinstance(inner, sp.Add):
                return step(sp.Add(*[sp.Derivative(a, v, *rest)
                                     for a in inner.args]))
            if isinstance(inner, sp.Mul):
                factors = inner.args
                out = sp.Add(*[
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v),)
                             + factors[i+1:]))
                    for i in range(len(factors))
                ])
                if rest:
                    out = sp.Derivative(out, *rest)
                return step(out)
            if isinstance(inner, sp.Pow):
                base, exponent = inner.args
                if isinstance(exponent, sp.Integer) and int(exponent) >= 2:
                    n_pow = int(exponent)
                    return step(n_pow * base**(n_pow - 1)
                                * sp.Derivative(base, v, *rest))
            return e
        return e
    prev = None
    cur = sp.expand(expr)
    while prev != cur:
        prev = cur
        cur = sp.expand(step(cur))
    return cur


def kill_trivially_zero_derivatives(expr):
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(sp.Derivative):
        inner = d.args[0]
        wrt = []
        for v in d.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        if inner.is_number or not any(inner.has(v) for v in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normalise(expr, surface_bc=None):
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    e = e.subs({sp.Derivative(b, t): 0})
    if surface_bc:
        e = e.subs(surface_bc)
    e = expand_derivatives(e)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


# ---------------------------------------------------------------------------
# Reference equations (paper, M=1, N=2 only)
# ---------------------------------------------------------------------------

def paper_references_1_2(u_coeffs, w_coeffs, p_coeffs):
    u0, u1 = u_coeffs
    w0, w1, _ = w_coeffs
    p0, p1, _ = p_coeffs
    refs = {}
    # Eq (4) row 1: ∂_t h + ∂_x(h u_0) = 0
    refs["cont_j0"] = sp.Derivative(h, t) + sp.Derivative(h * u0, x)
    # Eq (4) row 2: ∂_t(h u_0) + ∂_x(...) + g h ∂_x η = -2 p_1 ∂_x b
    refs["xmom_j0"] = (sp.Derivative(h * u0, t)
                      + sp.Derivative(h * u0**2
                                      + sp.Rational(1, 3) * h * u1**2
                                      + h * p0, x)
                      + g * h * sp.Derivative(eta, x)
                      - (-2 * p1 * sp.Derivative(b, x)))
    # Eq (4) row 3: ∂_t(h w_0) + ∂_x(...) = 2 p_1
    refs["zmom_j0"] = (sp.Derivative(h * w0, t)
                      + sp.Derivative(h * u0 * w0
                                      + sp.Rational(1, 3) * h * u1 * w1, x)
                      - 2 * p1)
    # Eq (5) line 1: I_1
    refs["I_1"] = (h * sp.Derivative(u0, x)
                  + sp.Rational(1, 3) * sp.Derivative(h * u1, x)
                  + sp.Rational(1, 3) * u1 * sp.Derivative(h, x)
                  + 2 * (w0 - u0 * sp.Derivative(b, x)))
    # Eq (5) line 2: I_2
    refs["I_2"] = (h * sp.Derivative(u0, x)
                  + u1 * sp.Derivative(h, x)
                  + 2 * (u1 * sp.Derivative(b, x) - w1))
    # Eq (5) line 3: w_2 closure
    refs["w_2"] = -(w0 + w1) + (u0 + u1) * sp.Derivative(b, x)
    return refs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--M", type=int, default=1, help="degree of u (default 1)")
    parser.add_argument("--N", type=int, default=2,
                        help="degree of w and p (default 2)")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on first mismatch (only for M=1, N=2)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress equation printing for non-paper cases")
    args = parser.parse_args()

    M, N = args.M, args.N
    print(f"=== VAM derivation — generic (M, N) = ({M}, {N}) ===\n")
    print(f"Unknowns: h, u_0..u_{M}, w_0..w_{N}, p_0..p_{N}  "
          f"= {M + 2*N + 4} total\n")
    print(f"Closure structure:")
    print(f"  Evolution: {M + N + 2} equations "
          f"(continuity j=0; x-mom j=0..{M}; z-mom j=0..{N-1})")
    print(f"  Closures:  {N + 2} equations "
          f"(continuity j=1..{N}; KBC bottom; surface BC for p_{N})\n")

    phi, u_coeffs, w_coeffs, p_coeffs, u, w, p, omega = setup_ansatz(M, N)

    # Continuity j=0..N — j=0 is evolution, j=1..N are constraints.
    cont_j = [derive_continuity_j(j, phi, u, omega) for j in range(N + 1)]
    # x-momentum j=0..M.
    xmom_j = [derive_xmom_j(j, phi, u, p, omega) for j in range(M + 1)]
    # z-momentum j=0..N-1 (z-mom j=N would evolve w_N which is closed
    # algebraically via KBC bottom; we drop it).
    zmom_j = [derive_zmom_j(j, phi, u, w, p, omega) for j in range(N)]

    # KBC closures.
    w_N_closure = kbc_bottom_solve_wN(omega, w_coeffs, N)
    omega1_for_check = kbc_top(omega)

    # ∂_t h substitution from continuity j=0.
    dt_h_rule = {sp.Derivative(h, t): -sp.Derivative(h * u_coeffs[0], x)}

    # Constraints from continuity j=1..N (after dt_h subst).
    constraints = []
    for j in range(1, N + 1):
        c = normalise(cont_j[j])
        c = c.subs(dt_h_rule)
        c = normalise(c)
        constraints.append(c)

    # Surface BC for p: p|_{ξ=1} = Σ_{i=0}^N (-1)^i p_i = 0 → solve for p_N.
    p_at_1 = sp.expand(p.subs(xi, 1))
    p_N_closure = sp.solve(p_at_1, p_coeffs[N])[0]

    # ----------------------------------------------------------------------
    # If (M, N) = (1, 2), compare to paper's eq (4)-(5).
    # ----------------------------------------------------------------------
    if (M, N) == (1, 2):
        refs = paper_references_1_2(u_coeffs, w_coeffs, p_coeffs)
        # Surface BC substitution: p_2 = p_1 - p_0.
        sbc = {p_coeffs[2]: p_coeffs[1] - p_coeffs[0]}

        n_mismatch = 0

        def check(label, derived, reference, surface_bc=None):
            nonlocal n_mismatch
            d = normalise(derived, surface_bc=surface_bc)
            r = normalise(reference, surface_bc=surface_bc)
            diff = sp.expand(d - r)
            ok = (diff == 0)
            mark = "✓ MATCH" if ok else "✗ MISMATCH"
            print(f"--- {label} ---  {mark}")
            if not ok:
                n_mismatch += 1
                print(f"   derived   = {d}")
                print(f"   reference = {r}")
                print(f"   diff      = {diff}")

        check("Eq (4) row 1 (cont j=0)",  cont_j[0], refs["cont_j0"])
        check("Eq (4) row 2 (x-mom j=0)", xmom_j[0], refs["xmom_j0"], sbc)
        check("Eq (4) row 3 (z-mom j=0)", zmom_j[0], refs["zmom_j0"], sbc)
        check("Eq (5) I_1 (cont j=1, dt_h subst)",  constraints[0], refs["I_1"])
        # Eq (5) I_2 — derived as cont j=N=2 after dt_h subst (equivalent
        # to KBC top combined with KBC bottom).  Substitute w_2 closure
        # to fold the bottom KBC into the constraint, then compare.
        # The continuity j=N projection has the opposite sign convention
        # from the paper's I_2 (which uses KBC top − KBC bottom); flip
        # the sign of the derived constraint to match the paper.
        c_j2 = -constraints[1].subs({w_coeffs[2]: w_N_closure})
        I_2_ref = refs["I_2"]
        check("Eq (5) I_2 (-cont j=2 with w_2 closure subst)",
              c_j2, I_2_ref)
        check("Eq (5) w_2 closure (KBC bottom)",
              w_N_closure, refs["w_2"])

        print()
        print(f"=== Summary: {n_mismatch} mismatch(es) at (M,N)=({M},{N}) ===")
        if args.strict and n_mismatch > 0:
            sys.exit(1)
        return

    # ----------------------------------------------------------------------
    # Generic case: print derived equations.
    # ----------------------------------------------------------------------
    if args.quiet:
        print(f"Derivation completed for (M, N) = ({M}, {N}).")
        print(f"  {len([cont_j[0]] + xmom_j + zmom_j)} evolution equations,")
        print(f"  {len(constraints)} continuity-j constraints,")
        print(f"  1 KBC bottom closure, 1 surface BC closure.")
        return

    print("--- Evolution equations (after normalise) ---\n")
    print("[Continuity j=0]")
    print("  " + str(normalise(cont_j[0])) + " = 0\n")
    for j in range(M + 1):
        print(f"[x-momentum j={j}]")
        print("  " + str(normalise(xmom_j[j])) + " = 0\n")
    for j in range(N):
        print(f"[z-momentum j={j}]")
        print("  " + str(normalise(zmom_j[j])) + " = 0\n")

    print("--- Closure constraints ---\n")
    for k, c in enumerate(constraints, start=1):
        print(f"[Continuity j={k} → constraint I_{k} (dt_h subst)]")
        print("  " + str(c) + " = 0\n")
    print(f"[KBC at ξ=0 → w_{N} closure]")
    print(f"  w_{N} = {w_N_closure}\n")
    print(f"[Surface BC p|_{{ξ=1}}=0 → p_{N} closure]")
    print(f"  p_{N} = {p_N_closure}\n")

    # Sanity check: after substituting w_N + p_N closures + dt_h subst,
    # KBC top (ω(1)=0) should be a linear combination of the
    # continuity-j constraints.  Just check it's NOT a fresh independent
    # condition by reducing it.
    check_top = sp.expand(omega1_for_check
                          .subs({w_coeffs[N]: w_N_closure})
                          .subs(dt_h_rule))
    check_top = normalise(check_top)
    print("--- Redundancy check: KBC top (ω(1)=0) after w_N + dt_h subst ---")
    print(f"  → reduces to: {check_top}")
    print(f"  (Should be a linear combination of the I_k constraints — "
          f"possibly = -I_{N} after w_{N} subst.)")


if __name__ == "__main__":
    main()
