"""VAM splitting → N×N Poisson system, generalised to (M, N).

Continues ``escalante2024_generic.py``.  For each constraint
``I_k(U)`` (k=1..N) substitutes the projection-correction update

    U^(k) = U^(k̃) − Δt T(U^(k̃), P^(k))

and verifies the result is **strictly linear** in P = (p_0, …, p_{N-1})
(the surface BC closes p_N).  The N constraints become an N × N
elliptic system in P — a direct generalisation of the 2 × 2 system
of paper eq (15).

Approach (parameter-free per equation):

1. Derive the projection equations directly from σ-coords +
   polynomial ansatz (same primitives as
   ``escalante2024_generic.py``).
2. Apply the surface BC ``p|_{ξ=1} = 0`` to substitute p_N.
3. For each evolved momentum (h u_j and h w_j), extract the
   *pressure source* by substituting p_0, …, p_{N-1} → 0 in the
   projection equation and subtracting from the full.  Divide by
   μ_j = 1/(2j+1) to get the per-conserved-variable T.
4. Define the corrector update:
     u_j^(k) = u_j^(tilde) − (Δt / h) · (1/μ_j) · TP_{x,j}
     w_j^(k) = w_j^(tilde) − (Δt / h) · (1/μ_j) · TP_{z,j}
5. Substitute into each I_k constraint (already in dt_h-substituted
   form from continuity j=k).
6. Verify each I_k(U^(k)) is in the linear span of
   {p_i, ∂_x p_i, ∂_xx p_i} for i=0..N-1, and read off the
   coefficients.

For (M, N) = (1, 2) this reproduces the 2 × 2 system of paper eq (15)
(verified equivalent to ``escalante2024_poisson.py``).
For (M, N) = (2, 3) it generates a 3 × 3 system.
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
dt = sp.Symbol(r"\Delta t", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b


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


def setup_ansatz(M, N):
    nmax = max(M, N)
    phi = shifted_legendre_basis(nmax)
    u_coeffs = [sp.Function(f"u_{i}", real=True)(t, x) for i in range(M + 1)]
    w_coeffs = [sp.Function(f"w_{i}", real=True)(t, x) for i in range(N + 1)]
    p_coeffs = [sp.Function(f"p_{i}", real=True)(t, x) for i in range(N + 1)]
    u_tilde = [sp.Function(rf"\tilde{{u}}_{i}", real=True)(t, x)
               for i in range(M + 1)]
    w_tilde = [sp.Function(rf"\tilde{{w}}_{i}", real=True)(t, x)
               for i in range(N + 1)]
    u = sum((u_coeffs[i] * phi[i] for i in range(M + 1)), sp.S.Zero)
    w = sum((w_coeffs[i] * phi[i] for i in range(N + 1)), sp.S.Zero)
    p = sum((p_coeffs[i] * phi[i] for i in range(N + 1)), sp.S.Zero)
    omega = (w
             - xi * sp.Derivative(h, t)
             - u * (xi * sp.Derivative(h, x) + sp.Derivative(b, x)))
    return phi, u_coeffs, w_coeffs, p_coeffs, u_tilde, w_tilde, u, w, p, omega


# ---------------------------------------------------------------------------
# Galerkin projection helpers
# ---------------------------------------------------------------------------

def derive_continuity_j(j, phi, u, omega):
    int_phi_j = integrate_polynomial(phi[j])
    int_phi_j_u = integrate_polynomial(phi[j] * u)
    dphi_j = sp.diff(phi[j], xi)
    int_dphi_omega = integrate_polynomial(dphi_j * omega)
    return (int_phi_j * sp.Derivative(h, t)
            + sp.Derivative(h * int_phi_j_u, x)
            - int_dphi_omega)


def derive_xmom_j(j, phi, u, p, omega):
    int_phi_j = integrate_polynomial(phi[j])
    int_phi_j_u = integrate_polynomial(phi[j] * u)
    int_phi_j_u2 = integrate_polynomial(phi[j] * u**2)
    int_phi_j_p = integrate_polynomial(phi[j] * p)
    p_at_0 = sp.expand(p.subs(xi, 0))
    p_at_1 = sp.expand(p.subs(xi, 1))
    phi_j_at_0 = sp.expand(phi[j].subs(xi, 0))
    phi_j_at_1 = sp.expand(phi[j].subs(xi, 1))
    boundary = (-phi_j_at_1 * p_at_1 * sp.Derivative(eta, x)
                + phi_j_at_0 * p_at_0 * sp.Derivative(b, x))
    dphi_j = sp.diff(phi[j], xi)
    inner = omega * u - p * (xi * sp.Derivative(h, x) + sp.Derivative(b, x))
    int_dphi_inner = integrate_polynomial(dphi_j * inner)
    return (sp.Derivative(h * int_phi_j_u, t)
            + sp.Derivative(h * int_phi_j_u2 + h * int_phi_j_p, x)
            + g * h * sp.Derivative(eta, x) * int_phi_j
            + boundary - int_dphi_inner)


def derive_zmom_j(j, phi, u, w, p, omega):
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


# ---------------------------------------------------------------------------
# Canonicalisation
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


def normalise(expr):
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    e = e.subs({sp.Derivative(b, t): 0})
    e = expand_derivatives(e)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


# ---------------------------------------------------------------------------
# Pressure-source extraction + linear-in-P decomposition
# ---------------------------------------------------------------------------

def extract_pressure_part(expr, pressure_funcs):
    """Return the part of ``expr`` that depends on any ``pi`` in
    ``pressure_funcs``.  Strategy: zero out all the pi (sympy
    propagates this through ``Derivative``), subtract from full."""
    e = normalise(expr)
    no_p = e.subs({pi: sp.S.Zero for pi in pressure_funcs})
    no_p = normalise(no_p)
    return sp.expand(e - no_p)


def collect_pressure_coeffs(expr, p_coeffs_active):
    """Decompose ``expr`` as a linear polynomial in
        (p_i, ∂_x p_i, ∂_xx p_i) for i=0..len(p_coeffs_active)-1.
    Returns ``(constant, [(name, coeff, atom), ...])``.  Asserts
    strict linearity."""
    e = normalise(expr)
    P_VARS = []
    for i, pi in enumerate(p_coeffs_active):
        P_VARS.append((f"p_{i}",      pi))
        P_VARS.append((f"d_x p_{i}",  sp.Derivative(pi, x)))
        P_VARS.append((f"d_xx p_{i}", sp.Derivative(pi, x, x)))
    dummies = [sp.Dummy(name) for name, _ in P_VARS]
    repl = {atom: d for d, (_, atom) in zip(dummies, P_VARS)}
    e_poly = e.xreplace(repl)
    leftover_p = [pi for pi in p_coeffs_active if e_poly.has(pi)]
    leftover_d = [d for d in e_poly.atoms(sp.Derivative)
                  if d.args[0] in p_coeffs_active]
    if leftover_p or leftover_d:
        raise AssertionError(
            f"Expression is not in the (p, ∂_x p, ∂_xx p) span: "
            f"unexpected atoms {leftover_p + leftover_d}")
    poly = sp.Poly(e_poly, *dummies)
    if poly.total_degree() > 1:
        raise AssertionError(
            f"Pressure dependency is non-linear (degree {poly.total_degree()}).")
    constant = poly.nth(*([0] * len(dummies)))
    out = []
    for i, (name, atom) in enumerate(P_VARS):
        idx = [0] * len(dummies)
        idx[i] = 1
        out.append((name, sp.simplify(poly.nth(*idx)), atom))
    return sp.simplify(constant), out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--print-coeffs", action="store_true",
                        help="Print all extracted coefficients")
    args = parser.parse_args()

    M, N = args.M, args.N
    print(f"=== VAM Poisson system — generic (M, N) = ({M}, {N}) ===\n")
    print(f"Pressure unknowns at corrector: p_0..p_{N-1}  ({N} of them)")
    print(f"Constraints to enforce:        I_1..I_{N}    ({N} of them)")
    print(f"Expected Poisson system:       {N} × {N}\n")

    (phi, u_coeffs, w_coeffs, p_coeffs,
     u_tilde, w_tilde, u, w, p, omega) = setup_ansatz(M, N)

    # Surface BC: p|_{ξ=1} = 0 → solve for p_N.
    p_at_1 = sp.expand(p.subs(xi, 1))
    p_N_closure = sp.solve(p_at_1, p_coeffs[N])[0]
    sbc = {p_coeffs[N]: p_N_closure}
    p_active = p_coeffs[:N]                  # the unknowns at the corrector

    # Continuity j=0..N (j=0 evolution; j≥1 constraints).
    cont_j = [derive_continuity_j(j, phi, u, omega) for j in range(N + 1)]
    xmom_j = [derive_xmom_j(j, phi, u, p, omega) for j in range(M + 1)]
    zmom_j = [derive_zmom_j(j, phi, u, w, p, omega) for j in range(N)]

    # Apply surface BC to all projection equations (so p_N is closed).
    cont_j = [eq.subs(sbc) for eq in cont_j]
    xmom_j = [eq.subs(sbc) for eq in xmom_j]
    zmom_j = [eq.subs(sbc) for eq in zmom_j]

    # Pressure source per evolved conserved variable.  Each projection
    # equation has the form  μ_j · (∂_t(h q_j) + non-pressure flux) +
    # pressure_part = 0, so the conservative pressure source is
    # (1/μ_j) · pressure_part.
    def mu(j):
        return sp.Rational(1, 2 * j + 1)

    TP_u = [extract_pressure_part(xmom_j[j], p_active) / mu(j)
            for j in range(M + 1)]
    TP_w = [extract_pressure_part(zmom_j[j], p_active) / mu(j)
            for j in range(N)]

    # Corrector updates: q_j^(k) = q_j^(tilde) − (Δt/h) · TP_q_j.
    u_corr = [u_tilde[j] - (dt / h) * TP_u[j] for j in range(M + 1)]
    w_corr = [w_tilde[j] - (dt / h) * TP_w[j] for j in range(N)]

    # ∂_t h substitution (continuity j=0 gives ∂_t h = -∂_x(h u_0)).
    dt_h_rule = {sp.Derivative(h, t): -sp.Derivative(h * u_coeffs[0], x)}

    # Build constraints I_k from continuity j=k after dt_h subst.
    constraints = []
    for j in range(1, N + 1):
        c = normalise(cont_j[j])
        c = c.subs(dt_h_rule)
        c = normalise(c)
        constraints.append(c)

    # Substitute corrector values for u_j and w_j into each I_k.  Also
    # substitute w_N from the KBC bottom closure (algebraic).
    omega_at_xi_0 = sp.expand(omega.subs(xi, 0))
    omega_at_xi_0 = omega_at_xi_0.subs({sp.Derivative(b, t): 0})
    w_N_closure_in_u_w = sp.solve(omega_at_xi_0, w_coeffs[N])[0]

    repl = {}
    for j in range(M + 1):
        repl[u_coeffs[j]] = u_corr[j]
    for j in range(N):
        repl[w_coeffs[j]] = w_corr[j]
    # w_N closure must be substituted AFTER expressing w_N in terms
    # of corrector u/w.  Simpler: do w_N first (still in original
    # u_coeffs / w_coeffs symbols), then plug u_corr / w_corr.
    repl_wN = {w_coeffs[N]: w_N_closure_in_u_w}

    n_fail = 0
    print("=" * 64)
    for k, I_k in enumerate(constraints, start=1):
        print(f"\n--- I_{k}(U^(k)) → Poisson row {k} ---")
        # 1) substitute w_N closure (algebraic, in terms of u, w).
        e = I_k.subs(repl_wN)
        # 2) substitute corrector values for u_j (j=0..M) and w_j (j=0..N-1).
        e = e.subs(repl)
        # 3) try to decompose linearly in P.
        try:
            constant, coeffs = collect_pressure_coeffs(e, p_active)
        except AssertionError as exc:
            print(f"  ✗ NOT POISSON-LINEAR: {exc}")
            n_fail += 1
            continue
        nonzero = [(n, c, a) for (n, c, a) in coeffs if c != 0]
        present = ", ".join(n for n, _, _ in nonzero) or "(none)"
        print(f"  ✓ Strictly linear in (p_0..p_{N-1}, ∂_x ·, ∂_xx ·)")
        print(f"  Active P-atoms: {present}")
        if args.print_coeffs:
            print(f"  Constant (≡ -RHS): {constant}")
            for name, c, _atom in coeffs:
                if c != 0:
                    print(f"    coeff[{name}] = {c}")

    print()
    print("=" * 64)
    print(f"Summary: {n_fail} non-linearity failure(s) at (M, N) = ({M}, {N}).")
    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
