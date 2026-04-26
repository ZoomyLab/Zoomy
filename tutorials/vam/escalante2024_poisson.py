"""VAM splitting → Poisson-like equations for the non-hydrostatic
pressure — Escalante et al. 2024, eq (15).

This continues ``escalante2024_derivation.py``.  Given the compact
system (6) of the paper, the projection-correction time-stepping
proceeds in two stages:

* **Predictor**  (eq 11): solve the hyperbolic part for ``U^(k̃)``
  from ``U^(k-1)``.
* **Corrector**  (eq 12): solve

      (U^(k) − U^(k̃)) / Δt  +  T(U^(k), P^(k), ∂_x P^(k), ∂_x b) = 0   (a)
      I_1(U^(k), ∂_x U^(k), ∂_x b) = 0                                  (b)
      I_2(U^(k), ∂_x U^(k), ∂_x b) = 0                                  (c)

Substituting (a) into (b)–(c) gives a linear system in
``P^(k) = (p_0^(k), p_1^(k))`` whose **structure** matches paper eq (15):

      a₁∂_xx p_0 + a₂∂_x p_0 + a₃p_0 + a₄∂_xx p_1 + a₅∂_x p_1 + a₆p_1 = RHS_1
      b₁∂_xx p_0 + b₂∂_x p_0 + b₃p_0 + b₄∂_xx p_1 + b₅∂_x p_1 + b₆p_1 = RHS_2

with the coefficients ``a_j, b_j`` and the right-hand sides ``RHS_j``
depending only on ``U^(k̃)``, ``∂_x U^(k̃)``, ``∂_x b``.

This script:

1. Sets up the conserved-variable form ``U = (h, hu_0, hu_1, hw_0, hw_1)``.
2. Writes the corrector update from eq (12a) in terms of ``U^(k̃)`` and
   ``P^(k)`` — h does not change in the corrector since the first row
   of T is zero.
3. Substitutes the corrector update into ``I_1(U^(k))`` and
   ``I_2(U^(k))``.
4. Extracts the coefficients ``a_j, b_j`` of (∂_xx p_0, ∂_x p_0, p_0,
   ∂_xx p_1, ∂_x p_1, p_1) and prints them.
5. Asserts each equation is **strictly linear** in (p_0, p_1) and
   their first/second derivatives — i.e. that the Poisson form (15)
   indeed holds.

Friction τ and viscous σ_xz set to zero (the inviscid VAM, identical
to the derivation script's setup).
"""
from __future__ import annotations

import argparse
import sys
import sympy as sp


# ---------------------------------------------------------------------------
# Symbols — predictor U^(k̃) (denoted with subscript ``_t`` for tilde)
# ---------------------------------------------------------------------------

t, x = sp.symbols("t x", real=True)
g = sp.Symbol("g", positive=True)
dt = sp.Symbol(r"\Delta t", positive=True)

b = sp.Function("b", real=True)(x)            # bottom topography (unchanged)
h = sp.Function("h", real=True)(t, x)          # h^(k̃) = h^(k) (no change)

# Predictor primary unknowns (h × velocity components).
q0_t = sp.Function(r"\tilde{q}_0", real=True)(t, x)   # h·u_0 at k̃
q1_t = sp.Function(r"\tilde{q}_1", real=True)(t, x)   # h·u_1 at k̃
r0_t = sp.Function(r"\tilde{r}_0", real=True)(t, x)   # h·w_0 at k̃
r1_t = sp.Function(r"\tilde{r}_1", real=True)(t, x)   # h·w_1 at k̃

# Non-hydrostatic pressure unknowns at corrector.
p0 = sp.Function("p_0", real=True)(t, x)
p1 = sp.Function("p_1", real=True)(t, x)


# ---------------------------------------------------------------------------
# Corrector update (eq 12a): U^(k) = U^(k̃) − Δt T(U^(k), P^(k), ∂_x P, ∂_x b)
#
# T from paper:
#     T_1 = 0
#     T_2 = ∂_x(h p_0) + 2 p_1 ∂_x b
#     T_3 = -2 p_1
#     T_4 = ∂_x(h p_1) − (3 p_0 − p_1) ∂_x h − 6(p_0 − p_1) ∂_x b
#     T_5 = 6(p_0 − p_1)
#
# h is unchanged (T_1 = 0), so h^(k) = h^(k̃) ≡ h.  In T we therefore use
# the predictor h freely.
# ---------------------------------------------------------------------------

T1 = sp.S.Zero
T2 = sp.Derivative(h * p0, x) + 2 * p1 * sp.Derivative(b, x)
T3 = -2 * p1
T4 = (sp.Derivative(h * p1, x)
      - (3 * p0 - p1) * sp.Derivative(h, x)
      - 6 * (p0 - p1) * sp.Derivative(b, x))
T5 = 6 * (p0 - p1)

# Corrector primary unknowns expressed via the predictor + Δt·T.
q0_k = q0_t - dt * T2
q1_k = q1_t - dt * T4
r0_k = r0_t - dt * T3            # = r0_t + 2 Δt p_1
r1_k = r1_t - dt * T5            # = r1_t − 6 Δt (p_0 − p_1)

# Velocities at the corrector are q_*^(k)/h.
u0_k = q0_k / h
u1_k = q1_k / h
w0_k = r0_k / h
w1_k = r1_k / h


# ---------------------------------------------------------------------------
# Constraints I_1, I_2 — evaluated at U^(k)
# ---------------------------------------------------------------------------

def I_1_at_k():
    """``I_1(U^(k), ∂_x U^(k), ∂_x b)`` per paper eq (5):

        h ∂_x u_0 + (1/3) ∂_x(h u_1) + (1/3) u_1 ∂_x h
            + 2(w_0 − u_0 ∂_x b)
    """
    return (h * sp.Derivative(u0_k, x)
            + sp.Rational(1, 3) * sp.Derivative(h * u1_k, x)
            + sp.Rational(1, 3) * u1_k * sp.Derivative(h, x)
            + 2 * (w0_k - u0_k * sp.Derivative(b, x)))


def I_2_at_k():
    """``I_2(U^(k), ∂_x U^(k), ∂_x b)`` per paper eq (5):

        h ∂_x u_0 + u_1 ∂_x h + 2(u_1 ∂_x b − w_1)
    """
    return (h * sp.Derivative(u0_k, x)
            + u1_k * sp.Derivative(h, x)
            + 2 * (u1_k * sp.Derivative(b, x) - w1_k))


# ---------------------------------------------------------------------------
# Linearity test + coefficient extraction
# ---------------------------------------------------------------------------

def expand_derivatives(expr):
    """Distribute Derivative(product, var) via the product rule, fixpoint."""
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
    """Replace ``Derivative(c, var)`` with 0 for c independent of var."""
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
    """Distribute derivatives and clear ∂_t b (b = b(x)).  Returns an
    expression in standard polynomial form over the pressure unknowns
    {p_0, p_1, ∂_x p_0, ∂_x p_1, ∂_xx p_0, ∂_xx p_1} with all other
    factors as opaque coefficients."""
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


# Pressure-derivative atoms used as "indeterminates" for coefficient
# extraction.  We treat (p_0, p_1, ∂_x p_0, ∂_x p_1, ∂_xx p_0, ∂_xx p_1)
# as 6 independent linear coordinates.
P_VARS = [
    ("p_0",     p0),
    ("d_x p_0", sp.Derivative(p0, x)),
    ("d_xx p_0", sp.Derivative(p0, x, x)),
    ("p_1",     p1),
    ("d_x p_1", sp.Derivative(p1, x)),
    ("d_xx p_1", sp.Derivative(p1, x, x)),
]


def collect_pressure_coeffs(expr):
    """Decompose ``expr`` as

        Σ_i c_i · μ_i

    where ``μ_i`` ranges over (1, p_0, ∂_x p_0, ∂_xx p_0, p_1, ∂_x p_1,
    ∂_xx p_1).  Returns (c_const, [(name, c_i, μ_i), ...]).  Asserts
    that ``expr`` contains no other p_*-bearing atom (i.e. is strictly
    linear in P; no p², no ∂_xxx p, no Derivative(p, t) etc.).
    """
    e = normalise(expr)
    # Replace each pressure atom with a Dummy in turn, decompose as a
    # sympy Poly in the Dummies, and read off coefficients.
    dummies = []
    repl = {}
    inv = {}
    for name, atom in P_VARS:
        d = sp.Dummy(name)
        dummies.append((name, atom, d))
        repl[atom] = d
        inv[d] = atom
    e_poly = e.xreplace(repl)
    # Sanity check: no remaining p_0(t,x) / p_1(t,x) atoms outside the
    # six listed (e.g. Derivative(p_0, t) or Derivative(p_0, x, x, x)).
    leftover = e_poly.atoms(sp.Function).intersection({p0, p1})
    leftover_d = [d for d in e_poly.atoms(sp.Derivative)
                  if d.args[0] in (p0, p1)]
    if leftover or leftover_d:
        raise AssertionError(
            f"Expression is not in the linear span of P_VARS — "
            f"unexpected pressure atoms: {leftover | set(leftover_d)}"
        )
    try:
        poly = sp.Poly(e_poly, *[d for _, _, d in dummies])
    except sp.PolynomialError as exc:
        raise AssertionError(f"Pressure dependency is not polynomial: {exc}")
    if poly.total_degree() > 1:
        raise AssertionError(
            f"Pressure dependency is non-linear (degree "
            f"{poly.total_degree()})."
        )
    # Read coefficients.
    constant = poly.nth(*([0] * len(dummies)))
    out = []
    for i, (name, atom, _d) in enumerate(dummies):
        idx = [0] * len(dummies)
        idx[i] = 1
        coeff = poly.nth(*idx)
        out.append((name, sp.simplify(coeff), atom))
    return sp.simplify(constant), out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on first failed assertion")
    parser.add_argument("--print-coeffs", action="store_true",
                        help="Print all extracted coefficients")
    args = parser.parse_args()

    print("=== VAM splitting → Poisson form (Escalante et al. 2024 eq 15) ===\n")

    n_fail = 0
    for label, eq_factory in (
        ("I_1(U^(k)) → eq (15) line 1", I_1_at_k),
        ("I_2(U^(k)) → eq (15) line 2", I_2_at_k),
    ):
        print(f"--- {label} ---")
        try:
            constant, coeffs = collect_pressure_coeffs(eq_factory())
        except AssertionError as exc:
            print(f"  ✗ NOT POISSON-LINEAR: {exc}")
            n_fail += 1
            print()
            continue
        print("  ✓ Strictly linear in P = (p_0, p_1) and ∂_x P, ∂_xx P")
        nonzero = [(n, c, a) for (n, c, a) in coeffs if c != 0]
        present = ", ".join(n for n, _, _ in nonzero) or "(none)"
        print(f"  Active P-atoms: {present}")
        if args.print_coeffs:
            print(f"  Constant (RHS, with sign moved): {constant}")
            for name, coeff, _atom in coeffs:
                print(f"    coeff[{name}] = {coeff}")
        print()

    print(f"=== Summary: {n_fail} failed Poisson-linearity test(s) ===")
    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
