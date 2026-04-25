"""Slim SME walkthrough — driven entirely by ``zoomy_core.symbolic`` primitives.

Phase 0 proof-of-concept for the redesigned symbolic-derivation
framework.  Each pipeline step is one explicit primitive call.  The
goal: close bug 3 (Wronskian-shape K&T-comparison asymmetry caused by
late-materialised ``∂_t h`` atoms) end-to-end without any
sympy auto-evaluation.

This script bypasses ``Expression.apply`` / ``_NodeProxy`` entirely
— it works directly on sympy expressions via the new primitives.
For the equation-system bookkeeping it uses ``FullINS`` only as a
convenient initial-equation builder; afterwards every transformation
runs through the primitive layer.

Pipeline (each step a single primitive call, no sympy magic):

* Step 1  — Subst({tau_ij: 0})
* Step 2  — hydrostatic + p substitution
              (PolynomialIntegrate, Subst, SolveFor)
* Step 3  — w-closure via PolynomialIntegrate(z, b, z) + SolveFor
* Step 4  — depth-integrate continuity + KBC
              (FundamentalTheorem / Leibniz / PolynomialIntegrate
               per term, Subst(kbc), UnSubs)
* Step 5  — Substitute w(z) closure into x-momentum
* Step 6  — Multiply by phi_k (per-test-mode), apply
              ProductRuleInverse on the ∂_z terms
* Step 7  — Depth-integrate x-momentum (per term, FT/Leibniz)
* Step 8  — Apply KBC
* Step 9  — Capture dt_h_relation from continuity
              (snapshot only — substitution applied as the LAST step)
* Step 10 — Insert velocity ansatz via FunctionExpand
* Step 11 — AffineChangeOfVariable(z, b, b+h, ζ̂)
* Step 12 — CanonicalizePhiDerivativeSubs + SplitIntegralOverAdd
* Step 13 — ProjectBasisIntegrand
* Step 14 — Bug-3 fixpoint:
              while ∂_t h appears in test_k:
                  DistributeDerivativeOverAdd
                  Subst(dt_h_relation)
                  ProjectBasisIntegrand on any fresh
                    Derivative(Integral(u), x) atom
                  Canonicalise
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic import (
    D,
    Int,
    affine_change_of_variable,
    alpha_rename,
    canonicalise,
    canonicalize_phi_derivative_subs,
    distribute_derivative_over_add,
    function_expand,
    fundamental_theorem,
    leibniz,
    leibniz_general,
    polynomial_integrate,
    product_rule_forward,
    product_rule_inverse,
    project_basis_integrand,
    solve_for,
    split_integral_over_add,
    subst,
    un_subs,
)
from zoomy_core.model.models.ins_generator import FullINS, StateSpace
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.basisfunctions import Legendre_shifted


LEVEL = 1
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
H = state.H


# ---------------------------------------------------------------------------
# Step 0 — Get raw INS as sympy expressions.
# ---------------------------------------------------------------------------

ins = FullINS(state)
# Reach through to the leaf Expressions via the tree.
mom_x = ins.momentum.x._node.expr      # ∂_t u + ∂_x(u·u) + ∂_z(u·w) + (1/ρ)∂_x p
mom_z = ins.momentum.z._node.expr      # ditto for w-momentum + g
cont = ins.continuity._node.expr        # ∂_x u + ∂_z w


def _print_step(label, *exprs):
    print(f"\n--- {label} ---")
    for name, e in exprs:
        print(f"  {name}: {e}")


# ---------------------------------------------------------------------------
# Step 1 — Drop all viscous stresses.
# ---------------------------------------------------------------------------

tau_zero = {state.tau[k]: 0 for k in state.tau._filter_dict()}
mom_x = canonicalise(subst(mom_x, tau_zero))
mom_z = canonicalise(subst(mom_z, tau_zero))
cont = canonicalise(subst(cont, tau_zero))


# ---------------------------------------------------------------------------
# Step 2 — Hydrostatic z-momentum: drop unsteady/convection, integrate to
#          get p(z), substitute p|η = 0, solve for p, plug into x-momentum.
# ---------------------------------------------------------------------------

# Hydrostatic scaling: drops ∂_t w and convection from z-mom, keeps gravity
# + (1/ρ) ∂_z p.  After tau=0 mom_z reads ∂_t w + ∂_x(uw) + ∂_z(w²)
# + (1/ρ)∂_z p + g.  Drop the convective/unsteady pieces.
mom_z_hydro = subst(
    mom_z,
    {
        D(state.w, t): 0,
        D(state.u * state.w, x): 0,
        D(state.w * state.w, z): 0,
    },
)
mom_z_hydro = canonicalise(mom_z_hydro)
# mom_z_hydro = (1/ρ) ∂_z p + g  (after canonicalise drops the 0s).
# Integrate from z to η, term by term:
#
#   Term A:  ∫_z^η (1/ρ) ∂_ξ p dξ  =  (1/ρ)·(p|_η − p|_z)   via FT
#   Term B:  ∫_z^η g dξ            =  g·(η − z)            via PolynomialIntegrate
xi = sp.Symbol("xi", real=True)
p_at_xi = state.p.subs(z, xi)
ft_p = fundamental_theorem(D(p_at_xi, xi), xi, z, state.eta)
const_g = polynomial_integrate(state.g, xi, z, state.eta)
mom_z_integrated = canonicalise(ft_p / state.rho + const_g)
# Apply atmospheric pressure p|η = 0.
mom_z_integrated = subst(
    mom_z_integrated,
    {state.p.subs(z, state.eta): 0},
)
mom_z_integrated = canonicalise(mom_z_integrated)
# Solve for p.
p_relation = solve_for(mom_z_integrated, state.p)
print("\n[Step 2] p =", p_relation[state.p])
mom_x = canonicalise(subst(mom_x, p_relation))


# ---------------------------------------------------------------------------
# Step 3 — w-closure from continuity: integrate ∂_z w + ∂_x u over (b, z).
# ---------------------------------------------------------------------------

# Integrate continuity from b to z to get w(z) - w(b) + ∫_b^z ∂_x u dẑ = 0.
# Use polynomial_integrate (the integrand ∂_z w + ∂_x u is linear in
# everything, but ∂_x u depends on z through u(t,x,z)).
# We need a running integral.  PolynomialIntegrate on the ∂_z w piece
# gives w(z) - w(b); on the ∂_x u piece keeps it as Int(∂_x u, (ẑ, b, z)).
zh = sp.Symbol(r"\hat{z}", real=True)
# Replace continuity z-derivative term with FT result and ∂_x u term with
# a held running integral.  Easiest: build the integrated form by hand.
# ∫_b^z (∂_x u + ∂_z w) dẑ = (w(z) - w(b)) + ∫_b^z ∂_x u(ẑ) dẑ
# Replace the cont integrand var z→ẑ first.
cont_zh = cont.subs(z, zh)
w_eq = (state.w - state.w.subs(z, state.b)
        + Int(D(state.u.subs(z, zh), x), (zh, state.b, z)))
# Sanity: differentiate w_eq w.r.t. z and check it equals cont (but
# cont is in terms of z, so we'd need ζ-frame — skip; the construction
# is direct from the math).
w_relation = solve_for(w_eq, state.w)
print("[Step 3] w(z) =", w_relation[state.w])
# Substitute w(z) into x-momentum.
mom_x = canonicalise(subst(mom_x, w_relation))


# ---------------------------------------------------------------------------
# Step 4 — Depth-integrate continuity + KBC for h-evolution.
# ---------------------------------------------------------------------------

# Apply Leibniz/FT term-by-term to ∫_b^η (∂_x u + ∂_z w) dz.
# ∫_b^η ∂_z w dz = w(η) - w(b)              (FundamentalTheorem)
# ∫_b^η ∂_x u dz = ∂_x ∫_b^η u dz - u(η)·∂_x η + u(b)·∂_x b  (Leibniz)
ub = state.u.subs(z, state.b)
ueta = state.u.subs(z, state.eta)
wb = state.w.subs(z, state.b)
weta = state.w.subs(z, state.eta)

# Leibniz on ∂_x u (general form: u is opaque, so leibniz() — which
# requires a polynomial integrand — won't fire; use leibniz_general):
leibniz_u = leibniz_general(D(state.u, x), z, state.b, state.eta)
# FT on ∂_z w:
ft_w = fundamental_theorem(D(state.w, z), z, state.b, state.eta)
cont_integrated = leibniz_u + ft_w
# KBC: w|_b = ∂_t b + u|_b · ∂_x b ; w|_η = ∂_t η + u|_η · ∂_x η
kbc = {
    wb: D(state.b, t) + ub * D(state.b, x),
    weta: D(state.eta, t) + ueta * D(state.eta, x),
}
cont_integrated = canonicalise(subst(cont_integrated, kbc))
print("[Step 4] depth-integrated continuity =", cont_integrated)


# ---------------------------------------------------------------------------
# Step 5 — w-closure already substituted in step 3.  mom_x is current.
# ---------------------------------------------------------------------------

print("\n[Step 5] mom_x has", len(sp.Add.make_args(canonicalise(mom_x))),
      "additive terms after w substitution")


# ---------------------------------------------------------------------------
# Step 6 — Galerkin test: phi_k · mom_x for k = 0, 1.
# ---------------------------------------------------------------------------

phi_fns = [sp.Function(f"phi_{k}") for k in range(LEVEL + 1)]
zoz = (z - state.b) / H
# Build per-test-mode equations.
test_eqs = {}
for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(phi_fns[k](zoz) * mom_x)

# Apply ProductRuleInverse term-by-term ONLY on terms with ∂_z derivatives
# (the conservative-form ∂_z(u·w) and ∂_z(-u·∫…) pieces).  Horizontal
# ProductRule introduces 1/h² transients we don't need until after the
# affine map.  But to actually fire Leibniz on ∂_x and ∂_t terms in step 7,
# we still need ProductRule there.  So apply ProductRuleInverse to ALL
# Mul-with-Derivative terms.
def apply_product_rule_inverse_term_wise(eq_expr, var_list):
    """Apply product_rule_inverse to each Add summand for each var."""
    out = sp.S.Zero
    for term in sp.Add.make_args(eq_expr):
        new_term = term
        for v in var_list:
            new_term = product_rule_inverse(new_term, v)
        out = out + new_term
    return out


for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(
        apply_product_rule_inverse_term_wise(test_eqs[k], [t, x, z])
    )

print("\n[Step 6] test_1 has", len(sp.Add.make_args(test_eqs[1])), "terms after ProductRuleInverse")


# ---------------------------------------------------------------------------
# Step 7 — Depth-integrate test_k over [b, η].
# ---------------------------------------------------------------------------

def depth_integrate_term(term):
    """Apply FT or Leibniz_general per term, falling back to
    PolynomialIntegrate or a held Int for the rest."""
    if isinstance(term, sp.Derivative):
        wrt = []
        for v in term.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        if z in wrt:
            r = fundamental_theorem(term, z, state.b, state.eta)
            if r is not None:
                return r
        if any(v in (t, x) for v in wrt):
            r = leibniz_general(term, z, state.b, state.eta)
            if r is not None:
                return r
    if isinstance(term, sp.Mul):
        derivs = [f for f in term.args if isinstance(f, sp.Derivative)]
        if len(derivs) == 1:
            d = derivs[0]
            wrt = []
            for v in d.args[1:]:
                wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
            coeff = sp.Mul(*[f for f in term.args if f is not d])
            if z in wrt:
                r = fundamental_theorem(d, z, state.b, state.eta)
                if r is not None:
                    return coeff * r
            if any(v in (t, x) for v in wrt):
                r = leibniz_general(d, z, state.b, state.eta)
                if r is not None:
                    return coeff * r
    # Fallback: polynomial-integrate or hold.
    r = polynomial_integrate(term, z, state.b, state.eta)
    if r is not None:
        return r
    return Int(term, (z, state.b, state.eta))


for k in range(LEVEL + 1):
    integrated = sp.S.Zero
    for term in sp.Add.make_args(test_eqs[k]):
        integrated = integrated + depth_integrate_term(term)
    test_eqs[k] = canonicalise(integrated)

print("[Step 7] test_1 has", len(sp.Add.make_args(test_eqs[1])), "terms after depth-integrate")


# ---------------------------------------------------------------------------
# Step 8 — Apply KBC again to surface boundary u·w pieces.
# ---------------------------------------------------------------------------

for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(subst(test_eqs[k], kbc))


# ---------------------------------------------------------------------------
# Step 9 — Snapshot dt_h_relation from depth-integrated continuity.
# ---------------------------------------------------------------------------

# After kbc applied to cont_integrated, distribute Derivative(b+h, t)
# so D(h, t) becomes a free atom solve_for can target.
cont_post_kbc = canonicalise(distribute_derivative_over_add(cont_integrated))
dt_h_relation = solve_for(cont_post_kbc, D(H, t))
print("\n[Step 9] dt_h_relation:", dt_h_relation)


# ---------------------------------------------------------------------------
# Step 10 — Insert velocity ansatz via function_expand.
# ---------------------------------------------------------------------------

basis_alpha = [sp.Function(f"alpha_{k}", real=True)(t, x) for k in range(LEVEL + 1)]


def u_ansatz(*args):
    arg = args[-1]
    rhs = basis_alpha[0]
    for k in range(1, LEVEL + 1):
        rhs = rhs + basis_alpha[k] * phi_fns[k]((arg - state.b) / H)
    return rhs


for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(function_expand(test_eqs[k], state.u.func, u_ansatz))


# ---------------------------------------------------------------------------
# Step 11 — AffineChangeOfVariable z → ζ̂h + b on (z, b, b+h) integrals.
# ---------------------------------------------------------------------------

zeta_hat = sp.Symbol(r"\hat{\zeta}", positive=True)
for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(
        affine_change_of_variable(test_eqs[k], z, state.b, state.b + H, zeta_hat)
    )

# The eta=b+h substitution in u(η) etc. — make explicit.
for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(subst(test_eqs[k], {state.eta: state.b + H}))


# ---------------------------------------------------------------------------
# Step 12 — Canonicalize phi-derivative-Subs + split for basis projection.
# ---------------------------------------------------------------------------

for k in range(LEVEL + 1):
    e = test_eqs[k]
    e = canonicalize_phi_derivative_subs(e)
    e = split_integral_over_add(e)
    test_eqs[k] = canonicalise(e)


# ---------------------------------------------------------------------------
# Step 13 — Project basis integrals via cache.
# ---------------------------------------------------------------------------

cache = BasisIntegralCache(Legendre_shifted(level=LEVEL))
for k in range(LEVEL + 1):
    test_eqs[k] = canonicalise(project_basis_integrand(test_eqs[k], cache))


# ---------------------------------------------------------------------------
# Step 14 — Bug-3 fixpoint: distribute Derivative atoms, substitute ∂_t h,
#           re-project any fresh Derivative(Integral(u), x) atoms.
# ---------------------------------------------------------------------------

def has_free_dt_h(expr):
    """True iff expr has a free Derivative(h, t) atom."""
    for d in expr.atoms(sp.Derivative):
        if d.args[0] == H:
            wrt = []
            for v in d.args[1:]:
                wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
            if t in wrt:
                return True
    return False


for k in range(LEVEL + 1):
    e = test_eqs[k]
    for _ in range(8):
        prev = e
        e = distribute_derivative_over_add(e)
        e = subst(e, dt_h_relation)
        # Substitution may have introduced fresh Integral atoms.  Apply
        # the basis projection again.
        e = function_expand(e, state.u.func, u_ansatz)
        e = affine_change_of_variable(e, z, state.b, state.b + H, zeta_hat)
        e = canonicalize_phi_derivative_subs(e)
        e = split_integral_over_add(e)
        e = project_basis_integrand(e, cache)
        e = canonicalise(e)
        if e == prev or not has_free_dt_h(e):
            break
    test_eqs[k] = e


# ---------------------------------------------------------------------------
# K&T comparison — Legendre boundary values + flat bottom.
# ---------------------------------------------------------------------------

phi_legendre = {
    phi_fns[0](sp.S.Zero): 1, phi_fns[0](sp.S.One): 1,
    phi_fns[1](sp.S.Zero): 1, phi_fns[1](sp.S.One): -1,
}
flat_bottom = {state.b: 0, D(state.b, x): 0, D(state.b, t): 0}


def kt_form(expr):
    return sp.expand(sp.expand(expr).subs(phi_legendre).subs(flat_bottom))


def expand_derivatives(expr):
    """Distribute Derivative(product, var) via product rule, fixpoint."""
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
                out = sp.Add(*[sp.Derivative(a, v, *rest) for a in inner.args])
                return step(out)
            if isinstance(inner, sp.Mul):
                factors = inner.args
                out = sp.Add(*[
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v),) + factors[i+1:]))
                    for i in range(len(factors))
                ])
                if rest:
                    out = sp.Derivative(out, *rest)
                return step(out)
            return e
        return e
    prev = None; cur = sp.expand(expr)
    while prev != cur:
        prev = cur
        cur = sp.expand(step(cur))
    return cur


a0, a1 = basis_alpha[0], basis_alpha[1]
dxH = D(H, x)
probes = [
    ("α_0² · ∂_x h",       a0**2 * dxH),
    ("α_0 · α_1 · ∂_x h",  a0 * a1 * dxH),
    ("α_1² · ∂_x h",       a1**2 * dxH),
    ("α_0 · ∂_x α_0 · h",  a0 * D(a0, x) * H),
    ("α_0 · ∂_x α_1 · h",  a0 * D(a1, x) * H),
    ("α_1 · ∂_x α_0 · h",  a1 * D(a0, x) * H),
    ("α_1 · ∂_x α_1 · h",  a1 * D(a1, x) * H),
    ("∂_t α_0 · h",        D(a0, t) * H),
    ("∂_t α_1 · h",        D(a1, t) * H),
]

print("\n=== K&T COMPARISON ===\n")
for k in (0, 1):
    e = test_eqs[k]
    e_kt = kt_form(e)
    e_expanded = expand_derivatives(e_kt)
    factor = 3 if k == 1 else 1   # mass-matrix normalisation
    label = "level-1 (×3)" if k == 1 else "level-0"
    print(f"test_{k} ({label}):")
    for plabel, key in probes:
        c = sp.simplify(factor * e_expanded.coeff(key))
        if c != 0:
            print(f"  {plabel:<25}  →  {c}")
    print()


# ---------------------------------------------------------------------------
# 1/h² audit.
# ---------------------------------------------------------------------------

def find_inverse_h_powers(expr, h_sym, n_min=2):
    hits = []
    for p in sp.preorder_traversal(expr):
        if isinstance(p, sp.Pow):
            base, exp = p.args
            if base == h_sym and exp.is_Integer and int(exp) <= -n_min:
                hits.append(p)
    return hits


print("=== 1/h² audit ===")
for k in (0, 1):
    bad = find_inverse_h_powers(test_eqs[k], H, n_min=2)
    if bad:
        print(f"[!] test_{k} contains {len(bad)} subexpression(s) with 1/h^≥2:")
        for b in bad[:5]:
            print(f"    {b}")
    else:
        print(f"[ok] test_{k} — no 1/h² or higher.")
