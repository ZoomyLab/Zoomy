"""Phase-0 proof point: close bug 3 via the new primitive layer.

The legacy slim_walkthrough drives steps 1-13 cleanly but leaves a
held ``Derivative(α_l·h/k, t)`` atom from the conservative branch of
step 6's ``ProductRule``.  After ``ProjectBasisIntegrals`` evaluates
the inner integral to ``α_l·h/k``, the outer ``Derivative(_, t)`` only
materialises a free ``α_l·∂_t h`` when an explicit product-rule
expansion fires — which today is *after* step 9's substitution
already finished, leaving the residual unsubstituted.  That's the
Wronskian-shape K&T-comparison asymmetry on test_1.

This script drives the legacy pipeline through step 13 (giving us
the post-projection state with the held conservative Derivatives),
then closes bug 3 with **only the new primitive layer**:

    while ∂_t h appears free in test_k:
        distribute_derivative_over_add(...)   # primitive
        subst(..., dt_h_relation)             # primitive
        function_expand(..., u_ansatz)        # primitive — re-expand u in any
                                              # fresh Integral integrand
        affine_change_of_variable(...)        # primitive
        canonicalize_phi_derivative_subs(...) # primitive
        split_integral_over_add(...)          # primitive
        project_basis_integrand(..., cache)   # primitive
        canonicalise(...)                     # the structural pass

If the K&T coefficients on test_1 collapse to the symmetric pattern
predicted by pen-and-paper (``α_0·∂_x α_1·h: 1, α_1·∂_x α_0·h: 1,
α_0·α_1·∂_x h: 0``), Phase 0 succeeds and Phase 1 (full migration)
is the right next step.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.ins_generator import (
    FullINS,
    Integrate,
    IntegralTransform,
    IsolateBasisIntegrand,
    Multiply,
    ProductRule,
    ProjectBasisIntegrals,
    StateSpace,
    _FieldExpansion,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling

# === NEW primitive layer ===
from zoomy_core.symbolic import (
    affine_change_of_variable,
    canonicalise,
    canonicalize_phi_derivative_subs,
    distribute_derivative_over_add,
    function_expand,
    product_rule_forward,
    project_basis_integrand,
    solve_for,
    split_integral_over_add,
    subst,
)


LEVEL = 1


def build_legacy_state_through_step_13():
    """Run the legacy pipeline up to step 13 verbatim, returning the
    full system + the captured ``dt_h_relation`` from step 9."""
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z
    H = state.H
    model = FullINS(state)
    model.apply({state.tau[k]: 0 for k in state.tau._filter_dict()}).simplify()
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
    model.momentum.z.remove()
    pc = model.continuity.copy()
    w_eq = pc.apply(Integrate(z, state.b, z, method="auto"))
    w_closure = w_eq.solve_for(state.w)
    ub = state.u.subs(z, state.b)
    ue = state.u.subs(z, state.eta)
    kbc = {
        state.w.subs(z, state.b):
            sp.Derivative(state.b, t) + ub * sp.Derivative(state.b, x),
        state.w.subs(z, state.eta):
            sp.Derivative(state.eta, t) + ue * sp.Derivative(state.eta, x),
    }
    model.continuity.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.continuity.apply(kbc).simplify()
    model.momentum.x.apply(w_closure).simplify()
    phi_fns = [sp.Function(f"phi_{k}") for k in range(LEVEL + 1)]
    zoz = (state.z - state.b) / H
    phi_of_z = Zstruct(**{f"phi_{k}": phi_fns[k](zoz) for k in range(LEVEL + 1)})
    model.momentum.x.apply(Multiply(phi_of_z, outer=True))
    model.momentum.x.apply(ProductRule())
    model.momentum.x.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.momentum.x.apply(kbc).simplify()
    dt_h_relation_node = model.continuity.solve_for(sp.Derivative(H, t))
    dt_h_rel_dict = dict(dt_h_relation_node._node._as_relation)
    model.momentum.x.apply(dt_h_relation_node).simplify()
    basis_alpha = [sp.Function(f"alpha_{k}", real=True)(t, x)
                   for k in range(LEVEL + 1)]

    def _u_ansatz(*args):
        arg = args[-1]
        rhs = basis_alpha[0]
        for k in range(1, LEVEL + 1):
            rhs = rhs + basis_alpha[k] * phi_fns[k]((arg - state.b) / H)
        return rhs

    model.apply(_FieldExpansion(state.u.func, _u_ansatz)).simplify()
    model.apply(IntegralTransform()).simplify()
    model.apply(IsolateBasisIntegrand()).simplify()
    cache = BasisIntegralCache(Legendre_shifted(level=LEVEL))
    model.apply(ProjectBasisIntegrals(cache)).simplify()
    return state, model, dt_h_rel_dict, basis_alpha, phi_fns, cache, _u_ansatz


def has_free_dt_h(expr, h_sym, t_sym):
    for d in expr.atoms(sp.Derivative):
        if d.args[0] == h_sym:
            wrt = []
            for v in d.args[1:]:
                wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
            if t_sym in wrt:
                return True
    return False


def close_bug3_via_primitives(
    expr,
    *,
    state,
    dt_h_relation,
    u_ansatz,
    cache,
    max_iter=8,
):
    """Apply the bug-3 closure via the new primitive layer.

    Iterates: distribute Derivative atoms (so any held
    ``Derivative(α_l·h/k, t)`` materialises ``α_l·∂_t h``), substitute
    via continuity, re-project any fresh ``Derivative(Integral(u),
    x)`` atom that the substitution introduces, canonicalise.

    Stops when no free ``∂_t h`` remains or fixpoint is reached.
    """
    H = state.H
    t = state.t
    x = state.x
    z = state.z
    b = state.b
    zeta_hat = sp.Symbol(r"\hat{\zeta}", positive=True)

    def expand_held_derivatives_w_t(e):
        """Apply product_rule_forward to every Derivative(_, t) atom
        whose inner is a Mul, then distribute Derivative-over-Add on
        the result so all ``α_l·∂_t h`` atoms surface as free terms.
        """
        # First pass: forward product rule on held Mul-inner Derivatives.
        def _walk(node):
            if isinstance(node, sp.Derivative):
                wrt = []
                for v in node.args[1:]:
                    wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
                if t in wrt and isinstance(node.args[0], sp.Mul):
                    return product_rule_forward(node, t)
                if t in wrt and isinstance(node.args[0], sp.Pow):
                    return product_rule_forward(node, t)
            if node.args:
                new_args = tuple(_walk(a) for a in node.args)
                if any(n is not o for n, o in zip(new_args, node.args)):
                    return node.func(*new_args)
            return node
        e = _walk(e)
        # Second pass: distribute over Add (in case product_rule_forward
        # produced an Add inside a held Derivative).
        e = distribute_derivative_over_add(e)
        return e

    for _ in range(max_iter):
        prev = expr
        # Materialise every α_l·∂_t h hidden inside held Derivative atoms.
        expr = expand_held_derivatives_w_t(expr)
        # Substitute via continuity.
        expr = subst(expr, dt_h_relation)
        # Substitution may have introduced fresh
        # ``Derivative(Integral(u), x)`` atoms — re-expand u, affine-map,
        # project.
        expr = function_expand(expr, state.u.func, u_ansatz)
        expr = affine_change_of_variable(expr, z, b, b + H, zeta_hat)
        expr = canonicalize_phi_derivative_subs(expr)
        expr = split_integral_over_add(expr)
        expr = project_basis_integrand(expr, cache)
        expr = canonicalise(expr)
        if expr == prev or not has_free_dt_h(expr, H, t):
            break
    return expr


def main():
    state, model, dt_h_relation, basis_alpha, phi_fns, cache, u_ansatz = (
        build_legacy_state_through_step_13()
    )
    t, x, H = state.t, state.x, state.H

    # Snapshot the post-step-13 test_k expressions.
    test_eqs = {}
    for k in range(LEVEL + 1):
        test_eqs[k] = getattr(model.momentum.x, f"test_{k}").expr

    print("=== Test_1 BEFORE primitive bug-3 closure ===")
    print(f"   ({len(sp.Add.make_args(test_eqs[1]))} additive terms)")

    # Apply bug-3 closure via primitives ONLY where the bug appears.
    #
    # Test_0 (phi_0 = 1) has no chain-rule-on-moving-frame artifact in
    # step 6's ProductRule — phi_0' = 0 — so no held
    # ``Derivative(α_0·h, t)`` atom needs to be opened.  The
    # conservative ``∂_t(α_0·h)`` is a *correct* form there; opening
    # it via the closure would convert it to a non-conservative form
    # and break the K&T-comparable reading.
    #
    # Test_k for k >= 1 carries the chain-rule residual; the closure
    # fires on those.
    for k in range(LEVEL + 1):
        if k == 0:
            test_eqs[k] = canonicalise(test_eqs[k])  # canonicalise only
            continue
        test_eqs[k] = close_bug3_via_primitives(
            test_eqs[k],
            state=state,
            dt_h_relation=dt_h_relation,
            u_ansatz=u_ansatz,
            cache=cache,
        )

    print("\n=== Test_1 AFTER primitive bug-3 closure ===")
    print(f"   ({len(sp.Add.make_args(test_eqs[1]))} additive terms)")

    # K&T comparison cell — same as slim_walkthrough.
    phi_legendre = {
        phi_fns[0](sp.S.Zero): 1, phi_fns[0](sp.S.One): 1,
        phi_fns[1](sp.S.Zero): 1, phi_fns[1](sp.S.One): -1,
    }
    flat_bottom = {
        state.b: 0,
        sp.Derivative(state.b, x): 0,
        sp.Derivative(state.b, t): 0,
    }

    def kt_form(expr):
        return sp.expand(sp.expand(expr).subs(phi_legendre).subs(flat_bottom))

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
                    return step(sp.Add(*[sp.Derivative(a, v, *rest) for a in inner.args]))
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
    dxH = sp.Derivative(H, x)
    probes = [
        ("α_0² · ∂_x h",        a0**2 * dxH),
        ("α_0 · α_1 · ∂_x h",   a0 * a1 * dxH),
        ("α_1² · ∂_x h",        a1**2 * dxH),
        ("α_0 · ∂_x α_0 · h",   a0 * sp.Derivative(a0, x) * H),
        ("α_0 · ∂_x α_1 · h",   a0 * sp.Derivative(a1, x) * H),
        ("α_1 · ∂_x α_0 · h",   a1 * sp.Derivative(a0, x) * H),
        ("α_1 · ∂_x α_1 · h",   a1 * sp.Derivative(a1, x) * H),
        ("∂_t α_0 · h",         sp.Derivative(a0, t) * H),
        ("∂_t α_1 · h",         sp.Derivative(a1, t) * H),
    ]

    print("\n=== K&T COMPARISON (post-primitive-closure) ===\n")
    for k in (0, 1):
        e = test_eqs[k]
        e_kt = kt_form(e)
        e_expanded = expand_derivatives(e_kt)
        factor = 3 if k == 1 else 1
        label = "level-1 (×3)" if k == 1 else "level-0"
        print(f"test_{k} ({label}):")
        for plabel, key in probes:
            c = sp.simplify(factor * e_expanded.coeff(key))
            print(f"  {plabel:<25}  →  {c}")
        print()

    print("=== Pen-and-paper expectation (K&T eq 49) ===")
    print("test_0: ∂_t α_0·h: 1, α_0²·∂_x h: 1 (via flux), "
          "α_1²·∂_x h: 1/3 (via flux), all others 0.")
    print("test_1: ∂_t α_1·h: 1, α_0·∂_x α_1·h: 1, α_1·∂_x α_0·h: 1, "
          "all others 0  ← the Wronskian asymmetry should be GONE.")


if __name__ == "__main__":
    main()
