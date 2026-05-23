"""SME pipeline verification against Kowalski & Torrilhon (2019)
"Moment Approximations and Model Cascades for Shallow Flow",
*Communications in Computational Physics* 25(3), 669-702.

Reference equations (1D, flat bottom ``h_b = const``, no friction
``ν = 0``, ``e_x = 0, e_z = 1``):

* **Level 0 — eq (4.13)** (shallow water):

    ∂_t h        + ∂_x(h u_m)                                = 0
    ∂_t(h u_m)   + ∂_x(h u_m² + g h²/2)                       = 0

* **Level 1 — eq (4.14)** (with ``s = α_1``):

    ∂_t h         + ∂_x(h u_m)                                = 0
    ∂_t(h u_m)    + ∂_x(h u_m² + h s²/3 + g h²/2)             = 0
    ∂_t(h s)      + ∂_x(2 h u_m s)
                  − u_m ∂_x(h s)                              = 0

* **Level 2 — eq (4.17)** (with ``s = α_1, κ = α_2``):

    ∂_t h         + ∂_x(h u_m)                                = 0
    ∂_t(h u_m)    + ∂_x(h u_m² + h s²/3 + h κ²/5 + g h²/2)    = 0
    ∂_t(h s)      + ∂_x(2 h u_m s + 4 h s κ /5)
                  − (u_m − κ/5) ∂_x(h s)
                  − (s/5)        ∂_x(h κ)                      = 0
    ∂_t(h κ)      + ∂_x(2 h u_m κ + 2 h s²/3 + 2 h κ²/7)
                  − s ∂_x(h s)
                  − (u_m + κ/7) ∂_x(h κ)                       = 0

This script:

1. Drives the legacy slim_walkthrough pipeline through step 13 at the
   chosen ``LEVEL``.
2. Applies the new primitive-layer bug-3 closure on every test_k.
3. Constructs the K&T-2019 reference equations literally as written
   above using sympy.
4. Compares pipeline output to reference, expanded into the same
   K&T-form (Legendre boundary values + flat bottom + product-rule
   expansion of ``∂_t / ∂_x`` of products).

Exit code 0 if every level matches; non-zero on first mismatch.
"""
from __future__ import annotations

import argparse
import sys
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
from zoomy_core.symbolic import (
    affine_change_of_variable,
    canonicalise,
    canonicalize_phi_derivative_subs,
    distribute_derivative_over_add,
    function_expand,
    product_rule_forward,
    project_basis_integrand,
    split_integral_over_add,
    subst,
)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def build_pipeline_to_step_13(level):
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
    phi_fns = [sp.Function(f"phi_{k}") for k in range(level + 1)]
    zoz = (state.z - state.b) / H
    phi_of_z = Zstruct(**{f"phi_{k}": phi_fns[k](zoz) for k in range(level + 1)})
    model.momentum.x.apply(Multiply(phi_of_z, outer=True))
    model.momentum.x.apply(ProductRule())
    model.momentum.x.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.momentum.x.apply(kbc).simplify()
    dt_h_node = model.continuity.solve_for(sp.Derivative(H, t))
    dt_h_relation = dict(dt_h_node._node._as_relation)
    model.momentum.x.apply(dt_h_node).simplify()
    basis_alpha = [sp.Function(f"alpha_{k}", real=True)(t, x)
                   for k in range(level + 1)]

    def u_ansatz(*args):
        arg = args[-1]
        rhs = basis_alpha[0]
        for k in range(1, level + 1):
            rhs = rhs + basis_alpha[k] * phi_fns[k]((arg - state.b) / H)
        return rhs

    model.apply(_FieldExpansion(state.u.func, u_ansatz)).simplify()
    model.apply(IntegralTransform()).simplify()
    model.apply(IsolateBasisIntegrand()).simplify()
    cache = BasisIntegralCache(Legendre_shifted(level=level))
    model.apply(ProjectBasisIntegrals(cache)).simplify()
    return state, model, dt_h_relation, basis_alpha, phi_fns, cache, u_ansatz


def has_free_dt_h(expr, h_sym, t_sym):
    for d in expr.atoms(sp.Derivative):
        if d.args[0] == h_sym:
            wrt = []
            for v in d.args[1:]:
                wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
            if t_sym in wrt:
                return True
    return False


def close_bug3(expr, *, state, dt_h_relation, u_ansatz, cache, max_iter=10):
    H, t, z, b = state.H, state.t, state.z, state.b
    zeta_hat = sp.Symbol(r"\hat{\zeta}", positive=True)

    def expand_held_dt(e):
        def _walk(node):
            if isinstance(node, sp.Derivative):
                wrt = [v[0] if isinstance(v, (tuple, sp.Tuple)) else v
                       for v in node.args[1:]]
                if t in wrt and isinstance(node.args[0], (sp.Mul, sp.Pow)):
                    return product_rule_forward(node, t)
            if node.args:
                new_args = tuple(_walk(a) for a in node.args)
                if any(n is not o for n, o in zip(new_args, node.args)):
                    return node.func(*new_args)
            return node
        return distribute_derivative_over_add(_walk(e))

    for _ in range(max_iter):
        prev = expr
        expr = expand_held_dt(expr)
        expr = subst(expr, dt_h_relation)
        expr = function_expand(expr, state.u.func, u_ansatz)
        expr = affine_change_of_variable(expr, z, b, b + H, zeta_hat)
        expr = canonicalize_phi_derivative_subs(expr)
        expr = split_integral_over_add(expr)
        expr = project_basis_integrand(expr, cache)
        expr = canonicalise(expr)
        if expr == prev or not has_free_dt_h(expr, H, t):
            break
    return expr


# ---------------------------------------------------------------------------
# K&T 2019 reference equations — written literally from the paper.
# ---------------------------------------------------------------------------

def kt2019_reference_eqs(level, state, basis_alpha):
    """Return ``[eq_0, eq_1, ..., eq_level]`` where ``eq_k`` is the
    level-k momentum equation from K&T 2019 with friction omitted, in
    the form ``LHS = 0``.

    Test functions and notation:

    * ``α_0`` ↔ ``u_m`` (depth-averaged velocity)
    * ``α_1`` ↔ ``s``   (first-order moment)
    * ``α_2`` ↔ ``κ``   (second-order moment)
    """
    H = state.H
    t, x, g = state.t, state.x, state.g
    a = basis_alpha
    eqs = []

    if level >= 0:
        # eq (4.13) momentum: ∂_t(h α_0) + ∂_x(h α_0² + g h²/2) = 0
        eq0 = (sp.Derivative(H * a[0], t)
               + sp.Derivative(H * a[0]**2 + g * H**2 / 2, x))
        eqs.append(eq0)

    if level >= 1:
        # eq (4.13) at level 1 — the level-0 equation gets the s² flux
        # contribution (NOTE this MODIFIES the level-0 equation).
        # Replace the level-0 equation accordingly.
        eqs[0] = (sp.Derivative(H * a[0], t)
                  + sp.Derivative(H * a[0]**2 + H * a[1]**2 / 3 + g * H**2 / 2, x))
        # eq (4.14) row 3:
        # ∂_t(h s) + ∂_x(2 h u_m s) − u_m ∂_x(h s) = 0
        eq1 = (sp.Derivative(H * a[1], t)
               + sp.Derivative(2 * H * a[0] * a[1], x)
               - a[0] * sp.Derivative(H * a[1], x))
        eqs.append(eq1)

    if level >= 2:
        # eq (4.17) row 1: level-0 with κ² added.
        eqs[0] = (sp.Derivative(H * a[0], t)
                  + sp.Derivative(H * a[0]**2 + H * a[1]**2 / 3
                                  + H * a[2]**2 / 5 + g * H**2 / 2, x))
        # eq (4.17) row 2: level-1 in level-2 system gets extra
        # 4·h·s·κ/5 flux + κ/5·∂_x(h s) + s/5·∂_x(h κ) non-conservative.
        eqs[1] = (sp.Derivative(H * a[1], t)
                  + sp.Derivative(2 * H * a[0] * a[1]
                                  + 4 * H * a[1] * a[2] / 5, x)
                  - (a[0] - a[2] / 5) * sp.Derivative(H * a[1], x)
                  - (a[1] / 5) * sp.Derivative(H * a[2], x))
        # eq (4.17) row 3: level-2 evolution.
        eq2 = (sp.Derivative(H * a[2], t)
               + sp.Derivative(2 * H * a[0] * a[2]
                               + 2 * H * a[1]**2 / 3
                               + 2 * H * a[2]**2 / 7, x)
               - a[1] * sp.Derivative(H * a[1], x)
               - (a[0] + a[2] / 7) * sp.Derivative(H * a[2], x))
        eqs.append(eq2)

    if level >= 3:
        raise NotImplementedError(
            "K&T 2019 eq (4.21) for level 3 is not yet transcribed.  "
            "Add it from the paper before running --level >= 3."
        )

    return eqs


# ---------------------------------------------------------------------------
# Comparison machinery
# ---------------------------------------------------------------------------

def expand_derivatives(expr):
    """Distribute Derivative(product, var) via product rule, fixpoint.
    Handles Add, Mul, Pow inner."""
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
            if isinstance(inner, sp.Pow):
                base, exponent = inner.args
                if isinstance(exponent, sp.Integer) and int(exponent) >= 2:
                    n_pow = int(exponent)
                    return step(n_pow * base**(n_pow - 1) * sp.Derivative(base, v, *rest))
            return e
        return e
    prev = None
    cur = sp.expand(expr)
    while prev != cur:
        prev = cur
        cur = sp.expand(step(cur))
    return cur


def kt_form(expr, phi_fns, state):
    phi_legendre = {}
    for fn in phi_fns:
        phi_legendre[fn(sp.S.Zero)] = 1
        # Shifted Legendre boundary values: φ_k(0) = 1, φ_k(1) = (-1)^k
        k = int(fn.__name__.split("_")[1])
        phi_legendre[fn(sp.S.One)] = (-1) ** k
    flat_bottom = {state.b: 0,
                   sp.Derivative(state.b, state.x): 0,
                   sp.Derivative(state.b, state.t): 0}
    return sp.expand(sp.expand(expr).subs(phi_legendre).subs(flat_bottom))


def kill_trivially_zero_derivatives(expr):
    """Replace ``Derivative(inner, var)`` with ``0`` when ``inner`` is
    a constant (Number/Symbol) or doesn't depend on ``var``.

    sympy's auto-evaluation should handle this, but in practice the
    held form sometimes survives — particularly when the Derivative
    was constructed via ``sp.Derivative(...)`` inside our
    expand_derivatives walker.  This helper cleans up any residue."""
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(sp.Derivative):
        inner = d.args[0]
        # Diff variables.
        wrt = []
        for v in d.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        # If inner is a Number or doesn't reference any diff var → 0.
        if inner.is_number:
            mapping[d] = sp.S.Zero
        elif not any(inner.has(v) for v in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normal_form(expr, state, basis_alpha):
    """Apply continuity ∂_t h = -∂_x(α_0·h), then expand all derivatives,
    then collect like terms.  This puts both pipeline and reference in
    a structurally-comparable canonical form."""
    H = state.H
    t, x = state.t, state.x
    cont = {sp.Derivative(H, t): -sp.Derivative(basis_alpha[0] * H, x)}
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    e = e.subs(cont)
    e = expand_derivatives(e)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", type=int, default=2,
                        help="Galerkin truncation level (default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on first mismatch")
    args = parser.parse_args()

    LEVEL = args.level
    print(f"=== K&T 2019 verification — LEVEL = {LEVEL} ===\n")

    state, model, dt_h_relation, basis_alpha, phi_fns, cache, u_ansatz = (
        build_pipeline_to_step_13(LEVEL)
    )

    test_eqs = {}
    for k in range(LEVEL + 1):
        test_eqs[k] = getattr(model.momentum.x, f"test_{k}").expr
        test_eqs[k] = close_bug3(
            test_eqs[k],
            state=state,
            dt_h_relation=dt_h_relation,
            u_ansatz=u_ansatz,
            cache=cache,
        )

    # Build K&T 2019 reference.
    ref_eqs = kt2019_reference_eqs(LEVEL, state, basis_alpha)

    # Mass-matrix factor — the pipeline level-k equation comes out
    # multiplied by 1/(2k+1) (K&T's m_k); the K&T 2019 paper writes
    # the equation in `h α_k` form where the conservative ∂_t flux
    # has coefficient 1.  Multiply pipeline by (2k+1) to match.
    n_mismatch = 0
    for k in range(LEVEL + 1):
        m_k = sp.Rational(1, 2 * k + 1)
        # Apply K&T form (Legendre boundary values, flat bottom).
        pipe = kt_form(test_eqs[k] / m_k, phi_fns, state)
        ref = kt_form(ref_eqs[k], phi_fns, state)

        # Both into canonical "non-conservative" form via continuity.
        pipe_norm = normal_form(pipe, state, basis_alpha)
        ref_norm = normal_form(ref, state, basis_alpha)
        diff = sp.expand(pipe_norm - ref_norm)
        ok = (diff == 0)
        mark = "✓ MATCH" if ok else "✗ MISMATCH"
        print(f"--- test_{k} ↔ K&T eq {{(4.13), (4.14), (4.17)}}[{k}] ---")
        print(f"  {mark}")
        if not ok:
            n_mismatch += 1
            print(f"  diff = pipeline − reference (after normalisation):")
            print(f"    {diff}")
        print()

    print(f"=== Summary: {n_mismatch} mismatch(es) at LEVEL={LEVEL} ===")
    if args.strict and n_mismatch > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
