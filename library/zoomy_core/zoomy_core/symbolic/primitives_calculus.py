"""Calculus-rule primitives.

Each function in this module implements ONE textbook calculus rule
that fires only when its structural pattern matches.  The rules
**never** call `.doit()`, `sp.simplify`, `sp.cancel`,
`sp.together`, or `sp.factor`.  Sympy is used only for:

* `sp.expand` (Mul-over-Add only, via
  `zoomy_core.symbolic.primitives_canonical.distribute_mul_over_add`),
* `sp.Poly(...).integrate()` for finite polynomial antiderivatives,
* `Add`/`Mul`/`Pow` constructors (commutative-ring arithmetic),
* `xreplace` for purely-structural substitution.

Refusal semantics: when the input contains no atom matching the
primitive's pattern, the function returns the input unchanged
(silent no-op — preserving the equation but doing nothing).  Use
`Expression.apply_to_term(i, primitive)` for explicit
single-term targeting.

Functions are lifted from
`zoomy_core.model.models.ins_generator` with two surgical
changes:

* `ProductRuleInverse` no longer calls ``Derivative(coeff,
  var).doit()`` on the residual (legacy bug source —
  `ins_generator.py:3402`); the residual is left as a held
  `Derivative` atom which Canonicalise reduces structurally.
* `Leibniz` no longer runs the inner `_doit_only_derivatives`
  walk (legacy bug source — `ins_generator.py:2026`); the caller
  must apply `DistributeDerivativeOverAdd` first if they want
  `var`-factor surfacing.
* `PolynomialIntegrate` no longer runs `_doit_derivs`
  (`ins_generator.py:2078`).  Same reason.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Derivative, Integral, Mul, S, Symbol

from zoomy_core.symbolic.errors import PrimitiveDoesNotMatch
from zoomy_core.symbolic.primitives_canonical import distribute_mul_over_add


__all__ = [
    "fundamental_theorem",
    "leibniz",
    "leibniz_general",
    "polynomial_integrate",
    "integration_by_parts",
    "product_rule_forward",
    "product_rule_inverse",
    "chain_rule",
    "distribute_derivative_over_add",
    "pull_scalar_out_of_derivative",
    "push_scalar_into_derivative",
]


# ---------------------------------------------------------------------------
# FundamentalTheorem
# ---------------------------------------------------------------------------

def fundamental_theorem(integrand, var, lower, upper):
    """`∫_lo^hi ∂_var g dvar = g|_hi − g|_lo`.

    Lifted verbatim from `_rule_fundamental_theorem`
    (ins_generator.py:1934).  Handles single-order `∂_var g` and
    higher-order-in-`var` (one `var` peels off).  Mixed-order
    `∂_var ∂_other g` strips one `var` from the diff-tuple.

    Returns `None` if the integrand isn't a `Derivative` whose
    diff-vars include `var` (caller convention preserved from
    ins_generator's rule-registry: `None` = "rule does not apply",
    not an exception).
    """
    if not isinstance(integrand, Derivative):
        return None
    diff_tuples = list(integrand.args[1:])
    match_index = None
    for i, dv in enumerate(diff_tuples):
        if len(dv) == 2 and dv[0] == var:
            match_index = i
            break
    if match_index is None:
        return None
    matched = diff_tuples[match_index]
    order = matched[1]
    if order <= 1:
        remaining = diff_tuples[:match_index] + diff_tuples[match_index + 1:]
    else:
        remaining = (diff_tuples[:match_index]
                     + [sp.Tuple(var, order - 1)]
                     + diff_tuples[match_index + 1:])
    inner = integrand.args[0]
    if remaining:
        antideriv = Derivative(inner, *remaining)
    else:
        antideriv = inner
    return antideriv.subs(var, upper) - antideriv.subs(var, lower)


# ---------------------------------------------------------------------------
# Leibniz
# ---------------------------------------------------------------------------

def leibniz(integrand, var, lower, upper):
    """`∫_lo^hi ∂_y f dvar = ∂_y(∫_lo^hi f dvar) − f(hi)·∂_y hi + f(lo)·∂_y lo`
    when `y ≠ var` and `f` is polynomial in `var`.

    Lifted from `_rule_derivative_of_polynomial` (ins_generator.py:1976)
    with the inner `_doit_only_derivatives` walk **removed**.  If
    the caller wants `var`-factors to surface from
    `Derivative(var·g, y)` etc. they must run
    `distribute_derivative_over_add` first.

    Returns `None` when:

    * integrand is not a `Derivative`,
    * diff-tuple has more than one entry,
    * differentiation is w.r.t. `var` itself
      (`fundamental_theorem` handles that),
    * the integrand is independent of `var`,
    * `sp.Poly` cannot build a polynomial in `var`.
    """
    if not isinstance(integrand, Derivative):
        return None
    diff_tuples = list(integrand.args[1:])
    if len(diff_tuples) != 1:
        return None
    dv = diff_tuples[0]
    if len(dv) != 2 or dv[1] != 1:
        return None
    y = dv[0]
    if y == var:
        return None
    inner = integrand.args[0]
    if not inner.has(var):
        return None
    inner = distribute_mul_over_add(inner)
    try:
        sp.Poly(inner, var)
    except (sp.PolynomialError, sp.CoercionFailed, sp.GeneratorsNeeded):
        return None
    inner_integrated = sp.Poly(inner, var).integrate().as_expr()
    upper = sp.sympify(upper)
    lower = sp.sympify(lower)
    inner_integrated = inner_integrated.subs(var, upper) - inner_integrated.subs(var, lower)
    main = Derivative(inner_integrated, y)
    surface = S.Zero
    if upper.has(y):
        surface -= inner.subs(var, upper) * Derivative(upper, y)
    if lower.has(y):
        surface += inner.subs(var, lower) * Derivative(lower, y)
    return main + surface


def leibniz_general(integrand, var, lower, upper):
    """Conservative-form Leibniz that does NOT require `f` to be
    polynomial in `var`.

    For `∫_lo^hi ∂_y f dvar` with `y ≠ var`:

    ```
        = ∂_y[ ∫_lo^hi f dvar ]   −  f|_hi · ∂_y hi   +   f|_lo · ∂_y lo
    ```

    The volume `∫f dvar` stays as a held `sympy.Integral`
    atom — no attempt to evaluate it.  Boundary terms are produced as
    held `sympy.Subs` atoms when the bounds depend on `y`.

    Use this when the integrand's antiderivative is unknown or
    opaque; use `leibniz` (poly-specialised) when `f` is
    polynomial in `var` and you want the closed-form volume.

    Returns `None` if:

    * integrand is not a `Derivative`,
    * diff-tuple has more than one entry,
    * differentiation is w.r.t. `var` itself
      (`fundamental_theorem` handles that).
    """
    if not isinstance(integrand, Derivative):
        return None
    diff_tuples = list(integrand.args[1:])
    if len(diff_tuples) != 1:
        return None
    dv = diff_tuples[0]
    if len(dv) != 2 or dv[1] != 1:
        return None
    y = dv[0]
    if y == var:
        return None
    inner = integrand.args[0]
    upper = sp.sympify(upper)
    lower = sp.sympify(lower)
    main = Derivative(Integral(inner, (var, lower, upper)), y)
    surface = S.Zero
    if upper.has(y):
        surface -= inner.subs(var, upper) * Derivative(upper, y)
    if lower.has(y):
        surface += inner.subs(var, lower) * Derivative(lower, y)
    return main + surface


# ---------------------------------------------------------------------------
# PolynomialIntegrate
# ---------------------------------------------------------------------------

def polynomial_integrate(integrand, var, lower, upper):
    """`∫_lo^hi p(var) dvar` via `sp.Poly.integrate` for polynomial
    `p` (with arbitrary opaque coefficients).

    Lifted from `_rule_polynomial_integrand` (ins_generator.py:2061)
    with the inner `_doit_derivs` walk **removed**.  If the caller
    wants `var`-factors to surface from `Derivative(var·g, y)`
    etc. they must run `distribute_derivative_over_add` first.

    Returns `(hi − lo) · integrand` for `var`-free integrands
    (constant integrand rule).  Returns `None` if `sp.Poly` cannot
    build a polynomial.
    """
    expr = distribute_mul_over_add(integrand)
    if not expr.has(var):
        return (upper - lower) * expr
    try:
        poly = sp.Poly(expr, var)
    except (sp.PolynomialError, sp.CoercionFailed, sp.GeneratorsNeeded):
        return None
    anti = poly.integrate().as_expr()
    return anti.subs(var, upper) - anti.subs(var, lower)


# ---------------------------------------------------------------------------
# IntegrationByParts
# ---------------------------------------------------------------------------

def integration_by_parts(f, g, var, lower, upper):
    """`∫_lo^hi (∂_var f)·g dvar = [f·g]_lo^hi − ∫_lo^hi f·∂_var g dvar`.

    Returns a 3-tuple ``(volume_integrand, boundary_upper,
    boundary_lower)`` so the caller can place each piece in the
    appropriate term tag.  `volume_integrand` already carries the
    `-` sign and is wrapped in a ``Integral(..., (var, lower,
    upper))``.

    Lifted from `integrate_by_parts` (ins_generator.py:4757), with
    the `IBPResult` wrapper unpacked since it lives in the legacy
    layer.
    """
    volume = -Integral(f * Derivative(g, var), (var, lower, upper))
    boundary_upper = (f * g).subs(var, upper)
    boundary_lower = (f * g).subs(var, lower)
    return volume, boundary_upper, boundary_lower


# ---------------------------------------------------------------------------
# ProductRule (forward + inverse, separated)
# ---------------------------------------------------------------------------

def product_rule_forward(term, var):
    """`∂_v(Π fᵢ) → Σᵢ (Π_{j≠i} fⱼ)·∂_v fᵢ` and
    `∂_v(f^n) → n·f^(n-1)·∂_v f` for integer `n ≥ 2`.

    Acts on a single `Derivative` atom.  If `term` is not a
    `Derivative`, returns `term` unchanged (no-op).  If the
    derivative has multiple variables or its inner is neither a Mul
    nor an integer-power Pow, returns `term` unchanged.

    Lifted from the forward branch of `ProductRule._one_term`
    (ins_generator.py:3358).  No `.doit()` involved — every output
    factor is a held `Derivative`.
    """
    if not isinstance(term, Derivative) or len(term.variables) != 1:
        return term
    if term.variables[0] != var:
        return term
    inner = term.args[0]
    if isinstance(inner, Mul) and len(inner.args) >= 2:
        factors = inner.args
        return sum(
            (Mul(*(factors[j] for j in range(len(factors)) if j != i))
             * Derivative(factors[i], var)
             for i in range(len(factors))),
            S.Zero,
        )
    if isinstance(inner, sp.Pow):
        base, exp = inner.args
        if isinstance(exp, sp.Integer) and int(exp) >= 2:
            n = int(exp)
            return n * base**(n - 1) * Derivative(base, var)
    return term


def product_rule_inverse(term, var):
    """`coeff · ∂_v f → ∂_v(coeff · f) − ∂_v(coeff) · f`.

    The legacy version (`ProductRule._one_term` inverse branch,
    ins_generator.py:3402) called `Derivative(coeff, var).doit()` on
    the residual.  That `.doit()` was the exact bug-3 source: when
    `coeff` is e.g. `φ_1((z-b)/h)`, `.doit()` fires the chain
    rule against the moving frame at exactly the wrong pipeline
    position.

    The redesigned primitive leaves the residual as a held
    `Derivative(coeff, var)` atom.  Canonicalise will reduce it to
    zero when `coeff` is genuinely `var`-free, otherwise the
    user calls `chain_rule` or `product_rule_forward`
    explicitly to expand it.
    """
    if not isinstance(term, Mul):
        return term
    factors = list(term.args)
    derivs = [f for f in factors if isinstance(f, Derivative)]
    if len(derivs) != 1:
        return term
    d = derivs[0]
    if len(d.variables) != 1 or d.variables[0] != var:
        return term
    inner = d.args[0]
    coeff_factors = [f for f in factors if f is not d]
    coeff = Mul(*coeff_factors) if coeff_factors else S.One
    return Derivative(coeff * inner, var) - Derivative(coeff, var) * inner


# ---------------------------------------------------------------------------
# ChainRule — the only blessed automatic chain-rule call
# ---------------------------------------------------------------------------

def chain_rule(term, outer_predicate, var):
    """`∂_v f(g(v)) → f'(g)·∂_v g` where `outer_predicate(f) → bool`
    selects the function families to expand.

    Walks the expression and applies the chain rule wherever the
    pattern matches.  `outer_predicate` is a callable taking a
    `Function` instance (the call site, e.g. `phi_1((z-b)/h)`)
    and returning `True` if this primitive should chain-rule it.

    For example, to chain-rule every `phi_*` family call:

    ```
        def is_basis(call):
            name = getattr(call.func, '__name__', '')
            return name.startswith('phi_')

        chain_rule(expr, is_basis, var=x)
    ```

    Implementation: walks `Derivative` atoms whose inner is a
    matching function call; replaces with the explicit chain-rule
    expansion using held `Derivative` atoms.  No `.doit()` —
    other `Derivative` atoms are not touched.
    """
    if not isinstance(term, sp.Basic):
        return term

    def _walk(e):
        if isinstance(e, Derivative) and len(e.variables) == 1 and e.variables[0] == var:
            inner = e.args[0]
            if isinstance(inner, sp.Function) and outer_predicate(inner):
                if len(inner.args) == 1:
                    g = inner.args[0]
                    # Build f'(g) as Subs(Derivative(f(ξ), ξ), ξ, g):
                    # sympy can't differentiate w.r.t. a Mul, so we
                    # introduce a fresh dummy to hold the
                    # ``f-prime-evaluated-at-g`` shape.
                    xi = sp.Dummy("xi", real=True)
                    fprime_at_g = sp.Subs(
                        Derivative(inner.func(xi), xi), xi, g
                    )
                    return fprime_at_g * Derivative(g, var)
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(term)


# ---------------------------------------------------------------------------
# Linearity of derivative
# ---------------------------------------------------------------------------

def distribute_derivative_over_add(expr):
    """`∂_v Add(t1, t2) → Add(∂_v t1, ∂_v t2)` — linearity.

    Hoisted from the Add branch of `_evaluate_linear_derivatives`
    (ins_generator.py:4651).  Distinguishes:

    * Inner is structurally an `Add` → distribute outer Derivative.
    * Inner is a `Mul` whose `as_coeff_Mul` separates a numeric
      scalar → pull it out (separate primitive
      `pull_scalar_out_of_derivative` does that; this one only
      does Add-distribution).
    * Trivially zero (`inner == 0` or inner doesn't reference any
      diff-var) → returns 0.

    No `.doit()` — distributed Derivative atoms are held.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, sp.Subs) and expr.args[0] == S.Zero:
        return S.Zero
    if isinstance(expr, Derivative):
        inner = expr.args[0]
        if inner == S.Zero:
            return S.Zero
        # Trivially zero: inner doesn't reference any diff-var.
        if not any(inner.has(v) for v in expr.variables):
            return S.Zero
        if isinstance(inner, sp.Add):
            inner_walked = distribute_derivative_over_add(inner)
            args_rest = expr.args[1:]
            return sp.Add(*(expr.func(t, *args_rest)
                            for t in sp.Add.make_args(inner_walked)))
        # Inner not an Add — recurse but don't synthesise a new derivative.
        inner_walked = distribute_derivative_over_add(inner)
        if inner_walked is inner:
            return expr
        return expr.func(inner_walked, *expr.args[1:])
    if expr.args:
        return expr.func(
            *(distribute_derivative_over_add(a) for a in expr.args)
        )
    return expr


def pull_scalar_out_of_derivative(expr):
    """`∂_v(c·f) → c·∂_v f` when `c` is a sympy `Number`
    (`Integer`/`Rational`/`Float`).

    Hoisted from the `Mul.as_coeff_Mul` branch of
    `_evaluate_linear_derivatives` (ins_generator.py:4644).
    Numeric-only — free symbols stay inside.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, Derivative):
        inner = expr.args[0]
        if isinstance(inner, sp.Mul):
            c, rest = inner.as_coeff_Mul()
            if c != S.One:
                walked = pull_scalar_out_of_derivative(
                    expr.func(rest, *expr.args[1:])
                )
                return c * walked
        inner_walked = pull_scalar_out_of_derivative(inner)
        if inner_walked is inner:
            return expr
        return expr.func(inner_walked, *expr.args[1:])
    if expr.args:
        return expr.func(
            *(pull_scalar_out_of_derivative(a) for a in expr.args)
        )
    return expr


def push_scalar_into_derivative(expr):
    """`c · ∂_v f → ∂_v(c · f)` when `c` is a sympy `Number`.

    Inverse of `pull_scalar_out_of_derivative`.  Lifted verbatim
    from `_fold_numeric_coeffs_into_derivative` (ins_generator.py:4674).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, sp.Mul):
        walked_args = [push_scalar_into_derivative(a) for a in expr.args]
        derivs = [a for a in walked_args if isinstance(a, Derivative)]
        if len(derivs) == 1:
            deriv = derivs[0]
            coeff = sp.Mul(*[a for a in walked_args if a is not deriv])
            if coeff.is_number and coeff != S.One:
                return Derivative(coeff * deriv.args[0], *deriv.args[1:])
        if any(n is not o for n, o in zip(walked_args, expr.args)):
            return expr.func(*walked_args)
        return expr
    if expr.args:
        walked_args = tuple(push_scalar_into_derivative(a) for a in expr.args)
        if any(n is not o for n, o in zip(walked_args, expr.args)):
            return expr.func(*walked_args)
    return expr
