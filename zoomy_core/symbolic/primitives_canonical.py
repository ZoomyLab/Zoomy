"""Purely-structural primitive operations.

These primitives implement structural rewrites that preserve
mathematical equality without firing any calculus rule.  They use only
`sympy.Basic.xreplace` and explicit pattern matching on the
expression tree — never `.doit()`, `sp.simplify`, or
`sp.expand` outside of carefully-scoped Mul-over-Add distribution.

Each primitive is a callable `primitive(expr) -> expr`.  Composition
with `zoomy_core.model.models.ins_generator.Operation`-style
wrappers happens elsewhere; here we keep the pure-function
implementations so they can be unit-tested in isolation.

Functions in this module are lifted verbatim from
`zoomy_core.model.models.ins_generator` (see file header for the
specific source-line range each one came from).  No logic changes —
just relocation behind a documented primitive interface.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Derivative, Integral, S, Symbol


__all__ = [
    "alpha_rename",
    "split_integral_over_add",
    "merge_integrals_over_add",
    "distribute_mul_over_add",
    "kill_zero_length_integral",
    "zero_derivative_of_free_symbol",
    "un_subs",
    "drop_zero_integrand",
    "drop_zero_derivative_inner",
    "constant_integrand",
    "protect_integrals",
]


# ---------------------------------------------------------------------------
# protect_integrals — utility used by several primitives
# ---------------------------------------------------------------------------

def protect_integrals(expr, *, also_derivative_of_integral=False):
    """Replace every `Integral` (and optionally
    `Derivative(Integral(...), ...)`) atom with a fresh `Dummy`
    and return `(protected_expr, restore_map)`.

    Restore via `protected_expr.xreplace(restore_map)`.  Used to run
    arithmetic (`sp.expand` Mul-over-Add) on the expression's outer
    structure without sympy fragmenting the integrand.

    Lifted from `_expand_preserve_integrals` (ins_generator.py:4124)
    and the protect-helper inside `_simplify_preserve_integrals`
    (ins_generator.py:4434).
    """
    if not isinstance(expr, sp.Basic):
        return expr, {}
    integral_map: dict[sp.Dummy, sp.Basic] = {}
    counter = [0]

    def _walk(e):
        if also_derivative_of_integral and isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr), integral_map


# ---------------------------------------------------------------------------
# AlphaRename
# ---------------------------------------------------------------------------

def alpha_rename(expr):
    """Rename every `Integral` 's bound variable to a depth-keyed
    canonical Dummy.

    Lifted verbatim from `_canonicalize_integral_dummies`
    (ins_generator.py:4154).  Pure `xreplace`; no math change —
    alpha-equivalence of bound variables is a definitional property
    of the integral.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    canon_by_depth: dict[int, sp.Dummy] = {}

    def _hat_name(old_var):
        # The canonical bound variable is the INPUT variable, hatted:
        # ``∫ … dζ`` → ``\hat{\zeta}``, ``∫ … dz`` → ``\hat{z}``.  Already
        # hatted vars (a fresh running-integral Dummy ``\hat{\zeta}``) are
        # kept verbatim so re-canonicalisation never double-hats.
        nm = getattr(old_var, "name", "")
        if nm.startswith(r"\hat{") and nm.endswith("}"):
            return nm
        return rf"\hat{{{sp.latex(old_var)}}}"

    def _canon(depth, old_var):
        # One Dummy per nesting depth (alpha-equivalence of sibling
        # integrals is preserved); the NAME follows the input variable.
        if depth not in canon_by_depth:
            canon_by_depth[depth] = sp.Dummy(_hat_name(old_var), positive=True)
        return canon_by_depth[depth]

    def _walk(e, depth=0):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0], depth=depth + 1)
            limits = e.args[1]
            if hasattr(limits, "__len__") and len(limits) == 3:
                old_var, lo, hi = limits
                new_var = _canon(depth, old_var)
                if old_var is new_var:
                    if integrand is not e.args[0]:
                        return Integral(integrand, *e.args[1:])
                    return e
                new_int = integrand.xreplace({old_var: new_var})
                new_lo = (lo.xreplace({old_var: new_var})
                          if isinstance(lo, sp.Basic) else lo)
                new_hi = (hi.xreplace({old_var: new_var})
                          if isinstance(hi, sp.Basic) else hi)
                return Integral(new_int, (new_var, new_lo, new_hi))
            if integrand is not e.args[0]:
                return Integral(integrand, *e.args[1:])
            return e
        if isinstance(e, Derivative):
            inner = _walk(e.args[0], depth=depth)
            if inner is not e.args[0]:
                return Derivative(inner, *e.args[1:])
            return e
        if e.args:
            new_args = tuple(_walk(a, depth=depth) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


# ---------------------------------------------------------------------------
# DistributeMulOverAdd
# ---------------------------------------------------------------------------

def distribute_mul_over_add(expr):
    """`c·(a+b) → c·a + c·b` — ring-axiom distribution.

    Uses `sp.expand` with all non-Mul branches disabled so the only
    rewrite that fires is Mul-over-Add distribution.  Integrals are
    protected during the expansion so their integrands stay intact
    (sympy's `expand` would otherwise fragment `Integral(f+g, lim)`
    into `Integral(f, lim) + Integral(g, lim)` — that's a separate
    primitive, `split_integral_over_add`).

    Lifted from `_expand_preserve_integrals` (ins_generator.py:4124).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    protected, restore_map = protect_integrals(expr)
    expanded = sp.expand(
        protected,
        mul=True,
        multinomial=False,
        power_base=False,
        power_exp=False,
        log=False,
        complex=False,
        trig=False,
        basic=False,
    )
    return expanded.xreplace(restore_map)


# ---------------------------------------------------------------------------
# SplitIntegralOverAdd
# ---------------------------------------------------------------------------

def split_integral_over_add(expr):
    """`∫(f+g) dv → ∫f dv + ∫g dv` — linearity of the integral.

    Distributes through `Derivative(Integral(...), v)` by linearity
    of the derivative.  Outer multiplicative factors stay attached.

    Lifted from `_split_integrals_expr` (ins_generator.py:4217).
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0])
            limits = e.args[1:]
            expanded = distribute_mul_over_add(integrand)
            terms = Add.make_args(expanded)
            if len(terms) <= 1:
                if integrand is not e.args[0]:
                    return Integral(integrand, *limits)
                return e
            return Add(*[Integral(t, *limits) for t in terms])
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            inner = _walk(e.args[0])
            wrt = e.args[1:]
            if isinstance(inner, Add):
                return Add(*[Derivative(a, *wrt) for a in inner.args])
            if inner is not e.args[0]:
                return Derivative(inner, *wrt)
            return e
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return distribute_mul_over_add(_walk(expr))


# ---------------------------------------------------------------------------
# MergeIntegralsOverAdd
# ---------------------------------------------------------------------------

def merge_integrals_over_add(expr):
    """Combine sibling Integrals with matching `(limits, deriv-wrapper)`
    signature into a single `Integral(Σ c_i · f_i, lim)`.

    Inverse of `split_integral_over_add`.  An outer factor moves
    inside the integral only if it doesn't depend on the integration
    variable — otherwise the factor stays out (passthrough); moving it
    inside would change the math.

    Lifted from `_merge_integrals_expr` (ins_generator.py:4263).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    expr = alpha_rename(expr)

    def _classify(term):
        coeff_factors = []
        target = term
        if isinstance(term, sp.Mul):
            cands = []
            for f in term.args:
                if isinstance(f, Integral):
                    cands.append(f)
                elif isinstance(f, Derivative) and isinstance(f.args[0], Integral):
                    cands.append(f)
                else:
                    coeff_factors.append(f)
            if len(cands) != 1:
                return None
            target = cands[0]
        elif isinstance(term, Integral):
            pass
        elif isinstance(term, Derivative) and isinstance(term.args[0], Integral):
            pass
        else:
            return None
        if isinstance(target, Derivative):
            inner = target.args[0]
            deriv_wrt = tuple(target.args[1:])
        else:
            inner = target
            deriv_wrt = ()
        if not isinstance(inner, Integral):
            return None
        limits = inner.args[1]
        if not (hasattr(limits, "__len__") and len(limits) == 3):
            return None
        var = limits[0]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else S.One
        if coeff.has(var):
            return None
        sig = (limits, deriv_wrt)
        return sig, coeff, inner.args[0], limits, deriv_wrt

    def _walk(e):
        if isinstance(e, Add):
            new_args = [_walk(a) for a in e.args]
            groups: dict = {}
            order = []
            passthrough = []
            for a in new_args:
                cls = _classify(a)
                if cls is None:
                    passthrough.append(a)
                    continue
                sig, coeff, integrand, limits, deriv_wrt = cls
                if sig not in groups:
                    groups[sig] = (limits, deriv_wrt, S.Zero)
                    order.append(sig)
                lim, dwrt, acc = groups[sig]
                groups[sig] = (lim, dwrt, acc + coeff * integrand)
            merged = []
            for sig in order:
                lim, dwrt, integrand_sum = groups[sig]
                # ``sp.expand`` here is restricted to Mul-over-Add inside
                # the integrand, which is pure ring distribution — safe.
                integrand_sum = distribute_mul_over_add(integrand_sum)
                if integrand_sum == 0:
                    continue
                term = Integral(integrand_sum, lim)
                if dwrt:
                    term = Derivative(term, *dwrt)
                merged.append(term)
            return Add(*merged, *passthrough)
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


# ---------------------------------------------------------------------------
# KillZeroLengthIntegral — STRUCTURAL ONLY (no sp.simplify)
# ---------------------------------------------------------------------------

def kill_zero_length_integral(expr):
    """`Integral(_, (var, a, a)) → 0` — structural equality only.

    This is a tightened version of the legacy
    `_kill_zero_length_integrals` (ins_generator.py:4366) which fell
    back to `sp.simplify(lo - hi) == 0`.  `sp.simplify` can fire
    arbitrary rules on the bounds — for the redesign we accept only
    structural equality (`lo == hi` or `lo - hi` reduces to
    `S.Zero` via sympy's `Add` canonicalisation alone, no
    further-simplification pass).

    Callers who want a deeper bound match must apply a `Subst` rule
    that makes the bounds structurally equal first.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for I in expr.atoms(Integral):
        limits = I.args[1]
        if hasattr(limits, "__len__") and len(limits) == 3:
            _, lo, hi = limits
            if lo == hi:
                mapping[I] = S.Zero
                continue
            # Add canonicalisation: if lo - hi reduces to plain 0
            # without invoking simplify, mark the integral as zero.
            diff = lo - hi
            if diff == 0 or diff is S.Zero:
                mapping[I] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


# ---------------------------------------------------------------------------
# ZeroDerivativeOfFreeSymbol
# ---------------------------------------------------------------------------

def zero_derivative_of_free_symbol(expr):
    """`∂_v sym → 0` when `sym` is a free `Symbol` distinct from
    every differentiation variable.

    These appear as Leibniz boundary residuals (`∂_x z` when `z`
    is a coordinate, not a function).  Sympy doesn't auto-reduce.

    Lifted verbatim from `_kill_free_derivatives` (ins_generator.py:4387).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(Derivative):
        inner = d.args[0]
        if not isinstance(inner, Symbol):
            continue
        variables = []
        for v in d.args[1:]:
            # ``Derivative.args[1:]`` is a tuple of either ``sp.Tuple(var,
            # order)`` or bare ``Symbol`` entries.  Both expose
            # ``__getitem__``; sp.Tuple is *not* a Python ``tuple``.
            if isinstance(v, (tuple, sp.Tuple)):
                variables.append(v[0])
            else:
                variables.append(v)
        if all(isinstance(v, Symbol) for v in variables) and all(v != inner for v in variables):
            mapping[d] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


# ---------------------------------------------------------------------------
# UnSubs
# ---------------------------------------------------------------------------

def un_subs(expr):
    """Unwrap every `Subs(f, var, val)` whose inner is safe to
    `xreplace(var → val)`.

    Safety = none of:

    * `var` appears as a `Derivative` differentiation variable in
      `f` (would commit to chain-rule-through-`val` ambiguity).
    * `var` is the integration variable of a nested `Integral` in
      `f` (would shadow the binder and produce nonsense limits).
    * `var` is the binding variable of a nested `Subs` in `f`
      (same shadowing concern).

    Lifted verbatim from `_resolve_subs_safe` (ins_generator.py:4539).
    The redesign promotes this to a primitive — the previously
    implicit calls (inside `Expression.apply`, `EvaluateIntegrals`,
    `ProjectBasisIntegrals`) all become explicit user steps.
    """
    if not isinstance(expr, sp.Basic) or not expr.has(sp.Subs):
        return expr
    mapping = {}
    for s in expr.atoms(sp.Subs):
        inner = s.args[0]
        vars_tup = s.args[1]
        vals_tup = s.args[2]
        deriv_vars = set()
        for d in inner.atoms(sp.Derivative):
            deriv_vars.update(d.variables)
        if any(v in deriv_vars for v in vars_tup):
            continue
        bound_vars = set()
        for I in inner.atoms(sp.Integral):
            for lim in I.args[1:]:
                if hasattr(lim, "__getitem__") and len(lim) >= 1:
                    bound_vars.add(lim[0])
        for nested in inner.atoms(sp.Subs):
            bound_vars.update(nested.args[1])
        if any(v in bound_vars for v in vars_tup):
            continue
        mapping[s] = inner.xreplace(dict(zip(vars_tup, vals_tup)))
    return expr.xreplace(mapping) if mapping else expr


# ---------------------------------------------------------------------------
# Trivial-zero rules — ∫0 dv = 0, ∂_v 0 = 0
# ---------------------------------------------------------------------------

def drop_zero_integrand(expr):
    """`Integral(0, …) → 0` — structural.

    Lifted from the `∫0` branch of `SimplifyIntegrals._leaf_sp`
    (ins_generator.py:2371) and `IsolateBasisIntegrand`
    (ins_generator.py:2698).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for I in expr.atoms(Integral):
        if I.args[0] == 0 or I.args[0] is S.Zero:
            mapping[I] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


def drop_zero_derivative_inner(expr):
    """`Derivative(0, …) → 0` — structural.

    Lifted from the `∂(0)` branch of `_evaluate_linear_derivatives`
    (ins_generator.py:4621).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(Derivative):
        if d.args[0] == 0 or d.args[0] is S.Zero:
            mapping[d] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


def constant_integrand(expr):
    """`∫_a^b c dvar → c·(b − a)` when integrand has no `var`.

    Lifted from the constant-integrand branch of
    `SimplifyIntegrals._leaf_sp` (ins_generator.py:2373) and
    `IsolateBasisIntegrand` (ins_generator.py:2698).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for I in expr.atoms(Integral):
        limits = I.args[1]
        if not (hasattr(limits, "__len__") and len(limits) == 3):
            continue
        var, lo, hi = limits
        if not I.args[0].has(var):
            mapping[I] = I.args[0] * (hi - lo)
    return expr.xreplace(mapping) if mapping else expr
