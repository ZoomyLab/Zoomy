"""Galerkin-projection operations on the modal ansatz: sum expansion, test-function projection,
bracket extraction and basis resolution. They carry a Model derivation from the unexpanded ansatz to
basis-resolved moment equations.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import (
    Operation, Multiply, is_bracket, is_bracket_body)


__all__ = [
    "Gram",
    "Weight",
    "is_bracket",
    "is_bracket_body",
    "bracket_atoms",
    "ExpandSums",
    "EvaluateSums",
    "Integrate",
    "Project",
    "PullConstants",
    "pull_out",
    "ExtractBrackets",
    "ResolveBasis",
]


# ── named bracket Function classes (closed-form-able) ──────────────────────


def _bracket_tex(name):
    """LaTeX renderer for the named bracket ``Gram`` or ``Weight``."""
    return {
        "Gram":   lambda A: rf"\langle \phi_{{{A[0]}}}, c\phi_{{{A[1]}}}\rangle",
        "Weight": lambda A: rf"\langle c, \phi_{{{A[0]}}}\rangle",
    }[name]


def _mk_bracket(name):
    """Build the sympy Function class for a named bracket (``Gram`` or ``Weight``)."""
    body = _bracket_tex(name)

    def _latex(expr, printer=None, exp=None):
        """LaTeX printer of the bracket as an inner product."""
        render = printer._print if printer is not None else sp.latex
        A = [str(render(a)) for a in expr.args]
        s = body(A)
        return s if exp is None else f"{s}^{{{exp}}}"

    return type(name, (sp.Function,), {"_latex": _latex, "_is_bracket": True})


# ⟨φ_i, c φ_l⟩ and ⟨c, φ_l⟩ — the only shapes with a closed form for an
# orthogonal basis (everything else stays an explicit ⟨…⟩ Integral).
Gram = _mk_bracket("Gram")
Weight = _mk_bracket("Weight")


def bracket_atoms(expr):
    """All named-bracket (``Gram``/``Weight``) atoms present in ``expr``."""
    return {a for a in expr.atoms(sp.Function)
            if getattr(a.func, "_is_bracket", False)}


# ── ExpandSums ─────────────────────────────────────────────────────────────


# Mode-index letters, in order (``l`` is the Galerkin test index — skipped).
_MODE_LETTERS = ["i", "j", "k", "m", "n", "o", "p", "q", "r", "s"]


def _fresh_index(idx, used):
    """A fresh summation index from the pool ``i, j, k, m, ...`` that is not in ``used``."""
    for nm in _MODE_LETTERS:
        cand = sp.Symbol(nm, integer=True, nonnegative=True)
        if cand not in used:
            return cand
    base = idx.name if isinstance(idx, sp.Symbol) else "i"
    n = 1
    while True:
        cand = sp.Symbol(f"{base}_{n}", integer=True, nonnegative=True)
        if cand not in used:
            return cand
        n += 1


def _merge_sums(sums):
    """Merge ``Sum`` factors into one multi-index Sum with distinct indices, so cross terms ``i !=
    j`` survive.
    """
    used = set()
    for s in sums:
        used |= set(s.free_symbols)
    claimed, summands, limits = set(), [], []
    for s in sums:
        repl, new_limits = {}, []
        for lim in s.limits:
            idx = lim[0]
            new = idx if idx not in claimed else _fresh_index(idx, used | claimed)
            claimed.add(new)
            used.add(new)
            repl[idx] = new
            new_limits.append((new,) + tuple(lim[1:]))
        summands.append(s.function.xreplace(repl))
        limits.extend(new_limits)
    return sp.Sum(sp.Mul(*summands), *limits)


def _expand_sum_products(e):
    """Rewrite products and positive integer powers of ``Sum`` into one multi-index Sum,
    recursively.
    """
    if not getattr(e, "args", None):
        return e
    e = e.func(*[_expand_sum_products(a) for a in e.args])
    if (isinstance(e, sp.Pow) and isinstance(e.base, sp.Sum)
            and e.exp.is_Integer and e.exp > 0):
        return _merge_sums([e.base] * int(e.exp))
    if isinstance(e, sp.Mul):
        sums, rest = [], []
        for f in e.args:
            if isinstance(f, sp.Sum):
                sums.append(f)
            elif (isinstance(f, sp.Pow) and isinstance(f.base, sp.Sum)
                  and f.exp.is_Integer and f.exp > 0):
                sums.extend([f.base] * int(f.exp))
            else:
                rest.append(f)
        if len(sums) >= 2:
            return sp.Mul(*rest) * _merge_sums(sums)
    return e


class ExpandSums(Operation):
    """Merge products and integer powers of unexpanded ``Sum`` factors into one multi-index ``Sum``
    with distinct dummies. Apply after the ansatz substitution and before :class:`Project`.
    """

    whole_leaf_op = True

    def __init__(self, name="expand_sums"):
        """Create the operation."""
        super().__init__(
            name=name,
            description="products/powers of Sum → one multi-index Sum",
        )

    def _leaf_sp(self, sp_expr):
        """Expand the sum products of one leaf expression."""
        return _expand_sum_products(sp_expr)


class EvaluateSums(Operation):
    """Unroll every finite ``Sum`` (concrete integer bound) to explicit modes, leaving ``Integral``
    and ``Derivative`` deferred. Narrower than a bare ``doit()``.
    """

    whole_leaf_op = True

    def __init__(self, name="evaluate_sums"):
        """Create the operation."""
        super().__init__(
            name=name,
            description="finite Sum.doit() → explicit modes",
        )

    def _leaf_sp(self, sp_expr):
        """Unroll finite sums with ``doit(deep=False)``, so inner integrals are not handed to sympy."""
        # ``deep=False`` is essential: a bare ``Sum.doit()`` first calls
        # ``self.function.doit()``, which descends into the opaque Galerkin
        # ``Integral(c φ_i φ_j φ_k …)`` and hands it to sympy's ``heurisch``
        # symbolic integrator — that recurses without bound on the unevaluated
        # ``φ`` heads and hangs.  We only want the SUMMATION unrolled to explicit
        # modes; the inner ``Integral`` / ``Derivative`` stay deferred (ResolveBasis
        # evaluates the brackets later, by antiderivative, not ``heurisch``).
        return sp_expr.replace(lambda e: isinstance(e, sp.Sum),
                               lambda e: e.doit(deep=False))


# ── Project (Multiply(test) → ∫ dζ through the Sum) ────────────────────────


def _integrate_one_term(term, var, lo, hi):
    """Integrate one additive term over ``(var, lo, hi)``: fundamental theorem for a bare ``var``
    derivative, commute a bare conservative ``d_w`` out, push through a single ``Sum``, else leave
    an opaque ``Integral``. The commute applies only to a bare outer derivative, since pulling
    ``d_x`` past an ``x``-dependent coefficient is not conservative.
    """
    coeff, rest = term.as_independent(var, as_Add=False)
    if rest == 1:
        return coeff * (hi - lo)
    # A bare outer Derivative as the whole var-dependent body.
    if isinstance(rest, sp.Derivative) and len(rest.variables) == 1:
        dvar = rest.variables[0]
        g = rest.args[0]
        if dvar == var:                       # FTC over the integration var
            return coeff * (g.subs(var, hi) - g.subs(var, lo))
        return coeff * sp.Derivative(         # conservative flux: ∂_x commutes out
            _integrate_through_sum(g, var, lo, hi), dvar)
    factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
    # A conservative ``∂_w`` (non-``var``) derivative FACTOR whose remaining
    # factors + coefficient are ``w``-INDEPENDENT commutes OUT of the
    # ζ-integral: ``∫ c φ_l ∂_x F dζ = ∂_x ∫ c φ_l F dζ``.  When the coefficient
    # is ``w``-DEPENDENT (an ``h`` pulled out by ``Multiply(h)``) the pull-out
    # is INVALID (``∂_x(h·…) ≠ h·∂_x(…)``) — leave the term abstract so a prior
    # ``ProductRule`` folds ``h`` inside the derivative first.
    for f in factors:
        if (isinstance(f, sp.Derivative) and len(f.variables) == 1
                and f.variables[0] != var):
            dw = f.variables[0]
            others = sp.Mul(*[g for g in factors if g is not f])
            if not (coeff * others).has(dw):
                inner = _integrate_through_sum(others * f.expr, var, lo, hi)
                return coeff * sp.Derivative(inner, dw)
            break
    # A single unexpanded Sum factor: push the integral inside.
    sums = [f for f in factors if isinstance(f, sp.Sum)]
    if len(sums) == 1:
        the_sum = sums[0]
        others = sp.Mul(*[f for f in factors if f is not the_sum])
        cc, dep = the_sum.function.as_independent(var, as_Add=False)
        oc, od = others.as_independent(var, as_Add=False)
        return coeff * oc * sp.Sum(
            cc * sp.Integral(od * dep, (var, lo, hi)), *the_sum.limits)
    # Otherwise leave the var-dependent body as an opaque Integral.
    return coeff * sp.Integral(rest, (var, lo, hi))


def _integrate_through_sum(expr, var, lo, hi):
    """Integrate ``expr`` over ``(var, lo, hi)`` term by term, pushing the integral through
    unexpanded ``Sum`` factors.
    """
    expr = sp.expand_mul(expr)
    out = sp.S.Zero
    for term in sp.Add.make_args(expr):
        out += _integrate_one_term(term, var, lo, hi)
    return out


class Integrate(Operation):
    """Purely abstract definite integral: wrap each additive term in an unevaluated ``Integral(term,
    (var, lo, hi))``. Evaluation is left to ``ResolveIntegral`` or :class:`ExtractBrackets`; a
    running integral (``var`` in the bounds) gets a fresh dummy variable.
    """

    whole_leaf_op = True

    def __init__(self, var=None, bounds=(0, 1), dummy=None, name="integrate"):
        """Create the operation.

        Args:
            var: Integration variable (default zeta).
            bounds: Integration interval.
            dummy: Symbol to bind the integrand to (default: a fresh dummy only when ``var`` is in
                the bounds).
        """
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._dummy = dummy
        super().__init__(
            name=name,
            description=f"∫ d{self._var} over {self._bounds} (abstract)")

    def _leaf_sp(self, sp_expr):
        """Wrap each additive term of the expression in an unevaluated ``Integral``."""
        var, (lo, hi) = self._var, self._bounds
        # bind to a fresh dummy ONLY when var appears in the bounds (collision).
        bound_syms = sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols
        if self._dummy is not None:
            bvar = self._dummy
        elif var in bound_syms:
            bvar = sp.Dummy(rf"\hat{{{sp.latex(var)}}}", real=True)
        else:
            bvar = var
        return sp.Add(*[sp.Integral(term.subs(var, bvar) if bvar is not var else term,
                                    (bvar, lo, hi))
                        for term in sp.Add.make_args(sp.expand(sp_expr))])


class Project(Operation):
    """Galerkin projection: multiply by the test function, then integrate over ``(zeta, 0, 1)``
    through the unexpanded ``Sum``. The opaque basis brackets stay unevaluated.

    Args:
        test_function: Full test weight, e.g. ``c(zeta) * phi(l, zeta)``.
        var: Integration variable (default zeta).
        bounds: Integration interval.

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    whole_leaf_op = True

    def __init__(self, test_function, var=None, bounds=(0, 1), name="project"):
        """Create the operation."""
        self._test = sp.sympify(test_function)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._multiply = Multiply(self._test)
        self._integrate = Integrate(var=self._var, bounds=self._bounds)
        super().__init__(
            name=name,
            description=f"PROJECT onto {self._test} over {self._var}",
        )

    def _leaf_sp(self, sp_expr):
        """Multiply by the test function and integrate."""
        scaled = self._multiply._leaf_sp(sp_expr)
        return self._integrate._leaf_sp(scaled)


# ── PullConstants (linearity normalisation, shared by ∫ and ∂) ─────────────


def _dist_deriv_over_sum(expr):
    """Rewrite ``d_v(Sum_j body)`` as ``Sum_j d_v body``, pulling the ``v``-independent factor of
    the summand out.
    """
    def _f(e):
        """Replacement for one derivative of a Sum."""
        body, wrt = e.args[0].function, e.variables
        const, dep = (body.as_independent(*wrt)
                      if isinstance(body, sp.Mul) else (sp.S.One, body))
        return sp.Sum(const * sp.Derivative(dep, *wrt), *e.args[0].limits)
    return expr.replace(
        lambda e: isinstance(e, sp.Derivative) and isinstance(e.args[0], sp.Sum),
        _f)


def _pull_piece(piece, v, lo, hi):
    """Normalise one additive term of an ``Integral`` over ``v``: push an independent ``Sum`` out,
    commute an independent ``d_w`` out, or hoist the ``v``-independent factor.

    Returns:
        ``(rewritten, changed)``.
    """
    facs = list(piece.args) if isinstance(piece, sp.Mul) else [piece]
    # An independent ``Σ_j`` (``j ⊥ v``) commutes OUT of the integral.
    the_sum = next((f for f in facs if isinstance(f, sp.Sum)), None)
    if the_sum is not None:
        others = sp.Mul(*[f for f in facs if f is not the_sum])
        inner = _normalize_one_integral(
            sp.Integral(others * the_sum.function, (v, lo, hi)))
        return sp.Sum(inner, *the_sum.limits), True
    # A commuting ``∂_w`` (non-``v`` var, ``v``-DEPENDENT argument, the rest
    # ``w``-independent) commutes OUT: ``∫_v c·∂_w F dv = ∂_w ∫_v c·F dv``.
    # A derivative of a ``v``-INDEPENDENT factor (``∂_x a_j``) is not a flux —
    # it is hoisted as a plain constant below, never commuted.
    deriv = next((f for f in facs
                  if isinstance(f, sp.Derivative)
                  and v not in f.variables and f.args[0].has(v)), None)
    if deriv is not None:
        rest = sp.Mul(*[f for f in facs if f is not deriv])
        if not rest.has(*deriv.variables):
            inner = _normalize_one_integral(
                sp.Integral(rest * deriv.expr, (v, lo, hi)))
            return sp.Derivative(inner, *deriv.variables), True
    # Hoist the ``v``-independent constant factor out.
    const, dep = (piece.as_independent(v, as_Add=False)
                  if isinstance(piece, sp.Mul) else (sp.S.One, piece))
    return const * sp.Integral(dep, (v, lo, hi)), const != sp.S.One


def _normalize_one_integral(integ):
    """Pull every ``v``-independent factor, independent ``Sum`` and commuting ``d_w`` out of one
    ``Integral`` over ``v``; basis-agnostic.
    """
    limits = integ.args[1]
    if len(limits) != 3:
        return integ
    v, lo, hi = limits
    integrand = sp.expand(_dist_deriv_over_sum(integ.args[0]))
    out, changed = sp.S.Zero, False
    for piece in sp.Add.make_args(integrand):
        new, did = _pull_piece(piece, v, lo, hi)
        out += new
        changed = changed or did
    return out if changed else integ


def pull_out(expr):
    """Pull every factor independent of a binding operator's variable out of that derivative or
    integral, to a fixpoint. Gentle: never applies the product rule to a variable-dependent product.
    """
    from .closure import pull_consts

    def _walk(e):
        """Bottom-up normalisation of every ``Integral``."""
        if hasattr(e, "args") and e.args:
            try:
                e = e.func(*(_walk(a) for a in e.args))
            except (TypeError, ValueError):
                return e
        if isinstance(e, sp.Integral):
            return _normalize_one_integral(e)
        return e

    prev = None
    while expr != prev:
        prev = expr
        expr = _dist_deriv_over_sum(expr)   # ∂_v Σ → Σ ∂_v  (coeff pulled out)
        expr = pull_consts(expr)            # ∂_v(c·f) → c·∂_v f
        expr = _walk(expr)                  # ∫ normalisation, bottom-up
    return expr


class PullConstants(Operation):
    """Pull operator-independent factors out of every ``Integral`` and ``Derivative`` (see
    :func:`pull_out`). Run before :class:`ExtractBrackets` so that only purely zeta-dependent bodies
    remain under a bracket.
    """

    whole_leaf_op = True

    def __init__(self, name="pull_constants"):
        """Create the operation."""
        super().__init__(
            name=name,
            description="pull operator-independent factors out of ∫ / ∂")

    def _leaf_sp(self, sp_expr):
        """Apply :func:`pull_out` to one leaf expression."""
        return pull_out(sp_expr)


# ── ExtractBrackets (naming only — assumes a PullConstants-normalised input) ─


class ExtractBrackets(Operation):
    """Name Galerkin brackets: a zeta-independent factor comes out as a constant, ``int phi_i c
    phi_l`` becomes ``Gram(i, l)``, ``int c phi_l`` becomes ``Weight(l)``, any other pure-zeta body
    stays an explicit ``Integral``. A non-zeta ``d_x`` wrapping the product commutes out
    (conservative flux).

    Args:
        basis: Opaque basis whose ``phi`` head is recognised.
        var: Integration variable (default zeta).
        weight_name: Name of the weight Function ``c`` (default read off the basis).
    """

    whole_leaf_op = True

    def __init__(self, basis, var=None, weight_name=None,
                 name="extract_brackets"):
        """Create the operation."""
        super().__init__(name=name, description="extract basis brackets")
        self._basis = basis
        self._phi = getattr(basis, "phi", None) or getattr(basis, "phi_fn", basis)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        if weight_name is None:
            weight_name = getattr(basis, "weight_name", "c")
        self._weight_name = weight_name

    def _phi_indices(self, factors):
        """Split ``factors`` into phi indices and remaining factors, with weight and
        weight-derivative flags.
        """
        idx, rest = [], []
        has_weight = has_dweight = False
        for a in factors:
            if isinstance(a, sp.Function) and a.func is self._phi:
                idx.append(a.args[0])
            elif (isinstance(a, sp.Pow) and isinstance(a.base, sp.Function)
                  and a.base.func is self._phi and a.exp.is_Integer
                  and a.exp > 0):
                idx.extend([a.base.args[0]] * int(a.exp))
            elif (isinstance(a, sp.Function)
                  and a.func.__name__ == self._weight_name):
                has_weight = True
            elif isinstance(a, sp.Derivative):
                has_dweight = True
                rest.append(a)
            else:
                rest.append(a)
        return idx, rest, has_weight, has_dweight

    def _to_bracket(self, integrand, v=None, lo=sp.S.Zero, hi=sp.S.One):
        """Split an integrand into a constant and a bracket (``Gram``/``Weight`` over (0, 1), else
        an opaque ``Integral``).

        Returns:
            The rewritten piece, or ``None`` when nothing is claimed.
        """
        if v is None:
            v = self._var
        const, dep = (sp.S.One, integrand)
        if isinstance(integrand, sp.Mul):
            const, dep = integrand.as_independent(v, as_Add=False)
        factors = list(dep.args) if isinstance(dep, sp.Mul) else [dep]
        idx, rest, has_w, has_dw = self._phi_indices(factors)
        canonical = (lo == 0 and hi == 1)
        if canonical and not has_dw:
            n = len(idx)
            # ⟨c, φ_l⟩ — closed form ``δ_{l,0}`` (orthogonal basis).
            if n == 1 and has_w and len(rest) == 0:
                return const * Weight(idx[0])
            # ⟨φ_i, c φ_l⟩ — closed form ``δ_{il}/(2l+1)``.
            if n == 2 and has_w and len(rest) == 0:
                return const * Gram(idx[0], idx[1])
        # A bracket is a PURE NUMBER: claim the integral as an opaque ⟨…⟩ ONLY
        # when its body depends on nothing but ``v`` (a ``t``/``x``-bearing body
        # is NOT a bracket — leave it a plain ``∫`` for a prior PullConstants to
        # finish hoisting).  The ``const`` prefactor stays outside.
        if not is_bracket_body(dep, v):
            return None
        return const * sp.Integral(dep, (v, lo, hi))

    def _name_one_integral(self, integ):
        """Name one ``Integral`` term by term, assuming a PullConstants-normalised input."""
        limits = integ.args[1]
        if len(limits) != 3:
            return integ
        v, lo, hi = limits
        out, changed = sp.S.Zero, False
        for piece in sp.Add.make_args(sp.expand(integ.args[0])):
            b = self._to_bracket(piece, v, lo, hi)
            if b is None:
                out += sp.Integral(piece, limits)
            else:
                out += b
                changed = True
        return out if changed else integ

    def _leaf_sp(self, expr):
        """Run :func:`pull_out`, then name every ``Integral`` in the expression."""
        # SHARP gate: first run the general :func:`pull_out` (PullConstants) so
        # every ζ-independent factor / sum is hoisted out — only THEN is a
        # ``∫_0^1`` over a purely ζ-dependent body, which :meth:`_to_bracket`
        # may name.  Composing the normaliser keeps a lone ``ExtractBrackets``
        # correct on its own, while the push-out logic still lives in one place.
        expr = pull_out(expr)

        def _walk(e):
            """Bottom-up naming of every ``Integral``."""
            if hasattr(e, "args") and e.args:
                try:
                    e = e.func(*(_walk(a) for a in e.args))
                except (TypeError, ValueError):
                    return e
            if isinstance(e, sp.Integral):
                return self._name_one_integral(e)
            return e
        return _walk(expr)


# ── ResolveBasis (resolve every Galerkin bracket to a number) ──────────────


class ResolveBasis(Operation):
    """Resolve every Galerkin bracket to a number against a concrete basis via
    ``Basisfunction.resolve``. Apply after ``ResolveModes`` has made every bracket index an integer.
    """

    whole_leaf_op = True

    def __init__(self, basis, var=None, name="resolve_basis"):
        """Create the operation."""
        self._basis = basis
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        super().__init__(
            name=name,
            description=f"resolve brackets via {getattr(basis, 'name', basis)}")

    def _leaf_sp(self, sp_expr):
        """Resolve the brackets of one leaf expression against the basis."""
        return self._basis.resolve(sp_expr, self._var)
