"""Memoizing integral cache over a basis: concretizes opaque phi_k(arg) with the basis polynomials
and integrates in polynomial form. Used by the Galerkin projection; it caches symbolic integrals,
not M/A/B/D arrays.
"""

import sympy as sp


_PHI_PREFIX = "phi_"


def _is_phi_call(e):
    """True iff the expression is an opaque call phi_k(arg)."""
    return (
        isinstance(e, sp.Function)
        and not isinstance(e, sp.Derivative)
        and getattr(e.func, "__name__", "").startswith(_PHI_PREFIX)
    )


def _phi_index(e):
    """Return k for phi_k(...) or its derivative."""
    name = e.func.__name__ if _is_phi_call(e) else e.args[0].func.__name__
    return int(name[len(_PHI_PREFIX):])


class BasisIntegralCache:
    """Per-basis, in-memory cache from (integrand, var, lower, upper) to the symbolic integral. The
    caller must have factored out all var-independent factors from the integrand.
    """

    def __init__(self, basis):
        """Store the basis polynomials and their z-derivatives."""
        self.basis = basis
        self._cache = {}
        # The basis stores its polynomials in a conventional "z" symbol.
        # We need to know which symbol to substitute away when concretizing.
        self._z_basis = sp.Symbol("z")
        self._phi_polys = [basis.get(k) for k in range(basis.level + 1)]
        self._dphi_polys = [
            sp.diff(p, self._z_basis) for p in self._phi_polys
        ]

    # ------------------------------------------------------------------
    # Public API

    def integrate(self, integrand, var, lower, upper):
        """Return the symbolic value of the integral of integrand over var from lower to upper,
        cached by that tuple. Opaque phi_k(arg) and Derivative(phi_k(arg), arg) nodes are replaced
        by the basis polynomials before integration.
        """
        key = (integrand, var, lower, upper)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._compute(integrand, var, lower, upper)
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Internals

    def _compute(self, integrand, var, lower, upper):
        """Concretize, expand and integrate in polynomial form; fall back to sympy.integrate for
        non-polynomial integrands.
        """
        concrete = self._concretize(integrand)
        # Flatten any nested Integrals whose integrand is now a
        # polynomial (after concretization, any ``phi_k(arg)`` → poly;
        # inner running integrals like ``∫_0^ζ poly(ζ') dζ'`` evaluate
        # via ``doit`` to a polynomial in ζ).
        if concrete.has(sp.Integral):
            concrete = concrete.doit()
        # Polynomial integration — fast path.
        expanded = sp.expand(concrete)
        try:
            poly = sp.Poly(expanded, var)
            anti = poly.integrate()
            value = anti.as_expr().subs(var, upper) - anti.as_expr().subs(var, lower)
        except (sp.GeneratorsNeeded, sp.PolynomialError):
            # Non-polynomial in var after concretize — should be rare;
            # fall back to direct antiderivative (still no opaque phi).
            anti = sp.integrate(expanded, var)
            value = anti.subs(var, upper) - anti.subs(var, lower)
        return sp.simplify(value)

    def _concretize(self, expr):
        """Replace every phi_k(arg) and Derivative(phi_k(arg), arg) by the basis polynomial or its
        derivative at arg.
        """
        def _rec(e):
            """Rebuild the expression tree with phi nodes replaced."""
            # Derivative(phi_k(arg), arg) → dphi_k/dz|_{z=arg}
            if isinstance(e, sp.Derivative):
                inner = e.args[0]
                if _is_phi_call(inner):
                    # Only handle the single-variable first-order case
                    # ``Derivative(phi_k(arg), arg)``.  Higher-order or
                    # mixed partials shouldn't arise here; fall through
                    # to generic recursion if they do.
                    wrts = e.args[1:]
                    if (len(wrts) == 1
                            and (wrts[0] == inner.args[0]
                                 or (isinstance(wrts[0], tuple)
                                     and wrts[0][0] == inner.args[0]
                                     and wrts[0][1] == 1))):
                        k = _phi_index(inner)
                        arg = inner.args[0]
                        return self._dphi_polys[k].subs(self._z_basis, arg)
                # Fall through for other derivatives — recurse into args.
                new_args = tuple(_rec(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
                return e
            if _is_phi_call(e):
                k = _phi_index(e)
                arg = e.args[0]
                return self._phi_polys[k].subs(self._z_basis, arg)
            if e.args:
                new_args = tuple(_rec(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e
        return _rec(expr)
