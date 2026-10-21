"""Change-of-variable primitive: affine map + Jacobian.

Replaces both `IntegralTransform` (ins_generator.py:2523) and
`AffineProjection` (ins_generator.py:2874) with a single primitive
that requires the caller to specify the source ``(var, lower,
upper)`` triple explicitly — no implicit `state.b/state.eta`
defaults, no `sp.simplify(bound_diff) == 0` matching for running
integrals (the legacy `AffineProjection` Pass 2 logic).

The redesigned primitive matches Integral atoms by **structural
equality** of the limits tuple.  Callers who have outer + nested
running integrals invoke the primitive twice with the appropriate
limits each time.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Derivative, Integral

__all__ = [
    "affine_change_of_variable",
]


def affine_change_of_variable(
    expr,
    var,
    lower,
    upper,
    ref_var,
    ref_lo=0,
    ref_hi=1,
):
    """Apply ``∫_{lower}^{upper} f(var) dvar
                 = ((upper-lower)/(ref_hi-ref_lo)) ·
                   ∫_{ref_lo}^{ref_hi} f(φ(ref_var)) d(ref_var)``
    where `φ(s) = lower + (upper-lower)·(s-ref_lo)/(ref_hi-ref_lo)`.

    Walks `expr` and rewrites every ``Integral(f, (var, lower,
    upper))`` whose limits structurally match the given tuple.
    Other Integral atoms are left untouched — call the primitive
    again with their limits if you want them transformed.

    Bottom-up walk: nested integrals' integrands are processed first
    so an inner running integral is in the new variable form before
    the outer transform runs.

    Examples
    --------
    >>> from zoomy_core.symbolic import Int
    >>> from zoomy_core.symbolic.primitives_change_of_variable import affine_change_of_variable
    >>> import sympy as sp
    >>> z, h = sp.symbols('z h', real=True)
    >>> b = sp.Function('b')(sp.Symbol('t'), sp.Symbol('x'))
    >>> zeta = sp.Symbol(r'\\hat{\\zeta}', real=True)
    >>> e = Int(sp.Function('f')(z), (z, b, b + h))
    >>> affine_change_of_variable(e, z, b, b + h, zeta)
    """
    if not isinstance(expr, sp.Basic):
        return expr

    span = upper - lower
    ref_span = ref_hi - ref_lo
    phi = lower + span * (ref_var - ref_lo) / ref_span
    jac = span / ref_span

    def _walk(e):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0])
            limits = e.args[1]
            if (hasattr(limits, "__len__") and len(limits) == 3
                    and limits[0] == var
                    and limits[1] == lower
                    and limits[2] == upper):
                new_integrand = integrand.subs(var, phi) * jac
                return Integral(new_integrand, (ref_var, ref_lo, ref_hi))
            if integrand is not e.args[0]:
                return Integral(integrand, *e.args[1:])
            return e
        if isinstance(e, Derivative):
            inner = _walk(e.args[0])
            if inner is not e.args[0]:
                return Derivative(inner, *e.args[1:])
            return e
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)
