"""Derivative sugar bound to the canonical coordinates.

    import zoomy_core.derivatives as d
    d.t(u)       # ->  Derivative(u, t)        (∂u/∂t)
    d.x(f, 2)    # ->  Derivative(f, (x, 2))   (∂²f/∂x²)

Every helper differentiates with respect to the matching symbol in
`zoomy_core.coords`, so `d.t` and your fields share the same `t`.
"""

import sympy as sp

from zoomy_core import coords as _c


def t(expr, n=1):
    """∂ⁿ/∂tⁿ of `expr`."""
    return sp.Derivative(expr, (_c.t, n))


def x(expr, n=1):
    """∂ⁿ/∂xⁿ of `expr`."""
    return sp.Derivative(expr, (_c.x, n))


def y(expr, n=1):
    """∂ⁿ/∂yⁿ of `expr`."""
    return sp.Derivative(expr, (_c.y, n))


def z(expr, n=1):
    """∂ⁿ/∂zⁿ of `expr`."""
    return sp.Derivative(expr, (_c.z, n))


def zeta(expr, n=1):
    """∂ⁿ/∂ζⁿ of `expr` (the mapped reference coordinate)."""
    return sp.Derivative(expr, (_c.zeta, n))


__all__ = ["t", "x", "y", "z", "zeta"]
