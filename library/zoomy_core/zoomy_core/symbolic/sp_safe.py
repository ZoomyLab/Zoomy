"""Held-form sympy constructors for the symbolic primitive layer.

Sympy's default constructors auto-evaluate where they can:
`sp.Derivative(x**2, x)` returns `2*x`, not a held atom.  This
auto-evaluation is the root of every "math fired at the wrong pipeline
position" bug we've hit — the framework can't keep an opaque atom
opaque if sympy decides to evaluate it on construction.

The constructors here always pass `evaluate=False`.  Every primitive
in this package must use them when it constructs a fresh `Derivative`,
`Integral` or `Subs` atom.  A CI grep forbids bare `sp.Derivative(...)`,
`sp.Integral(...)`, `sp.diff(...)` calls inside the symbolic
subpackage.
"""

from __future__ import annotations

import sympy as sp


def D(expr, *vars):
    """Held `Derivative(expr, *vars)`.

    Parameters
    ----------
    expr : sympy.Expr
    *vars
        Each entry is either a single `Symbol` (∂_v applied once) or
        a `(symbol, n)` tuple (∂_v^n).  Same shape sympy's
        `Derivative` accepts.

    Returns
    -------
    sympy.Derivative
        Always held — never auto-evaluated.  The primitive layer
        decides when to apply chain/product/Leibniz rules.
    """
    if not vars:
        raise ValueError("D requires at least one differentiation variable")
    return sp.Derivative(expr, *vars, evaluate=False)


def Int(integrand, *limits):
    """Held `Integral(integrand, *limits)`.

    Each limit is either a bare `Symbol` (indefinite integral, single
    arg) or a `(symbol, lower, upper)` triple (definite integral).

    `sp.Integral` is held by default (it never auto-evaluates on
    construction), so this is a thin documenting wrapper.  Only
    `.doit()` or `sp.integrate(...)` would evaluate it — both are
    forbidden inside the symbolic package.
    """
    if not limits:
        raise ValueError("Int requires at least one limit specification")
    return sp.Integral(integrand, *limits)


def Sub(expr, var, value):
    """Held `Subs(expr, var, value)`.

    `sp.Subs` is held by default, so this is a thin alias that
    documents intent: any "evaluate `expr` at `var = value`" should
    go through `Sub` and produce a held atom rendered as
    `expr|_{var=value}` by the latex printer.

    The primitive `UnSubs` is the only thing that may unwrap this.
    """
    if isinstance(var, (list, tuple)) and isinstance(value, (list, tuple)):
        return sp.Subs(expr, var, value)
    return sp.Subs(expr, (var,), (value,))


def held_function(name, *args, **kwargs):
    """Construct a held `UndefinedFunction` call `f(*args)`.

    A passthrough wrapper for completeness — sympy's default behaviour
    on UndefinedFunction calls is already non-evaluating, so this is
    just `sp.Function(name)(*args)`.
    """
    return sp.Function(name, **kwargs)(*args)
