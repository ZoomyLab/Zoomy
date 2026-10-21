"""Expression vocabulary of the template IR.

Template expressions are sympy expressions over NAMES (`sp.Symbol`) bound in
the enclosing scope.  Arithmetic, comparisons, `And`/`Or`/`Not`,
`Min`/`Max`, `Abs`, `Piecewise` and the elementary functions are plain
sympy.  The functions below add what a solver template needs on top: finiteness,
reductions, a value select, filled-array constructors and a safe divide.

Semantics (identical in every walker):

`IsFinite(x)`        elementwise isfinite (scalar -> bool)
`AnyTrue(m)`         global OR-reduction of a mask / predicate -> bool
`AllTrue(m)`         global AND-reduction -> bool
`ReduceMin(x)`       global min -> real (AD-safe in the jax backend)
`ReduceMax(x)`       global max -> real
`ReduceSum(x)`       global sum -> real
`Where(c, a, b)`     elementwise value select (both branches evaluated)
`FullLike(x, v)`     array of `x`'s shape filled with `v`
`ZerosLike(x)`       array of `x`'s shape filled with 0
`MaskLike(x, v)`     cell mask over the trailing (cell) axis of `x`
`SafeDiv(a, b, f)`   `a / b` where `b != 0`, else `f`; the divisor is
                       double-where'd, so reverse AD stays finite at `b == 0`

The reductions are GLOBAL: under domain decomposition they are collective, and
a value computed by one is replicated on every rank (the replicated-predicate
rule of the type checker relies on that).
"""
from __future__ import annotations

import sympy as sp
from sympy.logic.boolalg import BooleanFunction


class _BoolFn(BooleanFunction):
    @classmethod
    def eval(cls, *args):  # never auto-evaluate: the walkers lower these
        return None


class IsFinite(_BoolFn):
    nargs = 1


class AnyTrue(_BoolFn):
    nargs = 1


class AllTrue(_BoolFn):
    nargs = 1


class ReduceMin(sp.Function):
    nargs = 1


class ReduceMax(sp.Function):
    nargs = 1


class ReduceSum(sp.Function):
    nargs = 1


class Where(sp.Function):
    nargs = 3


class FullLike(sp.Function):
    nargs = 2


class ZerosLike(sp.Function):
    nargs = 1


class MaskLike(sp.Function):
    nargs = 2


class SafeDiv(sp.Function):
    nargs = 3


#: reductions (global, replicated results)
REDUCTIONS = (AnyTrue, AllTrue, ReduceMin, ReduceMax, ReduceSum)

#: every template-specific function class, by name
TEMPLATE_FUNCTIONS = {c.__name__: c for c in (
    IsFinite, AnyTrue, AllTrue, ReduceMin, ReduceMax, ReduceSum, Where,
    FullLike, ZerosLike, MaskLike, SafeDiv)}

#: elementary functions both walkers lower elementwise (sympy class name ->
#: numpy/jax name)
ELEMENTARY = {
    "exp": "exp", "log": "log", "sin": "sin", "cos": "cos", "tan": "tan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "atan": "arctan",
    "atan2": "arctan2", "asin": "arcsin", "acos": "arccos", "sign": "sign",
    "floor": "floor", "ceiling": "ceil", "erf": "erf",
}


def S(*names):
    """`S("Q", "dt")` -> `(Symbol("Q"), Symbol("dt"))`; one name -> one
    Symbol.  Shorthand for writing template expressions."""
    syms = tuple(sp.Symbol(n) for n in names)
    return syms[0] if len(syms) == 1 else syms


def as_expr(value):
    """Sympify a template expression; a bare name string becomes a Symbol,
    Python `bool` becomes `sp.true`/`sp.false`."""
    if isinstance(value, str):
        return sp.Symbol(value)
    if isinstance(value, bool):
        return sp.true if value else sp.false
    if isinstance(value, float):
        return sp.Float(value)
    return sp.sympify(value)


def names_in(expr) -> tuple:
    """Free NAMES an expression reads, sorted: Symbols plus the labels of
    indexed arrays (`x[i]`)."""
    if expr is None:
        return ()
    expr = as_expr(expr)
    out = set()
    for b in expr.atoms(sp.Indexed):
        out.add(str(b.base.label))
        for i in b.indices:
            out.update(str(s) for s in sp.sympify(i).free_symbols)
    for s in expr.free_symbols:
        if isinstance(s, sp.Symbol):
            out.add(str(s))
        elif isinstance(s, sp.IndexedBase):
            out.add(str(s.label))
    # an IndexedBase label symbol also appears in free_symbols; keep it
    return tuple(sorted(out))
