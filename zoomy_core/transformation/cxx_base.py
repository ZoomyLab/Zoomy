"""The C++ syntax of every Zoomy code printer, in one class.

Two printer families emit C++:

* the **solver lane** -- :class:`zoomy_core.symsolver.core.CxxPrinter`, which
  prints the solver procedures (``Solver.H``);
* the **kernel lane** -- :class:`zoomy_core.transformation.generic_c.
  GenericCppBase` and everything under it (the model / numerics / foam / amrex
  kernels, and the GLSL / JS out-parameter printers).

Both derive from :class:`CxxBase`.  They differ in *vocabulary*, not in syntax:
the kernel lane calls the qualified library (``std::max``, ``Foam::pow``) on
operands cast to the backend's real type; the solver lane calls the runtime's
own free functions (``maximum``, ``pow``) on bare operands.  The vocabulary is
the handful of class attributes below; what is printed from them lives here
once.
"""
import sympy as sp
from sympy.printing.cxx import CXX17CodePrinter


class CxxBase(CXX17CodePrinter):
    """C++ calls and n-ary ``Min`` / ``Max``; sympy's C++17 printer for the rest
    (numbers, relations, ``Piecewise``, the math functions).

    Vocabulary, overridden per lane:

    ``math_namespace``
        qualifier of the library calls built here (``std::``, ``Foam::``; empty
        for a runtime that defines them as free functions).
    ``_min_name`` / ``_max_name``
        the callee of ``Min`` / ``Max`` (after ``math_namespace``).
    ``_nest_left``
        how an n-ary ``Min`` / ``Max`` nests its binary calls:
        ``f(a, f(b, c))`` (default) or ``f(f(a, b), c)``.
    ``_typed_numeric_literals`` / ``real_type``
        cast a bare numeric operand to ``real_type`` (``std::max((T)0, x)``), so
        a templated ``std::max(int, double)`` instantiates.
    ``_callee(e)``
        the spelling of an applied function's name.
    """

    math_namespace = "std::"
    real_type = "double"
    _typed_numeric_literals = False
    _min_name, _max_name = "min", "max"
    _nest_left = False

    def _callee(self, e):
        return e.func.__name__

    def _print_Function(self, e):
        return f"{self._callee(e)}({', '.join(self._print(a) for a in e.args)})"

    def _print_Min(self, e):
        return self._print_min_max(self._min_name, e.args)

    def _print_Max(self, e):
        return self._print_min_max(self._max_name, e.args)

    def _print_min_max(self, func_name, args):
        """``Min`` / ``Max`` of any arity as nested binary calls."""
        call, ops = f"{self.math_namespace}{func_name}", [
            self._min_max_operand(a) for a in args]
        if self._nest_left:
            out = ops[0]
            for x in ops[1:]:
                out = f"{call}({out}, {x})"
            return out
        out = ops[-1]
        for x in reversed(ops[:-1]):
            out = f"{call}({x}, {out})"
        return out

    def _min_max_operand(self, arg):
        """One operand; a bare numeric literal is cast to ``real_type`` on typed
        backends (``std::max(int, double)`` does not instantiate)."""
        printed = self._print(arg)
        if self._typed_numeric_literals and isinstance(
            arg, (sp.Integer, sp.Float, sp.Rational)
        ):
            return f"({self.real_type}){printed}"
        return printed
