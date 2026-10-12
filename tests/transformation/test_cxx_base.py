"""The two C++ printer families -- the solver lane (``symsolver.core.CxxPrinter``)
and the kernel lane (``generic_c.GenericCppBase`` and everything under it) --
share ONE base, ``cxx_base.CxxBase``.  They differ in vocabulary, not syntax;
these pin the vocabulary each lane keeps (qualified / cast / right-nested vs
runtime / bare / left-nested) and the C++11 dialect of the kernel lane.
"""
import pytest
import sympy as sp

from zoomy_core.symsolver.core import CxxPrinter
from zoomy_core.transformation.cxx_base import CxxBase
from zoomy_core.transformation.generic_c import GenericCppBase, OutParamCodePrinter
from zoomy_core.transformation.to_glsl import GenericGlslBase
from zoomy_core.transformation.to_js import GenericJsBase
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter, FoamSystemModelPrinter)

pytestmark = pytest.mark.printer

x, y, z = sp.symbols("x y z")


def _solver(expr):
    """The solver lane prints whole procedures: the source of ``f(x, y, z)``."""
    return CxxPrinter().procedure("f", (x, y, z), lambda *_: expr, roots=())


def test_one_base():
    for cls in (CxxPrinter, GenericCppBase, OutParamCodePrinter, GenericJsBase,
                GenericGlslBase, FoamSystemModelPrinter, FoamNumericsPrinter):
        assert issubclass(cls, CxxBase), cls


def test_minmax_vocabulary_per_lane():
    kernel = GenericCppBase()
    assert kernel.doprint(sp.Max(x, y, z)) == "std::max(x, std::max(y, z))"   # right-nested
    assert kernel.doprint(sp.Max(0, x)) == "std::max((double)0, x)"           # literal cast
    assert "maximum(maximum(x, y), z)" in _solver(sp.Max(x, y, z))            # left-nested
    assert "maximum(0, x)" in _solver(sp.Max(0, x))                           # bare


def test_call_syntax_is_shared():
    f = sp.Function("f")
    assert GenericCppBase().doprint(f(x, y)) == "f(x, y)"
    assert "auto x1 = f(x, y);" in _solver(f(x, y))


def test_kernel_lane_keeps_the_cxx11_dialect():
    kernel, reg = GenericCppBase(), sp.Symbol("register")
    assert kernel.doprint(reg) == "register_"
    assert kernel.doprint(sp.beta(x, y)) == "beta(x, y)"
    assert CxxBase().doprint(sp.beta(x, y)) == "std::beta(x, y)"
