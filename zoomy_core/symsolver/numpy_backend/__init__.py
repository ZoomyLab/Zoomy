"""NumPy backend of the symbolic solvers (`zoomy_core.symsolver`): the printed templates on a
plain-numpy runtime, serial, with no JAX and no compiler.
"""
from zoomy_core.symsolver.numpy_backend.solver import (  # noqa: F401
    SymbolicChorinSolverNumpy, SymbolicHyperbolicSolverNumpy)
