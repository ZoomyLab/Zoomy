"""One template, two printers: the hyperbolic march of the C1 bump as Python
(what JAX runs) and as C++ (what foam and AMReX compile)."""
import sympy as sp

from bump import bump
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam

mesh, nsm = bump(order=1, nx=8)
template = HyperbolicTemplate(nsm, cfl=0.9, dimension=2)

Q, Qaux, p, time, time_end, m = sp.symbols("Q Qaux p time time_end mesh")
py = PyPrinter().procedure("hyperbolic", (Q, Qaux, p, time, time_end, m), template.march)
cxx = SymbolicHyperbolicSolverFoam(cfl=0.9, mood=False)._print(nsm, 2)

print(f"Python: {len(py.splitlines())} lines; C++: {len(cxx.splitlines())} lines\n")
print("\n".join(py.splitlines()[:14]))
start = next(i for i, ln in enumerate(cxx.splitlines()) if "face_inradius" in ln)
print("...\n")
print("\n".join(cxx.splitlines()[start:start + 12]))
