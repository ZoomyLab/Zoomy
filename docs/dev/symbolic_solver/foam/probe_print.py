import sys, re
sys.path.insert(0, "/Users/adam-obbpb5az1dhsjzf/git/Zoomy-solver-template/docs/dev/symbolic_solver/gates")
import common as C
import sympy as sp
from zoomy_core.symsolver import CxxPrinter, PyPrinter
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
mesh, nsm = C.dambreak_1d(0, 1, nx=20)
template = HyperbolicTemplate(nsm, cfl=0.9, dimension=1)
from zoomy_core.numerics import NumericalSystemModel
Q, Qaux, p, time, time_end, mesh_s = sp.symbols("Q Qaux p time time_end mesh")
def pr(P, side):
    return P().procedure("run", (Q, Qaux, p, time, mesh_s),
        lambda *a: template.precice_march(*a, side=side, subcycling=False))
mode = sys.argv[1]
if mode == "jaxfirst":
    PyPrinter().procedure("hyperbolic", (Q, Qaux, p, time, time_end, mesh_s), template.march)
outs = [pr(CxxPrinter, 1), pr(CxxPrinter, 0), pr(PyPrinter, 1)]
for o in outs:
    first = o.splitlines()[:6]
    bad = "tok_3" in o.splitlines()[4] or any(re.match(r"\s{4}auto x\d+ = couple_max_dt", l) for l in o.splitlines()[:6])
    print("BAD" if bad else "ok", len(o.splitlines()), "lines")
print(outs[0][:600])
