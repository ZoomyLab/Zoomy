"""First order (Audusse HR, no MOOD): min h after every step of the radial
dry dam break, template on JAX.  Positivity of the O1 HR scheme says
h >= 0 up to round-off."""
import numpy as np
import jax.numpy as jnp
import sympy as sp
import common as C
from zoomy_core.numerics import ReconstructedVariables, ReconstructionSpec
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

for variables in (ReconstructedVariables.model, ReconstructedVariables.conservative):
    mesh, nsm = C.dry_dambreak_2d()
    nsm.reconstruction = ReconstructionSpec(order=1, variables=variables)
    s = SymbolicHyperbolicSolverJax(time_end=0.05, cfl=0.45, mood=False)
    Q, A = s.setup(mesh, nsm)
    Qs, As, ps, ts, ms = sp.symbols("Q Qaux p time mesh")
    dim = int(s.mesh_np.dimension)
    template = HyperbolicTemplate(s.nsm, cfl=0.45, dimension=dim)
    src = PyPrinter().procedure("one", (Qs, As, ps, ts, ms), lambda Q, Qaux, p, time, mesh: template.step(
        Q, Qaux, p, time, template.dt(Q, Qaux, p, mesh), mesh))
    one = R.build(src, "one", s.module)
    h = [str(v) for v in nsm.state].index("h")
    t, worst, n = 0.0, np.inf, 0
    while t < 0.05:
        Q, A, dt, _ = one(Q, A, s.parameters, t, s.mesh, jnp.int64(0))
        t += float(dt)
        n += 1
        worst = min(worst, float(jnp.min(Q[h])))
    print(f"O1 HR ({variables} W): {n} steps to t={t:.3g}, min_t min_x h = {worst:.3e}, "
          f"cells h<0 at end {int((np.asarray(Q[h]) < 0).sum())}, mass {float(Q[h].sum()):.15g}")
