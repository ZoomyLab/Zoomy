"""O1 residual of the exact lake at rest around a dry island, per
reconstruction choice (all should give 0)."""
import numpy as np, jax.numpy as jnp, sympy as sp
import common as C
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.discretization.residual import residual
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from zoomy_core.numerics import BedReconstruction, ReconstructedVariables

for spec in (dict(variables=ReconstructedVariables.conservative), dict(bed=BedReconstruction.reconstructed), dict(bed=BedReconstruction.derived)):
    sm = SystemModel.from_model(C.sme(0, 2, ("left", "right"), "wall"))
    def ic(x):
        b = 0.8 * np.exp(-x[0] ** 2 / 0.02); return np.array([b, max(0.5 - b, 0.0), 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeRusanov,
                                                 reconstruction=ReconstructionSpec(order=1, **spec))
    s = SymbolicHyperbolicSolverJax(time_end=1.0, cfl=0.45)
    Q, A = s.setup(LSQMesh.create_1d(domain=(-1, 1), n_inner_cells=200), nsm)
    Qs, As, ps, ts, ms = sp.symbols("Q Qaux p time mesh")
    src = PyPrinter().procedure("res", (Qs, As, ps, ts, ms), lambda Q, Qaux, p, time, mesh: (
        residual(s.nsm, Q, Qaux, p, time, sp.Float(1e-3), mesh, 1),))
    L = np.asarray(R.build(src, "res", s.module)(Q, A, s.parameters, 0.0, s.mesh, jnp.int64(0))[0])
    print(spec, "max|dh/dt|", f"{np.abs(L[1]).max():.2e}", "max|dq/dt|", f"{np.abs(L[2]).max():.2e}",
          "riemann", type(nsm.build_numerics()).__name__, flush=True)
