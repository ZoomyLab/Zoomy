"""First NaN of the O1 lake at rest around a dry island: step by step."""
import numpy as np, jax.numpy as jnp, sympy as sp
import common as C
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver import hyperbolic as T
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

sm = SystemModel.from_model(C.sme(0, 2, ("left", "right"), "wall"))
names = [str(s) for s in sm.state]
def ic(x):
    q = np.zeros(len(names)); b = 0.8 * np.exp(-x[0] ** 2 / 0.02)
    q[0], q[1] = b, max(0.5 - b, 0.0); return q
sm.initial_conditions = IC.UserFunction(function=ic)
nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))
s = SymbolicHyperbolicSolverJax(time_end=1.0, cfl=0.45)
Q, A = s.setup(LSQMesh.create_1d(domain=(-1, 1), n_inner_cells=200), nsm)
print("aux names", [str(a) for a in nsm.aux_state], "aux finite", bool(jnp.isfinite(A).all()))
bad = np.where(~np.isfinite(np.asarray(A)).all(axis=0))[0]
print("nonfinite aux cells", bad[:10], np.asarray(A)[:, bad[:3]].T if len(bad) else "")
Qs, As, ps, ts, ms = sp.symbols("Q Qaux p time mesh")
src = PyPrinter().procedure("one", (Qs, As, ps, ts, ms), lambda Q, Qaux, p, time, mesh: T.advance(
    s.nsm, Q, Qaux, p, time, T.cfl_dt(s.nsm, Q, Qaux, p, mesh, 0.45, 1), mesh, 1, False))
one = R.build(src, "one", s.module)
t = 0.0
for k in range(400):
    Qn, An, dt, _ = one(Q, A, s.parameters, t, s.mesh, jnp.int64(0))
    if not (np.isfinite(np.asarray(Qn)).all() and np.isfinite(float(dt))):
        cells = np.where(~np.isfinite(np.asarray(Qn)).all(axis=0))[0]
        print(f"step {k}: dt {float(dt):.3e}, NaN cells {cells[:8]}")
        for c in cells[:2]:
            for cc in (c - 1, c, c + 1):
                print("   cell", cc, "Q", np.asarray(Q)[:, cc], "aux", np.asarray(A)[:, cc])
        break
    Q, A, t = Qn, An, t + float(dt)
else:
    print("no NaN in 400 steps, t", t)

# t = 0: residual of the exact lake at rest
Q0, A0 = s.setup(LSQMesh.create_1d(domain=(-1, 1), n_inner_cells=200), nsm)
src = PyPrinter().procedure("res", (Qs, As, ps, ts, ms), lambda Q, Qaux, p, time, mesh: (
    T.residual(s.nsm, Q, Qaux, p, time, sp.Float(1e-3), mesh, 1),))
res = R.build(src, "res", s.module)
L, _ = res(Q0, A0, s.parameters, 0.0, s.mesh, jnp.int64(0))
L, Q0 = np.asarray(L), np.asarray(Q0)
for i in np.argsort(-np.abs(L).max(axis=0))[:6]:
    print(f"cell {i}: b {Q0[0, i]:.4f} h {Q0[1, i]:.3e} eta {Q0[0, i] + Q0[1, i]:.4f} | dh/dt {L[1, i]:+.3e} dq/dt {L[2, i]:+.3e}")
