"""One reconstruction of the dry dam break state after a few steps: the
template's (printed + JAX blocks) vs the old EmittedWLSQMUSCLJAX, row by row."""
import numpy as np
import jax.numpy as jnp
import sympy as sp
import common as C
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.core import unpack
from zoomy_core.symsolver.reconstruction import reconstruct
from zoomy_core.symsolver.vocabulary import boundary_states
from zoomy_jax.fvm.reconstruction_jax import EmittedWLSQMUSCLJAX
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

mesh, nsm = C.dry_dambreak_2d(ties="old")
s = SymbolicHyperbolicSolverJax(time_end=0.004, cfl=0.45)
Q0, A0 = s.setup(mesh, nsm)
import os
if os.environ.get("AT_T0"):
    Q, A = Q0, A0
else:
    t, Q, A = s.run(Q0, A0)                     # a state with a moving front
names = [str(v) for v in nsm.state]
Qs, As, ps, ts, ms = sp.symbols("Q Qaux p time mesh")


def proc(Q, Qaux, p, time, mesh):
    Qb, Qauxb = unpack(boundary_states(nsm.call.boundary_conditions, Q, Qaux, p, time, mesh), 2)
    QL, QR = reconstruct(nsm, Q, Qaux, Qb, Qauxb, p, 2, sp.S.false, mesh)
    return QL, QR, Qb


src = PyPrinter().procedure("rec", (Qs, As, ps, ts, ms), proc)
print(src)
rec = R.build(src, "rec", s.module)
QL, QR, Qb, _ = rec(Q, A, s.parameters, 0.0, s.mesh, jnp.int64(0))
rt = s.rt
old = EmittedWLSQMUSCLJAX(s.mesh_j, 2, rt.reconstruction_variables, rt.state_from_reconstruction,
                          aux_of_q=rt.update_aux_variables, eps_wet=None,
                          b_index=names.index("b"), h_index=names.index("h"),
                          momentum_indices=[i for i, n in enumerate(names) if n.startswith("q_")],
                          piecewise_constant=False, limiter="minmod")
inner = np.asarray(s.mesh_np.face_cells[0])[np.asarray(s.mesh_np.boundary_face_face_indices)]
Ab = rt.update_aux_variables(Qb, A[:, inner], s.parameters)
oL, oR = old(Q, Qb, Qaux=A, parameters=s.parameters, bf_aux=Ab)
for nm, a, b in (("QL", QL, oL), ("QR", QR, oR)):
    d = np.abs(np.asarray(a) - np.asarray(b))
    print(nm, "max|template-old| per row:", " ".join(f"{n}={v:.2e}" for n, v in zip(names, d.max(axis=1))),
          "| faces differing:", int((d.max(axis=0) > 1e-12).sum()), "of", d.shape[1])
    f = int(np.argmax(d.max(axis=0)))
    print("   worst face", f, "template", np.asarray(a)[:, f], "old", np.asarray(b)[:, f])

# first order (MOOD-demoted): template takes the cell values, old maps W(Q) and back
def proc1(Q, Qaux, p, time, mesh):
    Qb, Qauxb = unpack(boundary_states(nsm.call.boundary_conditions, Q, Qaux, p, time, mesh), 2)
    QL, QR = reconstruct(nsm, Q, Qaux, Qb, Qauxb, p, 2, sp.S.true, mesh)
    return QL, QR


rec1 = R.build(PyPrinter().procedure("rec1", (Qs, As, ps, ts, ms), proc1), "rec1", s.module)
QL1, QR1, _ = rec1(Q, A, s.parameters, 0.0, s.mesh, jnp.int64(0))
oL1, oR1 = old(Q, Qb, force_o1=jnp.ones(Q.shape[1], bool), Qaux=A, parameters=s.parameters, bf_aux=Ab)
d = np.abs(np.asarray(QL1) - np.asarray(oL1))
print("O1 QL max|template-old| per row:", " ".join(f"{n}={v:.2e}" for n, v in zip(names, d.max(axis=1))),
      "| faces differing:", int((d.max(axis=0) > 1e-12).sum()))
f = int(np.argmax(d.max(axis=0)))
print("   worst face", f, "template", np.asarray(QL1)[:, f], "old", np.asarray(oL1)[:, f])
