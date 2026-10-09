"""One Chorin step, old vs symbolic: wall time and XLA cost analysis."""
import sys, os, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa
import jax, jax.numpy as jnp, numpy as np, sympy as sp
import escalante_bump as EB, gate7_chorin as g7
from zoomy_core.mesh import LSQMesh
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver import chorin as CH
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
pred, press, corr = EB.split(sp.Symbol("dt", positive=True))
o = ChorinSplitVAMSolverJax(pred, press, corr, pressure_tol=1e-9, pressure_maxit=200)
Q0 = np.array(o.setup_simulation(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS)))
xc = np.asarray(o._rt_mesh.cell_centers)[0, :o.nc]
Q0[:, :o.nc] = EB.initial_state(xc, Q0.shape[0])
Qo = jnp.asarray(Q0); Ao = o._qaux_from_q(Qo, o._rt_parameters)
p_o = o._rt_parameters
def old_step(Q, A):
    Qn = o.step(EB.DT, 0.0, Q, A, p_o)
    return o.post_step(EB.DT, EB.DT, Qn, Q, A, p_o)
pred, press, corr = EB.split(_DT_SYMBOL)
s = SymbolicChorinSolverJax(time_end=EB.T_END, dt=EB.DT, cfl=0.9, pressure_tol=1e-9, pressure_maxit=200)
Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS), g7.pred_nsm(pred, press, corr), press, corr)
Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
Qs, Qauxs, ps, time, mesh_s = sp.symbols("Q Qaux p time mesh")
src = PyPrinter().procedure("one", (Qs, Qauxs, ps, time, mesh_s), lambda Q, Qaux, p, time, mesh: CH.chorin_step(
    s.nsm, s.press, s.corr, Q, Qaux, p, time, sp.Float(EB.DT), mesh, 1, False, False, 1))
one = R.build(src, "one", s.module)
geometry = s.mesh
new_step = lambda Q, A: one(Q, A, s.parameters, 0.0, geometry, jnp.int64(0))[:2]
jitted = {}
for name, f, args in (("old", old_step, (Qo, Ao)), ("new", new_step, (Q, A))):
    jf = jax.jit(f)
    t0 = wall.perf_counter(); c = jf.lower(*args).compile(); tc = wall.perf_counter() - t0
    ca = c.cost_analysis(); ca = ca[0] if isinstance(ca, list) else ca
    jax.block_until_ready(jf(*args))
    jitted[name] = (jf, args, tc, ca)
best = {"old": 1e9, "new": 1e9}
for _ in range(4):                      # alternate; minimum = least disturbed
    for name, (jf, args, _, _) in jitted.items():
        t0 = wall.perf_counter()
        for _ in range(30):
            out = jf(*args)
        jax.block_until_ready(out)
        best[name] = min(best[name], (wall.perf_counter() - t0) / 30)
for name, (jf, args, tc, ca) in jitted.items():
    print(f"{name}: {1e3 * best[name]:.2f} ms/step, compile {tc:.1f}s, flops {ca.get('flops', 0):.3e}, "
          f"bytes {ca.get('bytes accessed', 0):.3e}", flush=True)
