"""Gate 22: AD through the distributed Chorin solver vs serial AD (gate 8's
case: Escalante VAM(1) bump, 1-D, 16 steps of 2^-7, pressure by GMRES).
Each rank differentiates its owned share of J = sum(h^2) + sum(q_0^2); the
halo exchange, the global reductions and the pressure solve's transpose
(custom_linear_solve, distributed FGMRES) carry the coupling.
    mpiexec -n 2 python gate22_parallel_chorin_ad.py"""
import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
import gate7_chorin as g7
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from zoomy_core.systemmodel.system_model import _DT_SYMBOL

comm = MPI.COMM_WORLD
N = 16


def solver(ad_mode, comm):
    pred, press, corr = g7.split(_DT_SYMBOL)
    g7.cases.set_state_width(pred)
    s = SymbolicChorinSolverJax(time_end=g7.DT * N, dt=g7.DT, cfl=0.9, ad_mode=ad_mode,
                                pressure_linear_solver="gmres", comm=comm)
    Q, A = s.setup(g7.mesh(), g7.pred_nsm(pred, press, corr), press, corr)
    nc = Q.shape[1]
    Q = jnp.asarray(g7.ic(nc, np.asarray(s.mesh_np.cell_centers)[:, :nc]))
    return s, Q, s.initial_aux(Q, A)


def derivatives(comm):
    sf, Q, A = solver("forward", comm)
    sr, _, _ = solver("reverse", comm)
    part = sf.partition
    n = part.n_own if part else Q.shape[1]
    total = (lambda x: x) if part is None else (lambda x: comm.allreduce(x))

    def J(s, p, X):
        _, Qn, _ = s.run(Q, A, parameters=p, X=X)
        return jnp.sum(Qn[1, :n] ** 2) + jnp.sum(Qn[2, :n] ** 2)
    p0, X0 = jnp.asarray(sf.parameters), sf.X
    ig = list(sf.rt.parameter_names).index("g")
    e = jnp.zeros_like(p0).at[ig].set(1.0)
    Jv = total(float(J(sf, p0, X0)))
    fwd = total(float(jax.jvp(lambda p: J(sf, p, X0), (p0,), (e,))[1]))
    gp = np.asarray(jax.grad(lambda p: J(sr, p, X0))(p0))
    Xg = np.asarray(sf.global_mesh.vertex_coordinates)[:1]
    v = np.random.default_rng(0).standard_normal(Xg.shape) * 1e-3
    vl = v if part is None else v[:, part.vertices]
    fX = total(float(jax.jvp(lambda X: J(sf, p0, X), (X0,), (jnp.asarray(vl),))[1]))
    gX = np.asarray(jax.grad(lambda X: J(sr, p0, X))(X0))
    if part is not None:
        gp = part.sum_to_global(gp, None, None)
        gX = part.sum_to_global(gX, part.vertices, Xg.shape[1])
    return Jv, fwd, float(gp[ig]), fX, float(np.vdot(gX, v))


par = derivatives(comm)
if comm.Get_rank() == 0:
    ser = derivatives(None)
    rel = [abs(a - b) / max(abs(b), 1e-300) for a, b in zip(par, ser)]
    print(f"Chorin AD, ranks {comm.Get_size()}: J {ser[0]:.6e}  dJ/dg {ser[1]:.6e} | rel par-ser: "
          f"J {rel[0]:.1e}  dJ/dg fwd {rel[1]:.1e} rev {rel[2]:.1e}  dJ/dX.v fwd {rel[3]:.1e} "
          f"rev {rel[4]:.1e}  (serial rev-fwd dJ/dg {abs(ser[2]-ser[1])/abs(ser[1]):.1e})", flush=True)
