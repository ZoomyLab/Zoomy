"""Gate 17: AD through the distributed solver vs serial AD.
    mpiexec -n 4 python gate17_parallel_ad.py
J = sum over cells of h^2 |V| at T; each rank differentiates its owned share
J_r (the halo exchange and the CFL minimum carry the coupling), the ranks sum.
Parameters: forward (jvp) and reverse (jax.grad, checkpointed loop).  Mesh:
d/dX of every vertex coordinate, the copies of a vertex on several ranks
summed (Partition.sum_to_global)."""
import numpy as np
import jax
import jax.numpy as jnp
from mpi4py import MPI
import common as C
from zoomy_jax.symsolver.runtime import geometry_of
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def objective(make, ad_mode, t_end, dt, comm):
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, dt=dt, ad_mode=ad_mode,
                                    ad_checkpoints=16, comm=comm)
    Q0, A0 = s.setup(mesh, nsm)
    h = s.nsm.sm.roles.depth
    n = s.partition.n_own if s.partition else Q0.shape[1]

    def Jp(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[h, :n] ** 2 * s.mesh.cell_volumes[:n])

    def JX(X):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, X=X), X=X)
        return jnp.sum(Q[h, :n] ** 2 * geometry_of(s.topo, X).cell_volumes[:n])
    return s, Jp, JX


def bump_direction(X):
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    s = jnp.prod(jnp.sin(jnp.pi * (X - lo) / (hi - lo)), axis=0)
    return jnp.stack([s * (1.0 + 0.5 * k) for k in range(X.shape[0])]) * (hi - lo) / 10


def derivatives(make, t_end, dt, comm, mesh_ad):
    """(J, dJ/dg forward, reverse[, dJ/dX.v forward, reverse]); distributed:
    summed over ranks, on every rank."""
    sf, Jpf, JXf = objective(make, "forward", t_end, dt, comm)
    sr, Jpr, JXr = objective(make, "reverse", t_end, dt, comm)
    part = sf.partition
    total = (lambda x: x) if part is None else (lambda x: comm.allreduce(x))
    p0 = sf.parameters.at[sf.rt.parameter_names.index("n")].set(0.03)
    e = jnp.zeros_like(p0).at[sf.rt.parameter_names.index("g")].set(1.0)
    J = total(float(Jpf(p0)))
    dp = total(float(jax.jvp(Jpf, (p0,), (e,))[1]))
    gp = np.asarray(jax.grad(Jpr)(p0))
    if part is not None:
        gp = part.sum_to_global(gp, None, None)
    out = [J, dp, float(gp @ np.asarray(e))]
    if mesh_ad:
        Xg = jnp.asarray(np.asarray(sf.global_mesh.vertex_coordinates)[:sf.global_mesh.dimension])
        v = bump_direction(Xg)
        vl = v if part is None else v[:, part.vertices]
        dX = total(float(jax.jvp(JXf, (sf.X,), (vl,))[1]))
        gX = np.asarray(jax.grad(JXr)(sr.X))
        if part is not None:
            gX = part.sum_to_global(gX, part.vertices, Xg.shape[1])
        out += [dX, float(np.vdot(gX, np.asarray(v)))]
    return out


import sys
CASES = {"o1": ("SWE 2-D bump O1, adaptive CFL, parameters + mesh",
                lambda: C.bump_2d(1, nx=12, flow=(0.1, 0.05)), 0.2, None, True),
         "o2": ("SWE 2-D bump O2 + MOOD, adaptive CFL, parameters",
                lambda: C.bump_2d(2, nx=12, flow=(0.1, 0.05)), 0.2, None, False)}
for key in sys.argv[1:] or list(CASES):
    label, make, t_end, dt, mesh_ad = CASES[key]
    par = derivatives(make, t_end, dt, comm, mesh_ad)
    if rank == 0:
        ser = derivatives(make, t_end, dt, None, mesh_ad)
        rel = [abs(a - b) / max(abs(b), 1e-300) for a, b in zip(par, ser)]
        msg = (f"{label}: ranks {comm.Get_size()}  J {ser[0]:.6e}  | rel par-ser: "
               f"J {rel[0]:.1e}  dJ/dg fwd {rel[1]:.1e} rev {rel[2]:.1e}")
        if mesh_ad:
            msg += f"  dJ/dX.v fwd {rel[3]:.1e} rev {rel[4]:.1e}"
        print(msg + f"  (serial rev-fwd dJ/dg {abs(ser[2]-ser[1])/abs(ser[1]):.1e})",
              flush=True)
