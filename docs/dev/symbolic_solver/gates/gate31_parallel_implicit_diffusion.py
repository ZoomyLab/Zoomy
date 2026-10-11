"""Gate 31: implicit diffusion (IMEX) as a runtime block -- Newton on the
Crank-Nicolson residual, each update a ``linsolve`` -- serial vs the legacy
DenseDiffusionOperatorJAX, and distributed vs serial.
    python gate31_parallel_implicit_diffusion.py            # block vs legacy
    mpiexec -n R python gate31_parallel_implicit_diffusion.py par   # march
    mpiexec -n R python gate31_parallel_implicit_diffusion.py ad    # AD
Model: SME(0) with the in-plane Newtonian stress kept (NewtonianInPlane):
its diffusion_matrix is state-dependent (A[q, h] = 2 nu q / h) and dense.
The extractor's sign was fixed (A[q, q] = +2 nu now; it took the residual
coefficient, -2 nu, which the +div(A : grad Q) blocks applied
anti-diffusively)."""
import sys
import time as clock
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (ManningFriction, Newtonian,
                                              NewtonianInPlane, StressFree)
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

NU = 1e-3
TIGHT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10,
             gmres_maxiter=60)


def viscous(dimension, order=2, nx=100):
    """Smooth velocity and depth variation, extrapolation boundaries."""
    tags = (("left", "right") if dimension == 2
            else ("left", "right", "bottom", "top"))
    sm = SystemModel.from_model(SME(
        level=0, dimension=dimension,
        closures=[Newtonian(), ManningFriction(), StressFree(), NewtonianInPlane()],
        boundary_conditions=BC.BoundaryConditions(
            [BC.Extrapolation(tag=t) for t in tags])))
    sm.parameter_values.nu = NU
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        r2 = (x[0] - 0.5) ** 2 + ((x[1] - 0.5) ** 2 if dimension == 3 else 0.0)
        h = 1.0 + 0.2 * np.exp(-r2 / 0.02)
        q[names.index("h")] = h
        for n, u in zip([n for n in names if n.startswith("q")], (0.3, 0.1)):
            q[names.index(n)] = h * (u + 0.2 * np.sin(4 * np.pi * x[0]))
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    mesh = (LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nx) if dimension == 2
            else LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx))
    return mesh, nsm


CASES = (("1-D", lambda: viscous(2, nx=100), 0.1),
         ("2-D", lambda: viscous(3, nx=20), 0.05))


def block_vs_legacy():
    """One implicit-diffusion step (dt = 0.01) from the initial state: the
    runtime block vs DenseDiffusionOperatorJAX.implicit_solve (Newton, exact
    jvp, GMRES), both at tight tolerances."""
    from zoomy_jax.fvm.reconstruction_jax import DenseDiffusionOperatorJAX
    for label, make, _ in CASES:
        mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=0.1, imex=True, **TIGHT)
        Q, A = s.setup(mesh, nsm)
        k, p, dt = s.rt.diffusion_matrix, s.parameters, 0.01
        new = np.asarray(jax.jit(lambda Q, A: s.module["implicit_diffusion"](
            k, Q, A, p, dt, s.mesh))(Q, A))
        op = DenseDiffusionOperatorJAX(s.mesh_np, s.mesh_np.dimension,
                                       s.rt.n_state, state_dependent=True)
        old = np.asarray(jax.jit(lambda Q, A: op.implicit_solve(
            Q, dt, lambda X: k(X, A, p), tol=1e-13, maxiter=60,
            newton_maxiter=10, newton_tol=1e-12))(Q, A))
        Q = np.asarray(Q)
        print(f"implicit diffusion {label}: block vs legacy max rel "
              f"{np.abs(new - old).max() / np.abs(old).max():.1e}  "
              f"(step changes Q by {np.abs(new - Q).max() / np.abs(Q).max():.1e})",
              flush=True)


def march(make, t_end, comm, **kw):
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, imex=True, comm=comm, **kw)
    Q, A = s.setup(mesh, nsm)
    t0 = clock.perf_counter()
    t, Q, A = s.run(Q, A)
    Q.block_until_ready()
    wall = clock.perf_counter() - t0
    return (s.partition.gather(Q) if s.partition else np.asarray(Q)), float(t), wall


def ad(make, comm):
    """dJ/dnu, dJ/dg (J = sum h^2 + q^2 over the owned cells, ranks summed):
    forward vs central FD, reverse vs forward; fixed dt."""
    res = {}
    for mode in ("forward", "reverse"):
        mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=0.01, dt=2e-3, imex=True,
                                        ad_mode=mode, ad_checkpoints=8,
                                        comm=comm, **TIGHT)
        Q0, A0 = s.setup(mesh, nsm)
        n = s.partition.n_own if s.partition else Q0.shape[1]

        def J(p, s=s, Q0=Q0, A0=A0, n=n):
            _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
            return jnp.sum(Q[1, :n] ** 2 + Q[2, :n] ** 2)
        res[mode] = (J, s)
    Jf, s = res["forward"]
    Jr, sr = res["reverse"]
    total = (lambda x: x) if s.partition is None else (lambda x: comm.allreduce(x))
    p0 = s.parameters
    g = np.asarray(jax.grad(Jr)(p0))
    if sr.partition is not None:
        g = sr.partition.sum_to_global(g, None, None)
    out = {}
    for nm in ("nu", "g"):
        i = s.rt.parameter_names.index(nm)
        e = jnp.zeros_like(p0).at[i].set(1.0)
        d = total(float(jax.jvp(Jf, (p0,), (e,))[1]))
        fd = min(abs(total(float(Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e)))
                     / (2 * r * p0[i]) - d) for r in (1e-3, 1e-4, 1e-5)) / abs(d)
        out[nm] = (d, fd, abs(float(g[i]) - d) / abs(d))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        block_vs_legacy()
        sys.exit()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for label, make, t_end in (CASES if sys.argv[1] == "par" else ()):
        Qp, tp, wp = march(make, t_end, comm, **TIGHT)
        if rank == 0:
            Qs, ts, _ = march(make, t_end, None, **TIGHT)
            print(f"IMEX implicit diffusion {label}: ranks {comm.Get_size()}  "
                  f"t={tp:.3g}/{ts:.3g}  max rel |par-ser| "
                  f"{np.abs(Qp - Qs).max() / np.abs(Qs).max():.1e}  finite "
                  f"{bool(np.isfinite(Qp).all())}  [run {wp:.1f}s]", flush=True)
    if sys.argv[1] != "ad":
        sys.exit()
    make = lambda: viscous(2, nx=40)                              # noqa: E731
    par = ad(make, comm)
    ser = ad(make, None) if rank == 0 else None
    if rank == 0:
        for nm in par:
            d, fd, rv = par[nm]
            print(f"AD dJ/d{nm} = {d: .6e}: ranks {comm.Get_size()}  |FD-jvp| {fd:.1e}  "
                  f"|rev-jvp| {rv:.1e}  |par-ser| {abs(d - ser[nm][0]) / abs(d):.1e}  "
                  f"(serial |FD-jvp| {ser[nm][1]:.1e} |rev-jvp| {ser[nm][2]:.1e})",
                  flush=True)
