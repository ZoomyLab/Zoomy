"""Run under `mpiexec -n N` by `test_numpy_mpi.py`: the NumPy backend distributed vs serial on one
rank, its halo exchange, reductions and linear solves.  Prints one JSON line on rank 0.

    mpiexec -n 2 python _numpy_mpi_run.py hyperbolic|linsolve
"""
import json
import sys

import numpy as np
from mpi4py import MPI

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel, Periodic
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.numpy_backend.parallel import MpiLinearSolver
from zoomy_core.systemmodel.system_model import SystemModel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def bump(nx=12, periodic=False):
    """`(mesh, nsm)`: a surface bump in a 10 x 10 box, walls (or periodic in x), O2 minmod."""
    walls = ("bottom", "top") if periodic else ("left", "right", "bottom", "top")
    bcs = [FromModel(tag=t, definition="wall") for t in walls]
    if periodic:
        bcs += [Periodic(tag="left", periodic_to_physical_tag="right"),
                Periodic(tag="right", periodic_to_physical_tag="left")]
    closures = [Newtonian(), ManningFriction(), StressFree()]
    model = SME(level=0, dimension=3, closures=closures,
                boundary_conditions=BoundaryConditions(bcs))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0.0, 0.5 + 0.05 * np.exp(-((x[0] - 4.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0),
         0.3 * (0.5 + 0.05 * np.exp(-((x[0] - 4.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0)), 0.0]))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


def march(c, **kw):
    """`(Q gathered, solver)` of the bump marched on communicator `c` (None: serial)."""
    mesh, nsm = bump(periodic=kw.pop("periodic", False))
    s = SymbolicHyperbolicSolverNumpy(comm=c, **kw)
    Q, A = s.setup(mesh, nsm)
    t, Q, A = s.run(Q, A)
    mass = s.integral(Q)
    Qg = s.partition.gather(Q) if s.partition else Q
    return t, Qg, mass, s


def hyperbolic():
    """O2 + MOOD and periodic O2: distributed = serial, the mass integral, halo copies = owners,
    NaN through reduce_min."""
    out = {}
    for name, kw in (("walls_mood", dict(time_end=1.0, mood=True)),
                     ("periodic", dict(time_end=1.0, mood=False, periodic=True))):
        t, Qd, mass_d, s = march(comm, **kw)
        if rank == 0:
            ts, Qs, mass_s, _ = march(None, **kw)
            out[name] = dict(rel=float(np.abs(Qd - Qs).max() / np.abs(Qs).max()),
                             t=[float(t), float(ts)],
                             mass=float(np.abs(mass_d - mass_s).max() / np.abs(mass_s).max()))
    d = s.module["_distributed"]
    gid = np.tile(s.partition.gids.astype(float), (2, 1))
    stale = np.where(np.arange(d.nc) < d.n_own, gid, -1.0)
    out["halo"] = bool(comm.allreduce(np.array_equal(d.halo(stale), gid), op=MPI.LAND))
    faces = np.ones(d.nf)
    if rank == comm.Get_size() - 1:
        faces[np.flatnonzero(d.own_face)[0]] = np.nan
    lo = d.reduce_min(faces)
    mpi_min = comm.allreduce(float(np.min(faces)), op=MPI.MIN)
    out["nan_min"] = bool(comm.allreduce(bool(np.isnan(lo)), op=MPI.LAND))
    out["mpi_min_swallows"] = bool(not np.isnan(mpi_min))
    out["any"] = bool(d.reduce_any(np.arange(d.nc) == (0 if rank == 1 else -1)))
    return out


def linsolve():
    """A diagonally dominant face-neighbour operator through the runtime's `linsolve` (halo inside
    the operator): petsc / krylov, distributed = serial."""
    mesh, nsm = bump()
    out = {}
    for method in (MpiLinearSolver.petsc, MpiLinearSolver.krylov):
        res = []
        for c in (comm, None):
            if c is None and rank != 0:
                continue
            s = SymbolicHyperbolicSolverNumpy(time_end=0.1, comm=c, mpi_linear_solver=method,
                                              gmres_tol=1e-13, gmres_maxiter=50)
            s.setup(mesh, nsm)
            m, mn = s.module, s.mesh_np
            nc = int(mn.n_inner_cells)
            fc = np.asarray(mn.face_cells)[:2]
            fin = fc[:, (fc[0] < nc) & (fc[1] < nc)]
            xc = np.asarray(mn.cell_centers)[:, :nc]

            def A(X):
                """`2 X - 0.4 sum_nbrs X + 1` per cell, two rows coupled."""
                X = m["halo"](X, s.mesh)
                Y = 2.0 * X + 1.0
                Y[0] += 0.1 * X[1]
                for a, b in (fin, fin[::-1]):
                    np.add.at(Y, (slice(None), a), -0.4 * X[:, b])
                return Y
            b = np.stack([np.sin(xc[0]) * np.cos(xc[1]), xc[0] - xc[1]])
            x = m["linsolve"](lambda X: A(X) - A(np.zeros_like(X)), b)
            g = s.partition.gather(x) if s.partition else x
            res.append(g)
        if rank == 0:
            out[str(method)] = float(np.abs(res[0] - res[1]).max() / np.abs(res[1]).max())
    return out


if __name__ == "__main__":
    result = globals()[sys.argv[1]]()
    if rank == 0:
        print(json.dumps(result), flush=True)
