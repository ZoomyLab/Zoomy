"""Gate 30: periodic BCs in distributed runs vs serial.
    mpiexec -n R python gate32_parallel_periodic.py
A smooth wave in a periodic box (SWE = SME(0); 1-D: left<->right, 2-D: both
directions), O1 and O2 + MOOD.  Each case runs on all ranks, is gathered to
rank 0 and compared with the serial run there; plus the mass of the
distributed run (owned cells, summed over ranks) at T against t = 0."""
import time as clock
import numpy as np
from mpi4py import MPI
import jax
jax.config.update("jax_enable_x64", True)
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def wave(dimension, order, nx):
    """SWE wave in the periodic unit box: 1-D (dimension 2) or 2-D (3)."""
    pairs = ([("left", "right")] if dimension == 2
             else [("left", "right"), ("bottom", "top")])
    bcs = [BC.Periodic(tag=a, periodic_to_physical_tag=b) for a, b in pairs]
    bcs += [BC.Periodic(tag=b, periodic_to_physical_tag=a) for a, b in pairs]
    sm = SystemModel.from_model(SME(
        level=0, dimension=dimension,
        closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BC.BoundaryConditions(bcs)))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        s = np.sin(2 * np.pi * x[0]) * (np.cos(2 * np.pi * x[1]) if dimension == 3 else 1.0)
        h = 1.0 + 0.2 * s
        q[names.index("h")] = h
        for n, u in zip([n for n in names if n.startswith("q")], (0.3, 0.1)):
            q[names.index(n)] = u * h
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    mesh = (LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nx) if dimension == 2
            else LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx))
    return mesh, nsm


CASES = (("1-D wave O1", lambda: wave(2, 1, 100), 0.5),
         ("1-D wave O2 + MOOD", lambda: wave(2, 2, 100), 0.5),
         ("2-D wave O2 + MOOD (x, y periodic)", lambda: wave(3, 2, 32), 0.3))

if __name__ == "__main__":
    for label, make, t_end in CASES:
        mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=t_end, comm=comm)
        Q0, A = s.setup(mesh, nsm)
        h, n = s.nsm.sm.roles.depth, s.partition.n_own
        vol = np.asarray(s.mesh.cell_volumes)[:n]
        m0 = comm.allreduce(float(np.sum(np.asarray(Q0)[h, :n] * vol)))
        t0 = clock.perf_counter()
        t, Q, A = s.run(Q0, A)
        Q.block_until_ready()
        wall = clock.perf_counter() - t0
        m1 = comm.allreduce(float(np.sum(np.asarray(Q)[h, :n] * vol)))
        Qg = s.partition.gather(Q)
        if rank == 0:
            mesh, nsm = make()
            r = SymbolicHyperbolicSolverJax(time_end=t_end)
            Qs, As = r.setup(mesh, nsm)
            _, Qs, _ = r.run(Qs, As)
            Qs = np.asarray(Qs)
            err = np.abs(Qg - Qs).max() / np.abs(Qs).max()
            print(f"{label}: ranks {comm.Get_size()}  t={float(t):.4g}  "
                  f"max rel |par-ser| {err:.1e}  mass rel drift {abs(m1 - m0) / m0:.1e}  "
                  f"finite {bool(np.isfinite(Qg).all())}  run {wall:.2f}s", flush=True)
