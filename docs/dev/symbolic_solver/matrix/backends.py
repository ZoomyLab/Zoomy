"""Backend adapters.  One class per backend, the whole interface:

    configs                      the configs it supports
    launch(config) -> argv|None  a command prefix to re-run matrix.py under
                                 (mpiexec, another env); None: run in-process
    run(r, mesh, systems, Q0, config, work) -> (t, Q) | None
                                 march Run ``r`` from Q0 (mesh order); return
                                 the final time and state in the mesh's cell
                                 order (None on non-root MPI ranks)

``systems`` is the case's NSM, or (pred, press, corr) when ``r.chorin``."""
import sys

import numpy as np


def _ranks(config):
    return int(config[3:]) if config.startswith("mpi") else 1


class Jax:
    configs = ("serial", "mpi2", "mpi4", "mpi8", "gpu1")

    def launch(self, config):
        if config.startswith("mpi"):
            return ["mpiexec", "-n", str(_ranks(config)), sys.executable]
        if config == "gpu1":
            return ["env", "JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "micromamba", "run", "-n", "zoomy-cuda", "python"]
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        import jax.numpy as jnp
        from zoomy_jax.symsolver.solver import (SymbolicChorinSolverJax,
                                                SymbolicHyperbolicSolverJax)
        comm = None
        if config.startswith("mpi"):
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        kw = dict(time_end=r.t_end, cfl=r.cfl, mood=r.mood, dt=r.dt, comm=comm)
        if r.chorin:
            s = SymbolicChorinSolverJax(pressure_linear_solver="gmres", **kw)
            Q, A = s.setup(mesh, *systems)
        else:
            s = SymbolicHyperbolicSolverJax(**kw)
            Q, A = s.setup(mesh, systems)
        if r.q0 is not None:           # on this rank's cells
            names = [str(v) for v in s.nsm.state]
            Q = jnp.asarray(r.q0(names, np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
            A = s.initial_aux(Q, A)
        elif comm is None:
            Q = jnp.asarray(Q0)
        t, Q, _ = s.run(Q, A)
        Q = s.partition.gather(np.asarray(Q)) if s.partition else np.asarray(Q)
        return None if Q is None else (float(t), Q)


class Foam:
    """OpenFOAM 13 (apptainer); mpiN needs SymbolicHyperbolicSolverFoam(nprocs=)."""
    configs = ("serial", "mpi2", "mpi4")

    def launch(self, config):
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        from zoomy_foam.symsolver import (SymbolicChorinSolverFoam,
                                          SymbolicHyperbolicSolverFoam)
        cls = SymbolicChorinSolverFoam if r.chorin else SymbolicHyperbolicSolverFoam
        kw = dict(time_end=r.t_end, cfl=r.cfl, mood=r.mood, dt=r.dt)
        if config != "serial":
            if "nprocs" not in cls.param:
                raise NotImplementedError("this zoomy_foam has no MPI runs (nprocs)")
            kw["nprocs"] = _ranks(config)
        t, Q = cls(**kw).run(mesh, systems, Q0, str(work))
        return float(t), np.asarray(Q)


class Amrex:
    """AMReX single level (apptainer); gpu1 = the CUDA build, 1 GPU."""
    configs = ("serial", "mpi2", "mpi4", "gpu1")

    def launch(self, config):
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        from zoomy_amrex.symsolver import (SymbolicChorinSolverAmrex,
                                           SymbolicHyperbolicSolverAmrex)
        kw = dict(time_end=r.t_end, cfl=r.cfl, mood=r.mood, dt=r.dt,
                  ranks=_ranks(config), gpu=config == "gpu1")
        s = (SymbolicChorinSolverAmrex(max_grid_size=8, **kw) if r.chorin
             else SymbolicHyperbolicSolverAmrex(**kw))
        t, Q = s.run(mesh, systems, Q0, work)
        return float(t), np.asarray(Q)


BACKENDS = {"jax": Jax, "foam": Foam, "amrex": Amrex}
