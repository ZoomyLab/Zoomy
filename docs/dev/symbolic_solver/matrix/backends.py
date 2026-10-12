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
    """mpiN: N CPU ranks; gpu1: 1 GPU; gpu2 (gpu-mpi): 2 ranks, a GPU each."""
    return int(config[3:]) if config.startswith(("mpi", "gpu")) else 1


def _solver_kw(r, cls):
    """The Run's solver settings; a setting the backend lacks is unsupported."""
    kw = dict(time_end=r.t_end, cfl=r.cfl, mood=r.mood, dt=r.dt, **(r.solver or {}))
    if r.imex:
        kw["imex"] = True
    if r.chorin:
        kw["time_order"] = r.time_order
        if r.pressure_solve != "linear":
            if "pressure_solve" not in cls.param:
                raise NotImplementedError(f"{cls.__name__}: no pressure_solve={r.pressure_solve!r}")
            kw["pressure_solve"] = r.pressure_solve
    return kw


class Jax:
    configs = ("serial", "mpi2", "mpi4", "mpi8", "gpu1", "gpu2")

    def launch(self, config):
        if config.startswith("mpi"):
            return ["mpiexec", "-n", str(_ranks(config)), sys.executable]
        gpu = ["env", "JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
               "micromamba", "run", "-n", "zoomy-cuda"]
        if config == "gpu1":
            return gpu + ["python"]
        if config == "gpu2":     # one GPU per rank, bound at launch
            return gpu + ["mpiexec", "-n", "2", "python", "-m", "zoomy_jax.gpurank"]
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        import jax.numpy as jnp
        from zoomy_jax.symsolver.solver import (SymbolicChorinSolverJax,
                                                SymbolicHyperbolicSolverJax)
        comm = None
        if _ranks(config) > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        cls = SymbolicChorinSolverJax if r.chorin else SymbolicHyperbolicSolverJax
        kw = dict(_solver_kw(r, cls), comm=comm)
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
    """OpenFOAM 13 (apptainer); mpiN = decomposePar + -parallel; no GPU."""
    configs = ("serial", "mpi2", "mpi4")

    def launch(self, config):
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        from zoomy_foam.symsolver import (SymbolicChorinSolverFoam,
                                          SymbolicHyperbolicSolverFoam)
        cls = SymbolicChorinSolverFoam if r.chorin else SymbolicHyperbolicSolverFoam
        kw = _solver_kw(r, cls)
        if config != "serial":
            kw["nprocs"] = _ranks(config)
        t, Q = cls(**kw).run(mesh, systems, Q0, str(work))
        return float(t), np.asarray(Q)


class Amrex:
    """AMReX single level, 1/2/3-D Cartesian (apptainer); gpuN = the CUDA build."""
    configs = ("serial", "mpi2", "mpi4", "gpu1", "gpu2")

    def launch(self, config):
        return None

    def run(self, r, mesh, systems, Q0, config, work):
        from zoomy_amrex.symsolver import (SymbolicChorinSolverAmrex,
                                           SymbolicHyperbolicSolverAmrex)
        cls = SymbolicChorinSolverAmrex if r.chorin else SymbolicHyperbolicSolverAmrex
        kw = dict(_solver_kw(r, cls), ranks=_ranks(config), gpu=config.startswith("gpu"))
        s = cls(max_grid_size=8, **kw) if r.chorin else cls(**kw)
        t, Q = s.run(mesh, systems, Q0, work)
        return float(t), np.asarray(Q)


BACKENDS = {"jax": Jax, "foam": Foam, "amrex": Amrex}
