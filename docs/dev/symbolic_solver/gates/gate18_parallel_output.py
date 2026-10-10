"""Gate 18: distributed output.  io.write snapshots gathered to rank 0 and
written on the global mesh; the file must equal the serial run's.
    mpiexec -n 3 python gate18_parallel_output.py [OUTDIR]   (default: a temp dir)"""
import os
import sys
import tempfile
import h5py
import numpy as np
from mpi4py import MPI
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
out = comm.bcast(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
                 else tempfile.mkdtemp() if comm.Get_rank() == 0 else None)
seen = []


def run(comm, path):
    mesh, nsm = C.bump_2d(2, nx=24, flow=(0.1, 0.05))
    s = SymbolicHyperbolicSolverJax(time_end=0.5, dt_write=0.1, output=path,
                                    comm=comm,
                                    on_write=lambda t, Q, A: seen.append(Q.shape[1]))
    Q, A = s.setup(mesh, nsm)
    s.run(Q, A)[1].block_until_ready()


def items(path):
    acc = {}
    with h5py.File(path) as f:
        f.visititems(lambda k, v: acc.__setitem__(k, v[()])
                     if isinstance(v, h5py.Dataset) else None)
    return acc


run(comm, os.path.join(out, "parallel.h5"))
comm.Barrier()
if comm.Get_rank() == 0:
    n_local = seen[-1]
    run(None, os.path.join(out, "serial.h5"))
    a, b = items(os.path.join(out, "parallel.h5")), items(os.path.join(out, "serial.h5"))
    diff = max((np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max()
                if np.asarray(a[k]).dtype.kind in "fi" else float(a[k] != b[k]))
               for k in b)
    print(f"output: ranks {comm.Get_size()}  datasets {len(a)} vs {len(b)} "
          f"(same names {sorted(a) == sorted(b)})  max |par-ser| {diff:.1e}  "
          f"snapshots {sum('fields/' in k and k.endswith('/Q') for k in a)}  "
          f"on_write saw local states of {n_local} cells", flush=True)
