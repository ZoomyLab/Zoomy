"""Gate 18: distributed output; the file must equal the serial run's.
In-memory mesh: snapshots gathered to rank 0.  Mesh file: every rank writes
its own block (h5py with MPI), else gathered; /mesh is the input's.
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


def run(comm, path, mesh_file=None):
    mesh, nsm = C.bump_2d(2, nx=24, flow=(0.1, 0.05))
    mesh = mesh if mesh_file is None else mesh_file
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


def compare(label, a, b, n_local):
    a, b = items(a), items(b)
    diff = max((np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max()
                if np.asarray(a[k]).dtype.kind in "fi" else float(not np.array_equal(a[k], b[k])))
               for k in b)
    print(f"output, {label}: ranks {comm.Get_size()}  datasets {len(a)} vs {len(b)} "
          f"(same names {sorted(a) == sorted(b)})  max |par-ser| {diff:.1e}  "
          f"snapshots {sum('fields/' in k and k.endswith('/Q') for k in a)}  "
          f"on_write saw local states of {n_local} cells", flush=True)


import h5py as _h5
from zoomy_core.mesh.distributed import write_mesh
mesh_file = os.path.join(out, "mesh.h5")
if comm.Get_rank() == 0:
    write_mesh(C.bump_2d(2, nx=24)[0], mesh_file)
comm.Barrier()
for label, mf in (("in-memory mesh", None), (
        "mesh file, " + ("blocks per rank" if _h5.get_config().mpi else "gathered"),
        mesh_file)):
    seen.clear()
    run(comm, os.path.join(out, "parallel.h5"), mf)
    comm.Barrier()
    if comm.Get_rank() == 0:
        n_local = seen[-1]
        run(None, os.path.join(out, "serial.h5"), mf)
        compare(label, os.path.join(out, "parallel.h5"),
                os.path.join(out, "serial.h5"), n_local)
    comm.Barrier()
