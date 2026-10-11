"""Gate 16: distributed hyperbolic solver (MPI ranks, mpi4jax) vs serial.
    mpiexec -n 4 python gate16_parallel.py
Each case runs on all ranks (the mesh split by coordinate bisection), is
gathered to rank 0 and compared with the serial run there."""
import os
import time as clock
import numpy as np
from mpi4py import MPI
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def tri_bump(order, nx=24):
    """The 2-D bump on triangles (each quad of a grid cut along alternating
    diagonals): an unstructured stencil, vertex rings of 6 to 12 cells."""
    from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
    from zoomy_core.mesh.base_mesh import BaseMesh
    _, nsm = C.bump_2d(order, nx=4, flow=(0.1, 0.05))
    xs = np.linspace(0, 10, nx + 1)
    vid = lambda i, k: i * (nx + 1) + k
    X = np.array([[x, y] for x in xs for y in xs]).T
    tris = []
    for i in range(nx):
        for k in range(nx):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            tris += ([(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)])
    tags = {}
    for m in range(nx):
        tags[tuple(sorted((vid(0, m), vid(0, m + 1))))] = 0
        tags[tuple(sorted((vid(nx, m), vid(nx, m + 1))))] = 1
        tags[tuple(sorted((vid(m, 0), vid(m + 1, 0))))] = 2
        tags[tuple(sorted((vid(m, nx), vid(m + 1, nx))))] = 3
    base = BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=tags,
                                tag_names={0: "left", 1: "right", 2: "bottom", 3: "top"})
    return ensure_lsq_mesh(base, nsm), nsm


if __name__ == "__main__":
    CASES = (("SWE 2-D bump O1, background flow", lambda: C.bump_2d(1, nx=40, flow=(0.1, 0.05)), 0.5),
             ("SWE 2-D bump O2 + MOOD", lambda: C.bump_2d(2, nx=40, flow=(0.1, 0.05)), 0.5),
             ("radial dry dam break O2 eta + MOOD", lambda: C.dry_dambreak_2d(nx=48), 0.02),
             ("SWE bump O2 + MOOD, triangles", lambda: tri_bump(2), 0.5))

    def from_file(make):
        """The same case read from a mesh file: each rank reads its own cells."""
        import h5py
        import tempfile
        from zoomy_core.mesh.distributed import write_mesh
        mesh, nsm = make()
        path = comm.bcast(os.path.join(tempfile.mkdtemp(), "mesh.h5")
                          if rank == 0 else None)
        if rank == 0:
            write_mesh(mesh, path)
            with h5py.File(path) as f:
                order = f["mesh/cell_order"][()]
        comm.Barrier()
        return path, nsm, (order if rank == 0 else None)


    CASES += (("SME(1) 1-D dam break O2", lambda: C.dambreak_1d(1, 2, nx=200), 0.1),)
    CASES += (("SWE bump O2 + MOOD, from a mesh file", lambda: C.bump_2d(2, nx=40, flow=(0.1, 0.05)), 0.5),)
    for label, make, t_end in CASES:
        order = None
        if "mesh file" in label:
            mesh, nsm, order = from_file(make)
        else:
            mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=t_end, comm=comm)
        Q, A = s.setup(mesh, nsm)
        t0 = clock.perf_counter()
        t, Q, A = s.run(Q, A)
        Q.block_until_ready()
        wall = clock.perf_counter() - t0
        Qg = s.partition.gather(Q) if s.partition else np.asarray(Q)
        if rank == 0:
            mesh, nsm = make()
            r = SymbolicHyperbolicSolverJax(time_end=t_end)
            Qs, As = r.setup(mesh, nsm)
            _, Qs, _ = r.run(Qs, As)
            Qs = np.asarray(Qs)
            if order is not None:          # file rows -> the mesh's cells
                Qs = Qs[:, order]
            err = np.abs(Qg - Qs).max() / max(np.abs(Qs).max(), 1e-300)
            own = s.partition.n_own if s.partition else Qs.shape[1]
            print(f"{label}: ranks {comm.Get_size()}  t={float(t):.4g}  "
                  f"max rel |par-ser| {err:.1e}  owned/rank0 {own} of {Qs.shape[1]}  "
                  f"run {wall:.2f}s", flush=True)
