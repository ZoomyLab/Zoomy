"""Gate 20: write_mesh_parallel (pieces on every rank, no rank holding the
mesh) writes the same file as the serial write_mesh: every dataset equal.
    mpiexec -n 3 python gate20_parallel_write.py [OUTDIR]"""
import os
import sys
import tempfile
import h5py
import numpy as np
from mpi4py import MPI
import gate16_parallel as G
from zoomy_core.mesh.base_mesh import BaseMesh, _local_faces_for_type
from zoomy_core.mesh.distributed import write_mesh, write_mesh_parallel

comm = MPI.COMM_WORLD
rank, P = comm.Get_rank(), comm.Get_size()
out = comm.bcast(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
                 else tempfile.mkdtemp() if rank == 0 else None)


def pieces(mesh):
    """This rank's blocks of a mesh: vertices, cells, boundary elements."""
    nc, n_bf = int(mesh.n_inner_cells), int(mesh.n_boundary_faces)
    X = np.asarray(mesh.vertex_coordinates)[:mesh.dimension]
    cv = np.asarray(mesh.cell_vertices)[:, :nc]
    nv = X.shape[1]
    v0, v1 = rank * nv // P, (rank + 1) * nv // P
    c0, c1 = rank * nc // P, (rank + 1) * nc // P
    f = np.asarray(mesh.boundary_face_face_indices)[:n_bf]
    fc = np.asarray(mesh.face_cells)[:, f]
    c = np.where(fc[0] < nc, fc[0], fc[1])
    lf = np.argmax(np.asarray(mesh.cell_faces)[:, c] == f[None], axis=0)
    local = np.array(_local_faces_for_type(mesh.type))
    fv = cv[local[lf].T, c]                                # (w, n_bf)
    tags = np.asarray(mesh.boundary_face_physical_tags)[:n_bf]
    b0, b1 = rank * n_bf // P, (rank + 1) * n_bf // P
    names = dict(zip((int(t) for t in mesh.boundary_conditions_sorted_physical_tags),
                     mesh.boundary_conditions_sorted_names))
    return X[:, v0:v1], v0, cv[:, c0:c1], c0, fv[:, b0:b1], tags[b0:b1], names, (c0, c1)


def items(path):
    acc = {}
    with h5py.File(path) as f:
        f.visititems(lambda k, v: acc.__setitem__(k, v[()]) if isinstance(v, h5py.Dataset) else None)
    return acc


rng = np.random.default_rng(0)
q = BaseMesh.create_2d((0, 20, 0, 10), 37, 23)
Xj = q.vertex_coordinates.copy()
Xj[:2] += 0.2 * rng.uniform(-1, 1, Xj[:2].shape)
tri = G.tri_bump(1, nx=17)[0]
CASES = {
    "quads": q,
    "jittered quads, one tag": BaseMesh.from_arrays(Xj[:2], q.cell_vertices, "quad",
                                                    default_tag=0, tag_names={0: "wall"}),
    "triangles": BaseMesh.from_arrays(tri.vertex_coordinates[:2],
                                      tri.cell_vertices[:, :tri.n_inner_cells], "triangle",
                                      default_tag=0, tag_names={0: "wall"}),
    "1-D": BaseMesh.create_1d((0, 1), 57),
    "hexahedra": BaseMesh.create_3d((0, 1, 0, 2, 0, 1), 5, 7, 3),
}
for label, mesh in CASES.items():
    X, v0, cv, c0, fv, tags, names, (a, b) = pieces(mesh)
    bathy = np.sin(np.arange(int(mesh.n_inner_cells)) * 0.37)
    ser, par = os.path.join(out, "serial.h5"), os.path.join(out, "parallel.h5")
    if rank == 0:
        write_mesh(mesh, ser, cell_data={"b": bathy})
    write_mesh_parallel(comm, par, mesh.type, X, v0, cv, c0, faces=fv, face_tags=tags,
                        tag_names=names, default_tag=-7, cell_data={"b": bathy[a:b]})
    comm.Barrier()
    if rank == 0:
        s, p = items(ser), items(par)
        bad = [k for k in s if k not in p or np.asarray(s[k]).shape != np.asarray(p[k]).shape
               or not np.array_equal(s[k], p[k])]
        print(f"parallel write, {label}: ranks {P}  datasets {len(p)}/{len(s)}  "
              f"{'all equal' if not bad and len(p) == len(s) else 'DIFFER: ' + str(bad)}  "
              f"(h5py MPI: {h5py.get_config().mpi})", flush=True)
    comm.Barrier()
