#!/usr/bin/env python3
"""Convert a TELEMAC-2D SELAFIN result into a Zoomy HDF5 snapshot store.

The Zoomy malpasset cases (thesis/cases/malpasset_jax) read their runs with
``zoomy_plotting.read_hdf5`` — a cell-centred store laid out as::

    mesh/...                         (vertices, cells, faces, neighbours, ...)
    fields/iteration_<k>/Q     (4, Ncells)   rows = [b, h, hu, hv]
    fields/iteration_<k>/Qaux  (1, Ncells)
    fields/iteration_<k>/time  ()

TELEMAC stores P1 *node* values (WATER DEPTH, VELOCITY U/V, BOTTOM); Zoomy is
P0 *cell-centred*. Because both meshes come from the SAME geo_malpasset-small.slf
(13541 nodes, 26000 triangles, identical ordering), the P1->P0 projection is the
exact triangle average: q_cell = mean over the cell's 3 vertices. We reuse a
reference Zoomy store's ``mesh/`` group verbatim so the cell ordering matches the
jax runs exactly — enabling a cell-for-cell comparison.

Two modes (kept in one file; each imports only what it needs):

  # 1) inside the TELEMAC apptainer (has TelemacFile, NO h5py):
  python3 selafin_to_zoomy.py dump  <result.slf> <nodes.npz>

  # 2) on the host / zoomy env (has h5py + numpy):
  python3 selafin_to_zoomy.py build <nodes.npz> <ref_store.h5> <out_dir>

The ``telemac_to_zoomy.sh`` wrapper runs both phases for you.
"""
import sys


# ----------------------------------------------------------------------------
# Mode 1: dump  (runs INSIDE the container — TelemacFile, no h5py dependency)
# ----------------------------------------------------------------------------
def dump(slf_path, npz_path):
    import numpy as np
    from data_manip.extraction.telemac_file import TelemacFile

    f = TelemacFile(slf_path)
    names = {v.strip().upper(): v for v in f.varnames}

    def pick(*cands):
        for c in cands:
            if c in names:
                return names[c]
        raise KeyError(f"none of {cands} in {list(names)}")

    vH = pick("WATER DEPTH", "HAUTEUR D'EAU")
    vU = pick("VELOCITY U", "VITESSE U")
    vV = pick("VELOCITY V", "VITESSE V")
    vB = pick("BOTTOM", "FOND")

    nt = f.ntimestep
    times = np.asarray(f.times, dtype="float64")
    H = np.stack([np.asarray(f.get_data_value(vH, k)) for k in range(nt)])  # (nt, Np)
    U = np.stack([np.asarray(f.get_data_value(vU, k)) for k in range(nt)])
    V = np.stack([np.asarray(f.get_data_value(vV, k)) for k in range(nt)])
    B = np.asarray(f.get_data_value(vB, nt - 1))                            # static bed

    np.savez(npz_path,
             x=np.asarray(f.meshx, dtype="float64"),
             y=np.asarray(f.meshy, dtype="float64"),
             times=times, H=H, U=U, V=V, B=B)
    print(f"[dump] {slf_path} -> {npz_path}: "
          f"{f.npoin3} nodes, {nt} snapshots, t in [{times[0]:.0f},{times[-1]:.0f}]s, "
          f"Hmax={H.max():.2f}m")


# ----------------------------------------------------------------------------
# Mode 2: build  (runs on the HOST / zoomy env — needs h5py + numpy)
# ----------------------------------------------------------------------------
def _node_to_store_index(sx, sy, vx, vy):
    """Map store-vertex order -> SELAFIN-node order (identity when bit-equal)."""
    import numpy as np
    if sx.size == vx.size:
        dmax = float(np.max(np.hypot(sx - vx, sy - vy)))
        if dmax < 1e-6:
            return np.arange(sx.size)            # same ordering — the common case
    # Fallback: match by rounded coordinate (no scipy dependency).
    key = {(round(float(a), 3), round(float(b), 3)): i for i, (a, b) in enumerate(zip(sx, sy))}
    idx = np.empty(vx.size, dtype="int64")
    for j in range(vx.size):
        idx[j] = key[(round(float(vx[j]), 3), round(float(vy[j]), 3))]
    return idx


def build(npz_path, ref_store, out_dir):
    import os, json
    import numpy as np
    import h5py

    d = np.load(npz_path)
    sx, sy = d["x"], d["y"]
    times = d["times"]
    H, U, V, B = d["H"], d["U"], d["V"], d["B"]
    nt = times.size

    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, "telemac.h5")

    with h5py.File(ref_store, "r") as ref, h5py.File(out_h5, "w") as out:
        # 1) reuse the reference mesh verbatim (same cell ordering as the jax runs)
        ref.copy("mesh", out)
        cell_vertices = np.asarray(ref["mesh/cell_vertices"])      # (3, Ncells), store-vertex indices
        vc = np.asarray(ref["mesh/vertex_coordinates"])           # (2, Nv)
        ncells = cell_vertices.shape[1]

    v2n = _node_to_store_index(sx, sy, vc[0], vc[1])              # store vtx -> SELAFIN node
    tri = v2n[cell_vertices]                                       # (3, Ncells) -> SELAFIN node ids

    def cell_avg(node_vals):                                       # P1 nodes -> P0 cell (triangle mean)
        return node_vals[tri].mean(axis=0)                         # (Ncells,)

    b_cell = cell_avg(B)
    with h5py.File(out_h5, "a") as out:
        g = out.create_group("fields")
        for k in range(nt):
            h = cell_avg(H[k])
            hu = cell_avg(H[k] * U[k])
            hv = cell_avg(H[k] * V[k])
            Q = np.stack([b_cell, h, hu, hv])                      # (4, Ncells): [b, h, hu, hv]
            it = g.create_group(f"iteration_{k}")
            it.create_dataset("Q", data=Q)
            it.create_dataset("Qaux", data=np.zeros((1, ncells)))
            it.create_dataset("time", data=float(times[k]))

    # 2) settings.h5 + ckpt.json (mirror the thesis/cases store layout)
    with h5py.File(os.path.join(out_dir, "settings.h5"), "w") as s:
        s.create_dataset("name", data="telemac")
        og = s.create_group("output")
        og.create_dataset("clean_directory", data=False)
        og.create_dataset("directory", data=out_dir)
        og.create_dataset("filename", data="telemac")
        og.create_dataset("snapshots", data=np.int64(nt))
    with open(out_h5 + ".ckpt.json", "w") as j:
        json.dump({"t_cumulative_start": float(times[0]),
                   "t_cumulative_end": float(times[-1]),
                   "this_run_seconds": float(times[-1] - times[0])}, j, indent=2)

    print(f"[build] {out_h5}: {ncells} cells, {nt} snapshots, "
          f"h in [{0:.2f},{H.max():.2f}]m -> Zoomy store ready (read_hdf5).")


def main(argv):
    if len(argv) >= 4 and argv[1] == "dump":
        dump(argv[2], argv[3])
    elif len(argv) >= 5 and argv[1] == "build":
        build(argv[2], argv[3], argv[4])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
