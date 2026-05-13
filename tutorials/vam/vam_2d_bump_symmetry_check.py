"""Verify rotational symmetry of the 2D radial-bump DAE output.

The bump is rotationally symmetric at t=0.  Under the chain DAE physics
(no preferred horizontal direction), this symmetry must be preserved
through time — except for the discrete mesh anisotropy of a Cartesian
quad grid (which sets the floor on how well the symmetry can be
resolved).

Reads the HDF5 produced by ``vam_2d_bump_dae.py`` and prints:

* ``h(x, 0, t)`` vs ``h(0, x, t)`` along the horizontal/vertical
  middle lines — these should agree pointwise under radial symmetry;
* ``|h(x, 0, t) − h(0, x, t)|_∞`` as a quantitative symmetry residual;
* The middle-line ``h(x, 0, t)`` plotted alongside the 1D simulation
  result at the same time (informal sanity check; the 2D bump is
  Gaussian while the 1D bump is cosine so the profiles differ — but
  the leading-edge wave speed should be comparable).
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np


HDF5_2D = "outputs_vam_2d_bump_dae/vam_2d_bump_dae.h5"


def _load_mesh(f):
    """Reconstruct cell centres + the (nx, ny) grid index map from a
    saved BaseMesh.create_2d HDF5 dump."""
    mesh = f["mesh"]
    nc = int(mesh["n_inner_cells"][()])
    # Try the precomputed cell_centers if available.
    if "_cell_centers" in mesh:
        cc = mesh["_cell_centers"][()]
    elif "cell_centers" in mesh:
        cc = mesh["cell_centers"][()]
    else:
        # Compute from vertices + cell_vertices.
        vc = mesh["vertex_coordinates"][()]
        cv = mesh["cell_vertices"][()]
        cc = np.zeros((vc.shape[0], cv.shape[1]))
        for j in range(cv.shape[1]):
            cc[:, j] = vc[:, cv[:, j]].mean(axis=1)
    return cc[:, :nc]


def _grid_indexing(cc, tol=1e-9):
    """Return (xs, ys, idx_grid) mapping (ix, iy) → cell index for a
    Cartesian quad mesh that BaseMesh.create_2d builds row-major over
    (ix, iy)."""
    xs = np.unique(np.round(cc[0], 6))
    ys = np.unique(np.round(cc[1], 6))
    nx, ny = len(xs), len(ys)
    idx_grid = np.full((nx, ny), -1, dtype=int)
    for c in range(cc.shape[1]):
        ix = int(np.argmin(np.abs(xs - cc[0, c])))
        iy = int(np.argmin(np.abs(ys - cc[1, c])))
        idx_grid[ix, iy] = c
    return xs, ys, idx_grid


def main():
    if not os.path.exists(HDF5_2D):
        print(f"missing {HDF5_2D}; run vam_2d_bump_dae.py first")
        return 1

    with h5py.File(HDF5_2D, "r") as f:
        cc = _load_mesh(f)
        xs, ys, idx_grid = _grid_indexing(cc)
        snaps = sorted(f["fields"].keys(),
                       key=lambda s: int(s.split("_")[1]))
        print(f"mesh: {len(xs)} x {len(ys)} cells, "
              f"x ∈ [{xs.min():.3f}, {xs.max():.3f}]")
        print(f"snapshots: {len(snaps)}")
        print()

        # The middle line in each axis.
        ix_mid = len(xs) // 2
        iy_mid = len(ys) // 2

        print(f"{'t':>8s}  {'h_min':>9s}  {'h_max':>9s}"
              f"  {'|h(x,0)−h(0,y)|_inf':>22s}  {'comment':>20s}")
        print("-" * 80)
        for s in snaps:
            t = float(f["fields"][s]["time"][()])
            Q = f["fields"][s]["Q"][()]
            # Q shape: (n_state, nc).  h is row 0.
            h_flat = Q[0]
            h_grid = np.full((len(xs), len(ys)), np.nan)
            for ix in range(len(xs)):
                for iy in range(len(ys)):
                    c = idx_grid[ix, iy]
                    if c >= 0:
                        h_grid[ix, iy] = h_flat[c]

            h_x = h_grid[:, iy_mid]                # h(x, 0)
            h_y = h_grid[ix_mid, :]                # h(0, y)
            # Symmetry residual: should be zero up to mesh anisotropy.
            sym_residual = float(np.max(np.abs(h_x - h_y)))
            comment = ("(initial)" if int(s.split('_')[1]) == 0
                       else "")
            print(f"{t:>8.4f}  {h_grid.min():>9.5f}  {h_grid.max():>9.5f}"
                  f"  {sym_residual:>22.3e}  {comment:>20s}")

        # Final snapshot middle line — print the profile.
        print()
        s = snaps[-1]
        t = float(f["fields"][s]["time"][()])
        Q = f["fields"][s]["Q"][()]
        h_flat = Q[0]
        h_grid = np.full((len(xs), len(ys)), np.nan)
        for ix in range(len(xs)):
            for iy in range(len(ys)):
                c = idx_grid[ix, iy]
                if c >= 0:
                    h_grid[ix, iy] = h_flat[c]
        h_x = h_grid[:, iy_mid]
        h_y = h_grid[ix_mid, :]
        print(f"Final snapshot t={t:.4f}, middle-line profiles:")
        print(f"{'x':>8s}  {'h(x,0)':>10s}  {'y':>8s}  {'h(0,y)':>10s}")
        for k in range(len(xs)):
            print(f"{xs[k]:>8.3f}  {h_x[k]:>10.6f}  {ys[k]:>8.3f}  {h_y[k]:>10.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
