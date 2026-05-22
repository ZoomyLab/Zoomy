#!/usr/bin/env python3
"""Render h and eta=h+b from a Firedrake .pvd output to PNG.

Usage::

    python render_pvd.py <pvd_path> <out_dir> [--frames i,j,k]

Reads the .pvd, picks a few timesteps (first / last / midpoints), merges
all per-rank .vtu pieces into a single triangulated mesh, and renders
``h`` (water depth) and ``eta = h+b`` (free-surface elevation) as
tricontourf plots with a shared color scale across snapshots.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _read_pvtu_merged(pvtu_path):
    """Read a .pvtu and concatenate all <Piece>.vtu data into one mesh."""
    base = os.path.dirname(pvtu_path)
    tree = ET.parse(pvtu_path)
    pieces = [p.get("Source") for p in tree.findall(".//Piece")]
    all_pts = []
    all_tri = []
    all_data = {}
    n_offset = 0
    for piece in pieces:
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(os.path.join(base, piece))
        r.Update()
        g = r.GetOutput()
        pts = vtk_to_numpy(g.GetPoints().GetData())
        all_pts.append(pts)
        # Cells: assume all triangles (VTK_TRIANGLE = 5).
        n_cells = g.GetNumberOfCells()
        tri = np.empty((n_cells, 3), dtype=np.int64)
        for ci in range(n_cells):
            cell = g.GetCell(ci)
            ids = cell.GetPointIds()
            for k in range(3):
                tri[ci, k] = ids.GetId(k)
        all_tri.append(tri + n_offset)
        n_offset += pts.shape[0]
        pd = g.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            name = pd.GetArray(i).GetName()
            arr = vtk_to_numpy(pd.GetArray(i))
            all_data.setdefault(name, []).append(arr)
    pts = np.concatenate(all_pts, axis=0)
    tri = np.concatenate(all_tri, axis=0)
    data = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    return pts, tri, data


def _select_frames(n_total, k):
    """Pick k evenly spaced indices in [0, n_total)."""
    if k >= n_total:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, k).astype(int)
    return list(idx)


def _render_field(pts, tri, values, title, vmin, vmax, cmap, out_path, mask=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    if mask is not None:
        triang.set_mask(mask)
    # tricontourf with explicit levels for consistent shading.
    levels = np.linspace(vmin, vmax, 32)
    cs = ax.tricontourf(
        triang, values, levels=levels, cmap=cmap, extend="both"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.85)
    cbar.set_label(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    pvd_path = argv[1]
    out_dir = argv[2]
    n_frames = 6 if len(argv) < 4 else int(argv[3])
    os.makedirs(out_dir, exist_ok=True)

    # Parse PVD: list of (time, pvtu_file).
    tree = ET.parse(pvd_path)
    entries = [
        (float(ds.get("timestep")), ds.get("file"))
        for ds in tree.findall(".//DataSet")
    ]
    entries.sort(key=lambda e: e[0])
    print(f"[render_pvd] {pvd_path}: {len(entries)} snapshots")

    # Pre-scan to find global colorbar ranges for h and eta.
    # Sample first + last + middle to bound cheaply.
    base = os.path.dirname(pvd_path)
    sample_idx = _select_frames(len(entries), min(5, len(entries)))
    h_lo, h_hi = np.inf, -np.inf
    eta_lo, eta_hi = np.inf, -np.inf
    for i in sample_idx:
        t, fname = entries[i]
        pts, tri, data = _read_pvtu_merged(os.path.join(base, fname))
        h = data["Q1"]
        b = data["Q0"]
        eta = b + h
        # Mask very thin / negative h regions for range computation.
        wet = h > 1e-3
        if wet.any():
            h_lo = min(h_lo, float(h[wet].min()))
            h_hi = max(h_hi, float(h[wet].max()))
            eta_lo = min(eta_lo, float(eta[wet].min()))
            eta_hi = max(eta_hi, float(eta[wet].max()))
    print(
        f"[render_pvd] global wet h in [{h_lo:.3f}, {h_hi:.3f}], "
        f"eta in [{eta_lo:.3f}, {eta_hi:.3f}]"
    )

    pick = _select_frames(len(entries), n_frames)
    for i in pick:
        t, fname = entries[i]
        pts, tri, data = _read_pvtu_merged(os.path.join(base, fname))
        h = data["Q1"]
        b = data["Q0"]
        eta = b + h
        tag = f"t{t:07.3f}".replace(".", "p")

        # h: water depth — clamp at zero floor for display.
        h_disp = np.clip(h, 0.0, None)
        _render_field(
            pts, tri, h_disp,
            title=f"h(t={t:.3f}s) [m]",
            vmin=0.0, vmax=h_hi,
            cmap="Blues",
            out_path=os.path.join(out_dir, f"h_{tag}.png"),
        )
        # eta: free-surface elevation.
        _render_field(
            pts, tri, eta,
            title=f"eta = h+b (t={t:.3f}s) [m]",
            vmin=eta_lo, vmax=eta_hi,
            cmap="viridis",
            out_path=os.path.join(out_dir, f"eta_{tag}.png"),
        )
        print(f"[render_pvd]   t={t:6.3f}s rendered (h, eta)")

    # Bathymetry once (stationary).
    pts0, tri0, data0 = _read_pvtu_merged(os.path.join(base, entries[0][1]))
    _render_field(
        pts0, tri0, data0["Q0"],
        title="b (bathymetry) [m]",
        vmin=float(data0["Q0"].min()), vmax=float(data0["Q0"].max()),
        cmap="terrain",
        out_path=os.path.join(out_dir, "b.png"),
    )
    print(f"[render_pvd] wrote {len(pick) * 2 + 1} PNGs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
