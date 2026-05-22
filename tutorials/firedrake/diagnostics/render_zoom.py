#!/usr/bin/env python3
"""Zoomed comparison renderer: DG(0) vs DG(1) at matched snapshots.

Produces side-by-side panels of h and eta at a fixed downstream
window so cell-scale features (ripples vs smooth wave) are visible.

Usage::

    python render_zoom.py <pvd_dg0> <pvd_dg1> <out_dir>
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

from render_pvd import _read_pvtu_merged, _select_frames


# Downstream window — wave-front region.
XLIM = (10500, 17500)
YLIM = (-2000, 5500)


def _crop(pts, tri, data, xlim, ylim):
    inx = (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1])
    iny = (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
    keep_pt = inx & iny
    # Drop triangles with any vertex outside the window.
    in_tri = keep_pt[tri].all(axis=1)
    return tri[in_tri]


def _read_at_time(pvd_path, t_target):
    base = os.path.dirname(pvd_path)
    tree = ET.parse(pvd_path)
    entries = [
        (float(ds.get("timestep")), ds.get("file"))
        for ds in tree.findall(".//DataSet")
    ]
    entries.sort(key=lambda e: e[0])
    # Pick nearest snapshot to t_target.
    i = int(np.argmin([abs(e[0] - t_target) for e in entries]))
    t, fname = entries[i]
    pts, tri, data = _read_pvtu_merged(os.path.join(base, fname))
    return t, pts, tri, data


def _make_pair(pvd0, pvd1, t_target, field, out_path):
    t0, p0, tri0, d0 = _read_at_time(pvd0, t_target)
    t1, p1, tri1, d1 = _read_at_time(pvd1, t_target)
    if field == "h":
        v0 = np.clip(d0["Q1"], 0.0, None)
        v1 = np.clip(d1["Q1"], 0.0, None)
        cmap = "Blues"
        title_root = "h [m]"
    else:  # eta = h+b
        v0 = d0["Q0"] + d0["Q1"]
        v1 = d1["Q0"] + d1["Q1"]
        cmap = "viridis"
        title_root = "eta = h+b [m]"

    tri0_c = _crop(p0, tri0, d0, XLIM, YLIM)
    tri1_c = _crop(p1, tri1, d1, XLIM, YLIM)

    # Shared color scale across the panels (within the window).
    in_p0 = (
        (p0[:, 0] >= XLIM[0]) & (p0[:, 0] <= XLIM[1])
        & (p0[:, 1] >= YLIM[0]) & (p0[:, 1] <= YLIM[1])
    )
    in_p1 = (
        (p1[:, 0] >= XLIM[0]) & (p1[:, 0] <= XLIM[1])
        & (p1[:, 1] >= YLIM[0]) & (p1[:, 1] <= YLIM[1])
    )
    vals = np.concatenate([v0[in_p0], v1[in_p1]])
    wet = vals > 1e-3
    if wet.any():
        vmin = max(float(vals[wet].min()), 0.0) if field == "h" else float(vals[wet].min())
        vmax = float(vals[wet].max())
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, p, tri_c, v, label, t_used in [
        (axes[0], p0, tri0_c, v0, "DG(0)", t0),
        (axes[1], p1, tri1_c, v1, "DG(1)", t1),
    ]:
        if tri_c.shape[0] == 0:
            ax.text(0.5, 0.5, "no cells in window", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        triang = mtri.Triangulation(p[:, 0], p[:, 1], tri_c)
        levels = np.linspace(vmin, vmax, 32)
        cs = ax.tricontourf(triang, v, levels=levels, cmap=cmap, extend="both")
        # Overlay mesh outline (light) so cell scale is visible.
        ax.triplot(triang, color="k", linewidth=0.05, alpha=0.25)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  t={t_used:.2f}s")
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    cbar = fig.colorbar(cs, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(title_root)
    fig.suptitle(f"{title_root}  —  zoom on downstream wave front  t≈{t_target:.1f}s")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 1
    pvd0 = argv[1]
    pvd1 = argv[2]
    out_dir = argv[3]
    os.makedirs(out_dir, exist_ok=True)
    # Snapshots that should show the wave-front in the downstream region.
    targets = [15.0, 30.0, 60.0, 100.0]
    for t in targets:
        for field in ("h", "eta"):
            tag = f"t{int(round(t)):03d}"
            _make_pair(
                pvd0, pvd1, t, field,
                os.path.join(out_dir, f"{field}_{tag}.png"),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
