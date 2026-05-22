#!/usr/bin/env python3
"""Tight per-panel zoom on a 2x2 km wave-front region.

Renders each (case, field, time) as one big PNG so cell-scale features
are clearly visible.  Mesh outline overlaid.
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

from render_pvd import _read_pvtu_merged


WINDOWS = {
    # Upstream reservoir — where the user spotted ripples (around id 4276,
    # coords ≈ (5000, 4300)).
    "reservoir_t05": dict(t=5.0, xlim=(3500, 6500), ylim=(3500, 5500)),
    "reservoir_t15": dict(t=15.0, xlim=(3500, 6500), ylim=(3500, 5500)),
    "reservoir_t30": dict(t=30.0, xlim=(3500, 6500), ylim=(3500, 5500)),
    "reservoir_t60": dict(t=60.0, xlim=(3500, 6500), ylim=(3500, 5500)),
    # Downstream wave front (previous focus).
    "front_t30": dict(t=30.0, xlim=(14000, 17500), ylim=(2500, 5500)),
    "front_t60": dict(t=60.0, xlim=(13000, 17500), ylim=(-1000, 4000)),
}


def _read_at_time(pvd_path, t_target):
    base = os.path.dirname(pvd_path)
    tree = ET.parse(pvd_path)
    entries = [
        (float(ds.get("timestep")), ds.get("file"))
        for ds in tree.findall(".//DataSet")
    ]
    entries.sort(key=lambda e: e[0])
    i = int(np.argmin([abs(e[0] - t_target) for e in entries]))
    t, fname = entries[i]
    pts, tri, data = _read_pvtu_merged(os.path.join(base, fname))
    return t, pts, tri, data


def _crop_tri(pts, tri, xlim, ylim):
    inx = (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1])
    iny = (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
    keep_pt = inx & iny
    in_tri = keep_pt[tri].all(axis=1)
    return tri[in_tri]


def _render_one(pts, tri_c, values, vmin, vmax, title, cmap, out_path,
                xlim, ylim, show_mesh=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    if tri_c.shape[0] == 0:
        ax.text(0.5, 0.5, "no cells in window", transform=ax.transAxes,
                ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri_c)
    levels = np.linspace(vmin, vmax, 41)
    cs = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend="both")
    if show_mesh:
        ax.triplot(triang, color="k", linewidth=0.15, alpha=0.4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(cs, ax=ax, shrink=0.85)
    cbar.set_label(title.split("(")[0].strip())
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv):
    pvd0 = argv[1]
    pvd1 = argv[2]
    out_dir = argv[3]
    os.makedirs(out_dir, exist_ok=True)

    for tag, win in WINDOWS.items():
        t_target = win["t"]
        xlim, ylim = win["xlim"], win["ylim"]
        for case, pvd in [("dg0", pvd0), ("dg1", pvd1)]:
            t, pts, tri, data = _read_at_time(pvd, t_target)
            tri_c = _crop_tri(pts, tri, xlim, ylim)
            h = np.clip(data["Q1"], 0.0, None)
            eta = data["Q0"] + data["Q1"]
            in_pt = (
                (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1])
                & (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
            )
            h_win = h[in_pt]
            eta_win = eta[in_pt]
            h_max = float(h_win.max()) if h_win.size else 1.0
            eta_lo = float(eta_win.min()) if eta_win.size else 0.0
            eta_hi = float(eta_win.max()) if eta_win.size else 1.0
            _render_one(
                pts, tri_c, h, 0.0, max(h_max, 1.0),
                f"h [m]  ({case.upper()}, t={t:.2f}s)",
                "Blues",
                os.path.join(out_dir, f"h_{tag}_{case}.png"),
                xlim, ylim,
            )
            _render_one(
                pts, tri_c, eta, eta_lo, eta_hi,
                f"eta=h+b [m]  ({case.upper()}, t={t:.2f}s)",
                "viridis",
                os.path.join(out_dir, f"eta_{tag}_{case}.png"),
                xlim, ylim,
            )
            print(f"  {tag} {case}: h_max={h_max:.2f}  eta=[{eta_lo:.2f},{eta_hi:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
