#!/usr/bin/env python3
"""1D cross-section through the Malpasset channel — DG(0) vs DG(1).

Picks a fixed line (y = const, x in range), samples h and eta along it
via barycentric interpolation in the unstructured mesh, and overplots
both cases so cell-scale wiggles vs smooth-mean is visible.
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

from render_pvd import _read_pvtu_merged


# Channel cross-section line: through the downstream wave region.
Y_LINE = 3000.0
X_RANGE = (10000.0, 17500.0)
N_SAMPLE = 600


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


def _sample_along_line(pts, tri, values, xs, y):
    """Linear interp values on (xs, y) — nearest cell, barycentric weights.

    Avoids matplotlib's triangulation validity check (which fails on
    multi-rank pieces with duplicate vertices) by using scipy's KDTree
    on triangle centroids, then doing barycentric interp within the
    nearest triangle.
    """
    from scipy.spatial import cKDTree
    cents = pts[tri].mean(axis=1)
    tree = cKDTree(cents[:, :2])
    out = np.full(xs.shape, np.nan)
    targets = np.column_stack([xs, np.full_like(xs, y)])
    # Search a few candidates per point and use the first one whose
    # barycentric weights are all >= -eps (i.e. the point is inside).
    _, candidates = tree.query(targets, k=12)
    for i, (xq, yq) in enumerate(targets):
        for ci in candidates[i]:
            a, b, c = pts[tri[ci]]
            denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denom) < 1e-14:
                continue
            wa = ((b[1] - c[1]) * (xq - c[0]) + (c[0] - b[0]) * (yq - c[1])) / denom
            wb = ((c[1] - a[1]) * (xq - c[0]) + (a[0] - c[0]) * (yq - c[1])) / denom
            wc = 1.0 - wa - wb
            if min(wa, wb, wc) >= -1e-6:
                va, vb, vc = values[tri[ci]]
                out[i] = wa * va + wb * vb + wc * vc
                break
    return out


def main(argv):
    pvd0 = argv[1]
    pvd1 = argv[2]
    out_dir = argv[3]
    os.makedirs(out_dir, exist_ok=True)

    times = [15.0, 30.0, 60.0, 100.0]
    fig, axes = plt.subplots(len(times), 1, figsize=(13, 10), sharex=True)
    xs = np.linspace(X_RANGE[0], X_RANGE[1], N_SAMPLE)
    for ax, t_target in zip(axes, times):
        t0, p0, tri0, d0 = _read_at_time(pvd0, t_target)
        t1, p1, tri1, d1 = _read_at_time(pvd1, t_target)
        eta0 = _sample_along_line(p0, tri0, d0["Q0"] + d0["Q1"], xs, Y_LINE)
        eta1 = _sample_along_line(p1, tri1, d1["Q0"] + d1["Q1"], xs, Y_LINE)
        b0 = _sample_along_line(p0, tri0, d0["Q0"], xs, Y_LINE)
        ax.plot(xs, b0, "k-", linewidth=1.0, alpha=0.5, label="b (bathy)")
        ax.plot(xs, eta0, "C0-", linewidth=1.3, label=f"DG(0) eta (t={t0:.1f}s)")
        ax.plot(xs, eta1, "C3-", linewidth=1.0, label=f"DG(1) eta (t={t1:.1f}s)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel(f"η, b [m]\n(t≈{t_target:.0f}s)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(f"x [m] along y = {Y_LINE:.0f} m")
    fig.suptitle("Cross-section eta — DG(0) vs DG(1) at matched times")
    fig.tight_layout()
    out = os.path.join(out_dir, "section_eta.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
