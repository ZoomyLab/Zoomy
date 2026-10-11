"""Deliverable: 3-D Euler on the JAX symbolic solver, from the store
(``store_runs.py`` -> ``store/*.h5``) -> ``euler3d.png``.

(a) Sod tube along x, y and z of a 100 x 2 x 2 hexahedral box (O2 + MOOD),
    density along the tube axis vs the exact Riemann solution;
(b) wall box 24^3, density on the mid-plane z = 0.4 through the pulse centre
    at the last snapshot.
    python deliverable.py"""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from euler_common import SOD, exact_riemann
from zoomy_plotting import figure_legend, label_panels, read_hdf5, subplots
from zoomy_plotting.plot import style

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store")

style.use("thesis")
fig, (ax, bx) = subplots(1, 2, preset="thesis")

for axis, name in enumerate("xyz"):
    with read_hdf5(os.path.join(STORE, f"sod_{name}.h5")) as st:
        k = st.n_snapshots - 1
        x = st.cell_centers[:, axis]
        rho = st.get_cell(k, "rho")
        o = np.argsort(x, kind="stable")
        ax.plot(x[o], rho[o], marker="None", lw=1.2 if axis == 0 else 0.8,
                ls=("-", "--", ":")[axis], label=f"tube along {name}")
        t = float(st.times[k])
xs = np.r_[np.linspace(0.0, 1.0, 30), 0.5 + 1.75 * t + np.array([-0.01, 0.01]),
           0.5 + 0.93 * t + np.array([-0.01, 0.01])]
ex = exact_riemann(SOD[0], SOD[1], (np.sort(xs) - 0.5) / t)
ax.plot(np.sort(xs), ex[0], ls="none", marker="o", mfc="none", zorder=10,
        color="k", label="exact")
ax.set_xlabel("coordinate along the tube")
ax.set_ylabel(r"$\rho$")
ax.set_title(f"Sod, t = {t:.2f}")

with read_hdf5(os.path.join(STORE, "wall_pulse.h5")) as st:
    k = st.n_snapshots - 1
    c = st.cell_centers
    rho = st.get_cell(k, "rho")
    n = round(len(rho) ** (1 / 3))
    zs = np.unique(np.round(c[:, 2], 12))
    zc = zs[np.argmin(np.abs(zs - 0.4))]
    sl = np.isclose(c[:, 2], zc)
    xi = np.rint(c[sl, 0] * n - 0.5).astype(int)
    yi = np.rint(c[sl, 1] * n - 0.5).astype(int)
    img = np.full((n, n), np.nan)
    img[yi, xi] = rho[sl]
    im = bx.imshow(img, origin="lower", extent=(0, 1, 0, 1),
                   cmap=style.CMAP_CONTINUOUS if hasattr(style, "CMAP_CONTINUOUS")
                   else "viridis")
    fig.colorbar(im, ax=bx, label=r"$\rho$")
    bx.set_xlabel("x")
    bx.set_ylabel("y")
    bx.set_title(f"wall box, z = {zc:.2f}")

label_panels(fig, [ax, bx])
figure_legend(fig)
out = os.path.join(HERE, "euler3d.png")
fig.savefig(out)
print(out)
