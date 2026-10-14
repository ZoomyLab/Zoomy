"""Clamp-saturated cells of a Malpasset SME(1) store (the thesis run's layout: Q rows b, h,
q_x_0, q_x_1, q_y_0, q_y_1): wet cells (h > eps) whose mode-0 momentum sits at the per-component
cap |q_*_0| >= u_max (h - eps), classified by their face neighbours:

  pit    every neighbour is a dry bank above the cell's free surface or wet at the same level
  spill  a dry neighbour whose bed lies below the cell's free surface
  other  wet neighbours at other free-surface levels, no spill

    python malpasset_clamp_cells.py STORE.h5 [MESH.h5]   (MESH: a store with /mesh/cell_neighbors,
                                                        e.g. the thesis run output/zp_l1/zp_l1.h5)

2026-10-14, t = 2000 s: old solver_jax (zp_l1) 578; symbolic solver 565 (pit 13, spill 287,
other 265); with reconstruction.shore_extension 194 (pit 19, spill 119, other 56);
limiter Barth-Jespersen or minmod 5.
"""
import sys

import h5py
import numpy as np

EPS, U_MAX = 1e-3, 30.0


def main(store, mesh_store=None):
    with h5py.File(mesh_store or store, "r") as f:
        nb = np.asarray(f["mesh"]["cell_neighbors"])
    with h5py.File(store, "r") as f:
        fl = f["fields"]
        snaps = sorted(fl, key=lambda s: float(fl[s]["time"][()]))
        for s in snaps[-1:]:
            Q = np.asarray(fl[s]["Q"])
            t = float(fl[s]["time"][()])
    n = Q.shape[1]
    h, b = Q[1], Q[0]
    eta = b + h
    cap = U_MAX * np.maximum(h - EPS, 0.0)
    sat = ((np.abs(Q[2]) >= cap - 1e-12) | (np.abs(Q[4]) >= cap - 1e-12)) & (h > EPS)
    nb = nb[:n]
    ok = (nb >= 0) & (nb < n)
    k = np.clip(nb, 0, n - 1)
    wetn = ok & (h[k] > EPS)
    other_level = np.sum(wetn & (np.abs(eta[k] - eta[:, None]) > 1e-9), 1)
    spill = np.sum(ok & ~wetn & (b[k] < eta[:, None]), 1) > 0
    pit = ~spill & (other_level == 0)
    print(f"t = {t:.1f} s: {sat.sum()} clamp-saturated cells: pit {(sat & pit).sum()}, "
          f"spill {(sat & spill).sum()}, other {(sat & ~pit & ~spill).sum()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
