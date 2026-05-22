#!/usr/bin/env python3
"""Malpasset — single-step diagnostic.

Runs ONE time step from the IC and inspects per-rank state deltas.
Goal: tell at a glance whether the convective step is actually
producing change, i.e. whether the dam-break is dynamic at all.
Lighter than a multi-second simulation — finishes in seconds.

Run::

    apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        python3 -u tutorials/firedrake/malpasset_one_step_diag.py
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import numpy as np  # noqa: E402
import firedrake as fd  # noqa: E402,F401
from mpi4py import MPI  # noqa: E402

from malpasset_viscous_v2 import (  # noqa: E402
    MalpassetSWE,
    MalpassetSolver,
    INPUT_MESH,
)
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveRusanov  # noqa: E402
from zoomy_core.misc.misc import Zstruct  # noqa: E402


DG_DEGREE = 0          # DG(0) keeps the wet/dry boundary diagnosable
                       # — DG(1) limiter masks per-cell evolution.
LIMITER = "none"
DT = 0.02
N_STEPS = 50           # ~1 s of physical time; lets the wave propagate
                       # a few cells beyond the dam so we can see whether
                       # it follows topography or fans out symmetrically.
TAG = "one_step_diag"


def _snapshot(Q):
    """Snapshot of the Q DOF data (per rank, no halo exchange)."""
    return Q.dat.data_ro.copy()


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    model = MalpassetSWE()
    settings = Settings(
        name=f"malpasset-{TAG}",
        output=Zstruct(
            directory=f"outputs/malpasset_{TAG}",
            snapshots=2,
            filename="dg",
            clean_directory=True,
        ),
    )
    solver = MalpassetSolver(
        settings=settings,
        time_end=1.0,  # unused — we call solver.step() directly
        CFL=0.5,
        dg_degree=DG_DEGREE,
        limiter=LIMITER,
        riemann_solver_cls=PositiveRusanov,
    )
    solver.setup_simulation(INPUT_MESH, model)

    # IC snapshot (Q layout: [b, h, hu, hv] per node).
    Q0 = _snapshot(solver._state.Qnp1)

    # Step forward N times at fixed dt.
    for _ in range(N_STEPS):
        solver.step(DT)

    Q1 = _snapshot(solver._state.Qnp1)
    dQ = Q1 - Q0

    # Per-rank stats.  Columns: b, h, hu, hv.
    NAMES = ["b", "h", "hu", "hv"]
    if rank == 0:
        print(f"[one_step_diag] mpi_size={size} dg={DG_DEGREE} "
              f"limiter={LIMITER} dt={DT} n_steps={N_STEPS} "
              f"(t={DT*N_STEPS:.2f}s)", flush=True)
        print(f"[one_step_diag] Q shape = {Q0.shape}", flush=True)
        # Wet cell evolution (b stays, so location of #wet growth is the wave front).
        h0 = Q0[:, 1]; h1 = Q1[:, 1]
        b = Q0[:, 0]
        for thr in (50.0, 10.0, 1.0, 0.01):
            n0 = int(np.sum(h0 > thr)); n1 = int(np.sum(h1 > thr))
            print(f"  #h>{thr:>5}:  t=0 -> {n0:>6d}   t=end -> {n1:>6d}   "
                  f"Δ={n1-n0:+d}", flush=True)
        # Bathymetry signature: where does new water go?
        newly_wet = (h0 <= 0.01) & (h1 > 0.01)
        drained   = (h0 >= 50.0) & (h1 < 50.0)
        # Coordinates per DOF — we just need to spot-check the wave region.
        coords = solver._state.Qnp1.function_space().mesh().coordinates.dat.data_ro
        # For DG(0) on a triangle, the # of DOFs is cells, and coords are
        # CG vertices (different count).  Skip coords access in DG(0) — use
        # b instead as the spatial proxy.
        print(f"  newly_wet cells: {int(newly_wet.sum())}", flush=True)
        if newly_wet.any():
            bnw = b[newly_wet]
            print(f"    b stats:    mean={bnw.mean():.2f}   "
                  f"min={bnw.min():.2f}   max={bnw.max():.2f}", flush=True)
        print(f"  drained-deep cells: {int(drained.sum())}", flush=True)
        # Global b distribution of dry cells at t=0 — for reference.
        dry0 = h0 <= 0.01
        if dry0.any():
            print(f"  dry@t=0 b stats: mean={b[dry0].mean():.2f}   "
                  f"min={b[dry0].min():.2f}   max={b[dry0].max():.2f}",
                  flush=True)
        # Where is max|Δh|?  What's b there?  Where is max|Δhu|?
        dh = Q1[:, 1] - Q0[:, 1]
        dhu = Q1[:, 2] - Q0[:, 2]
        i_dh  = int(np.abs(dh).argmax())
        i_dhu = int(np.abs(dhu).argmax())
        print(f"  max|Δh|   at: idx={i_dh}   b={b[i_dh]:.2f}   "
              f"h0={Q0[i_dh,1]:.2f}   Δh={dh[i_dh]:+.3f}", flush=True)
        print(f"  max|Δhu|  at: idx={i_dhu}  b={b[i_dhu]:.2f}  "
              f"h0={Q0[i_dhu,1]:.2f}  Δhu={dhu[i_dhu]:+.3f}", flush=True)
        # Histogram of newly_wet b in 20-unit bins to see distribution.
        bins = [0, 20, 40, 60, 80, 100, 110]
        if newly_wet.any():
            hist, _ = np.histogram(b[newly_wet], bins=bins)
            print("  newly_wet b histogram:", flush=True)
            for j, c in enumerate(hist):
                print(f"    b ∈ [{bins[j]:3d}, {bins[j+1]:3d}): {c} cells", flush=True)
        # Where IS the reservoir? Show b histogram of cells with h>50.
        deep = h0 > 50.0
        if deep.any():
            hist, _ = np.histogram(b[deep], bins=bins)
            print("  reservoir (h>50) b histogram:", flush=True)
            for j, c in enumerate(hist):
                print(f"    b ∈ [{bins[j]:3d}, {bins[j+1]:3d}): {c} cells", flush=True)
        # Where are the dry cells?  Distribution of b for h<0.01.
        if dry0.any():
            hist, _ = np.histogram(b[dry0], bins=bins)
            print("  dry@t=0 b histogram:", flush=True)
            for j, c in enumerate(hist):
                print(f"    b ∈ [{bins[j]:3d}, {bins[j+1]:3d}): {c} cells", flush=True)
        # Free-surface height (h+b) in the reservoir.
        if deep.any():
            fs = (Q0[deep, 0] + Q0[deep, 1])
            print(f"  free-surface h+b in reservoir: min={fs.min():.2f}  "
                  f"max={fs.max():.2f}  mean={fs.mean():.2f}", flush=True)

    # Reduce stats across ranks for a global view.
    comm = MPI.COMM_WORLD
    for i, name in enumerate(NAMES):
        loc_max_abs_dQ = float(np.abs(dQ[:, i]).max()) if Q0.size else 0.0
        loc_l1_dQ = float(np.abs(dQ[:, i]).sum())
        loc_argmax = int(np.abs(dQ[:, i]).argmax()) if Q0.size else 0
        loc_h_at_argmax = float(Q0[loc_argmax, 1]) if Q0.size else 0.0
        loc_b_at_argmax = float(Q0[loc_argmax, 0]) if Q0.size else 0.0
        glo_max = comm.allreduce(loc_max_abs_dQ, op=MPI.MAX)
        glo_l1 = comm.allreduce(loc_l1_dQ, op=MPI.SUM)
        if rank == 0:
            print(
                f"  Δ{name:>2s}: max|Δ| = {glo_max:.6e}   "
                f"L1|Δ|  = {glo_l1:.6e}",
                flush=True,
            )

    # Count cells whose h changed beyond a small threshold.
    if rank == 0:
        h_eps = 1e-8
        n_h_changed = int(np.abs(dQ[:, 1] > h_eps).sum())
        glo_n_h_changed = comm.allreduce(n_h_changed, op=MPI.SUM) if size > 1 else n_h_changed
        print(f"[one_step_diag] cells with |Δh|>{h_eps}: {glo_n_h_changed}",
              flush=True)
    else:
        comm.allreduce(0, op=MPI.SUM)

    return 0


if __name__ == "__main__":
    sys.exit(main())
