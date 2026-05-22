#!/usr/bin/env python3
"""Minimal channel dam-break — DG(0) vs DG(1) reproducer.

Goal: pin down the DG(1)+vertex stall on a mesh small enough to inspect
DOF-by-DOF.  Uses ``fd.RectangleMesh`` so there is no Gmsh / meshio
dependency, and re-uses ``MalpassetSWE`` so the bug is the same one
the production runner hits.

Geometry
--------
2-D channel ``[0, Lx] × [0, Ly]`` discretised by ``nx × ny``
triangles.  Default: ``Lx=100 m, Ly=10 m, nx=20, ny=2`` — 80 cells.

Initial condition
------------------
- ``h(x, y) = h_high if x < Lx/2 else 0`` — dam break in the middle.
- ``b(x, y) = b_high − (b_high − b_low) · x / Lx`` — linear drop from
  left (``b_high``) to right (``b_low``).  Modest slope so the wave
  has direction without dominating the dynamics.
- ``hu = hv = 0``.

Outputs
-------
``outputs/channel_dam_<TAG>/simulation.pvd`` plus a stdout summary of
``h, hu`` at three probe points after a few steps.

Run::

    apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        python3 -u tutorials/firedrake/channel_dam_break_diag.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import numpy as np  # noqa: E402
import firedrake as fd  # noqa: E402
from mpi4py import MPI  # noqa: E402

from malpasset_viscous_v2 import MalpassetSWE, bcs as MALPASSET_BCS  # noqa: E402
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL  # noqa: E402
from zoomy_core.misc.misc import Zstruct  # noqa: E402
from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver  # noqa: E402
from zoomy_firedrake.firedrake_compat import (  # noqa: E402
    safe_extract_component, safe_assign_component,
)


# ----------------------------------------------------------------------
# Geometry / IC parameters
# ----------------------------------------------------------------------
LX = 100.0          # channel length [m]
LY = 10.0           # channel width  [m]
NX = 20             # cells along x
NY = 2              # cells along y
H_HIGH = 5.0        # reservoir depth (left half) [m]
H_LOW  = 0.0        # downstream depth — set > 0 to disable wet/dry guards
B_HIGH = 0.0        # bathymetry on the left  [m]
B_LOW  = -1.0       # bathymetry on the right [m]
EPS_PROBE = 1e-3    # wet/dry probe threshold (just for the diagnostic)


def make_channel_mesh():
    """Triangular RectangleMesh — quadcells split into diagonal halves."""
    return fd.RectangleMesh(NX, NY, LX, LY, diagonal="left")


class ChannelDamBreakSolver(FiredrakeHyperbolicSolver):
    """``FiredrakeHyperbolicSolver`` with an analytic dam-break IC.

    Overrides only :meth:`set_initial_condition`: no meshio, no
    permutation, just ``fd.SpatialCoordinate`` → analytic h/b/hu/hv.
    """

    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        V_CG = fd.VectorFunctionSpace(
            mesh, "CG", 1, dim=Q.function_space().value_size
        )
        x, y = fd.SpatialCoordinate(mesh)
        b_expr = B_HIGH + (B_LOW - B_HIGH) * x / LX
        h_expr = fd.conditional(x < LX / 2, fd.Constant(H_HIGH),
                                fd.Constant(H_LOW))
        zero = fd.Constant(0.0)
        Q_CG = fd.Function(V_CG)
        Q_CG.interpolate(fd.as_vector([b_expr, h_expr, zero, zero]))
        Q.project(Q_CG)


def run_one(dg_degree, limiter, tag, time_end=2.0, snapshots=10, cfl=0.5):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print(f"\n==== {tag}: DG({dg_degree}) limiter={limiter} "
              f"t_end={time_end} ====", flush=True)

    mesh = make_channel_mesh()
    model = MalpassetSWE()
    # Without an explicit Wall BC every exterior facet leaks: the
    # weak form's boundary integral uses the interior trace as the
    # ghost state, i.e. transmissive boundary — water at h>0 sitting
    # on the boundary just flows out.  Re-use the Malpasset Wall BC
    # so the channel's exterior is closed (mass-conserving).
    model.boundary_conditions = MALPASSET_BCS
    output_dir = f"outputs/channel_dam_{tag}"
    settings = Settings(
        name=f"channel-{tag}",
        output=Zstruct(directory=output_dir, snapshots=snapshots,
                       filename="dg", clean_directory=True),
    )
    solver = ChannelDamBreakSolver(
        settings=settings, time_end=time_end, CFL=cfl,
        dg_degree=dg_degree, limiter=limiter,
        riemann_solver_cls=PositiveNonconservativeHLL,
    )
    solver.setup_simulation(mesh, model)

    # Snapshot at t = 0 (cell-centre probe).
    if rank == 0:
        _probe(solver, "t=0      ")

    t0 = time.perf_counter()
    solver.run_simulation()
    t1 = time.perf_counter()

    n_iter = int(getattr(solver._state, "last_iteration_count", 0))
    final_t = float(getattr(solver._state, "sim_time", 0.0))
    if rank == 0:
        print(f"  [done] wall={t1-t0:.2f}s  n_iter={n_iter}  "
              f"final_t={final_t:.3f}s", flush=True)
        _probe(solver, f"t=final ")


def _probe(solver, label):
    """Print h, hu, hv at three probe x-locations.

    Probes the cell-centre values via the DG(0) projection of the
    state — works for any ``dg_degree``.
    """
    Q = solver._state.Qnp1
    mesh = Q.function_space().mesh()
    n_state = Q.function_space().value_size
    V_DG0 = fd.FunctionSpace(mesh, "DG", 0)
    # Project each state component onto V_DG0 so we get cell-centre values.
    centres = []
    for i in range(n_state):
        qi_dg0 = fd.project(Q[i], V_DG0)
        centres.append(qi_dg0.dat.data_ro.copy())
    # Cell centroid coords (DG0 element).
    coords = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0))
    coords.interpolate(fd.SpatialCoordinate(mesh))
    xy = coords.dat.data_ro
    # Pick probes along the channel midline.
    probe_xs = [0.10 * LX, 0.25 * LX, 0.40 * LX, 0.55 * LX,
                0.70 * LX, 0.90 * LX]
    for px in probe_xs:
        # Find the cell whose centroid is closest to (px, LY/2).
        d = (xy[:, 0] - px) ** 2 + (xy[:, 1] - LY / 2) ** 2
        ic = int(np.argmin(d))
        b, h, hu, hv = (float(centres[k][ic]) for k in range(4))
        print(f"  {label} x≈{px:5.1f}  cell={ic:3d}  "
              f"b={b:+.4f}  h={h:+.4f}  hu={hu:+.4f}  hv={hv:+.4f}",
              flush=True)


def main():
    # Sweep with H_LOW=0 (dry) — both the wet/dry guards and the
    # limiter are active; H_LOW=1e-1 (small but wet) — disables the
    # wet/dry guards while keeping a strong h-jump for the dam break.
    global H_LOW
    for h_low, label in [(0.0, "DRY"), (1e-1, "WET_DOWNSTREAM")]:
        H_LOW = h_low
        print(f"\n#### {label}: H_LOW = {h_low} ####", flush=True)
        run_one(dg_degree=0, limiter="none",   tag=f"dg0_{label}",
                time_end=0.5)
        run_one(dg_degree=1, limiter="none",   tag=f"dg1n_{label}",
                time_end=0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
