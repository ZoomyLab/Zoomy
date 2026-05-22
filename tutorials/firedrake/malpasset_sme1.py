#!/usr/bin/env python3
"""Malpasset SME(L=1) — full library NCP form on the production mesh.

Companion to ``tutorials/firedrake/run_malpasset_mpi.py``: same
26k-triangle mesh, same dam-break IC, same Audusse + Rusanov
positivity-preserving Riemann route — but the symbolic Model is now
``SME1Viscous2D`` (level-1 Shallow Moment with τ_xx + τ_xz, full
library cross-coupling), not the depth-mean ``MalpassetSWE``.

State (per cell):  ``[b, h, hu_0, hv_0, hu_1, hv_1]``  (6 conserved).

The 1D Galerkin projection of ``notebooks/010`` is extended to 2D
in ``tutorials/sme/sme1_viscous.py`` by symmetry — the x-row carries
the verified 1D library extras, the y-row mirrors with ``u↔v, x↔y``.
A full 2D re-derivation in the same notebook style is the natural
next step but lives in a different ticket.

Initial conditions:  reuse the Malpasset SWE IC for ``[b, h, hu, hv]``
and zero out the shear moments ``hu_1 = hv_1 = 0``.

Run::

    PYTHONPATH=library/zoomy_core:library/zoomy_firedrake \\
        mpirun -n 4 apptainer exec --bind $PWD:$PWD \\
            zoomy_firedrake.sif \\
            python3 -u tutorials/firedrake/malpasset_sme1.py
"""
from __future__ import annotations

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

# Force the editable install to win over the SIF's system zoomy_core
# (which is missing the ``legacy/`` folder).  Same trick the
# README documents.
for _p in (
    os.path.join(_REPO_ROOT, "library", "zoomy_core"),
    os.path.join(_REPO_ROOT, "library", "zoomy_firedrake"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import firedrake as fd  # noqa: E402
from mpi4py import MPI  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "tutorials", "sme"))
from sme1_viscous import SME1Viscous2D  # noqa: E402

# Reuse Malpasset IC loader + bcs from the SWE production runner.
from malpasset_viscous_v2 import (  # noqa: E402
    INPUT_MESH, _meshio_mesh, _build_vertex_permutation,
    _total_water_volume, MANNING_N, NU, EPS_WD,
)
import zoomy_core.model.boundary_conditions as BC  # noqa: E402
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov  # noqa: E402
from zoomy_core.misc.misc import Zstruct  # noqa: E402
from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver  # noqa: E402


TIME_END = 20.0
CFL = 0.4
DG_DEGREE = 0          # DG(0) cell-mean — matches existing Malpasset DG(0) baseline
LIMITER = "none"       # no limiter at DG(0)
SNAPSHOTS = 20
OUTPUT_TAG = "malpasset_sme1"


class MalpassetSME1Solver(FiredrakeHyperbolicSolver):
    """Loads Malpasset IC into the 6-state ``[b, h, hu_0, hv_0, hu_1, hv_1]``.

    Reuse the legacy mesh permutation + point-data loader from
    ``malpasset_viscous_v2``; zero out the shear moments.
    """

    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        V_CG = fd.VectorFunctionSpace(
            mesh, "CG", 1, dim=Q.function_space().value_size
        )
        Q_CG = fd.Function(V_CG)
        mio = _meshio_mesh()
        perm = _build_vertex_permutation(mesh, mio)
        pd = mio.point_data
        Q_CG.dat.data[:, 0] = pd["B"][perm]                # b
        Q_CG.dat.data[:, 1] = pd["H"][perm]                # h
        Q_CG.dat.data[:, 2] = (pd["H"] * pd["U"])[perm]    # hu_0
        Q_CG.dat.data[:, 3] = (pd["H"] * pd["V"])[perm]    # hv_0
        Q_CG.dat.data[:, 4] = 0.0                          # hu_1
        Q_CG.dat.data[:, 5] = 0.0                          # hv_1
        Q.project(Q_CG)


# Wall BCs on the exterior — apply to *both* moment vectors as
# distinct velocity vectors.
BCS = BC.BoundaryConditions(
    [BC.Wall(tag="wall",
             momentum_field_indices=[[2, 3], [4, 5]],
             permeability=0.0, wall_slip=1.0)]
)


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print(
            f"[malpasset_sme1] mpi_size={size}  dg={DG_DEGREE}  "
            f"limiter={LIMITER}  CFL={CFL}  t_end={TIME_END}  "
            f"snapshots={SNAPSHOTS}  output_tag={OUTPUT_TAG}",
            flush=True,
        )
        print(f"[malpasset_sme1] mesh={INPUT_MESH}", flush=True)

    # K&T form first — the full-library cross-coupling extras (set
    # cross_coupling=True) carry ``-6 u_0² ∂_x b`` terms that grow
    # unboundedly on Malpasset's steep bathymetry; the published
    # K&T form drops them precisely because they're ill-posed in
    # complex topography.
    model = SME1Viscous2D(g=9.81, nu=NU, eps=EPS_WD, tau_b=0.0,
                          cross_coupling=False,
                          boundary_conditions=BCS)

    output_dir = f"outputs/{OUTPUT_TAG}"
    settings = Settings(
        name=OUTPUT_TAG,
        output=Zstruct(
            directory=output_dir,
            snapshots=SNAPSHOTS,
            filename="dg",
            clean_directory=True,
        ),
    )

    solver = MalpassetSME1Solver(
        settings=settings,
        time_end=TIME_END,
        CFL=CFL,
        dg_degree=DG_DEGREE,
        limiter=LIMITER,
        riemann_solver_cls=PositiveNonconservativeRusanov,
    )

    solver.setup_simulation(INPUT_MESH, model)
    V0 = _total_water_volume(solver)
    t0 = time.perf_counter()
    solver.run_simulation()
    t1 = time.perf_counter()
    V1 = _total_water_volume(solver)
    dV_rel = (V1 - V0) / V0 if V0 != 0.0 else float("nan")
    n_iter = int(getattr(solver._state, "last_iteration_count", 0))
    final_t = float(getattr(solver._state, "sim_time", 0.0))

    if rank == 0:
        print(
            f"[malpasset_sme1 DONE] tag={OUTPUT_TAG}  "
            f"wall={t1 - t0:.2f}s n_iter={n_iter}  final_t={final_t:.3f}s  "
            f"V0={V0:.6e} V1={V1:.6e} ΔV/V0={dV_rel:+.3e}",
            flush=True,
        )
        print(
            f"[malpasset_sme1 DONE] output: "
            f"{os.path.abspath(output_dir)}/simulation.pvd",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
