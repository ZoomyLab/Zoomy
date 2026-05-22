#!/usr/bin/env python3
"""SME(1) 1D dam break — smoke test for the full-library SystemModel.

Runs ``tutorials.sme.sme1_viscous.SME1Viscous`` (the K&T 2019 eq
(4.14) form augmented with the full library projection's
cross-coupling NCP terms — see ``notebooks/010``) on a 1D interval
mesh from t=0 to t=TIME_END.

IC: classic Stoker dam break, ``h(x) = h_L`` for ``x < x_dam``,
``h_R`` otherwise; ``b ≡ 0``; ``hu_0 = hu_1 = 0``.

Expected: with ``hu_1 ≡ 0`` initially SME(1) starts SWE-like; the
j=1 cross-coupling spins up vertical shear over time.
"""
from __future__ import annotations

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import firedrake as fd  # noqa: E402
from mpi4py import MPI  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "tutorials", "sme"))
from sme1_viscous import SME1Viscous  # noqa: E402

import zoomy_core.model.boundary_conditions as BC  # noqa: E402
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov  # noqa: E402
from zoomy_core.misc.misc import Zstruct  # noqa: E402
from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver  # noqa: E402


# ---------------------------------------------------------------------
TIME_END = 20.0
CFL = 0.4
DG_DEGREE = 0
LIMITER = "none"
SNAPSHOTS = 40

L_CHANNEL = 100.0
N_CELLS = 400

H_LEFT = 2.0       # gentler IC than the classical 10:1
H_RIGHT = 1.0
X_DAM = L_CHANNEL / 2.0
G_GRAVITY = 9.81
NU_VISC = 0.01      # small ν first; check dam-break wave is stable

OUTPUT_TAG = "sme1_dam_break"


class SME1Solver(FiredrakeHyperbolicSolver):
    """FiredrakeHyperbolicSolver with the 1D dam-break IC hardcoded."""

    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        # CG1 staging space — same trick MalpassetSolver uses.
        V_CG = fd.VectorFunctionSpace(
            mesh, "CG", 1, dim=Q.function_space().value_size
        )
        Q_CG = fd.Function(V_CG)
        # Vertex coordinates of the CG(1) space.
        x_coords = fd.Function(
            fd.FunctionSpace(mesh, "CG", 1)
        ).interpolate(fd.SpatialCoordinate(mesh)[0])
        xs = x_coords.dat.data_ro[:]
        # State layout: [b, h, hu_0, hu_1].
        Q_CG.dat.data[:, 0] = 0.0                                        # b
        Q_CG.dat.data[:, 1] = (xs < X_DAM) * H_LEFT + (xs >= X_DAM) * H_RIGHT
        Q_CG.dat.data[:, 2] = 0.0                                        # hu_0
        Q_CG.dat.data[:, 3] = 0.0                                        # hu_1
        Q.project(Q_CG)


# Zero-Neumann (extrapolation) BCs on both ends — water leaves the
# domain freely once the dam-break front reaches the wall.  Wall BC
# is not used because its momentum-reflection assumes a velocity
# vector, while ``hu_0`` and ``hu_1`` are *different* moments of the
# same velocity profile (depth-mean and first shear) — they don't
# reflect together as a vector.
BCS = BC.BoundaryConditions(
    [BC.Extrapolation(tag="open")]
)


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print(
            f"[sme1_dam_break] L={L_CHANNEL}, N={N_CELLS}, t_end={TIME_END}, "
            f"CFL={CFL}, DG={DG_DEGREE}, ν={NU_VISC}",
            flush=True,
        )

    mesh = fd.IntervalMesh(N_CELLS, L_CHANNEL)
    model = SME1Viscous(g=G_GRAVITY, nu=NU_VISC, tau_b=1e-12,
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

    solver = SME1Solver(
        settings=settings,
        time_end=TIME_END,
        CFL=CFL,
        dg_degree=DG_DEGREE,
        limiter=LIMITER,
        riemann_solver_cls=PositiveNonconservativeRusanov,
    )

    t0 = time.perf_counter()
    solver.setup_simulation(mesh, model)
    solver.run_simulation()
    t1 = time.perf_counter()

    if rank == 0:
        s = solver._state
        n_iter = int(getattr(s, "last_iteration_count", 0))
        final_t = float(getattr(s, "sim_time", 0.0))
        print(
            f"[sme1_dam_break DONE] wall={t1 - t0:.2f}s n_iter={n_iter} "
            f"final_t={final_t:.3f}s "
            f"output={os.path.abspath(output_dir)}/simulation.pvd",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
