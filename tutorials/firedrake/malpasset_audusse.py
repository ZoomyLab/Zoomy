#!/usr/bin/env python3
"""Malpasset — **Audusse + hydrostatic_pressure** formulation.

Sibling experiment to ``malpasset_viscous_v2.py`` (NCP-based baseline).
This file inherits the NCP version's IC loader / boundary conditions /
viscous & friction operators and **only** overrides the operator slots
that distinguish the two well-balancing schemes:

==========================  =====================  ===========================
slot                        NCP baseline (v2)      Audusse (this file)
==========================  =====================  ===========================
``flux``                    convective only        convective only
``hydrostatic_pressure``    zeros (default)        ``½ g h² I`` on diagonals
``nonconservative_matrix``  ``g h ∂(b+h)``         zeros
==========================  =====================  ===========================

Under Audusse the well-balancing is anchored on ``hydrostatic_pressure``
read at the HR'd face states (``h_star = max(0, h + b − b_star)`` with
``b_star = max(b_L, b_R)``).  ``PositiveHLL`` / ``PositiveRusanov``
evaluate ``(flux + hydrostatic_pressure) @ n`` on the HR'd states and
add the ``(P_raw - P_star) @ n`` jump as a per-cell fluctuation —
together these give lake-at-rest preservation without any explicit
``grad(b)`` source.

Run::

    conda activate zoomy-host-mpi
    mpirun -n 4 apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        python3 -u tutorials/firedrake/malpasset_audusse.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import sympy as sp  # noqa: E402
from sympy import Matrix  # noqa: E402
import firedrake as fd  # noqa: E402,F401
from mpi4py import MPI  # noqa: E402

from malpasset_viscous_v2 import (  # noqa: E402
    MalpassetSWE,
    MalpassetSolver,
    bcs,  # noqa: F401
    INPUT_MESH,
    _total_water_volume,
    MANNING_N,
    EPS_WD,
    H_FRICTION_FLOOR,
    NU,
)
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL  # noqa: E402
from zoomy_core.misc.misc import Zstruct, ZArray  # noqa: E402


# ----------------------------------------------------------------------
# Test-case parameters.
# ----------------------------------------------------------------------
TIME_END = 2.0
CFL = 0.5
DG_DEGREE = 0          # DG(0) for this Audusse debugging pass
LIMITER = "none"
SNAPSHOTS = 20
OUTPUT_TAG = "audusse"


class AudusseMalpassetSWE(MalpassetSWE):
    """``MalpassetSWE`` with Audusse + ``hydrostatic_pressure`` wiring.

    Drops the ``grad(b)`` / ``grad(h)`` entries from
    ``nonconservative_matrix`` and exposes ``g·h²/2`` on the momentum
    diagonals of ``hydrostatic_pressure`` instead — see module
    docstring for why that's the right slot.
    """

    def flux(self):
        _, h, hu, hv, hinv = self._primitives()
        F = Matrix.zeros(4, 2)
        F[1, 0] = hu
        F[1, 1] = hv
        # Pure convective — pressure moved to hydrostatic_pressure.
        F[2, 0] = hu * hu * hinv
        F[2, 1] = hu * hv * hinv
        F[3, 0] = hu * hv * hinv
        F[3, 1] = hv * hv * hinv
        return ZArray(F)

    def hydrostatic_pressure(self):
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        P = Matrix.zeros(4, 2)
        P[2, 0] = sp.Rational(1, 2) * g * h * h
        P[3, 1] = sp.Rational(1, 2) * g * h * h
        return ZArray(P)

    def nonconservative_matrix(self):
        # Empty — Audusse HR absorbs grad(b) via h_star.
        return ZArray.zeros(4, 4, 2)


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print(
            f"[malpasset_audusse] mpi_size={size}  dg={DG_DEGREE}  "
            f"limiter={LIMITER}  CFL={CFL}  t_end={TIME_END}  "
            f"tag={OUTPUT_TAG}",
            flush=True,
        )
        print(
            f"[malpasset_audusse] params: ν={NU} Manning_n={MANNING_N} "
            f"eps_wd={EPS_WD} h_friction_floor={H_FRICTION_FLOOR}",
            flush=True,
        )

    model = AudusseMalpassetSWE()
    output_dir = f"outputs/malpasset_{OUTPUT_TAG}"
    settings = Settings(
        name=f"malpasset-{OUTPUT_TAG}",
        output=Zstruct(
            directory=output_dir,
            snapshots=SNAPSHOTS,
            filename="dg",
            clean_directory=True,
        ),
    )
    solver = MalpassetSolver(
        settings=settings,
        time_end=TIME_END,
        CFL=CFL,
        dg_degree=DG_DEGREE,
        limiter=LIMITER,
        riemann_solver_cls=PositiveNonconservativeHLL,
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
            f"[malpasset_audusse DONE] tag={OUTPUT_TAG}  "
            f"wall={t1 - t0:.2f}s  n_iter={n_iter}  "
            f"final_t={final_t:.3f}s  V0={V0:.6e}  V1={V1:.6e}  "
            f"ΔV/V0={dV_rel:+.3e}",
            flush=True,
        )
        print(
            f"[malpasset_audusse DONE] output: "
            f"{os.path.abspath(output_dir)}/simulation.pvd",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
