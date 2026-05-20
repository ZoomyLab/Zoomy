#!/usr/bin/env python3
"""Malpasset dam-break — MPI production runner.

A thin CLI wrapper around ``MalpassetSWE`` / ``MalpassetSolver`` from
``malpasset_viscous_v2.py``.  No multiple back-to-back configurations,
no benchmark prints — just one simulation per invocation, controllable
from the command line so it can be dropped into a job script.

Quick local check (Docker)
--------------------------

::

    docker run --rm -v $PWD:/work -w /work zoomy-fd-test:local bash -c '
      pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
      pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1
      mpirun -n 4 --allow-run-as-root --oversubscribe \\
          python3 -u tutorials/firedrake/run_malpasset_mpi.py \\
          --time-end 10 --dg-degree 1 --limiter vertex --output-tag prod_test
    '

Apptainer (HPC / different machine)
-----------------------------------

The project publishes a Firedrake apptainer-compatible image at
``ghcr.io/zoomylab/zoomy_firedrake:latest`` — see ``README.md`` and
``docs/book/installation.md``.  **Don't build a custom SIF**; use the
published one:

**One-time** — pull and convert to SIF::

    apptainer build zoomy_firedrake.sif \\
        docker://ghcr.io/zoomylab/zoomy_firedrake:latest

**Run** — single-host, 4 ranks::

    mpirun -n 4 apptainer exec \\
        --bind $PWD:/work \\
        zoomy_firedrake.sif \\
        bash -c '
          # Editable-install overlay: needed only while the
          # DG(1) MPI fixes are unpublished.  Drop these two lines
          # once a wheel containing them is on PyPI / in the image.
          pip install -q -e /work/library/zoomy_core --no-deps 2>&1 | tail -1
          pip install -q --no-deps -e /work/library/zoomy_firedrake 2>&1 | tail -1
          ZOOMY_DIR=/work python3 -u \\
              /work/tutorials/firedrake/run_malpasset_mpi.py \\
              --time-end 100 --dg-degree 1 --limiter vertex \\
              --output-tag prod
        '

**Run on a Slurm cluster** — typical pattern::

    srun -n $SLURM_NTASKS apptainer exec \\
        --bind /scratch/$USER/Zoomy:/work \\
        zoomy_firedrake.sif \\
        bash -c '
          pip install -q -e /work/library/zoomy_core --no-deps 2>&1 | tail -1
          pip install -q --no-deps -e /work/library/zoomy_firedrake 2>&1 | tail -1
          ZOOMY_DIR=/work python3 -u \\
              /work/tutorials/firedrake/run_malpasset_mpi.py \\
              --time-end 100 --dg-degree 1 --limiter vertex \\
              --output-tag $SLURM_JOB_ID
        '

Output lands in ``outputs/malpasset_<tag>/simulation.pvd`` (open in
ParaView).  Per-step iteration logs go to stdout via Firedrake's logger.
"""

import argparse
import os
import sys
import time


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--time-end", type=float, default=100.0,
                   help="Physical simulation end time in seconds. "
                        "Default: 100 s.")
    p.add_argument("--cfl", type=float, default=0.5,
                   help="CFL safety factor on top of the theoretical "
                        "RKDG bound (≤ 1).  Default: 0.5.")
    p.add_argument("--dg-degree", type=int, default=1, choices=[0, 1],
                   help="DG polynomial degree.  Default: 1.")
    p.add_argument("--limiter", default="vertex",
                   choices=["none", "vertex", "p_weighted"],
                   help="Slope limiter (only consulted for "
                        "dg-degree >= 1).  Default: vertex (Kuzmin).")
    p.add_argument("--snapshots", type=int, default=20,
                   help="Number of evenly-spaced ParaView snapshots "
                        "to write (including t=0 and t=time-end).  "
                        "Default: 20.")
    p.add_argument("--output-tag", default="prod",
                   help="Label appended to the output directory.  "
                        "Default: 'prod'.")
    p.add_argument("--mesh", default=None,
                   help="Path to the Malpasset .msh file.  "
                        "Default: data/malpasset/geo_malpasset-small.msh "
                        "under $ZOOMY_DIR.")
    return p.parse_args()


# Parse CLI args FIRST, then strip them from ``sys.argv`` before
# importing Firedrake — otherwise PETSc inspects ``sys.argv`` and
# emits "unused options" warnings for our argparse flags.
_ARGS = _parse_args()
sys.argv = [sys.argv[0]]

# The MPI halo workaround for PETSc 3.20+ MUST run *before* the first
# ``import firedrake`` anywhere in the process — see
# ``tutorials/firedrake/_mpi_halo_patch.py``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import firedrake as fd  # noqa: E402
from firedrake.petsc import PETSc  # noqa: E402
from mpi4py import MPI  # noqa: E402

# Reuse the production model + solver wiring.  These already pull in
# the SystemModel pipeline, the auto-stationary limiter exclude
# (``b`` is auto-detected as stationary and skipped), and the
# ``firedrake_compat`` dat-slice workarounds.
from malpasset_viscous_v2 import (  # noqa: E402
    MalpassetSWE,
    MalpassetSolver,
    bcs,
    INPUT_MESH,
    _total_water_volume,
    MANNING_N,
    EPS_WD,
    H_FRICTION_FLOOR,
    NU,
)
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL  # noqa: E402
from zoomy_core.misc.misc import Zstruct  # noqa: E402


def main():
    args = _ARGS
    if args.mesh is None:
        args.mesh = INPUT_MESH
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        print(
            f"[run_malpasset_mpi] mpi_size={size}  "
            f"dg_degree={args.dg_degree}  limiter={args.limiter}  "
            f"CFL={args.cfl}  time_end={args.time_end}  "
            f"snapshots={args.snapshots}  output_tag={args.output_tag}",
            flush=True,
        )
        print(
            f"[run_malpasset_mpi] params: ν={NU}  Manning_n={MANNING_N}  "
            f"eps_wd={EPS_WD}  h_friction_floor={H_FRICTION_FLOOR}",
            flush=True,
        )

    model = MalpassetSWE()
    output_dir = f"outputs/malpasset_{args.output_tag}"
    settings = Settings(
        name=f"malpasset-{args.output_tag}",
        output=Zstruct(
            directory=output_dir,
            snapshots=args.snapshots,
            filename="dg",
            clean_directory=True,
        ),
    )

    solver = MalpassetSolver(
        settings=settings,
        time_end=args.time_end,
        CFL=args.cfl,
        dg_degree=args.dg_degree,
        limiter=args.limiter,
        riemann_solver_cls=PositiveNonconservativeHLL,
    )

    # Setup once so we can sample initial mass + bath before stepping.
    solver.setup_simulation(args.mesh, model)
    V0 = _total_water_volume(solver)

    t0 = time.perf_counter()
    solver.run_simulation()
    t1 = time.perf_counter()
    V1 = _total_water_volume(solver)

    dV_rel = (V1 - V0) / V0 if V0 != 0.0 else float("nan")
    n_iter = int(getattr(solver._state, "last_iteration_count", 0))
    final_t = float(getattr(solver._state, "sim_time", 0.0))
    avg_dt = (final_t / n_iter) if n_iter > 0 else float("nan")

    if rank == 0:
        print(
            f"[run_malpasset_mpi DONE] tag={args.output_tag}  "
            f"wall={t1 - t0:.2f}s  n_iter={n_iter}  "
            f"final_t={final_t:.3f}s  avg_dt={avg_dt:.4f}s  "
            f"ms/iter={(t1 - t0) * 1000.0 / max(n_iter, 1):.1f}  "
            f"V0={V0:.6e}  V1={V1:.6e}  ΔV/V0={dV_rel:+.3e}",
            flush=True,
        )
        print(
            f"[run_malpasset_mpi DONE] output: "
            f"{os.path.abspath(output_dir)}/simulation.pvd",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
