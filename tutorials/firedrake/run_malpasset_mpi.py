#!/usr/bin/env python3
"""Malpasset dam-break — MPI production runner (self-contained test case).

Reconstructs the full Malpasset production test case from a single
file: no CLI arguments, no shell-side parameter strings.  All
parameters are hard-coded as module-level constants below — copy this
file and edit those constants to define a sibling configuration.

Run — single host, 4 ranks (Apptainer + SIF)
--------------------------------------------

The project publishes a Firedrake apptainer-compatible image at
``ghcr.io/zoomylab/zoomy_firedrake:latest`` — see ``README.md`` and
``docs/book/installation.md``.  Don't build a custom SIF; pull the
published one::

    apptainer build zoomy_firedrake.sif \\
        docker://ghcr.io/zoomylab/zoomy_firedrake:latest

One-time host setup — create a slim conda env with the **same Open
MPI version** the container ships (4.1.6).  Open MPI 4.1.x is
ABI-stable, so the host launcher and the container's libmpi speak
the same wire protocol::

    mamba env create -f install/zoomy_host_mpi.yml
    conda activate zoomy-host-mpi
    mpirun --version    # confirm Open MPI 4.1.x

Until the latest ``zoomy_core`` / ``zoomy_firedrake`` are published
to PyPI and a new SIF is built that bundles them, install the
in-repo sources editable into the container's user site-packages.
Apptainer bind-mounts ``$HOME``, so ``--user`` writes land in
``~/.local/lib/...`` on the host and are visible on subsequent
``apptainer exec`` invocations.  One-time::

    apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        pip install --user --no-deps -e library/zoomy_core
    apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        pip install --user --no-deps -e library/zoomy_firedrake

Once the changes are upstream in the published image, drop those two
lines — the SIF's bundled wheels are sufficient.

Then::

    conda activate zoomy-host-mpi
    mpirun -n 4 apptainer exec --bind $PWD:$PWD zoomy_firedrake.sif \\
        python3 -u tutorials/firedrake/run_malpasset_mpi.py

That is the entire run command.  ``ZOOMY_DIR`` (used by
``misc.get_main_directory`` to locate ``data/malpasset/*.msh``) is
auto-set from ``__file__`` inside this script, so it does not need to
be exported on the host.

Slurm
-----

Same idea, ``srun`` replacing ``mpirun`` — and on a Slurm cluster the
system MPI module typically provides the matching Open MPI build
instead of the conda env, so load it first (``module load
openmpi/4.1.6`` or similar) and skip the ``zoomy-host-mpi`` env::

    srun -n $SLURM_NTASKS apptainer exec \\
        --bind /scratch/$USER/Zoomy:/scratch/$USER/Zoomy \\
        zoomy_firedrake.sif \\
        python3 -u /scratch/$USER/Zoomy/tutorials/firedrake/run_malpasset_mpi.py

Output lands in ``outputs/malpasset_<OUTPUT_TAG>/simulation.pvd`` (open
in ParaView).  Per-step iteration logs go to stdout via Firedrake's
logger.
"""

import os
import sys
import time


# ----------------------------------------------------------------------
# Test-case parameters — edit these to define a sibling configuration.
# ----------------------------------------------------------------------
TIME_END = 100.0      # Physical simulation end time [s]
CFL = 0.4             # CFL safety factor on top of the RKDG bound (≤ 1).
                      # 0.4 is ≤ Xing-Zhang 2013 triangle bound ŵ₁=1/2 in
                      # the |K|/perimeter form (CellDiameter-based formula
                      # overshoots that by ~15% on shape-regular triangles).
DG_DEGREE = 1         # DG polynomial degree (0 or 1)
LIMITER = "ofdg"      # "none" | "vertex" | "p_weighted" | "ofdg"
SNAPSHOTS = 20        # Number of evenly-spaced ParaView snapshots
OUTPUT_TAG = "prod"
MESH_PATH = None      # None ⇒ data/malpasset/geo_malpasset-small.msh under $ZOOMY_DIR


# Make data-dir lookups (``misc.get_main_directory``) deterministic
# without requiring the user to ``export ZOOMY_DIR`` or relying on the
# ``git rev-parse`` fallback (git may be absent inside the SIF).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
os.environ.setdefault("ZOOMY_DIR", _REPO_ROOT)

# The MPI halo workaround for PETSc 3.20+ MUST run *before* the first
# ``import firedrake`` anywhere in the process — see
# ``tutorials/firedrake/_mpi_halo_patch.py``.  Python auto-prepends the
# script's directory to ``sys.path`` when invoked as
# ``python3 tutorials/firedrake/run_malpasset_mpi.py``, so this sibling
# import resolves without any explicit ``sys.path`` manipulation.
import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import firedrake as fd  # noqa: E402,F401
from firedrake.petsc import PETSc  # noqa: E402,F401
from mpi4py import MPI  # noqa: E402

# Reuse the production model + solver wiring.  These already pull in
# the SystemModel pipeline, the auto-stationary limiter exclude
# (``b`` is auto-detected as stationary and skipped), and the
# ``firedrake_compat`` dat-slice workarounds.
from malpasset_viscous_v2 import (  # noqa: E402
    MalpassetSWE,
    MalpassetSolver,
    bcs,  # noqa: F401  — imported for side-effects / parity with prior runner
    INPUT_MESH,
    _total_water_volume,
    MANNING_N,
    EPS_WD,
    H_FRICTION_FLOOR,
    NU,
)
from zoomy_core.fvm.solver_numpy import Settings  # noqa: E402
from zoomy_core.fvm.riemann_solvers import (  # noqa: E402
    PositiveNonconservativeHLL,
    PositiveNonconservativeRusanov,
)
from zoomy_core.misc.misc import Zstruct  # noqa: E402


def main():
    mesh_path = MESH_PATH if MESH_PATH is not None else INPUT_MESH
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        print(
            f"[run_malpasset_mpi] mpi_size={size}  "
            f"dg_degree={DG_DEGREE}  limiter={LIMITER}  "
            f"CFL={CFL}  time_end={TIME_END}  "
            f"snapshots={SNAPSHOTS}  output_tag={OUTPUT_TAG}",
            flush=True,
        )
        print(
            f"[run_malpasset_mpi] params: ν={NU}  Manning_n={MANNING_N}  "
            f"eps_wd={EPS_WD}  h_friction_floor={H_FRICTION_FLOOR}",
            flush=True,
        )
        print(f"[run_malpasset_mpi] mesh={mesh_path}", flush=True)

    model = MalpassetSWE()
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
        # Xing-Zhang 2013 (JSC 57:19-41) — the cell-mean positivity
        # proof for DG-SWE on unstructured triangles is for the
        # Lax-Friedrichs / Rusanov flux with Audusse hydrostatic
        # reconstruction.  HLL+Audusse is *not* within scope of that
        # theorem and produces ``h̄ < 0`` (up to ~30cm) at wet/dry
        # wave fronts on triangle meshes (eigenvalue-gating breaks the
        # convex-combination decomposition the proof relies on).
        # Switching to ``PositiveNonconservativeRusanov`` restores
        # the paper-proven combination.
        riemann_solver_cls=PositiveNonconservativeRusanov,
    )

    # Setup once so we can sample initial mass + bath before stepping.
    solver.setup_simulation(mesh_path, model)
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
            f"[run_malpasset_mpi DONE] tag={OUTPUT_TAG}  "
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
