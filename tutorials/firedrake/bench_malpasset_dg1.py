"""Benchmark harness for ``malpasset_viscous_v2`` at DG(1) — profiles
the per-stage cost and exercises a sweep of PETSc solver parameters.

Run as a script with the integer index of the configuration to
benchmark (defaults to ``0`` = baseline):

.. code-block:: bash

    ZOOMY_DIR=/work python3 tutorials/firedrake/bench_malpasset_dg1.py 0   # baseline
    ZOOMY_DIR=/work python3 tutorials/firedrake/bench_malpasset_dg1.py 1   # variation 1

Outputs per-stage cumulative timing every ``REPORT_EVERY`` iterations
and a final summary line with mass-conservation + total wall.

Used overnight to find a Firedrake/PETSc configuration that brings
down the DG(1) wall time before the longer Malpasset runs.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MPI halo workaround MUST be applied before any Firedrake import
# that touches the halo SF — i.e. before ``import firedrake``.
import _mpi_halo_patch  # noqa: E402
_mpi_halo_patch.apply()

import numpy as np  # noqa: E402
import firedrake as fd  # noqa: E402
from attrs import define, field  # noqa: E402

from malpasset_viscous_v2 import (
    MalpassetSWE,
    MalpassetSolver,
    INPUT_MESH,
    CFL,
)
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_core.fvm.riemann_solvers import PositiveHLL


# ──────────────────────────────────────────────────────────────────────
# Per-stage timing solver subclass
# ──────────────────────────────────────────────────────────────────────

@define(frozen=True, slots=True, kw_only=True)
class TimedMalpassetSolver(MalpassetSolver):
    """``MalpassetSolver`` with per-stage timing baked into ``step``.

    Each ``step`` records wall time for the four expensive phases —
    convective linear solve, slope limiter, source nonlinear solve,
    and the post-step ``update_Q`` / ``update_Qaux`` — into
    ``self._stage_times`` (a dict of cumulative seconds).

    ``linear_solver_parameters`` / ``nonlinear_solver_parameters`` let
    each benchmark configuration override the PETSc options that
    :meth:`_get_linear_solver` / :meth:`_get_nonlinear_solver` use.
    """

    linear_solver_parameters: dict = field(factory=dict)
    nonlinear_solver_parameters: dict = field(factory=dict)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self, "_stage_times",
            {"convective": 0.0, "limiter": 0.0, "update_Q_conv": 0.0,
             "source": 0.0, "update_Q_src": 0.0, "compute_dt": 0.0,
             "n_steps": 0},
        )

    # -- Override solver constructors to honor the per-config options --
    def _get_linear_solver(self, weak_form, Qnp1, Qaux_np1):
        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        sp_ = self.linear_solver_parameters or {
            "ksp_type": "bcgs", "pc_type": "lu",
        }
        return fd.LinearVariationalSolver(problem, solver_parameters=dict(sp_))

    def _get_nonlinear_solver(self, weak_form, Qnp1, Qaux_np1):
        J = fd.derivative(weak_form, Qnp1)
        problem = fd.NonlinearVariationalProblem(weak_form, Qnp1, J=J)
        sp_ = self.nonlinear_solver_parameters or {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_damping": 0.8,
            "snes_max_it": 25,
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_stol": 1e-12,
            "ksp_type": "gmres",
            "pc_type": "lu",
        }
        return fd.NonlinearVariationalSolver(problem, solver_parameters=dict(sp_))

    # -- Timed step ----------------------------------------------------
    def step(self, dt_value):
        s = self._state
        t = self._stage_times

        s.Qn.assign(s.Qnp1)
        s.Qaux_n.assign(s.Qaux_np1)
        s.dt.assign(dt_value)

        t0 = time.perf_counter()
        s.solver_convective.solve()
        t["convective"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._apply_slope_limiter(s.Qs)
        t["limiter"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self.update_Q(s.Qs, s.Qaux_s, s.runtime_model)
        self.update_Qaux(s.Qs, s.Qaux_s, s.runtime_model)
        t["update_Q_conv"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        s.solver_source.solve()
        t["source"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._apply_slope_limiter(s.Qnp1)
        self.update_Q(s.Qnp1, s.Qaux_np1, s.runtime_model)
        self.update_Qaux(s.Qnp1, s.Qaux_np1, s.runtime_model)
        t["update_Q_src"] += time.perf_counter() - t0

        t["n_steps"] += 1


# ──────────────────────────────────────────────────────────────────────
# Configuration sweep
# ──────────────────────────────────────────────────────────────────────

# Per-config tuples: (label, linear_params, nonlinear_params)
CONFIGS = [
    # 0 — baseline (matches what's currently in firedrake_solver.py).
    ("baseline-LU+GMRES+LU",
     {"ksp_type": "bcgs", "pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "gmres", "pc_type": "lu"}),

    # 1 — convective via block-Jacobi + sub-LU (DG mass is
    # block-diagonal per cell, so this should be ~exact and faster
    # than a global LU factorization).
    ("conv-BJacobi+LU",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu", "sub_pc_factor_mat_solver_type": "petsc"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "gmres", "pc_type": "lu"}),

    # 2 — convective preonly+LU (skip the iterative outer loop on
    # an already-direct PC; one LU solve per step).
    ("conv-preonly+LU",
     {"ksp_type": "preonly", "pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "gmres", "pc_type": "lu"}),

    # 3 — source LU via MUMPS (often 2-3x faster than PETSc's
    # built-in LU on serial / threaded runs).
    ("src-MUMPS",
     {"ksp_type": "preonly", "pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}),

    # 4 — source SNES with basic line search + KSP rtol relaxed,
    # plus MUMPS.  Fewer Newton iters when the residual is mildly
    # nonlinear (Manning is the only true nonlinearity here).
    ("src-MUMPS+basic-LS+relaxed-KSP",
     {"ksp_type": "preonly", "pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "basic",
      "snes_max_it": 25, "snes_rtol": 1e-6, "snes_atol": 1e-8,
      "snes_stol": 1e-10,
      "ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}),

    # 5 — Field-split: separate the depth row from momentum rows,
    # additive Schwarz on each block.  Sometimes worth it when the
    # implicit diffusion strongly couples momentum but not depth.
    ("src-FieldSplit",
     {"ksp_type": "preonly", "pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "fgmres",
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "additive",
      "fieldsplit_0_pc_type": "lu",
      "fieldsplit_1_pc_type": "lu"}),

    # 6 — DG(0)-targeted: BJacobi+LU on the source step.  On DG(0)
    # the source-step Jacobian is block-diagonal (cell-local mass +
    # friction Jacobian) plus the TPFA face stencil; ILU(0) or
    # block-Jacobi with cell blocks captures most of it without
    # paying the full LU cost.
    ("src-BJacobi+LU",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "gmres", "pc_type": "bjacobi", "sub_pc_type": "lu"}),

    # 7 — DG(0)-targeted: GMRES with ILU(0) PC on the source step.
    # Cheap PC, often a winner when LU is expensive.
    ("src-GMRES+ILU",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "bt",
      "snes_linesearch_damping": 0.8, "snes_max_it": 25,
      "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-12,
      "ksp_type": "gmres", "ksp_rtol": 1e-5,
      "pc_type": "bjacobi", "sub_pc_type": "ilu"}),

    # 8 — Combined best-of-DG(0): conv BJacobi+LU + src MUMPS +
    # basic line-search + relaxed Newton tolerances.
    ("combined-best-DG0",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "basic",
      "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
      "snes_stol": 1e-10,
      "ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}),

    # 9 — MPI-targeted: GMRES + ASM (additive Schwarz, 1-level
    # overlap) with sub-LU.  Should scale well under MPI because
    # the PC is local-by-construction; serial fallback is reasonable.
    ("MPI-GMRES+ASM+sub-LU",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "basic",
      "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
      "snes_stol": 1e-10,
      "ksp_type": "gmres", "ksp_rtol": 1e-6,
      "pc_type": "asm", "sub_pc_type": "lu"}),

    # 10 — MPI-targeted: GMRES + GAMG (algebraic multigrid).
    # Strongest scaling PC for elliptic-ish problems.  The source
    # step Jacobian here (block-diag mass + Manning + boundary
    # diffusion) isn't perfectly elliptic but GAMG still often beats
    # direct LU for moderate-to-large N.
    ("MPI-GMRES+GAMG",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "basic",
      "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
      "snes_stol": 1e-10,
      "ksp_type": "gmres", "ksp_rtol": 1e-6,
      "pc_type": "gamg"}),

    # 11 — MPI-targeted: parallel direct via MUMPS (MUMPS supports
    # distributed factorisation, unlike PETSc's default LU).
    ("MPI-MUMPS-direct",
     {"ksp_type": "preonly", "pc_type": "bjacobi",
      "sub_pc_type": "lu"},
     {"snes_type": "newtonls", "snes_linesearch_type": "basic",
      "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
      "snes_stol": 1e-10,
      "ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}),
]


# ──────────────────────────────────────────────────────────────────────
# Bench driver
# ──────────────────────────────────────────────────────────────────────

def run_bench(config_idx: int, time_end: float, tag_suffix: str = ""):
    label, lin_params, nlin_params = CONFIGS[config_idx]
    print(f"\n[bench] config={config_idx}  label={label!r}  time_end={time_end}s",
          flush=True)
    print(f"[bench]   linear_params={lin_params}", flush=True)
    print(f"[bench]   nonlin_params={nlin_params}", flush=True)

    out_tag = f"bench_cfg{config_idx}_{label.replace('+', '_').replace(' ', '')}{tag_suffix}"
    s = Settings(name=out_tag,
                 output=Zstruct(directory=f"outputs/{out_tag}",
                                snapshots=5, filename="dg",
                                clean_directory=True))
    dg_degree = int(os.environ.get("BENCH_DG_DEGREE", "1"))
    limiter = os.environ.get("BENCH_LIMITER", "vertex" if dg_degree >= 1 else "none")
    solver = TimedMalpassetSolver(
        settings=s,
        time_end=time_end,
        CFL=CFL,
        dg_degree=dg_degree,
        limiter=limiter,
        riemann_solver_cls=PositiveHLL,
        linear_solver_parameters=lin_params,
        nonlinear_solver_parameters=nlin_params,
    )
    solver.setup_simulation(INPUT_MESH, MalpassetSWE())

    # Initial water volume.
    V0 = float(fd.assemble(solver._state.Qnp1[1] * fd.dx))

    state = solver._state
    t_now = 0.0
    iter_n = 0
    t_start = time.perf_counter()
    REPORT_EVERY = 50

    try:
        while t_now < time_end:
            t0 = time.perf_counter()
            dt = state.compute_dt(state.Qnp1, state.Qaux_np1)
            dt = min(dt, time_end - t_now)
            solver._stage_times["compute_dt"] += time.perf_counter() - t0

            solver.step(dt)
            t_now += dt
            iter_n += 1

            if iter_n % REPORT_EVERY == 0:
                tt = solver._stage_times
                wall = time.perf_counter() - t_start
                total_in_step = (tt["convective"] + tt["limiter"]
                                 + tt["update_Q_conv"] + tt["source"]
                                 + tt["update_Q_src"] + tt["compute_dt"])
                pct = lambda x: 100.0 * x / max(total_in_step, 1e-12)
                print(
                    f"[bench]  it={iter_n:5d}  t={t_now:7.3f}  dt={dt:.3e}  "
                    f"wall={wall:7.1f}s   "
                    f"conv={tt['convective']:6.1f}s ({pct(tt['convective']):4.1f}%)  "
                    f"limit={tt['limiter']:6.1f}s ({pct(tt['limiter']):4.1f}%)  "
                    f"src={tt['source']:6.1f}s ({pct(tt['source']):4.1f}%)  "
                    f"updQ={tt['update_Q_conv']+tt['update_Q_src']:5.1f}s",
                    flush=True,
                )
    except Exception as e:
        wall = time.perf_counter() - t_start
        print(f"[bench]  FAIL after it={iter_n}, t={t_now:.3f}, wall={wall:.1f}s: "
              f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        return

    wall = time.perf_counter() - t_start
    V1 = float(fd.assemble(state.Qnp1[1] * fd.dx))
    dV_rel = (V1 - V0) / V0 if V0 != 0.0 else float("nan")
    tt = solver._stage_times

    print(f"\n[bench]  DONE  cfg={config_idx}  label={label!r}  "
          f"wall={wall:.1f}s  iters={iter_n}  s_per_phys_s={wall/time_end:.1f}",
          flush=True)
    print(f"[bench]  mass:  V0={V0:.6e}  V1={V1:.6e}  ΔV/V0={dV_rel:+.3e}",
          flush=True)
    print(f"[bench]  per-stage cumulative (over {iter_n} steps):",
          flush=True)
    for stage in ("convective", "limiter", "update_Q_conv",
                  "source", "update_Q_src", "compute_dt"):
        secs = tt[stage]
        per_step_ms = 1000.0 * secs / max(iter_n, 1)
        print(f"[bench]    {stage:15s}  {secs:8.2f}s  ({per_step_ms:7.2f} ms/step)",
              flush=True)


if __name__ == "__main__":
    cfg_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    run_bench(cfg_idx, t_end, tag)
