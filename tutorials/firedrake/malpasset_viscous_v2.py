# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Malpasset dam break (viscous) — new SystemModel + symbolic-Riemann pipeline
#
# Port of `tutorials/firedrake/malpasset_viscous.py` to the new
# `FiredrakeHyperbolicSolver` (SystemModel → symbolic Riemann → UFL
# runtime).  Same physical setup and geometry as the original.
#
# The original uses a hand-rolled SWE class with `[b, h, hu, hv]`
# state, `[hinv]` aux, and inserts a hand-written DG(0) TPFA viscous
# block into the weak form.  Here we:
#
# - define an equivalent `MalpassetSWE` Model that returns
#   `flux`, `nonconservative_matrix`, `source` and `diffusion_matrix_explicit`
#   in the new operator-form convention;
# - let `SystemModel.from_model` extract the operators;
# - let `PositiveHLL(SystemModel).to_runtime_ufl()` lower the
#   Audusse-Bristeau-Klein well-balanced Riemann numerics to UFL;
# - rely on the solver's built-in TPFA (DG(0)) and IP-DG (DG(1+))
#   diffusion paths — no per-app weak-form override.
#
# ## Running
#
# Serial::
#
#     python tutorials/firedrake/malpasset_viscous_v2.py
#
# Parallel (MPI, recommended for larger meshes / longer runs)::
#
#     mpirun -n 4 python tutorials/firedrake/malpasset_viscous_v2.py
#
# The MPI halo workaround for PETSc 3.20+ is applied automatically at
# import time.  Both serial and parallel paths use the solver's GAMG
# defaults (see
# :attr:`FiredrakeHyperbolicSolver.DEFAULT_NONLINEAR_SOLVER_PARAMETERS`)
# which were tuned via the
# ``tutorials/firedrake/bench_*`` optimisation campaign:
# **~9× faster** than the previous LU-based default in serial; scales
# cleanly under MPI.  Override per-run by passing
# ``linear_solver_parameters=...`` / ``nonlinear_solver_parameters=...``
# to the solver constructor.
#
# Two runs at the bottom: DG(0), then DG(1) with vertex-based limiter.

# %%
import os
import sys
import time

# MPI halo-exchange workaround for PETSc 3.20+: must run BEFORE
# importing firedrake.  No-op when COMM_WORLD.size == 1, so safe to
# import unconditionally — both ``python script.py`` and
# ``mpirun -n N python script.py`` use this entry point.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mpi_halo_patch
_mpi_halo_patch.apply()

import numpy as np
import sympy as sp
from sympy import Matrix, sqrt, Piecewise

import firedrake as fd
from firedrake.petsc import PETSc
from mpi4py import MPI
# ``meshio`` is only needed for reading the Malpasset mesh's vertex
# point-data (initial condition).  Imported lazily inside the IC
# loader so that importing this module — e.g. ``from
# malpasset_viscous_v2 import MalpassetSWE`` from the production
# runner — works even on images that don't ship meshio.

from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basemodel import Model  # noqa: F401 (compat re-export)
from zoomy_core.model.models import MalpassetSWE as _CanonicalMalpassetSWE
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.misc.misc as misc

from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver


# %% [markdown]
# ## Physical parameters and inputs

# %%
MANNING_N = float(os.environ.get("MALPASSET_MANNING", "0.033"))
EPS_WD = float(os.environ.get("MALPASSET_EPS_WD", "1e-2"))
H_FRICTION_FLOOR = float(os.environ.get("MALPASSET_H_FRICTION", "0.5"))
NU = float(os.environ.get("MALPASSET_NU", "1.0"))
TIME_END = float(os.environ.get("MALPASSET_TIME_END", "100.0"))
CFL = float(os.environ.get("MALPASSET_CFL", "0.5"))

main_dir = misc.get_main_directory()
INPUT_MESH = os.path.join(main_dir, "data", "malpasset",
                          "geo_malpasset-small.msh")
assert os.path.exists(INPUT_MESH), f"missing mesh: {INPUT_MESH}"

# Loaded lazily inside ``MalpassetSolver.set_initial_condition``
# to keep ``import malpasset_viscous_v2`` working on images that
# don't ship meshio (e.g. the published GHCR zoomy_firedrake image
# pre-meshio dependency).
_MESHIO_MESH_CACHE = None


def _meshio_mesh():
    """Lazy ``meshio.read(INPUT_MESH)`` with module-level caching."""
    global _MESHIO_MESH_CACHE
    if _MESHIO_MESH_CACHE is None:
        import meshio  # local import — keeps top-level cheap
        _MESHIO_MESH_CACHE = meshio.read(INPUT_MESH)
    return _MESHIO_MESH_CACHE


# %% [markdown]
# ## SWE Model — 4-component state, aux ``hinv``, depth-weighted
# viscosity
#
# Matches the original `malpasset_viscous.py` operator-by-operator,
# but in the new ``Model → SystemModel`` convention.

# %%
class MalpassetSWE(_CanonicalMalpassetSWE):
    """Tutorial compat shim over the canonical
    :class:`zoomy_core.model.models.MalpassetSWE`.

    The model PHYSICS (flux, hydrostatic pressure, bed-slope NCP, Manning
    source, eddy viscosity, wet/dry aux + capping + eigenvalues, the
    free-surface ``reconstruction_variables``) now lives ONCE in the
    canonical ``zoomy_core/model/models`` package — there are no hand-rolled
    model operators in tutorial / thesis scripts any more.  This subclass
    adds NOTHING physical: it only forwards the tutorial's
    environment-driven numeric defaults (``MALPASSET_MANNING`` /
    ``MALPASSET_NU`` / ``MALPASSET_EPS_WD`` / ``MALPASSET_H_FRICTION`` /
    ``MALPASSET_EV_GATE``) so existing ``from malpasset_viscous_v2 import
    MalpassetSWE`` consumers keep byte-for-byte behaviour.  New code should
    import the model directly: ``from zoomy_core.model.models import
    MalpassetSWE``.
    """

    def __init__(self, *, g=9.81, n=MANNING_N, nu=NU, eps=EPS_WD,
                 h_friction_floor=H_FRICTION_FLOOR,
                 ev_gate=(os.environ.get("MALPASSET_EV_GATE", "1") != "0"),
                 **kw):
        super().__init__(g=g, n=n, nu=nu, eps=eps,
                         h_friction_floor=h_friction_floor,
                         ev_gate=ev_gate, **kw)


# %% [markdown]
# ## Solver subclass: meshio-driven initial condition
#
# Only override `set_initial_condition` to load `B, H, HU, HV` from the
# point data of the meshio file.  Everything else (Riemann solver,
# weak forms, diffusion path) comes from the base solver.

# %%
def _build_vertex_permutation(fd_mesh, meshio_mesh, decimal=12):
    """See `malpasset_baseline.py` — Firedrake reorders nodes."""
    dim = fd_mesh.geometric_dimension
    coords_fd = np.round(fd_mesh.coordinates.dat.data_ro[:, :dim], decimal)
    coords_mio = np.round(meshio_mesh.points[:, :dim], decimal)
    lookup = {tuple(c): i for i, c in enumerate(coords_mio)}
    perm = np.empty(coords_fd.shape[0], dtype=np.int64)
    for j, c in enumerate(coords_fd):
        perm[j] = lookup[tuple(c)]
    return perm


class MalpassetSolver(FiredrakeHyperbolicSolver):
    """FiredrakeHyperbolicSolver with only the Malpasset IC loader
    overridden — the dam-break-specific state hygiene (momentum cap
    against ``|u| = hu/h`` runaway in nearly-dry cells) lives on the
    SystemModel slot ``update_variables`` (:meth:`MalpassetSWE.update_variables`),
    so every backend honors it uniformly via
    ``runtime_model.update_variables(Q, Qaux, p)``.
    """

    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        # CG1 staging space so we can write the raw vertex point data
        # before projecting onto the (possibly DG1) target space.
        V_CG = fd.VectorFunctionSpace(mesh, "CG", 1,
                                      dim=Q.function_space().value_size)
        Q_CG = fd.Function(V_CG)
        mio = _meshio_mesh()
        perm = _build_vertex_permutation(mesh, mio)
        pd = mio.point_data
        Q_CG.dat.data[:, 0] = pd["B"][perm]
        Q_CG.dat.data[:, 1] = pd["H"][perm]
        Q_CG.dat.data[:, 2] = (pd["H"] * pd["U"])[perm]
        Q_CG.dat.data[:, 3] = (pd["H"] * pd["V"])[perm]
        Q.project(Q_CG)


# %% [markdown]
# ## Boundary conditions
#
# Single ``Wall`` BC covering the whole exterior — the Malpasset mesh
# has no named ``Physical Curve`` groups, so
# :meth:`FiredrakeHyperbolicSolver.get_map_boundary_tag_to_boundary_function_index`
# falls back to mapping this BC over ``fd.ds`` (every unmarked exterior
# facet).  With ``permeability=0`` (no normal flow) and
# ``wall_slip=1`` (free tangential slip) and bathymetry / depth scalar
# rows untouched, the discrete mass flux through ∂Ω is zero — total
# water volume must be conserved up to discretization error.  Momentum
# indices ``[2, 3]`` correspond to ``hu, hv`` in the
# ``[b, h, hu, hv]`` state layout.

# %%
bcs = BC.BoundaryConditions(
    [BC.Wall(tag="wall", momentum_field_indices=[[2, 3]],
             permeability=0.0, wall_slip=1.0)]
)


# %% [markdown]
# ## Run helpers — including a mass-conservation diagnostic

# %%
def _total_water_volume(solver):
    """∫_Ω h dx — total water volume on the active state ``Qnp1``."""
    s = solver._state
    h = s.Qnp1[1]                                # index 1 in [b, h, hu, hv]
    return float(fd.assemble(h * fd.dx))


def _b_stats(solver):
    """Per-rank and global min/max/integral of the bathymetry component
    ``Q[0] = b`` on the active state.  Used to detect spurious b
    modification at DG(1): b is stationary in the model (S_b = 0,
    A[b, *] = 0, F[b, *] = 0), so b at step 1 must equal b at step 0
    bit-for-bit.  Any drift fingers the slope limiter / update_Q /
    halo path as the culprit."""
    s = solver._state
    b = s.Qnp1[0]
    b_func = fd.Function(fd.FunctionSpace(
        s.Qnp1.function_space().mesh(), "DG", solver.dg_degree
    )).interpolate(b)
    arr = b_func.dat.data_ro
    comm = s.Qnp1.function_space().mesh().comm
    rank = comm.Get_rank()
    local = (float(arr.min()), float(arr.max()), float(arr.mean()))
    g_min = comm.allreduce(local[0], op=MPI.MIN)
    g_max = comm.allreduce(local[1], op=MPI.MAX)
    g_int = float(fd.assemble(b * fd.dx))
    return rank, local, (g_min, g_max, g_int)


def run(dg_degree=0, limiter="none", time_end=TIME_END, tag=""):
    model = MalpassetSWE()
    out_tag = tag or f"dg{dg_degree}_lim{limiter}"
    s = Settings(
        name=f"malpasset-{out_tag}",
        output=Zstruct(directory=f"outputs/firedrake_viscous_v2_{out_tag}",
                       snapshots=int(os.environ.get("MALPASSET_SNAPSHOTS", "10")),
                       filename="dg", clean_directory=True),
    )
    # ``PositiveHLL`` — HLL with Audusse-Bristeau-Klein hydrostatic
    # reconstruction.  Required for the Malpasset dam-break: the bare
    # ``Rusanov`` Riemann does not preserve positivity at the wet/dry
    # shoreline, so the simulation hit ``DIVERGED_FNORM_NAN`` once the
    # front reached a dry cell (around t ≈ 8.78 s).  ``PositiveHLL``
    # enforces ``h_face ≥ 0`` at every interior facet and is
    # well-balanced for the lake-at-rest steady state.
    # Solver-option presets for the perf study (MALPASSET_SNES_PRESET):
    #   default  — class defaults (newtonls + fresh GAMG each solve)
    #   pcreuse  — keep the GAMG hierarchy/preconditioner across solves
    #   lagjac   — pcreuse + lag the Jacobian (rebuild every 5th solve)
    #   ksponly  — single linearised solve per stage (no Newton loop)
    _PRESETS = {
        "default": None,
        "pcreuse": {
            "snes_type": "newtonls", "snes_linesearch_type": "basic",
            "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
            "snes_stol": 1e-10, "ksp_type": "gmres", "ksp_rtol": 1e-6,
            "pc_type": "gamg", "pc_gamg_reuse_interpolation": True,
            "snes_lag_preconditioner": 10,
            "snes_lag_preconditioner_persists": True,
        },
        "lagjac": {
            "snes_type": "newtonls", "snes_linesearch_type": "basic",
            "snes_max_it": 15, "snes_rtol": 1e-6, "snes_atol": 1e-8,
            "snes_stol": 1e-10, "ksp_type": "gmres", "ksp_rtol": 1e-6,
            "pc_type": "gamg", "pc_gamg_reuse_interpolation": True,
            "snes_lag_preconditioner": 10,
            "snes_lag_preconditioner_persists": True,
            "snes_lag_jacobian": 5, "snes_lag_jacobian_persists": True,
        },
        "ksponly": {
            "snes_type": "ksponly",
            "ksp_type": "gmres", "ksp_rtol": 1e-6,
            "pc_type": "gamg", "pc_gamg_reuse_interpolation": True,
        },
        # combined: single linearised solve + persistent lagged PC
        "fast": {
            "snes_type": "ksponly",
            "ksp_type": "gmres", "ksp_rtol": 1e-6,
            "pc_type": "gamg", "pc_gamg_reuse_interpolation": True,
            "snes_lag_preconditioner": 10,
            "snes_lag_preconditioner_persists": True,
        },
    }
    preset = os.environ.get("MALPASSET_SNES_PRESET", "default")
    solver = MalpassetSolver(
        settings=s,
        time_end=time_end,
        CFL=CFL,
        dg_degree=dg_degree,
        limiter=limiter,
        riemann_solver_cls=PositiveNonconservativeHLL,
        nonlinear_solver_parameters=_PRESETS[preset],
    )
    print(f"[malpasset] snes preset = {preset}")
    # Setup pre-time-loop so we can sample initial mass before stepping.
    solver.setup_simulation(INPUT_MESH, model)
    V0 = _total_water_volume(solver)
    rank0, b0_local, b0_global = _b_stats(solver)
    t0 = time.perf_counter()
    solver.run_simulation()
    t1 = time.perf_counter()
    V1 = _total_water_volume(solver)
    rank1, b1_local, b1_global = _b_stats(solver)
    dV_rel = (V1 - V0) / V0 if V0 != 0.0 else float("nan")
    # Per-rank b stats printed by every rank so MPI-rank-localised
    # corruption (e.g. halo) is visible.  Use PETSc.Sys.syncPrint to
    # interleave cleanly.
    PETSc.Sys.syncPrint(
        f"[malpasset {out_tag}] rank={rank0:2d}  "
        f"b_local_before=({b0_local[0]:+.6e}, {b0_local[1]:+.6e}, mean={b0_local[2]:+.6e})  "
        f"b_local_after =({b1_local[0]:+.6e}, {b1_local[1]:+.6e}, mean={b1_local[2]:+.6e})  "
        f"Δb_max_rank={max(abs(b1_local[0]-b0_local[0]), abs(b1_local[1]-b0_local[1])):.3e}"
    )
    PETSc.Sys.syncFlush()
    print(
        f"[malpasset {out_tag}] wall_time={t1 - t0:.2f}s  "
        f"V0={V0:.6e}  V1={V1:.6e}  ΔV/V0={dV_rel:+.3e}  "
        f"b_global_before=(min={b0_global[0]:+.3e}, max={b0_global[1]:+.3e}, ∫b={b0_global[2]:+.6e})  "
        f"b_global_after =(min={b1_global[0]:+.3e}, max={b1_global[1]:+.3e}, ∫b={b1_global[2]:+.6e})  "
        f"Δ∫b/∫b={(b1_global[2]-b0_global[2])/abs(b0_global[2]) if b0_global[2] != 0 else float('nan'):+.3e}"
    )
    # Compact per-step throughput line for benchmark tables.
    n_iter = int(getattr(solver._state, "last_iteration_count", 0))
    final_t = float(getattr(solver._state, "sim_time", 0.0))
    avg_dt = (final_t / n_iter) if n_iter > 0 else float("nan")
    ms_per_iter = ((t1 - t0) * 1000.0 / n_iter) if n_iter > 0 else float("nan")
    print(
        f"[malpasset {out_tag} BENCH] n_iter={n_iter}  final_t={final_t:.3f}  "
        f"avg_dt={avg_dt:.4f}s  wall={t1 - t0:.2f}s  "
        f"ms/iter={ms_per_iter:.1f}  wall/sim_s={(t1 - t0) / final_t if final_t > 0 else float('nan'):.2f}"
    )
    return solver


# %%
if __name__ == "__main__":
    # ONE_STEP mode: short t_end (~1–2 steps) for both DG(0) and DG(1).
    # Goal is to compare bathymetry b before and after a single step
    # at DG(0) (where b is provably conserved) vs DG(1) (where the
    # user observes immediate b flicker at MPI rank boundaries +
    # spurious mass injection).  Set MALPASSET_ONE_STEP=0 to fall
    # back to the full TIME_END run.
    one_step = bool(int(os.environ.get("MALPASSET_ONE_STEP", "1")))
    t_end = 0.05 if one_step else TIME_END
    # Which variants to run: comma list out of {dg0, dg1_nolim, dg1_vert}.
    variants = os.environ.get(
        "MALPASSET_VARIANTS", "dg0,dg1_nolim,dg1_vert").split(",")
    print(f"[malpasset] ν={NU}  time_end={t_end}  CFL={CFL}  BC=wall"
          f"  one_step={one_step}  variants={variants}")
    if "dg0" in variants:
        solver_dg0 = run(dg_degree=0, limiter="none", time_end=t_end,
                         tag="dg0_tpfa_wall")
    # DG(1) probe — also run with limiter="none" to discriminate
    # whether the residual b-flicker + mass-injection come from the
    # limiter (excluded for b but still active on h) or from somewhere
    # else (MPI halo, source Newton, etc.).  If ΔV/V0 → 0 here, the
    # h-limiter is the culprit.
    if "dg1_nolim" in variants:
        solver_dg1_nolim = run(dg_degree=1, limiter="none", time_end=t_end,
                               tag="dg1_ipdg_nolim_wall")
    if "dg1_vert" in variants:
        solver_dg1 = run(dg_degree=1, limiter="vertex", time_end=t_end,
                         tag="dg1_ipdg_vert_wall")
