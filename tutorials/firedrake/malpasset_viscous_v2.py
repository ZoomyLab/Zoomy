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
import meshio

from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basemodel import Model
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

# Loaded once and reused for IC projection.
MESHIO_MESH = meshio.read(INPUT_MESH)


# %% [markdown]
# ## SWE Model — 4-component state, aux ``hinv``, depth-weighted
# viscosity
#
# Matches the original `malpasset_viscous.py` operator-by-operator,
# but in the new ``Model → SystemModel`` convention.

# %%
class MalpassetSWE(Model):
    """SWE with `[b, h, hu, hv]` state and `[hinv]` aux.

    - ``flux``: convective only (``hu⊗u``).  The hydrostatic pressure
      ``½ g h² I`` is moved into ``nonconservative_matrix`` so the
      well-balancing handshake with the bathymetry slope is preserved
      (Audusse-style).
    - ``nonconservative_matrix``: ``g h ∂_d b`` and ``g h ∂_d h`` on the
      momentum rows.
    - ``source``: Manning bed friction with floored ``h^{-1/3}``.
    - ``diffusion_matrix``: ``A[hu_i, hu_i, d, d] = ν · h``.
    - ``eigenvalues``: switched off where ``h ≤ eps`` (dry cells get
      zero wave speed).
    """

    def __init__(self, *, g=9.81, n=MANNING_N, nu=NU, eps=EPS_WD, **kw):
        super().__init__(
            dimension=2,
            variables=["b", "h", "hu", "hv"],
            aux_variables=["hinv"],
            parameters={
                "g":  (float(g),  "positive"),
                "n":  (float(n),  "non-negative"),
                "nu": (float(nu), "non-negative"),
                "eps": (float(eps), "positive"),
            },
            eigenvalue_mode="symbolic",
            **kw,
        )

    # -- Convenience -----------------------------------------------------
    def _primitives(self):
        v = self.variables
        a = self.aux_variables
        return v.b, v.h, v.hu, v.hv, a.hinv

    # -- Operators -------------------------------------------------------
    def flux(self):
        _, h, hu, hv, hinv = self._primitives()
        F = Matrix.zeros(4, 2)
        # mass equation
        F[1, 0] = hu
        F[1, 1] = hv
        # momentum: pure convective (no pressure here — see NCP)
        F[2, 0] = hu * hu * hinv
        F[2, 1] = hu * hv * hinv
        F[3, 0] = hu * hv * hinv
        F[3, 1] = hv * hv * hinv
        return ZArray(F)

    def nonconservative_matrix(self):
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 2)
        # Momentum_x: depends on ∂_x b and ∂_x h
        N[2, 0, 0] = g * h
        N[2, 1, 0] = g * h
        # Momentum_y: depends on ∂_y b and ∂_y h
        N[3, 0, 1] = g * h
        N[3, 1, 1] = g * h
        return N

    def source(self):
        _, h, hu, hv, hinv = self._primitives()
        p = self._parameter_symbols
        u = hu * hinv
        w = hv * hinv
        u_mag = sqrt(u * u + w * w + 1e-12)
        # Manning friction with bounded ``h^(-1/3)``: floor the depth at
        # ``H_FRICTION_FLOOR`` so friction stays finite at the wet/dry
        # interface.  Wet/dry "deactivation" happens naturally —
        # ``hu = hv = 0`` ⇒ ``u_mag = 0`` ⇒ friction = 0 — and at very
        # low depths ``hinv`` (from ``update_aux_variables``) is also
        # already floored.  We deliberately avoid ``Piecewise`` /
        # ``conditional`` here because the SystemModel auto-derives the
        # source Jacobian via ``sp.diff`` and conditionals are not
        # differentiable in SymPy.
        h_safe = sp.Max(h, sp.Float(H_FRICTION_FLOOR))
        friction_div = h_safe ** (-sp.Rational(1, 3))
        factor = -p.n ** 2 * p.g * friction_div * u_mag
        S_b = sp.S.Zero
        S_h = sp.S.Zero
        S_hu = factor * hu
        S_hv = factor * hv
        return ZArray([S_b, S_h, S_hu, S_hv])

    def diffusion_matrix_explicit(self):
        """Depth-weighted eddy viscosity on momentum rows — **explicit
        treatment** (folded into the convective step at ``Qn``).

        Routed via the SystemModel ``diffusion_matrix_explicit`` slot
        (the explicit IMEX companion to ``diffusion_matrix``).  For the
        Malpasset dam-break (ν=1, h_cell≈50–200 m) the parabolic CFL
        ``dt ≤ h²/(2ν)`` is ≈1250–20000 s — ~30 000× looser than the
        hyperbolic CFL (~0.04 s) — so explicit treatment never
        constrains the step.  Putting diffusion in the explicit slot
        leaves the implicit source-step Jacobian block-diagonal
        (mass + Manning friction) → cell-local block-Jacobi PC
        becomes exact and Newton converges in 1 iter.
        """
        _, h, _, _, _ = self._primitives()
        nu = self._parameter_symbols.nu
        A = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        for i_row in (2, 3):       # hu, hv
            for d in (0, 1):
                A[i_row, i_row, d, d] = nu * h
        return ZArray(A)

    # ---- State hygiene exposed via the SystemModel slot --------------
    #
    # ``Model.update_variables`` is the symbolic per-cell state-remap
    # the SystemModel carries (slot ``sm.update_variables``).  Every
    # backend calls it once per step on the post-convective state to
    # apply state-level constraints.  For the dam-break we cap
    # ``|hu_i| ≤ h · u_max`` (component-wise) so leftover momentum in
    # nearly-dry cells doesn't blow up ``|u| = hu/h`` — h itself is
    # **never** modified.  Setting ``max_hu = max(h, 0)·u_max`` makes
    # the cap zero out hu in dry cells (h ≤ 0) for free, without an
    # explicit h-threshold or wet/dry mask.

    def update_variables(self):
        v = self.variables
        h, hu, hv = v.h, v.hu, v.hv
        u_max = sp.Float(30.0)
        # ``max(h, 0)`` zeroes the cap in dry cells (h ≤ 0) without
        # touching ``h`` itself.
        h_eff = sp.Max(h, sp.S.Zero)
        max_hu = h_eff * u_max

        def cap(c):
            # Symmetric clamp to ``[-max_hu, max_hu]`` using only ``Min``
            # / ``Max`` (which lambdify cleanly through every backend's
            # module dict).  Avoids ``sp.sign`` / ``sp.Abs`` whose UFL
            # lowering of the eager numerical-coercion path was failing
            # with ``Indexed.__float__ returned NotImplemented``.
            return sp.Max(-max_hu, sp.Min(c, max_hu))

        return ZArray([v.b, h, cap(hu), cap(hv)])

    def update_variables_jacobian_wrt_variables(self):
        # ``sp.derive_by_array(update_variables, [b, h, hu, hv])`` would
        # produce ``Piecewise`` / ``sign`` derivatives that the numpy
        # / UFL printers cannot serialise.  This Jacobian is only used
        # for IMEX implicit-update Newton (which the Firedrake backend
        # does not run on ``update_variables``), so returning the
        # explicit zero suppresses the auto-derivation cleanly without
        # disabling the Model's other auto-derived operators.
        n = self.n_variables
        return ZArray.zeros(n, n)

    def eigenvalues(self):
        _, h, hu, hv, hinv = self._primitives()
        p = self._parameter_symbols
        n = self.normal
        # Build velocity from the conservative momentum.  ``hinv`` (an
        # aux variable computed by ``update_aux_variables`` as
        # ``1/max(h, eps)``) is the bounded divisor used everywhere
        # else; we use it here too for consistency.  This is NOT a
        # state-level safeguard on h — h itself is never modified —
        # just a regularization of the velocity expression.
        u = hu * hinv
        w = hv * hinv
        un = u * n.n0 + w * n.n1
        c = sqrt(p.g * sp.Max(h, p.eps))
        raw_ev = [sp.S.Zero, un, un - c, un + c]

        # Wet/dry wave-speed gate.  Cells with ``h ≤ eps`` carry only
        # numerical-noise momentum (leftover ``hu`` from a passing wave
        # that's no longer in the cell); ``|u| = hu/h`` then blows up
        # to physically meaningless values (we measured |u| > 1e7
        # m/s in dry cells right after iteration 2 of the Malpasset
        # run).  Letting those bogus wavespeeds set the global CFL
        # collapses ``dt → 0``.  Gating ``λ → 0`` in dry cells is
        # the standard remedy: dry cells propagate no waves and must
        # not constrain the global time step.
        #
        # ``conditional`` (an opaque ``sp.Function``) is non-
        # differentiable, but eigenvalues are not differentiated by
        # ``SystemModel._compute_source_jacobian`` / ``_compute_
        # quasilinear_matrix`` (only ``source``, ``flux`` and
        # ``hydrostatic_pressure`` are) and ``expose_aux_atoms`` only
        # scans those operator matrices, so the conditional is safe
        # here and lowers cleanly through the UFL ``_ufl_conditional``
        # / numpy ``np.where`` shims at lambdify time.
        cond = sp.Function("conditional")
        gated = [cond(h > p.eps, e, sp.S.Zero) for e in raw_ev]
        return ZArray(gated)

    def update_aux_variables(self):
        """hinv = 1 / max(h, eps)."""
        v = self.variables
        p = self._parameter_symbols
        h_safe = sp.Max(v.h, p.eps)
        return ZArray([1 / h_safe])


# %% [markdown]
# ## Solver subclass: meshio-driven initial condition
#
# Only override `set_initial_condition` to load `B, H, HU, HV` from the
# point data of the meshio file.  Everything else (Riemann solver,
# weak forms, diffusion path) comes from the base solver.

# %%
def _build_vertex_permutation(fd_mesh, meshio_mesh, decimal=12):
    """See `malpasset_baseline.py` — Firedrake reorders nodes."""
    dim = fd_mesh.geometric_dimension()
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
        perm = _build_vertex_permutation(mesh, MESHIO_MESH)
        pd = MESHIO_MESH.point_data
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


def run(dg_degree=0, limiter="none", time_end=TIME_END, tag=""):
    model = MalpassetSWE()
    out_tag = tag or f"dg{dg_degree}_lim{limiter}"
    s = Settings(
        name=f"malpasset-{out_tag}",
        output=Zstruct(directory=f"outputs/firedrake_viscous_v2_{out_tag}",
                       snapshots=10, filename="dg", clean_directory=True),
    )
    # ``PositiveHLL`` — HLL with Audusse-Bristeau-Klein hydrostatic
    # reconstruction.  Required for the Malpasset dam-break: the bare
    # ``Rusanov`` Riemann does not preserve positivity at the wet/dry
    # shoreline, so the simulation hit ``DIVERGED_FNORM_NAN`` once the
    # front reached a dry cell (around t ≈ 8.78 s).  ``PositiveHLL``
    # enforces ``h_face ≥ 0`` at every interior facet and is
    # well-balanced for the lake-at-rest steady state.
    solver = MalpassetSolver(
        settings=s,
        time_end=time_end,
        CFL=CFL,
        dg_degree=dg_degree,
        limiter=limiter,
        riemann_solver_cls=PositiveNonconservativeHLL,
    )
    # Setup pre-time-loop so we can sample initial mass before stepping.
    solver.setup_simulation(INPUT_MESH, model)
    V0 = _total_water_volume(solver)
    t0 = time.perf_counter()
    solver.run_simulation()
    t1 = time.perf_counter()
    V1 = _total_water_volume(solver)
    dV_rel = (V1 - V0) / V0 if V0 != 0.0 else float("nan")
    print(
        f"[malpasset {out_tag}] wall_time={t1 - t0:.2f}s  "
        f"V0={V0:.6e}  V1={V1:.6e}  ΔV/V0={dV_rel:+.3e}"
    )
    return solver


# %%
if __name__ == "__main__":
    print(f"[malpasset] ν={NU}  time_end={TIME_END}  CFL={CFL}  BC=wall")
    solver_dg0 = run(dg_degree=0, limiter="none", tag="dg0_tpfa_wall")
    #solver_dg1 = run(dg_degree=1, limiter="vertex", tag="dg1_ipdg_vert_wall")
