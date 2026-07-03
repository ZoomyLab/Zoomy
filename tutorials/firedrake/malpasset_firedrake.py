# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---
# Self-contained Malpasset dam-break on Firedrake DG0.  No dependency on any
# other Malpasset notebook: the shallow-water model, the reservoir initial
# condition and the wall boundary condition are all defined here.  The
# numerics come from the single source of truth — FiredrakeHyperbolicSolver
# (the up-to-date IMEX solver: explicit hyperbolic + implicit source, MPI).

# %% 1. imports
import os
import numpy as np
import sympy as sp
from sympy import sqrt
import firedrake as fd
import meshio

from zoomy_core.misc.misc import ZArray
import zoomy_core.misc.misc as misc
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.closures import (
    ManningFriction, EddyViscosity, swe_closure_state)
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver

# %% 2. load mesh
MESH = os.path.join(misc.get_main_directory(), "data", "malpasset",
                    "geo_malpasset-small.msh")

# %% 3. shallow-water model (from scratch) -> SystemModel
# State [b, h, hu, hv] + aux [hinv] (KP-desingularised 1/h). Well-balanced
# encoding: convective flux only; hydrostatic pressure 1/2 g h^2 as a SEPARATE
# operator; bed slope g h db as the nonconservative product; Manning bed
# friction + eddy viscosity via composable closures.
class ShallowWater(SWE):
    variables = ["b", "h", "hu", "hv"]
    parameters = {
        "g":   (9.81, "positive"),
        "n":   (0.033, "nonnegative"),      # Manning roughness
        "nu":  (1.0, "nonnegative"),        # eddy viscosity
        "wet_dry_eps": (1e-2, "positive"),  # wet/dry depth threshold
    }
    U_MAX = 30.0                            # per-cell velocity cap (m/s)

    def __init__(self, *, g=9.81, n=0.033, nu=1.0, eps=1e-2,
                 h_friction_floor=0.5, ev_gate=True):
        self._h_friction_floor = float(h_friction_floor)
        self._ev_gate = bool(ev_gate)
        self.closures = [ManningFriction(h_floor=float(h_friction_floor)),
                         EddyViscosity()]
        super().__init__(
            dimension=2, aux_variables=["hinv"], eigenvalue_mode="symbolic",
            parameters={"g": (float(g), "positive"),
                        "n": (float(n), "nonnegative"),
                        "nu": (float(nu), "nonnegative"),
                        "wet_dry_eps": (float(eps), "positive")})

    def _build_function_groups(self):
        return {}

    @property
    def _parameter_symbols(self):
        return self.parameters

    def _primitives(self):
        v, a = self.variables, self.aux_variables
        return v.b, v.h, v.hu, v.hv, a.hinv

    def flux(self):
        _, h, hu, hv, hinv = self._primitives()
        F = sp.Matrix.zeros(4, 2)
        F[1, 0], F[1, 1] = hu, hv                          # mass
        F[2, 0], F[2, 1] = hu * hu * hinv, hu * hv * hinv  # momentum (convective)
        F[3, 0], F[3, 1] = hu * hv * hinv, hv * hv * hinv
        return ZArray(F)

    def hydrostatic_pressure(self):
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        P = ZArray.zeros(4, 2)
        P[2, 0] = g * h ** 2 / 2
        P[3, 1] = g * h ** 2 / 2
        return P

    def nonconservative_matrix(self):
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 2)
        N[2, 0, 0] = g * h            # g h db/dx
        N[3, 0, 1] = g * h            # g h db/dy
        return N

    def source(self):
        # Bed friction is applied EXPLICITLY via source_explicit (below), not
        # here.  The fully-implicit Lie-split source solve is unstable at wet/dry
        # shorelines — it spuriously drains still water (the sea collapses ~12 m
        # over the Malpasset run) even though the friction value is correct.
        # jax applies friction explicitly and is stable; matching that here.
        return ZArray([sp.S.Zero] * 4)

    def source_explicit(self):
        # Manning bed friction, evaluated at Qn in the (explicit) convective
        # step — jax's treatment.  Stable at wet/dry and damps identically.
        _, h, hu, hv, hinv = self._primitives()
        u, w = hu * hinv, hv * hinv
        st = swe_closure_state(self)
        rate = sum((c.expression(st) for c in self.closures
                    if c.closes == "bottom"), sp.S.Zero)
        return ZArray([sp.S.Zero, sp.S.Zero, rate * u, rate * w])

    def diffusion_matrix_explicit(self):
        # FULL horizontal deviatoric viscous stress divergence
        #   ∇·( ν h (∇u + (∇u)ᵀ) )
        # i.e. WITH the normal stresses τ_xx = 2ν ∂ₓu, τ_yy = 2ν ∂_yv and the
        # transpose-gradient cross-coupling — not just the Laplacian ∇·(νh∇u).
        # Component form (u=hu/h, v=hv/h):
        #   u-mom: ∂ₓ(2νh ∂ₓu) + ∂_y(νh ∂_yu) + ∂_y(νh ∂ₓv)
        #   v-mom: ∂ₓ(νh ∂ₓv) + ∂ₓ(νh ∂_yu) + ∂_y(2νh ∂_yv)
        # Diffusion is in the VELOCITY, so each ∂_e term carries the h
        # chain-rule (−ν·vel on the h column) that turns ∂(h·vel) into h·∂vel.
        # Every term ∝ a velocity gradient ⇒ vanishes at rest ⇒ well-balanced.
        _, h, hu, hv, hinv = self._primitives()
        u, w = hu * hinv, hv * hinv
        st = swe_closure_state(self)
        nu = sum((c.expression(st) for c in self.closures
                  if c.closes == "horizontal"), sp.S.Zero)
        A = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        vel = {2: u, 3: w}                       # velocity carried by momentum m

        def add(i, m, d, e, c=1):                # += ∂_d( c·νh ∂_e vel[m] ) to eq i
            A[i, m, d, e] += c * nu
            A[i, 1, d, e] += -c * nu * vel[m]    # h chain-rule column

        add(2, 2, 0, 0, 2); add(2, 2, 1, 1, 1); add(2, 3, 1, 0, 1)   # u-momentum
        add(3, 3, 0, 0, 1); add(3, 2, 0, 1, 1); add(3, 3, 1, 1, 2)   # v-momentum
        return ZArray(A)

    def update_variables(self):
        v, p = self.variables, self._parameter_symbols
        h, hu, hv = v.h, v.hu, v.hv
        u_max = sp.Float(self.U_MAX)
        max_hu = sp.Max(h - p.wet_dry_eps, sp.S.Zero) * u_max
        cap = lambda c: sp.Max(-max_hu, sp.Min(c, max_hu))
        return ZArray([v.b, h, cap(hu), cap(hv)])

    def update_variables_jacobian_wrt_variables(self):
        return ZArray.zeros(self.n_variables, self.n_variables)

    def eigenvalues(self):
        _, h, hu, hv, hinv = self._primitives()
        p, nrm = self._parameter_symbols, self.normal
        un = hu * hinv * nrm.n0 + hv * hinv * nrm.n1
        c = sqrt(p.g * sp.Max(h, p.wet_dry_eps))
        raw = [sp.S.Zero, un, un - c, un + c]
        if not self._ev_gate:
            return ZArray(raw)
        cond = sp.Function("conditional")
        return ZArray([cond(h > p.wet_dry_eps, e, sp.S.Zero) for e in raw])

    def update_aux_variables(self):
        v, p = self.variables, self._parameter_symbols
        h_floor = sp.Max(v.h, p.wet_dry_eps)
        return ZArray([sqrt(2) * v.h / sqrt(v.h ** 4 + h_floor ** 4)])


sm = ShallowWater().system_model

# %% 4. NumericalSystemModel
nsm = NumericalSystemModel.from_system_model(sm)

# %% 5. settings + default solver
# The model declares no initial_conditions, so the solver zeros every field
# first; this hook then overwrites with the reservoir state [b, h, hu, hv]
# from the mesh point-data (B, H, U, V) — the one Malpasset-specific bit.
def reservoir_ic(Q, model):
    mesh = Q.function_space().mesh()
    mio = meshio.read(MESH)
    dim = mesh.geometric_dimension
    cfd = np.round(mesh.coordinates.dat.data_ro[:, :dim], 12)
    cmio = np.round(mio.points[:, :dim], 12)
    lut = {tuple(c): i for i, c in enumerate(cmio)}
    perm = np.array([lut[tuple(c)] for c in cfd], dtype=np.int64)
    pd = mio.point_data
    # Well-balanced IC: build the DG0 cell state in the EQUILIBRIUM variable
    # eta = b + h, NOT in h.  L2-projecting h directly cell-averages
    # max(0, eta-b), and by Jensen on the convex max the cell-mean of h exceeds
    # max(0, -b_cellmean) -> the free surface OVERSHOOTS at every wet/dry
    # shoreline (spurious eta != still-water level).  That seed drains the sea
    # over the run.  Instead: cell-average b and eta separately, then set
    # h = max(0, eta - b) per cell so still water stays exactly at rest.
    CG = fd.FunctionSpace(mesh, "CG", 1)
    DG = fd.FunctionSpace(mesh, "DG", 0)

    def to_dg0(vertex_values):
        f = fd.Function(CG)
        f.dat.data[:] = vertex_values
        return np.asarray(fd.Function(DG).project(f).dat.data)

    B = pd["B"][perm]
    H = pd["H"][perm]
    b0 = to_dg0(B)
    eta0 = to_dg0(B + H)                       # free surface, projected as one field
    h0 = np.maximum(eta0 - b0, 0.0)            # consistent per cell -> no overshoot
    Q.dat.data[:, 0] = b0
    Q.dat.data[:, 1] = h0
    wet = h0 > 0.0
    Q.dat.data[:, 2] = np.where(wet, to_dg0((pd["H"] * pd["U"])[perm]), 0.0)
    Q.dat.data[:, 3] = np.where(wet, to_dg0((pd["H"] * pd["V"])[perm]), 0.0)


solver = FiredrakeHyperbolicSolver(
    settings=Settings(name="malpasset-firedrake", output=Zstruct(
        directory="outputs/malpasset_firedrake", snapshots=40,
        filename="dg", clean_directory=True)),
    time_end=2000.0, CFL=0.5, dg_degree=0, limiter="none",
    riemann_solver_cls=PositiveNonconservativeHLL,
    initial_condition_overwrite=reservoir_ic)

# %% 6. boundary conditions + solve
bcs = BC.BoundaryConditions(
    [BC.Wall(tag="wall", momentum_field_indices=[[2, 3]],
             permeability=0.0, wall_slip=1.0)])
solver.setup_simulation(MESH, nsm, boundary_conditions=bcs)
solver.run_simulation()
