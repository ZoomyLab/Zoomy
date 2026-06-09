"""2nd-order LSQ-MUSCL probes on a tiny structured 50x10 channel.

Probes (run in order; each is independent, all share the same SymPy
SystemModel built locally):

  P1  Lake-at-rest, flat bathymetry b=const.
      h + b = const, hu = hv = 0.  After ~100 steps: hu must stay
      < 1e-8, max|h - h0| < 1e-10, mass conserved to machine eps.

  P2  Lake-at-rest, sloped bathymetry b(x) = 0.1 * x.
      h + b = const, hu = hv = 0.  Same robustness check —
      this is where well-balancing across non-constant cells
      bites if the Audusse S~ does not survive MUSCL.

  P3  Wet/dry dam break (flat bed, hL = 1, hR = 0, hu = hv = 0).
      h >= 0 everywhere, no spurious negative h.  Front advances
      at ~ Ritter speed 2 sqrt(g hL).  (Just a sanity probe;
      we mostly assert positivity + no NaN.)

  P4  Wet/dry dam break on sloped bathymetry (small Malpasset-lite).
      Combines P2 + P3.  h >= 0 everywhere.

Each probe is run twice: order=1 (reference) and order=2 (target).
The pass/fail criterion focuses on the order=2 path; order=1 is the
reference / known-good baseline.

This script reuses everything in zoomy_jax + zoomy_core; no library
edits.  It is the TDD probe before any reconstruction / NCP edits.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import sympy as sp
from sympy import Matrix, sqrt

import jax
import jax.numpy as jnp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.initial_conditions import InitialConditions
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.mesh import LSQMesh
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.mesh import convert_mesh_to_jax


# ── Constants -------------------------------------------------------

MANNING_N = 0.0   # friction off in probes (purely test the spatial scheme)
NU = 0.0          # viscosity off
EPS_WD = 1e-3
H_FRICTION_FLOOR = 0.5
U_MAX = 30.0
G = 9.81


# ── SystemModel constructor (matches malpasset_spmd_small_4dev_snapshots.build_malpasset_sm) ──

class _ZeroIC(InitialConditions):
    def apply(self, X, Q):
        Q[:] = 0.0
        return Q


def build_swe_sm():
    """Build the 4-state SWE SystemModel: [b, h, hu, hv].

    Mirrors the malpasset SPMD reference but with friction / nu turned
    off so the probes test only the spatial scheme + NCP wiring.
    """
    t = sp.Symbol("t", real=True)
    x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    n0, n1 = sp.Symbol("n0", real=True), sp.Symbol("n1", real=True)
    dX = sp.Symbol("dX", real=True)
    X0, X1, X2 = (sp.Symbol(f"X{d}", real=True) for d in range(3))
    position = Zstruct(X0=X0, X1=X1, X2=X2)
    position._symbolic_name = "X"

    b = sp.Symbol("b", real=True)
    h = sp.Symbol("h", real=True)
    hu = sp.Symbol("hu", real=True)
    hv = sp.Symbol("hv", real=True)
    state = [b, h, hu, hv]
    variables = Zstruct(b=b, h=h, hu=hu, hv=hv)
    variables._symbolic_name = "Q"
    aux_variables = Zstruct()
    aux_variables._symbolic_name = "Qaux"

    g = sp.Symbol("g", positive=True)
    n_man = sp.Symbol("n", nonnegative=True)
    nu = sp.Symbol("nu", nonnegative=True)
    eps = sp.Symbol("eps", positive=True)
    parameters = Zstruct(g=g, n=n_man, nu=nu, eps=eps)
    parameters._symbolic_name = "p"
    parameter_values = Zstruct(g=G, n=MANNING_N, nu=NU, eps=EPS_WD)
    normal = Zstruct(n0=n0, n1=n1)
    normal._symbolic_name = "n"

    # KP-desingularised 1/h, inlined.
    h_floor = sp.Max(h, eps)
    hinv = sqrt(2) * h / sqrt(h ** 4 + h_floor ** 4)
    u = hu * hinv
    w = hv * hinv

    F = Matrix.zeros(4, 2)
    F[1, 0] = hu;            F[1, 1] = hv
    F[2, 0] = hu * hu * hinv; F[2, 1] = hu * hv * hinv
    F[3, 0] = hu * hv * hinv; F[3, 1] = hv * hv * hinv
    P = sp.zeros(4, 2)

    Bncp = sp.MutableDenseNDimArray.zeros(4, 4, 2)
    Bncp[2, 0, 0] = g * h;  Bncp[2, 1, 0] = g * h
    Bncp[3, 0, 1] = g * h;  Bncp[3, 1, 1] = g * h

    # Friction / viscosity are zero in test (parameter values = 0).
    u_mag = sqrt(u * u + w * w + sp.Float(1e-12))
    h_safe = sp.Max(h, sp.Float(H_FRICTION_FLOOR))
    factor = -n_man ** 2 * g * h_safe ** (-sp.Rational(1, 3)) * u_mag
    S = Matrix([sp.S.Zero, sp.S.Zero, factor * u, factor * w])

    h_wet = sp.Max(h - sp.Float(EPS_WD), sp.S.Zero)
    max_hu = h_wet * sp.Float(U_MAX)

    def cap(c):
        return sp.Max(-max_hu, sp.Min(c, max_hu))

    UV = Matrix([b, h, cap(hu), cap(hv)])

    un = u * n0 + w * n1
    c_speed = sqrt(g * sp.Max(h, eps))
    cond = sp.Function("conditional")
    raw_ev = [sp.S.Zero, un, un - c_speed, un + c_speed]
    EV = Matrix([cond(h > eps, e, sp.S.Zero) for e in raw_ev])

    A = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
    for d in (0, 1):
        A[2, 2, d, d] = nu
        A[3, 3, d, d] = nu
        A[2, 1, d, d] = -nu * u
        A[3, 1, d, d] = -nu * w

    # Wall on all four sides (mass-tight box).
    bcs = BC.BoundaryConditions([
        BC.Wall(tag="left",   momentum_field_indices=[[2, 3]],
                permeability=0.0, wall_slip=1.0),
        BC.Wall(tag="right",  momentum_field_indices=[[2, 3]],
                permeability=0.0, wall_slip=1.0),
        BC.Wall(tag="bottom", momentum_field_indices=[[2, 3]],
                permeability=0.0, wall_slip=1.0),
        BC.Wall(tag="top",    momentum_field_indices=[[2, 3]],
                permeability=0.0, wall_slip=1.0),
    ])
    aux_bcs = BC.BoundaryConditions([
        AuxBC.Extrapolation(tag="left"),
        AuxBC.Extrapolation(tag="right"),
        AuxBC.Extrapolation(tag="bottom"),
        AuxBC.Extrapolation(tag="top"),
    ])
    bc_kernel = bcs.get_boundary_condition_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="boundary_conditions",
    )
    bgrad_kernel = bcs.get_boundary_gradient_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="boundary_gradients",
    )
    aux_bc_kernel = aux_bcs.get_boundary_condition_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="aux_boundary_conditions",
    )

    sm = SystemModel(
        time=t, space=[x, y],
        state=state, aux_state=[],
        parameters=parameters,
        flux=F,
        hydrostatic_pressure=P,
        nonconservative_matrix=Bncp,
        source=S,
        mass_matrix=sp.eye(4),
        eigenvalues=EV,
        normal=normal,
        parameter_values=parameter_values,
        update_variables=UV,
        diffusion_matrix=sp.MutableDenseNDimArray.zeros(4, 4, 2, 2),
        diffusion_matrix_explicit=A,
        source_explicit=sp.zeros(4, 1),
        initial_conditions=_ZeroIC(),
        aux_initial_conditions=_ZeroIC(),
        boundary_conditions=bc_kernel,
        boundary_gradients=bgrad_kernel,
        aux_boundary_conditions=aux_bc_kernel,
    )
    sm.expose_aux_atoms()
    return sm


# ── Setup helpers --------------------------------------------------

def make_mesh(nx=50, ny=10, Lx=5.0, Ly=1.0):
    return LSQMesh.create_2d((0.0, Lx, 0.0, Ly), nx, ny)


def setup_solver(mesh, order):
    sm = build_swe_sm()
    nsm = NumericalSystemModel.from_system_model(
        sm,
        riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=order, limiter="minmod"),
    )
    solver = HyperbolicSolver()
    Q_glob, Qaux_glob = solver.setup_simulation(mesh, nsm)
    return solver, Q_glob, Qaux_glob, sm


def set_ic(solver, Q, b_field, h_field, hu_field=None, hv_field=None):
    """Set initial conditions from per-cell arrays.  All arrays are
    shape (n_inner_cells,).  hu, hv default to zero."""
    nc = int(solver._rt_mesh.n_inner_cells)
    Q = Q.at[0, :nc].set(jnp.asarray(b_field, dtype=Q.dtype))
    Q = Q.at[1, :nc].set(jnp.asarray(h_field, dtype=Q.dtype))
    if hu_field is None:
        hu_field = np.zeros(nc)
    if hv_field is None:
        hv_field = np.zeros(nc)
    Q = Q.at[2, :nc].set(jnp.asarray(hu_field, dtype=Q.dtype))
    Q = Q.at[3, :nc].set(jnp.asarray(hv_field, dtype=Q.dtype))
    return Q


def run_steps(solver, Q, Qaux, n_steps, cfl=0.4):
    """Adaptive-dt run on a single device.  Uses the same recipe as the
    SPMD persistent runner: eigenvalue-based dt, explicit flux add,
    update_q at the end of each step."""
    mesh_jax = convert_mesh_to_jax(solver._rt_mesh)
    runtime = solver._rt_model
    flux_op = solver.get_flux_operator(mesh_jax, runtime)
    eig_op = solver.get_compute_max_abs_eigenvalue(mesh_jax, runtime)
    parameters = solver._rt_parameters
    n_owned = int(solver._rt_mesh.n_inner_cells)
    min_inradius = float(np.min(np.asarray(mesh_jax.cell_inradius)))

    update_var = runtime.update_variables
    have_update = update_var is not None

    @jax.jit
    def step(dt, t, Q, Qaux):
        dQ = flux_op(dt, t, Q, Qaux, parameters, jnp.zeros_like(Q))
        Q1 = Q.at[:, :n_owned].add(dt * dQ[:, :n_owned])
        if have_update:
            Q1 = Q1.at[:, :n_owned].set(
                update_var(Q1[:, :n_owned], Qaux[:, :n_owned], parameters))
        return Q1

    t_phys = 0.0
    for k in range(n_steps):
        max_eig = float(eig_op(Q, Qaux, parameters))
        if not (max_eig > 0):
            max_eig = 1e-12
        dt = cfl * min_inradius / max_eig
        Q = step(jnp.asarray(dt, dtype=Q.dtype),
                 jnp.asarray(t_phys, dtype=Q.dtype), Q, Qaux)
        t_phys += dt
    return Q, t_phys


# ── Probe drivers ---------------------------------------------------

def probe_lake_at_rest(slope, order, nx=50, ny=10, n_steps=200, verbose=True):
    """Probe P1 (slope=0) and P2 (slope>0)."""
    mesh = make_mesh(nx=nx, ny=ny)
    nc = int(mesh.n_inner_cells)
    xc = mesh.cell_centers[0, :nc]
    yc = mesh.cell_centers[1, :nc]

    b0 = slope * xc            # bathymetry
    eta0 = 1.0                 # free surface elevation (constant)
    h0 = eta0 - b0             # depth
    hu0 = np.zeros(nc)
    hv0 = np.zeros(nc)

    solver, Q, Qaux, sm = setup_solver(mesh, order=order)
    Q = set_ic(solver, Q, b0, h0, hu0, hv0)
    Q0 = np.asarray(Q)[:, :nc].copy()

    Q, t_end = run_steps(solver, Q, Qaux, n_steps=n_steps)

    Q_np = np.asarray(Q)[:, :nc]
    b1 = Q_np[0]; h1 = Q_np[1]; hu1 = Q_np[2]; hv1 = Q_np[3]
    eta1 = b1 + h1

    metrics = dict(
        t_end=t_end,
        max_abs_hu=float(np.max(np.abs(hu1))),
        max_abs_hv=float(np.max(np.abs(hv1))),
        max_eta_drift=float(np.max(np.abs(eta1 - eta0))),
        max_h_drift=float(np.max(np.abs(h1 - h0))),
        h_min=float(h1.min()),
        h_max=float(h1.max()),
        mass_initial=float(np.sum(h0 * mesh.cell_volumes[:nc])),
        mass_final=float(np.sum(h1 * mesh.cell_volumes[:nc])),
    )
    metrics["mass_drift_rel"] = (
        metrics["mass_final"] - metrics["mass_initial"]
    ) / metrics["mass_initial"]
    if verbose:
        print(f"  t_end={metrics['t_end']:.4g}  "
              f"max|hu|={metrics['max_abs_hu']:.3e}  "
              f"max|hv|={metrics['max_abs_hv']:.3e}")
        print(f"  max|eta-eta0|={metrics['max_eta_drift']:.3e}  "
              f"max|h-h0|={metrics['max_h_drift']:.3e}")
        print(f"  h_min={metrics['h_min']:.6f}  h_max={metrics['h_max']:.6f}")
        print(f"  mass_drift_rel={metrics['mass_drift_rel']:.3e}")
    return metrics


def probe_dam_break(slope, hL=1.0, hR=0.0, order=2,
                    nx=200, ny=10, n_steps=500, verbose=True):
    """Probe P3 (slope=0) and P4 (slope>0).  Wet/dry dam break."""
    mesh = make_mesh(nx=nx, ny=ny, Lx=10.0)  # longer channel
    nc = int(mesh.n_inner_cells)
    xc = mesh.cell_centers[0, :nc]

    b0 = slope * xc
    # Dam at x = 5.0: h = hL if x < 5 else hR (with depth measured above b).
    # For wet/dry on a slope we want the LEFT side to be wet:
    eta_left = b0[xc < 5.0].max() + hL if (xc < 5.0).any() else hL
    h0 = np.where(xc < 5.0,
                  np.maximum(eta_left - b0, hL * 0.5),
                  np.maximum(hR, 0.0))
    # Ensure no negative h initially.
    h0 = np.maximum(h0, 0.0)

    hu0 = np.zeros(nc)
    hv0 = np.zeros(nc)

    solver, Q, Qaux, sm = setup_solver(mesh, order=order)
    Q = set_ic(solver, Q, b0, h0, hu0, hv0)

    Q, t_end = run_steps(solver, Q, Qaux, n_steps=n_steps, cfl=0.3)

    Q_np = np.asarray(Q)[:, :nc]
    h1 = Q_np[1]; hu1 = Q_np[2]

    metrics = dict(
        t_end=t_end,
        h_min=float(h1.min()),
        h_max=float(h1.max()),
        n_negative_h=int(np.sum(h1 < -1e-10)),
        max_abs_hu=float(np.max(np.abs(hu1))),
        any_nan=bool(not np.all(np.isfinite(h1)) or
                     not np.all(np.isfinite(hu1))),
    )
    if verbose:
        print(f"  t_end={metrics['t_end']:.4g}  "
              f"h in [{metrics['h_min']:.3e}, {metrics['h_max']:.3e}]")
        print(f"  n_h_neg(<-1e-10)={metrics['n_negative_h']}  "
              f"max|hu|={metrics['max_abs_hu']:.3g}  any_nan={metrics['any_nan']}")
    return metrics


# ── Main -----------------------------------------------------------

def main():
    print("=" * 72)
    print("Probe P1: Lake-at-rest, flat bed")
    print("=" * 72)
    for order in (1, 2):
        print(f"\n[order={order}]")
        m1 = probe_lake_at_rest(slope=0.0, order=order, n_steps=200)

    print("\n" + "=" * 72)
    print("Probe P2: Lake-at-rest, sloped bed b=0.1*x")
    print("=" * 72)
    for order in (1, 2):
        print(f"\n[order={order}]")
        m2 = probe_lake_at_rest(slope=0.1, order=order, n_steps=200)

    print("\n" + "=" * 72)
    print("Probe P3: Wet/dry dam break, flat bed, hL=1, hR=0")
    print("=" * 72)
    for order in (1, 2):
        print(f"\n[order={order}]")
        m3 = probe_dam_break(slope=0.0, order=order, n_steps=500)

    print("\n" + "=" * 72)
    print("Probe P4: Wet/dry dam break, sloped bed b=0.05*x, hL=1, hR=0")
    print("=" * 72)
    for order in (1, 2):
        print(f"\n[order={order}]")
        m4 = probe_dam_break(slope=0.05, order=order, n_steps=500)


if __name__ == "__main__":
    main()
