"""Fast wet/dry dam-break probe for iterating on positivity-preserving
reconstruction (Xing-Zhang and friends).

A 200x2 quasi-1D channel mesh (≈ 800 triangles).  Initial condition:
``h = h_L`` for ``x < x_dam`` and ``h = 0`` for ``x > x_dam`` —
the canonical Ritter dam break.  No bathymetry, no friction, no
viscosity.  Single-device JAX (no SPMD) so each step is ~ms.

Reports per-config:

* steady wall-clock per step (after JIT warmup),
* ``h_min`` over the whole run (the positivity diagnostic),
* mass drift,
* a small ``h(x, t_end)`` print of the front region so you can eyeball
  the shock shape vs the analytic Ritter solution.

Compare configurations via env vars (defaults run a small comparison):

  python tests/scripts/zoomy_jax/wetdry_dambreak_probe.py

  # custom single config
  WD_ORDER=2 WD_CFL=0.25 WD_POSITIVITY=1 WD_N_STEPS=5000 \
      python tests/scripts/zoomy_jax/wetdry_dambreak_probe.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import sympy as sp
from sympy import Matrix, sqrt

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.initial_conditions import InitialConditions
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_jax.fvm.solver_jax import HyperbolicSolver


# ── A minimal SWE-with-bathymetry SystemModel ─────────────────────────
class _ZeroIC(InitialConditions):
    def apply(self, X, Q):
        Q[:] = 0.0
        return Q


def build_swe_sm(g=9.81, eps=1e-6, u_max=30.0, h_dry=1e-6):
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
    variables = Zstruct(b=b, h=h, hu=hu, hv=hv); variables._symbolic_name = "Q"
    aux_variables = Zstruct(); aux_variables._symbolic_name = "Qaux"

    g_sym = sp.Symbol("g", positive=True)
    eps_sym = sp.Symbol("eps", positive=True)
    parameters = Zstruct(g=g_sym, eps=eps_sym); parameters._symbolic_name = "p"
    parameter_values = Zstruct(g=float(g), eps=float(eps))
    normal = Zstruct(n0=n0, n1=n1); normal._symbolic_name = "n"

    h_floor = sp.Max(h, eps_sym)
    hinv = sqrt(2) * h / sqrt(h ** 4 + h_floor ** 4)
    u = hu * hinv
    w = hv * hinv

    F = Matrix.zeros(4, 2)
    F[1, 0] = hu; F[1, 1] = hv
    F[2, 0] = hu * hu * hinv; F[2, 1] = hu * hv * hinv
    F[3, 0] = hu * hv * hinv; F[3, 1] = hv * hv * hinv

    Bncp = sp.MutableDenseNDimArray.zeros(4, 4, 2)
    Bncp[2, 0, 0] = g_sym * h; Bncp[2, 1, 0] = g_sym * h
    Bncp[3, 0, 1] = g_sym * h; Bncp[3, 1, 1] = g_sym * h

    # Momentum cap consistent with the SPMD driver (zeros dry-cell momentum).
    h_wet = sp.Max(h - sp.Float(h_dry), sp.S.Zero)
    max_hu = h_wet * sp.Float(u_max)
    def cap(c):
        return sp.Max(-max_hu, sp.Min(c, max_hu))
    UV = Matrix([b, h, cap(hu), cap(hv)])

    un = u * n0 + w * n1
    c_speed = sqrt(g_sym * sp.Max(h, eps_sym))
    cond = sp.Function("conditional")
    raw_ev = [sp.S.Zero, un, un - c_speed, un + c_speed]
    EV = Matrix([cond(h > eps_sym, e, sp.S.Zero) for e in raw_ev])

    # ``LSQMesh.create_2d`` tags the four sides as left/right/bottom/top —
    # the BC list must cover ALL of them (or the BC-index lookup runs
    # off the end of the Piecewise kernel and produces NaN at runtime).
    bcs = BC.BoundaryConditions([
        BC.Wall(tag=tag, momentum_field_indices=[[2, 3]],
                permeability=0.0, wall_slip=1.0)
        for tag in ("left", "right", "bottom", "top")
    ])
    aux_bcs = BC.BoundaryConditions(
        [AuxBC.Extrapolation(tag=tag)
         for tag in ("left", "right", "bottom", "top")]
    )
    bc_kernel = bcs.get_boundary_condition_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="boundary_conditions")
    bgrad_kernel = bcs.get_boundary_gradient_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="boundary_gradients")
    aux_bc_kernel = aux_bcs.get_boundary_condition_function(
        t, position, dX, variables, aux_variables, parameters, normal,
        function_name="aux_boundary_conditions")

    sm = SystemModel(
        time=t, space=[x, y], state=state, aux_state=[],
        parameters=parameters,
        flux=F,
        hydrostatic_pressure=sp.zeros(4, 2),
        nonconservative_matrix=Bncp,
        source=sp.zeros(4, 1),
        mass_matrix=sp.eye(4),
        eigenvalues=EV,
        normal=normal,
        parameter_values=parameter_values,
        update_variables=UV,
        diffusion_matrix=sp.MutableDenseNDimArray.zeros(4, 4, 2, 2),
        diffusion_matrix_explicit=sp.MutableDenseNDimArray.zeros(4, 4, 2, 2),
        source_explicit=sp.zeros(4, 1),
        initial_conditions=_ZeroIC(),
        aux_initial_conditions=_ZeroIC(),
        boundary_conditions=bc_kernel,
        boundary_gradients=bgrad_kernel,
        aux_boundary_conditions=aux_bc_kernel,
    )
    sm.expose_aux_atoms()
    return sm


def make_mesh(nx=200, ny=2, Lx=10.0, Ly=0.1):
    return LSQMesh.create_2d((0.0, Lx, 0.0, Ly), nx, ny)


def setup_solver(mesh, order, reconstruction_mode="conservative",
                 limiter="minmod"):
    """``reconstruction_mode``:
      * ``conservative`` — plain LSQ-MUSCL on (b, h, hu, hv).
      * ``xz`` — Xing-Zhang slope scaling on h (cell-mean positivity).
      * ``eta`` — primitive MUSCL on (b, η=h+b, hu, hv) — Audusse-Bouchut.
    """
    sm = build_swe_sm()
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=order, limiter=limiter),
    )
    kwargs = {}
    if order >= 2 and reconstruction_mode != "conservative":
        kwargs.update(
            free_surface_h_index=1,
            free_surface_b_index=0,
            free_surface_momentum_indices=[2, 3],
            reconstruction_variables=reconstruction_mode,
        )
    solver = HyperbolicSolver(**kwargs)
    Q, Qaux = solver.setup_simulation(mesh, nsm)
    return solver, Q, Qaux, sm


def set_dambreak_ic(solver, Q, x_dam=5.0, h_L=1.0, h_R=0.0, b_slope=0.0):
    """Wet/dry dam break IC.

    ``b_slope`` adds a downward-sloping bed ``b(x) = -b_slope * x``,
    so the right half is below the upstream free surface (the wave
    runs onto a falling slope — Malpasset-ish).
    """
    nc = int(solver._rt_mesh.n_inner_cells)
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc]
    b0 = -b_slope * xc
    h0 = np.where(xc < x_dam, h_L, h_R).astype(float)
    Q = Q.at[0, :nc].set(jnp.asarray(b0, dtype=Q.dtype))
    Q = Q.at[1, :nc].set(jnp.asarray(h0, dtype=Q.dtype))
    Q = Q.at[2, :nc].set(0.0)
    Q = Q.at[3, :nc].set(0.0)
    return Q


def run_steps(solver, Q, Qaux, n_steps, cfl=0.5, eig_refresh=1):
    """Adaptive-dt loop (single device).  Returns (Q_final, t_final,
    h_min_seen, h_max_seen, mass_t0, mass_tend, ms_per_step)."""
    from zoomy_jax.mesh.mesh import convert_mesh_to_jax
    mesh_jax = convert_mesh_to_jax(solver._rt_mesh)
    runtime = solver._rt_model
    flux_op = solver.get_flux_operator(mesh_jax, runtime)
    eig_op = solver.get_compute_max_abs_eigenvalue(mesh_jax, runtime)
    parameters = solver._rt_parameters
    n_owned = int(solver._rt_mesh.n_inner_cells)
    min_inradius = float(np.min(np.asarray(mesh_jax.cell_inradius)))
    cell_vols = np.asarray(mesh_jax.cell_volumes[:n_owned])

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

    h_min_seen = float("inf")
    h_max_seen = -float("inf")
    h_init = np.asarray(Q)[1, :n_owned].copy()
    mass_t0 = float(np.sum(h_init * cell_vols))

    # Warmup (don't count in timing).
    max_eig = float(eig_op(Q, Qaux, parameters))
    dt0 = jnp.asarray(cfl * min_inradius / max(max_eig, 1e-12), dtype=Q.dtype)
    t0 = jnp.asarray(0.0, dtype=Q.dtype)
    Q = step(dt0, t0, Q, Qaux); Q.block_until_ready()

    t_phys = 0.0
    cached_dt = None
    cached_dt_host = 0.0
    steps_to_next_eig = 0
    t_start = time.perf_counter()
    for k in range(n_steps):
        if steps_to_next_eig == 0:
            max_eig = float(eig_op(Q, Qaux, parameters))
            if not (max_eig > 0):
                max_eig = 1e-12
            cached_dt_host = cfl * min_inradius / max_eig
            cached_dt = jnp.asarray(cached_dt_host, dtype=Q.dtype)
            steps_to_next_eig = eig_refresh
        dt_arr = cached_dt
        t_arr = jnp.asarray(t_phys, dtype=Q.dtype)
        Q = step(dt_arr, t_arr, Q, Qaux)
        t_phys += cached_dt_host
        steps_to_next_eig -= 1
        # Periodic positivity probe (every 50 steps).
        if k % 50 == 0:
            h_now = np.asarray(Q)[1, :n_owned]
            h_min_seen = min(h_min_seen, float(np.min(h_now)))
            h_max_seen = max(h_max_seen, float(np.max(h_now)))
    Q.block_until_ready()
    wall = time.perf_counter() - t_start
    h_final = np.asarray(Q)[1, :n_owned]
    h_min_seen = min(h_min_seen, float(np.min(h_final)))
    h_max_seen = max(h_max_seen, float(np.max(h_final)))
    mass_tend = float(np.sum(h_final * cell_vols))
    ms_per_step = 1000.0 * wall / max(n_steps, 1)
    return Q, t_phys, h_min_seen, h_max_seen, mass_t0, mass_tend, ms_per_step


def run_config(label, order, reconstruction_mode, cfl, n_steps,
               eig_refresh=10, b_slope=0.0):
    mesh = make_mesh()
    nc = int(mesh.n_inner_cells)
    solver, Q, Qaux, sm = setup_solver(
        mesh, order=order, reconstruction_mode=reconstruction_mode)
    Q = set_dambreak_ic(solver, Q, b_slope=b_slope)
    Q, t_end, h_min, h_max, m0, m1, ms = run_steps(
        solver, Q, Qaux, n_steps=n_steps, cfl=cfl, eig_refresh=eig_refresh)
    mass_drift = (m1 - m0) / m0
    h_final = np.asarray(Q)[1, :nc]
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc]
    # Front sampling: take the first cell column (ny=2 wraps).
    order_x = np.argsort(xc)
    x_sorted = xc[order_x]
    h_sorted = h_final[order_x]
    # Bin by x to get a 1D profile (average over y).
    n_bins = 40
    bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    bin_idx = np.digitize(x_sorted, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    h_profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for j, b in enumerate(bin_idx):
        h_profile[b] += h_sorted[j]
        counts[b] += 1
    h_profile = h_profile / np.maximum(counts, 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print(f"  {label:36s}  steps={n_steps:>5}  t_end={t_end:6.3f}  "
          f"h∈[{h_min:.3e}, {h_max:.3e}]  mass_drift={mass_drift:+.2e}  "
          f"ms/step={ms:5.2f}")
    return {
        "label": label, "order": order, "mode": reconstruction_mode,
        "cfl": cfl, "n_steps": n_steps, "t_end": t_end,
        "h_min": h_min, "h_max": h_max, "mass_drift": mass_drift,
        "ms_per_step": ms,
        "x_profile": bin_centers, "h_profile": h_profile,
    }


def main():
    n_steps = int(os.environ.get("WD_N_STEPS", "2000"))
    b_slope = float(os.environ.get("WD_B_SLOPE", "0.0"))
    bed = f"flat" if b_slope == 0.0 else f"sloped b'={-b_slope}"
    print(f"Wet/dry dam-break probe  (200x2 channel, {bed}, "
          f"n_steps={n_steps})")
    print(f"{'config':<38}  {'steps':>5}  {'t_end':>6}  "
          f"{'h range':>26}  {'mass_drift':>12}  {'ms/step':>9}")
    print("-" * 110)

    runs = [
        ("O1 (constant)",                 1, "conservative",  0.5,  n_steps),
        ("O2 vanilla CFL=0.4",            2, "conservative",  0.4,  n_steps),
        ("O2 + Xing-Zhang CFL=0.4",       2, "xz",            0.40, n_steps),
        ("O2 + Xing-Zhang CFL=0.25",      2, "xz",            0.25, n_steps),
        ("O2 + η-MUSCL CFL=0.4",          2, "eta",           0.40, n_steps),
        ("O2 + η-MUSCL CFL=0.25",         2, "eta",           0.25, n_steps),
    ]

    results = []
    for label, order, mode, cfl, steps in runs:
        results.append(run_config(label, order, mode, cfl, steps,
                                  eig_refresh=10, b_slope=b_slope))

    print("\nFinal h-profile (avg over y, 40 bins) — bin centers in [0, 10]:")
    print(f"  {'x':>5}", end="")
    for r in results:
        print(f"  {r['label'][:18]:>20}", end="")
    print()
    x_axis = results[0]["x_profile"]
    for i in range(len(x_axis)):
        print(f"  {x_axis[i]:5.2f}", end="")
        for r in results:
            print(f"  {r['h_profile'][i]:20.3e}", end="")
        print()


if __name__ == "__main__":
    main()
