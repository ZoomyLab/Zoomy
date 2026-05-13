# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # VAM(1, 2, 2) — flow over a Gaussian bump (Escalante 2024 benchmark)
#
# ::: {.callout-note title="Reference"}
# Escalante, C., Morales De Luna, T., Cantero-Chinchilla, F. and
# Castro-Orgaz, O. (2024). *Vertically averaged and moment equations:
# New derivation, efficient numerical solution and comparison with
# other physical approximations for modeling non-hydrostatic free
# surface flows.* Journal of Computational Physics 504, 112882.
# DOI: 10.1016/j.jcp.2024.112882
# :::
#
# This notebook reproduces the **flow over a bump** test from the
# original `tutorials/vam/simple.ipynb` (commit `c80fe12c`).
# Twenty-four reference points $(x, \eta)$ and $(x, p_b)$ are baked
# in from `tests/pdesoft/plots_paper.py:vam_analytical_eta /
# vam_analytical_p`.
#
# ## What this notebook tests
#
# Rather than running to steady state ($T = 50$, expensive on the
# implicit DAE solver), we use the analytical reference profile as
# **initial condition** (cubic-spline interpolated through the
# digitized reference points) and run the chain DAE for a short
# physical time $T = 0.5$.  The drift away from the analytical
# steady state **is** the discretization error: a faithful
# discretization preserves a steady state, a sloppy one drifts.
#
# Plot overlays at the end:
#
# * 24 reference markers (Escalante paper digitization).
# * Initial condition (cubic spline through the markers).
# * Numerical state after $T$.
# * Bathymetry $b(x)$ Gaussian bump.

# %% [markdown]
# ## 1. Imports & configuration

# %%
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.fvm.solver_dae_numpy import DAESolver
import zoomy_core.fvm.timestepping as ts


class DAESolverWithBathymetry(DAESolver):
    """Demonstrates the canonical user override of
    :meth:`Solver.update_qaux`: fill the function-aux row ``b`` from
    a per-cell topography callable, then defer to the registry-driven
    super-class implementation to compute every derivative-aux row
    (``b_x``, ``h_x``, …) via LSQ."""

    def __init__(self, *, bathymetry, **kwargs):
        super().__init__(**kwargs)
        self._bathymetry = bathymetry

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model,
                    parameters, time, dt):
        # Function-aux rows from the registry.
        nc = Q.shape[1]
        for entry in self.sm.aux_registry:
            if entry["kind"] == "function" and entry["name"] == "b":
                Qaux[entry["row"], :] = self._bathymetry(
                    mesh.cell_centers[:, :nc]
                )
        # Derivative-aux rows via the default LSQ walker.
        return super().update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt,
        )

# %%
# Domain + numerics.  Original Escalante setup used NX=60, but
# running 60 cells × T=0.5 through the implicit DAE Newton at the
# current performance is multi-hour; we use a coarser grid here so
# the notebook stays runnable.  Bump the grid up for production.
NX = 24
DOMAIN = (-1.5, 1.5)
G = 9.81
RHO = 1000.0

# Bathymetry: Gaussian bump.
B_HEIGHT, B_SIGMA = 0.20, 0.20


def bathymetry(cell_centers):
    """``b(x) = 0.20 · exp(-x²/(2·0.2²))`` evaluated at every cell.

    Accepts either a single point ``x`` (shape ``(dim,)``) or an array
    of cell centres ``(dim, n_cells)``.
    """
    arr = np.asarray(cell_centers)
    x = arr[0] if arr.ndim == 2 else arr[0]
    return B_HEIGHT * np.exp(-x ** 2 / (2 * B_SIGMA ** 2))


# %% [markdown]
# ## 2. Analytical reference points (Escalante 2024, digitized)
#
# Twenty-four $(x, \eta)$ and $(x, p_b/(\rho g))$ points lifted
# verbatim from the `vam_analytical_eta()` and `vam_analytical_p()`
# helpers of `tests/pdesoft/plots_paper.py`.

# %%
def vam_analytical_eta():
    x = [-0.5928667563930013, -0.5430686406460297, -0.4946164199192463,
         -0.4448183041722746, -0.3990578734858681, -0.34522207267833105,
         -0.2981157469717362, -0.2510094212651413, -0.19851951547779273,
         -0.15141318977119783, -0.10430686406460293, -0.0531628532974428,
         -0.0006729475100942239, 0.04643337819650073, 0.09757738896366086,
         0.14737550471063254, 0.19851951547779279, 0.24562584118438757,
         0.29811574697173626, 0.34791386271870794, 0.3963660834454913,
         0.446164199192463, 0.49865410497981155, 0.5511440107671601]
    y = [0.3418918918918919, 0.34121621621621623, 0.3398648648648649,
         0.3418918918918919, 0.3398648648648649, 0.3398648648648649,
         0.33851351351351355, 0.33783783783783783, 0.3337837837837838,
         0.32770270270270274, 0.322972972972973, 0.31486486486486487,
         0.3054054054054054, 0.29054054054054057, 0.26891891891891895,
         0.2425675675675676, 0.21621621621621623, 0.18581081081081083,
         0.15540540540540543, 0.13108108108108107, 0.10608108108108108,
         0.0918918918918919, 0.07297297297297298, 0.06554054054054054]
    return np.asarray(x), np.asarray(y)


def vam_analytical_p():
    x = [-0.6001390820584145, -0.5556328233657858, -0.5041724617524339,
         -0.4485396383866481, -0.3998609179415855, -0.35396383866481224,
         -0.30389429763560505, -0.24965229485396384, -0.2051460361613352,
         -0.15229485396383868, -0.1008344923504868, -0.05354659248956889,
         -0.0006954102920723737, 0.0521557719054242, 0.09805285118219742,
         0.15090403337969394, 0.20236439499304582, 0.24826147426981915,
         0.30111265646731566, 0.3539638386648122, 0.4026425591098748,
         0.45271210013908203, 0.5013908205841446, 0.5514603616133518]
    y = [0.3319327731092437, 0.33053221288515405, 0.32212885154061627,
         0.30952380952380953, 0.29061624649859946, 0.27450980392156865,
         0.25, 0.22338935574229693, 0.18417366946778713, 0.15756302521008403,
         0.12394957983193278, 0.09453781512605042, 0.0700280112044818,
         0.04201680672268908, 0.04481792717086835, 0.03431372549019608,
         0.04831932773109244, 0.058823529411764705, 0.06022408963585434,
         0.06932773109243698, 0.0742296918767507, 0.0861344537815126,
         0.08473389355742297, 0.07913165266106442]
    return np.asarray(x), np.asarray(y)


# %% [markdown]
# ## 3. Smooth interpolants for the IC
#
# Cubic-spline interpolation through the reference points, clamped to
# the constant far-field values outside the data interval (the bump
# is local to $|x| < 0.6$, so the rest of the domain is flat
# subcritical or supercritical).

# %%
x_eta_ref, eta_ref = vam_analytical_eta()
x_pb_ref, pb_ref = vam_analytical_p()

# Far-field free-surface heights.  Take the leftmost / rightmost
# digitized values as constant inflow / outflow surface.
eta_left, eta_right = eta_ref[0], eta_ref[-1]
pb_left, pb_right = pb_ref[0], pb_ref[-1]

eta_spline = CubicSpline(x_eta_ref, eta_ref, bc_type="natural",
                         extrapolate=False)
pb_spline = CubicSpline(x_pb_ref, pb_ref, bc_type="natural",
                        extrapolate=False)


def eta_analytical(x):
    """Smooth free-surface profile from the reference data."""
    x = np.asarray(x)
    y = eta_spline(x)
    y = np.where(x < x_eta_ref[0], eta_left, y)
    y = np.where(x > x_eta_ref[-1], eta_right, y)
    return y


def pb_analytical(x):
    """Smooth bottom-pressure profile from the reference data."""
    x = np.asarray(x)
    y = pb_spline(x)
    y = np.where(x < x_pb_ref[0], pb_left, y)
    y = np.where(x > x_pb_ref[-1], pb_right, y)
    return y


# Quick sanity check.
x_check = np.linspace(*DOMAIN, 300)
plt.figure(figsize=(10, 3))
plt.plot(x_check, eta_analytical(x_check), label=r"$\eta$ spline")
plt.plot(x_eta_ref, eta_ref, "ko", label=r"$\eta^\text{exp}$ markers")
plt.plot(x_check, bathymetry((x_check,)), label=r"$b$ bump")
plt.legend(); plt.xlabel("x"); plt.ylabel("m"); plt.title("Analytical IC")
plt.tight_layout(); plt.savefig("/tmp/vam_bump_ic_check.png", dpi=120)

# %% [markdown]
# ## 4. Initial condition for the chain state
#
# Steady-state mass flux $Q = h \cdot u_0 = 0.11197$ from the
# Escalante setup.  We map the analytical $\eta$, $p_b$ to the
# chain's 7-state primitive form:
#
# * $h(x) = \eta(x) - b(x)$,
# * $U_0(x) = Q / h(x)$ (depth-averaged horizontal velocity),
# * $U_1, W_0, W_1 = 0$ as initial guess (the projection step
#   adjusts the algebraic $P_k$'s anyway),
# * $P_1(x) = \tfrac{g}{2}(p_b(x) - h(x))$ from
#   $p_b/(\rho g) = h + 2 p_1/g$,
# * $P_0(x) = 0$ initial guess.

# %%
Q_FLUX = 0.11197    # prescribed steady-state h·u_0 from Escalante setup

def ic_state(x):
    """Return the 7-entry primitive state at a single cell-centre x."""
    eta = float(eta_analytical(x[0]))
    b = float(bathymetry(x))
    h = max(eta - b, 0.015)
    pb = float(pb_analytical(x[0]))
    P_1 = 0.5 * G * (pb - h)
    return np.array([
        h,                          # h
        Q_FLUX / h,                 # U_0
        0.0,                        # U_1
        0.0,                        # W_0
        0.0,                        # W_1
        0.0,                        # P_0
        P_1,                        # P_1
    ])


# %% [markdown]
# ## 5. Build model + mesh + solver

# %%
model = VAMModelGalerkin(level=1, dimension=2)
model.parameters.g = G
model.parameters.rho = RHO
model.initial_conditions = UserFunction(function=ic_state)
model.boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])

mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=NX)

solver = DAESolverWithBathymetry(
    bathymetry=bathymetry,
    time_end=0.02,
    method="ars232",
    compute_dt=ts.constant(dt=0.002),
    newton_tol=1e-6,
    newton_maxit=50,
)

# %% [markdown]
# ## 6. Run.  Bathymetry through the topography callable.

# %%
solver.setup_simulation(mesh, model, write_output=False)
nc = solver.nc
x_cells = solver._sim_mesh.cell_centers[0, :nc]

# Snapshot the projected IC for the plot.
Q_init = solver._sim_Q.copy()
# Bathymetry row in Qaux (located via the auto-scan registry).
_b_row = next(e["row"] for e in solver.sm.aux_registry
              if e["kind"] == "function" and e["name"] == "b")
b_cells = solver._sim_Qaux[_b_row, :]

# Drive the time loop ourselves and capture multiple snapshots.
snapshots = [(0.0, Q_init.copy())]
T_FINAL = solver.time_end
n_snaps = 5
times_to_capture = np.linspace(0.0, T_FINAL, n_snaps)[1:]
next_capture = list(times_to_capture)
parameters = solver._parameters_array()

t = 0.0
while t < T_FINAL - 1e-12:
    dt = solver.compute_dt(
        solver._sim_Q, solver._sim_Qaux, parameters,
        solver._sim_min_inradius, solver._sim_max_eigenvalue,
    )
    dt = float(min(dt, T_FINAL - t))
    if not np.isfinite(dt) or dt <= 0:
        break
    solver._sim_time = t
    solver.step(dt)
    t += dt
    while next_capture and t >= next_capture[0] - 1e-12:
        snapshots.append((t, solver._sim_Q.copy()))
        next_capture.pop(0)

print(f"completed: {len(snapshots)} snapshots, t_final = {t:.3f}")

# %% [markdown]
# ## 7. Reproduce the original two-panel plot
#
# Left:  $\eta = h + b$, $b$, and the 24 analytical $\eta^\text{exp}$ markers.
# Right: $p_b/(\rho g) = h + 2 P_1/g$, and the analytical $p_b^\text{exp}$
# markers.
#
# All numerical snapshots plotted in time-graded line shading; the
# initial condition (cubic-spline through the markers) should sit on
# the marker points exactly; later snapshots show the drift due to
# discretization + boundary effects.

# %%
def panel_data(Q):
    h = Q[0, :]
    P_1 = Q[6, :]
    eta = h + b_cells
    pb = h + 2.0 * P_1 / G
    return h, eta, pb


fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True,
                         sharey=True)
cmap = plt.cm.viridis
n_snap = len(snapshots)

for i, (t_i, Q_i) in enumerate(snapshots):
    h_i, eta_i, pb_i = panel_data(Q_i)
    c = cmap(i / max(1, n_snap - 1))
    lw = 2.0 if i == 0 else 1.2
    label = f"t = {t_i:.3f}"
    axes[0].plot(x_cells, eta_i, color=c, lw=lw, label=label)
    axes[1].plot(x_cells, pb_i, color=c, lw=lw, label=label)

axes[0].plot(x_cells, b_cells, "k-", lw=1.0, label=r"$b$")
axes[0].plot(x_eta_ref, eta_ref, "k*", ms=8,
             label=r"$\eta^\text{exp}$ (Escalante)")
axes[1].plot(x_pb_ref, pb_ref, "k*", ms=8,
             label=r"$p_b^\text{exp}$ (Escalante)")

axes[0].set_xlabel(r"$x$ [m]")
axes[0].set_ylabel(r"$\eta$, $b$  [m]")
axes[0].set_title(r"Free surface")
axes[0].set_ylim(0, 0.4)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(alpha=0.3)

axes[1].set_xlabel(r"$x$ [m]")
axes[1].set_ylabel(r"$p_b / (\rho g)$  [m]")
axes[1].set_title(r"Bottom pressure")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(alpha=0.3)

fig.suptitle(
    "VAM(1, 2, 2) — flow over a bump.  IC = analytical interpolant; "
    f"drift after T = {T_FINAL} s on the chain-DAE solver."
)
fig.savefig("outputs_vam_bump_analytical.png", dpi=140,
            bbox_inches="tight")
print("plot → outputs_vam_bump_analytical.png")

# %% [markdown]
# ## 8. Drift summary
#
# $L^\infty$ deviation of $\eta_\text{num}(x, T)$ from the analytical
# spline at the cell centres — a direct measure of the
# discretization error after a short transient.

# %%
eta_analytical_cells = eta_analytical(x_cells)
pb_analytical_cells = pb_analytical(x_cells)

print(f"{'t':>6}  {'|η_num − η_ana|_∞':>20}  {'|p_b_num − p_b_ana|_∞':>22}")
for t_i, Q_i in snapshots:
    _, eta_i, pb_i = panel_data(Q_i)
    deta = float(np.max(np.abs(eta_i - eta_analytical_cells)))
    dpb = float(np.max(np.abs(pb_i - pb_analytical_cells)))
    print(f"{t_i:>6.3f}  {deta:>20.4e}  {dpb:>22.4e}")
