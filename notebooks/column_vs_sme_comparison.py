# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Column Model vs SME vs Multi-Layer Comparison
#
# Compares three approaches to hydrostatic shallow-water moment equations on a Gaussian-bump
# test case in a closed basin:
#
# 1. **SME** at levels L0, L1, L2 — continuous Legendre polynomial basis, single layer
# 2. **Multi-Layer** at N=1, 2, 3 layers — piecewise-constant (step) basis, `level=0`
# 3. **Column Model** — Multi-Layer with many layers (N=8) + `ColumnStructure` integration
#
# All models share the same derivation pipeline; only the **basis** used in the final
# projection step differs. The notebook covers:
#
# - Time evolution of $h$ and mean-$\bar u$ on a 4×4 snapshot grid
# - Vertical velocity profile comparison at multiple $x$-locations
# - Comparison with / without friction (Newtonian + Navier slip)
# - Model derivation graphs (Mermaid) with full equations
# - `ColumnStructure` numerical integration convergence

# %% [markdown]
# ## 1. Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from zoomy_core.model.models.sme_model import SMEInviscid, SMEModel
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.initial_conditions import UserFunction
import zoomy_core.model.boundary_conditions as BC

from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts

from zoomy_core.mesh.column_structure import ColumnStructure

# %% [markdown]
# ## 2. Test Case Parameters
#
# Closed basin $[0, 10]$ with wall boundaries. Initial Gaussian bump:
#
# $$h(x, 0) = 1 + A \exp\!\left(-\frac{(x - x_0)^2}{2\sigma^2}\right), \quad u = 0, \quad b = 0.$$

# %%
N_HORIZONTAL = 200
L_DOMAIN = 10.0
X0 = 5.0
SIGMA = 0.5
AMPLITUDE = 0.1
T_END = 2.0
CFL = 0.3
N_LAYERS_COLUMN = 8
N_SNAPSHOTS = 16  # for 4x4 grid


def ic_gauss(x, n_vars):
    Q = np.zeros(n_vars)
    Q[0] = 0.0
    Q[1] = 1.0 + AMPLITUDE * np.exp(-((x[0] - X0) ** 2) / (2 * SIGMA ** 2))
    return Q


# %% [markdown]
# ## 3. Solver Helper with In-Memory Snapshots
#
# The solver already has snapshot machinery: `settings.output.snapshots` controls how
# many snapshots to record, and `_sim_save_fields` is the callback invoked at each
# snapshot time. We just replace the default HDF5 writer with an in-memory collector —
# no re-implementation of the time loop.

# %%
from zoomy_core.misc.misc import Settings, Zstruct


def in_memory_snapshot_collector(Q_list, times_list):
    """Return a save_fields callback that appends (time, Q) to the given lists."""
    def save(time, time_stamp, i_snapshot, Q, Qaux):
        Q_list.append(np.array(Q))
        times_list.append(float(time))
        return i_snapshot + 1
    return save


def solve_with_snapshots(level, n_layers=1, friction=False,
                         n_snapshots=N_SNAPSHOTS, t_end=T_END):
    """Set up and run a solver, collecting in-memory snapshots via the
    solver's own ``_sim_save_fields`` callback."""
    ModelCls = SMEModel if friction else SMEInviscid
    model = ModelCls(level=level, n_layers=n_layers)

    if friction:
        model.parameters.update({"nu": 1e-4, "lamda": 1e-2})

    model.initial_conditions = UserFunction(
        lambda x: ic_gauss(x, model.n_variables))
    model._system.boundary_conditions.apply(BC.SystemWall(), tag="left")
    model._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    model.boundary_conditions = BC.compile_system_bcs(
        model._system.boundary_conditions,
        model._equation_variable_map,
        model.dimension,
    )

    mesh = BaseMesh.create_1d(domain=(0.0, L_DOMAIN), n_inner_cells=N_HORIZONTAL)
    settings = Settings(output=Zstruct(
        directory="output_tmp", filename="sme", snapshots=n_snapshots,
        clean_directory=True,
    ))
    solver = FreeSurfaceFlowSolver(
        time_end=t_end, reconstruction_order=2,
        compute_dt=ts.adaptive(CFL=CFL), settings=settings,
    )
    solver.setup_simulation(mesh, model, write_output=False)

    # Replace HDF5 writer with in-memory collector
    Q_snapshots = [np.array(solver._sim_Q)]
    times_recorded = [0.0]
    solver._sim_save_fields = in_memory_snapshot_collector(
        Q_snapshots, times_recorded)

    # Run using the solver's own time loop
    solver.run_simulation()

    return solver._sim_mesh, model, np.array(times_recorded), Q_snapshots


# %% [markdown]
# ## 4. Run All Configurations (Frictionless + Friction)
#
# Full sweep: 3 SME levels × 3 Multi-Layer + Column + 2 friction modes.
# This runs the majority of the simulation cost of the notebook.

# %%
def run_family(friction):
    """Run SME L0/L1/L2, Multi-Layer N=1,2,3, and Column N=8."""
    fam = {"friction": friction, "sme": {}, "ml": {}, "column": None}
    print(f"Running {'friction' if friction else 'frictionless'} suite...")
    for level in [0, 1, 2]:
        mesh, model, t, Qs = solve_with_snapshots(level=level, n_layers=1,
                                                    friction=friction)
        fam["sme"][level] = dict(mesh=mesh, model=model, times=t, snapshots=Qs,
                                 label=f"SME L{level}")
    for n_layers in [1, 2, 3]:
        mesh, model, t, Qs = solve_with_snapshots(level=0, n_layers=n_layers,
                                                    friction=friction)
        fam["ml"][n_layers] = dict(mesh=mesh, model=model, times=t, snapshots=Qs,
                                   label=f"ML N={n_layers}")
    mesh, model, t, Qs = solve_with_snapshots(level=0, n_layers=N_LAYERS_COLUMN,
                                                friction=friction)
    fam["column"] = dict(mesh=mesh, model=model, times=t, snapshots=Qs,
                         label=f"Column N={N_LAYERS_COLUMN}")
    return fam


results_inv = run_family(friction=False)
results_fric = run_family(friction=True)

# %% [markdown]
# ## 5. Time Evolution: 4×4 Grid of $h(x, t)$ Snapshots
#
# Each subplot shows $h(x)$ at one snapshot time. Four model families are overlaid per panel.

# %%
def plot_h_evolution_grid(family, title):
    mesh = family["column"]["mesh"]
    nc = mesh.n_inner_cells
    x = mesh.cell_centers[0, :nc]
    times = family["column"]["times"]
    n_snaps = len(times)
    n_show = min(n_snaps, 16)

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f"h(x, t) — {title}", fontsize=14)

    for k in range(16):
        ax = axes[k // 4, k % 4]
        if k >= n_show:
            ax.axis("off")
            continue
        t = times[k]
        # Plot SME L0, SME L2, Multi-Layer N=2, Column N=8
        ax.plot(x, family["sme"][0]["snapshots"][k][1, :nc],
                "-", lw=1, label="SME L0", color="tab:blue")
        ax.plot(x, family["sme"][2]["snapshots"][k][1, :nc],
                "-", lw=1, label="SME L2", color="tab:green")
        ax.plot(x, family["ml"][3]["snapshots"][k][1, :nc],
                "--", lw=1, label="ML N=3", color="tab:orange")
        ax.plot(x, family["column"]["snapshots"][k][1, :nc],
                "-.", lw=1.5, label=f"Col N={N_LAYERS_COLUMN}", color="red")
        ax.set_title(f"t = {t:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for r in range(4):
        axes[r, 0].set_ylabel("h")
    for c in range(4):
        axes[-1, c].set_xlabel("x")
    plt.tight_layout()
    plt.show()


plot_h_evolution_grid(results_inv, "frictionless")
plot_h_evolution_grid(results_fric, "with friction")


# %% [markdown]
# ## 6. Time Evolution: 4×4 Grid of Mean Velocity $\bar u(x, t)$
#
# $\bar u = (h u)_0 / h$ — the mean horizontal velocity (first moment / depth).

# %%
def plot_u_evolution_grid(family, title):
    mesh = family["column"]["mesh"]
    nc = mesh.n_inner_cells
    x = mesh.cell_centers[0, :nc]
    times = family["column"]["times"]
    n_snaps = len(times)
    n_show = min(n_snaps, 16)

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f"Mean velocity u̅(x, t) — {title}", fontsize=14)

    for k in range(16):
        ax = axes[k // 4, k % 4]
        if k >= n_show:
            ax.axis("off")
            continue
        t = times[k]
        # Mean velocity = (hu_0) / h
        def ubar(Q):
            return Q[2, :nc] / Q[1, :nc]
        ax.plot(x, ubar(family["sme"][0]["snapshots"][k]),
                "-", lw=1, label="SME L0", color="tab:blue")
        ax.plot(x, ubar(family["sme"][2]["snapshots"][k]),
                "-", lw=1, label="SME L2", color="tab:green")
        ax.plot(x, ubar(family["ml"][3]["snapshots"][k]),
                "--", lw=1, label="ML N=3", color="tab:orange")
        ax.plot(x, ubar(family["column"]["snapshots"][k]),
                "-.", lw=1.5, label=f"Col N={N_LAYERS_COLUMN}", color="red")
        ax.set_title(f"t = {t:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for r in range(4):
        axes[r, 0].set_ylabel("u̅")
    for c in range(4):
        axes[-1, c].set_xlabel("x")
    plt.tight_layout()
    plt.show()


plot_u_evolution_grid(results_inv, "frictionless")
plot_u_evolution_grid(results_fric, "with friction")


# %% [markdown]
# ## 7. Vertical Velocity Profile Comparison
#
# At selected locations, reconstruct the full $u(\zeta)$ profile from each model:
#
# - **SME L$k$**: $u(\zeta) = \sum_{j=0}^{k} \alpha_j \phi_j(\zeta)$ with Legendre basis
# - **Multi-Layer N**: piecewise-constant $u(\zeta) = \bar u_k$ on layer $k$
# - **Column N**: same piecewise-constant structure with many layers
#
# We plot at four $x$-locations: near the left wall, at $x = 2.5$, at the bump peak
# ($x = 5$), and near the right wall.

# %%
def reconstruct_profile_sme(Q_col, level, zeta_eval):
    """Reconstruct u(zeta) from SME Legendre moments at one column.

    Q_col: shape (n_vars,) — one column of Q, layout [b, h, h*alpha_0, ...].
    level: polynomial level.
    Returns: (len(zeta_eval),) — u values at each zeta.
    """
    h = Q_col[1]
    alphas = np.array([Q_col[2 + k] / h for k in range(level + 1)])
    basis = Legendre_shifted(level=level)
    u = np.zeros_like(zeta_eval, dtype=float)
    for k, alpha in enumerate(alphas):
        # Evaluate shifted Legendre phi_k at zeta_eval
        phi_vals = np.array([float(basis.eval(k, z)) for z in zeta_eval])
        u += alpha * phi_vals
    return u


def reconstruct_profile_multilayer(Q_col, n_layers, zeta_eval):
    """Reconstruct u(zeta) from Multi-Layer piecewise-constant moments.

    Q_col: shape (n_vars,), layout [b, h, h*u_0, h*u_1, ..., h*u_{N-1}].
    Each layer occupies zeta in [k/N, (k+1)/N].
    """
    h = Q_col[1]
    u_layers = np.array([Q_col[2 + k] / h for k in range(n_layers)])
    u = np.zeros_like(zeta_eval, dtype=float)
    for i, zeta in enumerate(zeta_eval):
        k = min(int(zeta * n_layers), n_layers - 1)
        u[i] = u_layers[k]
    return u


def plot_velocity_profiles(family, title):
    x_targets = np.array([0.5, 2.5, 5.0, 9.5])
    mesh = family["column"]["mesh"]
    nc = mesh.n_inner_cells
    x_cells = mesh.cell_centers[0, :nc]
    # Last snapshot
    idx_last = -1
    t_last = family["column"]["times"][idx_last]

    # Find column indices closest to each target x
    col_indices = [int(np.argmin(np.abs(x_cells - xt))) for xt in x_targets]

    zeta_fine = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, len(x_targets), figsize=(16, 4), sharey=True)
    fig.suptitle(f"Vertical velocity profile at t = {t_last:.3f} — {title}",
                 fontsize=13)

    for j, (xt, ci) in enumerate(zip(x_targets, col_indices)):
        ax = axes[j]
        # SME L0/L1/L2
        for level, color in zip([0, 1, 2],
                                 ["tab:blue", "tab:green", "tab:purple"]):
            Q_col = family["sme"][level]["snapshots"][idx_last][:, ci]
            u = reconstruct_profile_sme(Q_col, level, zeta_fine)
            ax.plot(u, zeta_fine, "-", lw=1.5, color=color,
                    label=f"SME L{level}")
        # Multi-Layer N=3
        Q_col = family["ml"][3]["snapshots"][idx_last][:, ci]
        u_ml = reconstruct_profile_multilayer(Q_col, 3, zeta_fine)
        ax.plot(u_ml, zeta_fine, "--", lw=1.5, color="tab:orange",
                label="ML N=3")
        # Column N=8
        Q_col = family["column"]["snapshots"][idx_last][:, ci]
        u_col = reconstruct_profile_multilayer(Q_col, N_LAYERS_COLUMN, zeta_fine)
        ax.plot(u_col, zeta_fine, "-.", lw=2, color="red",
                label=f"Col N={N_LAYERS_COLUMN}")

        ax.set_title(f"x = {x_cells[ci]:.2f}")
        ax.set_xlabel("u(ζ)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("ζ = (z-b)/h")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


plot_velocity_profiles(results_inv, "frictionless")
plot_velocity_profiles(results_fric, "with friction")

# %% [markdown]
# ## 7.5 Interlude: What is the Multi-Layer approach actually doing?
#
# Looking at the active `derived_model.py`, the current
# `SMEModel(level=0, n_layers=N)` does **not** use a Heaviside / Dirac-delta
# projection. Each layer uses the **same** Legendre basis on $[0,1]$ and
# contributes to depth integrals with a uniform layer weight $w_k = 1/N$:
#
# $$\int_0^1 f(\zeta) \, d\zeta \;\approx\; \sum_{k=1}^{N} w_k \int_{\zeta_{k-1}}^{\zeta_k} f \, d\zeta \;=\; \frac{1}{N} \sum_{k=1}^{N} \bar{f}_k.$$
#
# At `level=0`, each layer has a single constant velocity $\bar u_k$, and
# the state vector per horizontal cell is $[b, h, h\bar u_1, \dots, h\bar u_N]$.
# The advective flux for layer $k$ involves **only** layer $k$'s own moments
# (no inter-layer coupling in the advection). Layers couple only through:
#
# 1. The shared $h$ (mass equation sums all layers)
# 2. The shared free-surface pressure gradient
# 3. Viscous / slip source terms (through the basis $D$ matrix)
#
# **There is no explicit Riemann solver at layer interfaces.** The horizontal
# Riemann solver treats the full $(2 + N)$-dimensional state as a single vector.
#
# The *legacy* `pde_generator.py` did implement a Heaviside-windowed ansatz
# where $\partial/\partial \zeta$ of the window produced Dirac deltas at
# interfaces (regularized via the sifting property). **The current active
# path does not use that.**
#
# ### Column approach vs Multi-Layer — they are genuinely different
#
# | | Multi-Layer ($n_\text{layers}=N$) | Column (ColumnIntegratingSolver) |
# |-|-|-|
# | Mesh | 1D horizontal only | 2D/3D extruded mesh |
# | Vertical DOFs | $N$ moments per horizontal cell | $N_z$ explicit vertical cells |
# | Vertical fluxes | None — depth integrals via $w_k$ | Computed numerically per face |
# | Interface treatment | No per-interface Riemann; coupling via symbolic flux | Riemann solver on every face |
# | 2nd order | MUSCL on horizontal state vector | LSQ-MUSCL in **both** directions |
# | DOFs | $(2 + N) \cdot N_\text{h}$ | $(2 + N_h) \cdot N_z \cdot N_\text{h}$ |
#
# **Key takeaway:** Multi-Layer achieves "vertical resolution" by adding more
# depth-averaged moments; there's no vertical numerical flux. Column is a true
# 2D/3D method where the vertical direction is treated identically to the
# horizontal. They converge only in the hydrostatic regime; for non-hydrostatic
# or strongly overturning flows, only Column can resolve the physics.
#
# ### 2nd order for Multi-Layer
#
# Yes — but only in the horizontal direction. MUSCL is applied uniformly to
# the full state vector $[b, h, h\bar u_1, \dots, h\bar u_N]$ without
# awareness of which entries represent which layer. There is no notion of
# "vertical 2nd order" because there is no vertical discretization.
#
# For the Column solver, the LSQ-MUSCL reconstruction in 2D/3D naturally
# provides 2nd order in both horizontal and vertical directions on the
# extruded mesh.
#
# ## 8. Observation on Friction
#
# - **Frictionless case**: SME L0 = L1 = L2 and all Multi-Layer configurations collapse to a
#   flat $u(\zeta)$ profile. Without a shear source, the higher moments stay zero and all
#   models reduce to plain SWE.
#
# - **With friction**: The Newtonian stress $\tau = \rho\nu\, \partial u/\partial z$ acting
#   at the bottom (via Navier slip) drives a vertical shear. In SME this excites the
#   $\alpha_1, \alpha_2$ moments; in Multi-Layer it differentiates the per-layer velocities.
#   SME L0 still has no $\zeta$-dependence — its profile is always constant.
#
# This notebook uses $\nu_\text{bulk} = 10^{-4}$ and wall slip length
# $\lambda = 10^{-2}$ (small $\lambda$ = strong friction). The slip stress is
# $\tau_\text{wall} = -u_\text{bottom}/\lambda$, so with $\lambda = 10^{-2}$ the
# wall friction is 10× stronger than the bulk viscosity effect — clearly visible
# in the profile plots.
#
# You can reconfigure: `model.parameters.update({"nu": 1e-2, "lamda": 1e-1})`
# or per attribute: `model.parameters.nu = 1e-3`.

# %% [markdown]
# ## 9. ColumnStructure Convergence
#
# `ColumnStructure.integrate()` evaluates $\int_0^1 Q(\zeta)\,d\zeta$ numerically by the
# midpoint rule. For a smooth integrand, it converges at $O(N_z^{-2})$.

# %%
exact = 1.0 / 30.0  # ∫₀¹ [ζ(1-ζ)]² dζ
fig, ax = plt.subplots(figsize=(6, 5))
Nzs = [2, 4, 8, 16, 32, 64, 128]
errs = []
for n_z in Nzs:
    cs = ColumnStructure(n_horizontal=1, n_layers=n_z)
    Q_u2 = np.zeros((1, n_z))
    for iz in range(n_z):
        zeta = cs.zeta_midpoints[iz]
        Q_u2[0, iz] = (zeta * (1 - zeta)) ** 2
    num = cs.integrate(Q_u2)[0, 0]
    errs.append(abs(num - exact))

ax.loglog(Nzs, errs, "o-", lw=2, label="midpoint rule")
ax.loglog(Nzs, [errs[0] * (Nzs[0] / nz) ** 2 for nz in Nzs],
          "k--", lw=1, label=r"$O(N_z^{-2})$")
ax.set_xlabel("N_z (number of layers)")
ax.set_ylabel("|error| in  ∫₀¹ [ζ(1-ζ)]² dζ")
ax.set_title("ColumnStructure: convergence of numerical integration")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Model Derivation Graphs (Mermaid)
#
# Each `model.describe(...)` call produces a rendered markdown cell containing:
# - **Mermaid** graph of the derivation (class hierarchy + applied operations)
# - **Assumptions** (hydrostatic pressure, kinematic BCs, material model, ...)
# - **Final equations** in LaTeX (continuity, x-momentum)

# %% [markdown]
# ### 10.1 SME L0 (frictionless) — plain SWE

# %%
display(results_inv["sme"][0]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ### 10.2 SME L1 (frictionless)

# %%
display(results_inv["sme"][1]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ### 10.3 SME L2 (frictionless)

# %%
display(results_inv["sme"][2]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ### 10.4 Multi-Layer N=3 (frictionless)

# %%
display(results_inv["ml"][3]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ### 10.5 Column Model N=8 (frictionless)

# %%
display(results_inv["column"]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ### 10.6 SME L1 WITH friction — derivation includes Newtonian material

# %%
display(results_fric["sme"][1]["model"].describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ## 11. Summary
#
# | Model | Basis | # Moments | Variables |
# |-------|-------|-----------|-----------|
# | SME L0 | Legendre $\phi_0$ | 1 | $[b, h, h\bar u]$ |
# | SME L1 | Legendre $\phi_0, \phi_1$ | 2 | $[b, h, h\alpha_0, h\alpha_1]$ |
# | SME L2 | Legendre $\phi_0, \phi_1, \phi_2$ | 3 | $[b, h, h\alpha_0, h\alpha_1, h\alpha_2]$ |
# | Multi-Layer N=1 | 1 indicator (= SWE) | 1 | $[b, h, h\bar u_1]$ |
# | Multi-Layer N=2 | 2 layer indicators | 2 | $[b, h, h\bar u_1, h\bar u_2]$ |
# | Multi-Layer N=3 | 3 layer indicators | 3 | $[b, h, h\bar u_1, h\bar u_2, h\bar u_3]$ |
# | **Column Model N=8** | **8 layer indicators** | **8** | $[b, h, h\bar u_1, \dots, h\bar u_8]$ |
#
# **Frictionless case**: All hydrostatic models yield identical $h$ — no vertical shear source,
# so all higher moments stay zero and every model reduces to plain SWE.
#
# **Friction case**: Bottom stress excites vertical shear. SME L1/L2 and Multi-Layer N>1
# differ from plain SWE; the Column Model at high $N$ provides the most detailed vertical
# profile at the cost of more DOF.
#
# **ColumnStructure**: $O(N_z^{-2})$ convergence for smooth integrands (midpoint rule).

# %% [markdown]
# ## 12. Convergence Test for Higher SME Levels
#
# Run SME at levels L0, L1, L2 on progressively refined meshes and measure the
# $L^2$ error of $h$ against a reference solution (SME L0 at the finest resolution).
#
# **What this shows:** Higher $n_\text{order}$ does not reduce the error for flows
# without vertical shear — they all converge to the same solution in the mesh
# refinement limit. The mesh-refinement convergence rate is dominated by the
# spatial discretization (MUSCL + RK2 → O(h²) in the interior).

# %%
from zoomy_core.mesh import ensure_lsq_mesh

def l2_h(Q_a, mesh_a, Q_ref, mesh_ref):
    """L2 error in h between two solutions on DIFFERENT meshes.
    Interpolates Q_a onto the centres of mesh_ref (1D)."""
    nc_a = mesh_a.n_inner_cells
    nc_ref = mesh_ref.n_inner_cells
    x_a = mesh_a.cell_centers[0, :nc_a]
    x_ref = mesh_ref.cell_centers[0, :nc_ref]
    h_a_on_ref = np.interp(x_ref, x_a, Q_a[1, :nc_a])
    dx = mesh_ref.cell_volumes[:nc_ref]
    err = np.sqrt(np.sum((h_a_on_ref - Q_ref[1, :nc_ref]) ** 2 * dx) / np.sum(dx))
    return float(err)


def solve_at_resolution(level, N_h, t_end=T_END):
    """Run SMEInviscid at the given level + horizontal resolution."""
    model = SMEInviscid(level=level, n_layers=1)
    model.initial_conditions = UserFunction(
        lambda x: ic_gauss(x, model.n_variables))
    model._system.boundary_conditions.apply(BC.SystemWall(), tag="left")
    model._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    model.boundary_conditions = BC.compile_system_bcs(
        model._system.boundary_conditions,
        model._equation_variable_map,
        model.dimension,
    )
    mesh = BaseMesh.create_1d(domain=(0.0, L_DOMAIN), n_inner_cells=N_h)
    solver = FreeSurfaceFlowSolver(
        time_end=t_end, reconstruction_order=2,
        compute_dt=ts.adaptive(CFL=CFL),
    )
    Q, _ = solver.solve(mesh, model, write_output=False)
    return ensure_lsq_mesh(mesh, model), Q


# Reference solution: highest-resolution L0 run
N_REF = 800
T_CONV = 0.5  # shorter time to avoid the reflected wave overlapping
mesh_ref, Q_ref = solve_at_resolution(level=0, N_h=N_REF, t_end=T_CONV)

Ns = [50, 100, 200, 400]
errors_by_level = {}
for level in [0, 1, 2]:
    errs = []
    for N in Ns:
        mesh_a, Q_a = solve_at_resolution(level=level, N_h=N, t_end=T_CONV)
        err = l2_h(Q_a, mesh_a, Q_ref, mesh_ref)
        errs.append(err)
    errors_by_level[level] = errs

# %% [markdown]
# ### Convergence table

# %%
print(f"L2 error of h vs L0 reference at N={N_REF}, t={T_CONV}")
print("-" * 60)
print(f"{'N_h':<8}" + "".join(f"{'L'+str(lev):>16}" for lev in [0, 1, 2]))
for i, N in enumerate(Ns):
    row = f"{N:<8}" + "".join(
        f"{errors_by_level[lev][i]:>16.4e}" for lev in [0, 1, 2])
    print(row)

# %% [markdown]
# ### Convergence plot

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for level, errs in errors_by_level.items():
    ax.loglog(Ns, errs, "o-", lw=1.5, label=f"SME L{level}")
# Reference slope O(N^{-2})
ax.loglog(Ns, [errs[0] * (Ns[0] / n) ** 2 for n in Ns], "k--", lw=1,
          label=r"$O(N^{-2})$")
ax.set_xlabel("$N_h$ (horizontal cells)")
ax.set_ylabel(r"$\|h - h_\mathrm{ref}\|_{L^2}$")
ax.set_title("SME convergence vs L0 reference")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation**: For this shear-free Gaussian-bump test, L0/L1/L2 converge at
# the same rate (MUSCL spatial O(h²)) to the same solution. Higher-order bases
# only pay off when the flow has vertical structure the lower levels cannot
# represent — e.g. viscous flows with strong bottom shear.

# %% [markdown]
# ## 13. Notes
#
# - The `ColumnStructure` (in `zoomy_core/mesh/column_structure.py`) is model-agnostic — it
#   operates on any `(n_vars, n_3d_cells)` array with the extrusion convention
#   `cell_3d = iz * n_horizontal + i_horizontal`.
# - The `ColumnIntegratingSolver` (in `zoomy_core/fvm/solver_column.py`) inherits from
#   `IMEXSolver` and exposes `integrate_vertical()`, `partial_integrate()`,
#   `depth_average()` methods.
# - For the full "raw NS on extruded mesh + numerical depth integration" workflow, a `Model`
#   wrapper that consumes `FullINS` equations without `with_basis()` projection is the next
#   step.
# - The symbolic ML-SWE / ML-SME derivation using a true piecewise basis is in
#   `notebooks/ml_sme_derivation.py` (companion notebook to this one).
