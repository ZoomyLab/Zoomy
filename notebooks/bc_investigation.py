# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Boundary Condition Investigation
#
# Systematic test of three BC types (Extrapolation, Wall, Periodic) with the
# ghost-cell-free NumPy solver on:
#
# - **SME L0** (plain SWE, 3 variables: $[b, h, hu]$)
# - **SME L1** (4 variables: $[b, h, hu_0, hu_1]$)
# - **SME L2** (5 variables)
# - **Column** (ML with 8 layers, level=0)
#
# at both **O1** and **O2** reconstruction.
#
# **Test case:** Gaussian bump $h(x,0) = 1 + 0.1 \exp(-(x-L/2)^2 / (2\sigma^2))$
# in a long domain $[0, L]$. At time $t_\text{check}$ the wavefront has not
# yet reached the boundary $\to$ boundary cells must be undisturbed. At
# $t_\text{end}$ the wave interacts with the boundary $\to$ check behaviour
# depends on BC type.

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from zoomy_core.model.models.sme_model import SMEInviscid
from zoomy_core.model.initial_conditions import UserFunction
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
from zoomy_core.misc.misc import Settings, Zstruct

# Test parameters — kept small for fast iteration
L_DOMAIN = 10.0
SIGMA = 0.3
AMPLITUDE = 0.05
N_HORIZONTAL = 50
CFL = 0.45

# Wave speed ≈ sqrt(g) ≈ 3.1 m/s
# Wavefront reaches boundary at ≈ 5/3.1 ≈ 1.6 s
T_CHECK = 0.5       # well before wave hits boundary
T_END = 2.0         # after wave reaches boundary


def ic_gauss(x, n_vars):
    Q = np.zeros(n_vars)
    Q[0] = 0.0
    Q[1] = 1.0 + AMPLITUDE * np.exp(-((x[0] - L_DOMAIN / 2) ** 2) / (2 * SIGMA ** 2))
    return Q


# %% [markdown]
# ## 2. Solver Helper
#
# Returns the state at two times: $t_\text{check}$ (before wall interaction) and
# $t_\text{end}$ (after wall interaction).

# %%
def solve_case(level, n_layers, bc_type, order):
    """Run one configuration. Returns (mesh, Q_check, Q_end, label).

    Runs a single solve to T_END and captures an early snapshot at T_CHECK.
    """
    model = SMEInviscid(level=level, n_layers=n_layers)

    model.initial_conditions = UserFunction(
        lambda x: ic_gauss(x, model.n_variables))

    if bc_type == "extrapolation":
        model._system.boundary_conditions.apply(
            BC.SystemExtrapolation(), tag="left")
        model._system.boundary_conditions.apply(
            BC.SystemExtrapolation(), tag="right")
    elif bc_type == "wall":
        model._system.boundary_conditions.apply(BC.SystemWall(), tag="left")
        model._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    elif bc_type == "periodic":
        model._system.boundary_conditions.apply(
            BC.SystemPeriodic(periodic_to_physical_tag="right"), tag="left")
        model._system.boundary_conditions.apply(
            BC.SystemPeriodic(periodic_to_physical_tag="left"), tag="right")

    model.boundary_conditions = BC.compile_system_bcs(
        model._system.boundary_conditions,
        model._equation_variable_map,
        model.dimension,
    )

    mesh = BaseMesh.create_1d(domain=(0.0, L_DOMAIN), n_inner_cells=N_HORIZONTAL)
    label = f"L{level}" if n_layers == 1 else f"Col{n_layers}"

    solver = FreeSurfaceFlowSolver(
        time_end=T_END, reconstruction_order=order,
        compute_dt=ts.adaptive(CFL=CFL),
    )
    solver.setup_simulation(mesh, model, write_output=False)

    # Capture snapshot at T_CHECK via in-memory callback
    Q_check = [None]
    orig_save = solver._sim_save_fields

    def save_with_capture(time, ts_, i_snap, Q, Qaux):
        if Q_check[0] is None and time >= T_CHECK - 1e-10:
            Q_check[0] = np.array(Q)
        return i_snap + 1

    solver._sim_save_fields = save_with_capture
    solver.run_simulation()

    if Q_check[0] is None:
        Q_check[0] = np.array(solver._sim_Q)

    return solver._sim_mesh, Q_check[0], solver._sim_Q, label


# %% [markdown]
# ## 3. Run All Configurations

# %%
configs = [
    ("SME L0", 0, 1),
    ("SME L1", 1, 1),
    ("SME L2", 2, 1),
    ("Column 8", 0, 8),
]
bc_types = ["extrapolation", "wall", "periodic"]
orders = [1, 2]

results = {}
for bc_type in bc_types:
    for order in orders:
        for name, level, n_layers in configs:
            key = (bc_type, order, name)
            print(f"  {bc_type:>14s}  O{order}  {name:>10s}", end="", flush=True)
            try:
                mesh, Q_check, Q_end, label = solve_case(
                    level, n_layers, bc_type, order)
                results[key] = dict(mesh=mesh, Q_check=Q_check, Q_end=Q_end,
                                    label=label)
                print(f"  h_check=[{Q_check[1].min():.5f}, {Q_check[1].max():.5f}]"
                      f"  h_end=[{Q_end[1].min():.5f}, {Q_end[1].max():.5f}]")
            except Exception as e:
                results[key] = None
                print(f"  FAILED: {e}")


# %% [markdown]
# ## 4. Boundary Integrity at $t_\text{check}$ (Before Wave Reaches Boundary)
#
# The wavefront travels at $c \approx \sqrt{g} \approx 3.1\,$m/s. At
# $t_\text{check} = 1\,$s it has moved $\approx 3\,$m from the centre —
# still $\approx 7\,$m from each boundary. The boundary cells (first/last 5)
# should be UNDISTURBED: $h = 1$, $hu = 0$.

# %%
print("Boundary integrity check at t_check =", T_CHECK)
print("=" * 80)
margin = 5  # number of boundary cells to check
tol = 1e-10

n_issues = 0
for bc_type in bc_types:
    for order in orders:
        for name, level, n_layers in configs:
            key = (bc_type, order, name)
            r = results.get(key)
            if r is None:
                continue
            Q = r["Q_check"]
            nc = r["mesh"].n_inner_cells
            # Check left boundary cells
            h_left = Q[1, :margin]
            hu_left = Q[2, :margin]
            # Check right boundary cells
            h_right = Q[1, nc - margin:]
            hu_right = Q[2, nc - margin:]

            h_err_left = np.max(np.abs(h_left - 1.0))
            h_err_right = np.max(np.abs(h_right - 1.0))
            hu_err_left = np.max(np.abs(hu_left))
            hu_err_right = np.max(np.abs(hu_right))

            ok = (h_err_left < tol and h_err_right < tol and
                  hu_err_left < tol and hu_err_right < tol)
            status = "OK" if ok else "ISSUE"
            if not ok:
                n_issues += 1
                print(f"  {bc_type:>14s} O{order} {name:>10s}: {status}"
                      f"  h_err_L={h_err_left:.2e}  h_err_R={h_err_right:.2e}"
                      f"  hu_err_L={hu_err_left:.2e}  hu_err_R={hu_err_right:.2e}")

if n_issues == 0:
    print("  All configurations: boundary cells undisturbed (h=1, hu=0) — OK")
else:
    print(f"\n  {n_issues} issue(s) detected!")


# %% [markdown]
# ## 5. Plots: $h(x)$ at $t_\text{check}$ and $t_\text{end}$
#
# ### 5.1 O1 — all BCs

# %%
def plot_bc_comparison(order_label, order):
    fig, axes = plt.subplots(len(bc_types), 2, figsize=(14, 4 * len(bc_types)),
                             sharex=True)
    fig.suptitle(f"h(x) — O{order} ({order_label})", fontsize=14, y=1.02)

    for row, bc_type in enumerate(bc_types):
        for col, (t_key, t_label) in enumerate([
            ("Q_check", f"t = {T_CHECK}"),
            ("Q_end", f"t = {T_END}"),
        ]):
            ax = axes[row, col]
            for name, level, n_layers in configs:
                key = (bc_type, order, name)
                r = results.get(key)
                if r is None:
                    continue
                nc = r["mesh"].n_inner_cells
                x = r["mesh"].cell_centers[0, :nc]
                ax.plot(x, r[t_key][1, :nc], "-", lw=1.2, label=name)
            ax.set_title(f"{bc_type} — {t_label}")
            ax.set_ylabel("h")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        axes[row, -1].set_xlabel("x")

    plt.tight_layout()
    plt.show()


plot_bc_comparison("first order", 1)

# %% [markdown]
# ### 5.2 O2 — all BCs

# %%
plot_bc_comparison("second order", 2)


# %% [markdown]
# ## 6. Zoom on Boundary Behaviour at $t_\text{end}$
#
# Close-up of the left and right boundary regions after the wave has interacted.

# %%
def plot_boundary_zoom(order):
    fig, axes = plt.subplots(len(bc_types), 2, figsize=(12, 4 * len(bc_types)),
                             sharey="row")
    fig.suptitle(f"Boundary zoom — O{order} at t = {T_END}", fontsize=14, y=1.02)

    for row, bc_type in enumerate(bc_types):
        for col, (x_range, side) in enumerate([
            ((0, L_DOMAIN * 0.15), "left"),
            ((L_DOMAIN * 0.85, L_DOMAIN), "right"),
        ]):
            ax = axes[row, col]
            for name, level, n_layers in configs:
                key = (bc_type, order, name)
                r = results.get(key)
                if r is None:
                    continue
                nc = r["mesh"].n_inner_cells
                x = r["mesh"].cell_centers[0, :nc]
                mask = (x >= x_range[0]) & (x <= x_range[1])
                ax.plot(x[mask], r["Q_end"][1, :nc][mask], "-o", ms=2,
                        lw=1, label=name)
            ax.set_title(f"{bc_type} — {side}")
            ax.set_xlabel("x")
            ax.set_ylabel("h")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


plot_boundary_zoom(1)
plot_boundary_zoom(2)

# %% [markdown]
# ## 7. Summary
#
# **Expected behaviour:**
#
# | BC type | Before wave hits boundary | After wave hits boundary |
# |---------|--------------------------|--------------------------|
# | Extrapolation | h=1, hu=0 at boundary (zero flux) | Wave exits domain; h returns toward 1 |
# | Wall | h=1, hu=0 at boundary | Wave reflects; amplitude doubles momentarily at wall |
# | Periodic | h=1, hu=0 at boundary | Wave wraps around and re-enters from opposite side |
#
# **What to look for in the plots:**
# - Any oscillation at the boundary BEFORE the wave arrives → solver/BC bug
# - Non-physical behaviour at the boundary AFTER the wave arrives → BC implementation issue
# - Differences between SME levels (L0 vs L1 vs L2) for this shear-free test → should be identical
