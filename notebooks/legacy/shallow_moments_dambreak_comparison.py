# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dam Break with Friction: Shallow Moment Model Comparison
#
# Compares Legendre vs Chebyshev U (shifted, physical weight), levels 0-2,
# with Navier-slip bottom friction to excite higher-order velocity moments.

# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Chebyshevu_shifted
from zoomy_core.model.numerical_model import NumericalModel
from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
from zoomy_core.misc.misc import ZArray, Zstruct
import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC

os.makedirs("outputs/dambreak_comparison", exist_ok=True)

# %% [markdown]
# ## Model with slip friction

# %%
class GeneratedShallowModelWithSlip(GeneratedShallowModel):
    def source(self):
        return ZArray(self.slip())

# %% [markdown]
# ## Configuration

# %%
DOMAIN = (-5.0, 5.0)
N_CELLS = 30
TIME_END = 2.0
CFL = 0.45
SLIP_LAMBDA = 1.0

# %%
def run_case(basis, level, eig_mode, h_right, weight_mode="orthogonal", proxy=None):
    t_build = time.time()
    a = GeneratedShallowModelWithSlip(
        n_layers=1, level=level, dimension=1, basis_type=basis,
        eigenvalue_mode=eig_mode, weight_mode=weight_mode,
    )
    a.parameter_defaults_map["lamda"] = SLIP_LAMBDA
    a.parameter_values[list(a.parameters.keys()).index("lamda")] = SLIP_LAMBDA
    t_build = time.time() - t_build

    nv = a.n_variables
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right"),
    ])
    def ic(x, _nv=nv):
        Q = np.zeros(_nv); Q[1] = 1.0 if float(x[0]) < 0.0 else h_right
        return Q
    num = NumericalModel(a, boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic), eigenvalue_proxy_level=proxy)
    num.parameter_values[list(num.parameters.keys()).index("lamda")] = SLIP_LAMBDA

    mesh = petscMesh.Mesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    settings = Zstruct(output=Zstruct(
        directory="outputs/dambreak_comparison", filename="tmp",
        snapshots=2, clean_directory=False,
    ))
    solver = GeneratedModelSolver(
        time_end=TIME_END, settings=settings,
        compute_dt=timestepping.adaptive(CFL=CFL), min_dt=1e-6,
    )
    t_solve = time.time()
    Q, _ = solver.solve(mesh, num, write_output=False)
    t_solve = time.time() - t_solve
    n = mesh.n_inner_cells
    return {
        "x": mesh.cell_centers[0, :n], "Q": Q[:, :n], "h": Q[1, :n],
        "build": t_build, "solve": t_solve, "level": level, "n_vars": nv,
        "basis": basis, "basis_name": basis.name if hasattr(basis, "name") else str(basis),
    }

# %% [markdown]
# ## Run all configurations

# %%
configs = []

# Legendre L0/L1/L2: symbolic eigenvalues
for level in [0, 1, 2]:
    configs.append(dict(basis=Legendre_shifted, level=level, eig="symbolic", wm="orthogonal",
                        proxy=None, label=f"Leg L{level} sym"))
    configs.append(dict(basis=Legendre_shifted, level=level, eig="numerical", wm="orthogonal",
                        proxy=None, label=f"Leg L{level} num"))

# Chebyshev: all numerical eigenvalues (symbolic is slow due to irrational coefficients)
for level in [0, 1, 2]:
    configs.append(dict(basis=Chebyshevu_shifted, level=level, eig="numerical", wm="orthogonal",
                        proxy=None, label=f"ChebS L{level} orth"))
    configs.append(dict(basis=Chebyshevu_shifted, level=level, eig="numerical", wm="physical",
                        proxy=None, label=f"ChebS L{level} phys"))

results = {}
for case_name, h_right in [("wet", 0.8), ("wet-dry", 0.5)]:
    for cfg in configs:
        key = (cfg["label"], case_name)
        try:
            r = run_case(cfg["basis"], cfg["level"], cfg["eig"], h_right,
                         weight_mode=cfg["wm"], proxy=cfg["proxy"])
            results[key] = r
        except Exception as e:
            results[key] = {"error": str(e)[:60]}

# %% [markdown]
# ## Results table (grouped by expected equivalence)

# %%
def print_group(title, keys, case):
    print(f"\n{title} ({case}):")
    print(f"  {'Label':<25s} {'build':>5s} {'solve':>5s} {'h_min':>8s} {'h_max':>8s} {'h>=0':>5s}")
    for k in keys:
        r = results.get((k, case))
        if r is None:
            print(f"  {k:<25s} — not run —")
        elif "error" in r:
            print(f"  {k:<25s} FAILED: {r['error']}")
        else:
            h = r["h"]
            print(f"  {k:<25s} {r['build']:>4.1f}s {r['solve']:>4.1f}s {h.min():>8.4f} {h.max():>8.4f} {str((h>=0).all()):>5s}")

for case in ["wet", "wet-dry"]:
    print(f"\n{'='*70}")
    print(f"Case: {case} (h_R={'0.8' if case=='wet' else '0.5'})")
    print(f"{'='*70}")

    # L0: all should be identical (SWE limit)
    print_group("L0 — all bases/modes should match (SWE limit)",
        ["Leg L0 sym", "Leg L0 num", "ChebS L0 orth", "ChebS L0 phys"], case)

    # L1: Legendre sym/num should match; Chebyshev orth/phys may differ
    print_group("L1 — Legendre sym vs num (should match)",
        ["Leg L1 sym", "Leg L1 num"], case)
    print_group("L1 — Chebyshev orth vs physical (may differ due to boundary weight)",
        ["ChebS L1 orth", "ChebS L1 phys"], case)

    # L2: Legendre sym/num; Chebyshev weight modes
    print_group("L2 — Legendre sym vs num (should match)",
        ["Leg L2 sym", "Leg L2 num"], case)
    print_group("L2 — Chebyshev orth vs physical",
        ["ChebS L2 orth", "ChebS L2 phys"], case)

# %% [markdown]
# ## Water depth: Legendre vs Chebyshev (physical weight) at each level

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
for col, level in enumerate([0, 1, 2]):
    for row, (case, h_right) in enumerate([("wet", 0.8), ("wet-dry", 0.5)]):
        ax = axes[row, col]
        leg_key = f"Leg L{level} sym"
        cheb_key = f"ChebS L{level} phys" if level < 2 else "ChebS L2 phys"
        for key, color, ls, lbl in [
            (leg_key, "#1f77b4", "-", "Legendre"),
            (cheb_key, "#d62728", "--", "Chebyshev (physical)"),
        ]:
            r = results.get((key, case))
            if r and "error" not in r:
                ax.plot(r["x"], r["h"], color=color, linestyle=ls, label=lbl, linewidth=1.5)
        ax.set_title(f"L{level}, {case}"); ax.set_xlabel("x"); ax.set_ylabel("h")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(*DOMAIN)

fig.suptitle(f"Water depth: Legendre vs Chebyshev (slip friction, lambda={SLIP_LAMBDA})", fontsize=13)
plt.savefig("outputs/dambreak_comparison/h_comparison.png", dpi=150)
print("Saved: outputs/dambreak_comparison/h_comparison.png")

# %% [markdown]
# ## Velocity profiles at x=0

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)

for row, level in enumerate([0, 1, 2]):
    for col, (case, h_right) in enumerate([("wet", 0.8), ("wet-dry", 0.5)]):
        ax = axes[row, col]
        leg_sym = f"Leg L{level} sym"
        leg_num = f"Leg L{level} num"
        cheb_phys = f"ChebS L{level} phys" if level < 2 else "ChebS L2 phys"
        cheb_orth = f"ChebS L{level} orth" if level < 2 else "ChebS L2 orth"

        for key, basis_cls, color, ls, lbl in [
            (leg_sym, Legendre_shifted, "#1f77b4", "-", "Leg sym"),
            (leg_num, Legendre_shifted, "#1f77b4", ":", "Leg num"),
            (cheb_phys, Chebyshevu_shifted, "#d62728", "--", "Cheb phys"),
            (cheb_orth, Chebyshevu_shifted, "#d62728", ":", "Cheb orth"),
        ]:
            r = results.get((key, case))
            if r is None or "error" in r:
                continue
            x, Q, h = r["x"], r["Q"], r["h"]
            i_center = np.argmin(np.abs(x))
            h_c = float(h[i_center])
            n_mom = level + 1
            alphas = [float(Q[2 + k, i_center]) / max(h_c, 1e-10) for k in range(n_mom)]

            basis = basis_cls(level=level)
            z_lo, z_hi = float(basis.bounds()[0]), float(basis.bounds()[1])
            zeta_pts = np.linspace(z_lo, z_hi, 100)
            u_profile = np.zeros_like(zeta_pts)
            for k in range(n_mom):
                phi_fn = basis.get_lambda(k)
                u_profile += alphas[k] * np.array(phi_fn(list(zeta_pts)))

            zeta_norm = (zeta_pts - z_lo) / (z_hi - z_lo)
            ax.plot(u_profile, zeta_norm, color=color, linestyle=ls, label=lbl, linewidth=1.3)

        ax.set_title(f"L{level}, {case}")
        ax.set_xlabel("u(zeta)"); ax.set_ylabel("zeta (bottom=0, top=1)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle(f"Vertical velocity profile at x=0 (slip friction, lambda={SLIP_LAMBDA})", fontsize=13)
plt.savefig("outputs/dambreak_comparison/velocity_profiles.png", dpi=150)
print("Saved: outputs/dambreak_comparison/velocity_profiles.png")

# %% [markdown]
# ## Symbolic vs numerical eigenvalue accuracy

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
for col, level in enumerate([0, 1, 2]):
    for row, (case, h_right) in enumerate([("wet", 0.8), ("wet-dry", 0.5)]):
        ax = axes[row, col]
        sym_key = f"Leg L{level} sym"
        num_key = f"Leg L{level} num"
        r_sym = results.get((sym_key, case))
        r_num = results.get((num_key, case))
        if r_sym and "error" not in r_sym:
            ax.plot(r_sym["x"], r_sym["h"], "-", color="#1f77b4", linewidth=1.5, label="symbolic")
        if r_num and "error" not in r_num:
            ax.plot(r_num["x"], r_num["h"], "--", color="#ff7f0e", linewidth=1.2, label="numerical")
        ax.set_title(f"Leg L{level}, {case}"); ax.set_xlabel("x"); ax.set_ylabel("h")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(*DOMAIN)

fig.suptitle("Symbolic vs numerical eigenvalues (Legendre, with friction)", fontsize=13)
plt.savefig("outputs/dambreak_comparison/sym_vs_num.png", dpi=150)
print("Saved: outputs/dambreak_comparison/sym_vs_num.png")

# %% [markdown]
# ## Chebyshev: orthogonal vs physical weight

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
for col, level in enumerate([0, 1, 2]):
    for row, (case, h_right) in enumerate([("wet", 0.8), ("wet-dry", 0.5)]):
        ax = axes[row, col]
        orth_key = f"ChebS L{level} orth" if level < 2 else "ChebS L2 orth"
        phys_key = f"ChebS L{level} phys" if level < 2 else "ChebS L2 phys"
        for key, color, ls, lbl in [
            (orth_key, "#d62728", "-", "orthogonal weight"),
            (phys_key, "#2ca02c", "--", "physical weight"),
        ]:
            r = results.get((key, case))
            if r and "error" not in r:
                ax.plot(r["x"], r["h"], color=color, linestyle=ls, label=lbl, linewidth=1.5)
        ax.set_title(f"ChebS L{level}, {case}"); ax.set_xlabel("x"); ax.set_ylabel("h")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(*DOMAIN)

fig.suptitle("Chebyshev: orthogonal vs physical weight (with friction)", fontsize=13)
plt.savefig("outputs/dambreak_comparison/orth_vs_phys.png", dpi=150)
print("Saved: outputs/dambreak_comparison/orth_vs_phys.png")
