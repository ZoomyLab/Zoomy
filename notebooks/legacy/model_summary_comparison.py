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
# # Model Summary & Compilation Benchmark
#
# Compares all model configurations: Legendre vs Chebyshev U (shifted), levels 0-2.
# Shows full model descriptions via `summarize_model(tex=True)`.

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display, Math, Markdown
from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Chebyshevu_shifted

import os
os.makedirs("outputs/dambreak_comparison", exist_ok=True)

sp.init_printing()

# %% [markdown]
# ## Compilation Benchmark

# %%
rows = []
for basis_cls, bname in [(Legendre_shifted, "Legendre"), (Chebyshevu_shifted, "Chebyshev")]:
    for level in [0, 1, 2]:
        for eig_mode in ["symbolic", "numerical"]:
            if bname == "Chebyshev" and level == 2 and eig_mode == "symbolic":
                rows.append(f"| {bname} L{level} | symbolic | skipped (15 min basis matrices) | — | — |")
                continue
            t0 = time.time()
            m = GeneratedShallowModel(
                n_layers=1, level=level, dimension=1,
                basis_type=basis_cls, eigenvalue_mode=eig_mode,
            )
            t_model = time.time() - t0
            t0 = time.time()
            _ = m.summarize_model(tex=True)
            t_summary = time.time() - t0
            rows.append(f"| {bname} L{level} | {eig_mode} | {t_model:.2f}s | {t_summary:.3f}s | {m.n_variables} vars |")

print("| Config | Eigenvalues | Model build | Summary | Size |")
print("|--------|------------|-------------|---------|------|")
for r in rows:
    print(r)

# %% [markdown]
# ## Model Summaries

# %%
def display_summary(model, title=""):
    s = model.summarize_model(tex=True)
    display(Markdown(f"### {title}"))
    display(Math(s["pde_form"]))
    display(Markdown(f"$\\mathbf{{Q}} = {s['Q']}$"))
    display(Markdown(f"$\\mathbf{{F}} = {s['F']}$"))
    if s["P"]:
        display(Markdown(f"$\\mathbf{{P}} = {s['P']}$"))
    if s["B"]:
        for d, B_tex in s["B"].items():
            display(Markdown(f"$\\mathbf{{B}}_{d} = {B_tex}$"))
    if isinstance(s["eigenvalues"], list):
        display(Markdown("**Eigenvalues (symbolic):**"))
        for i, ev in enumerate(s["eigenvalues"]):
            display(Math(rf"\lambda_{i} = {ev}"))
    else:
        display(Markdown(f"**Eigenvalues:** {s['eigenvalues']}"))
    display(Markdown(f"**Parameters:** {s['parameters']}"))
    display(Markdown(f"**Config:** {s['config']}"))
    display(Markdown("---"))

# %% [markdown]
# ### Legendre basis (shifted, weight=1, domain [0,1])

# %%
for level in [0, 1, 2]:
    m = GeneratedShallowModel(
        n_layers=1, level=level, dimension=1,
        basis_type=Legendre_shifted,
    )
    display_summary(m, f"Legendre L{level} ({m.n_variables} variables)")

# %% [markdown]
# ### Chebyshev U basis (shifted, weight=$\sqrt{{z(1-z)}}$, domain [0,1])
#
# Using numerical eigenvalue mode for L2 (symbolic basis matrix computation
# takes 15 minutes; once cached, model builds in <1s).

# %%
for level in [0, 1, 2]:
    eig = "symbolic" if level <= 1 else "numerical"
    m = GeneratedShallowModel(
        n_layers=1, level=level, dimension=1,
        basis_type=Chebyshevu_shifted,
        eigenvalue_mode=eig,
    )
    display_summary(m, f"Chebyshev L{level} ({m.n_variables} vars, eig={eig})")

# %% [markdown]
# ### Basis function comparison
#
# Both bases satisfy $\phi_0 = 1$ (SWE limit correct).

# %%
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

for col, (basis_cls, bname) in enumerate([(Legendre_shifted, "Legendre"), (Chebyshevu_shifted, "Chebyshev")]):
    b = basis_cls(level=2)
    z_pts = np.linspace(float(b.bounds()[0]), float(b.bounds()[1]), 200)
    for k in range(3):
        phi_fn = b.get_lambda(k)
        y = phi_fn(list(z_pts))
        ax[col].plot(z_pts, y, label=f"$\\phi_{k}$", linewidth=1.5)
    ax[col].set_title(f"{bname} (level=2)")
    ax[col].set_xlabel("z"); ax[col].set_ylabel("$\\phi_k(z)$")
    ax[col].legend(); ax[col].grid(True, alpha=0.3)

plt.savefig("outputs/dambreak_comparison/basis_functions.png", dpi=150)
print("Saved: outputs/dambreak_comparison/basis_functions.png")

# %% [markdown]
# ## Architecture: opaque numerical functions
#
# | SymPy function | NumPy implementation | Purpose |
# |---------------|---------------------|---------|
# | `max_wavespeed(Q, Qaux, p, n)` | `np.linalg.eigvals` or compiled symbolic | Rusanov dissipation + CFL |
# | `clamp_positive(h)` | `np.maximum(h, 0)` | Prevent negative water depth |
# | `clamp_momentum(hu, h, u_max)` | `np.clip(hu, -h*u_max, h*u_max)` | Velocity cap at dry fronts |
# | `conditional(c, t, f)` | `np.where(c, t, f)` | Branching (existing) |
#
# These are **opaque to SymPy** — never simplified away, even with `positive=True` assumptions.
# Each backend (NumPy, JAX, C) provides its own implementation.
