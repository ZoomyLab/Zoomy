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
# # Inclined Plane Rundown: Spectral Accuracy Test
#
# Analytical solution: $u(\zeta) = \frac{g_x h^2}{\nu}\left(\zeta - \frac{\zeta^2}{2} + \frac{\lambda}{h}\right)$
#
# This is quadratic in $\zeta$, so level 2 should capture it exactly.
# Level 3 moment should be zero.

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

DATA_DIR = "outputs/inclined_plane"

# %%
if not os.path.exists(os.path.join(DATA_DIR, "manifest.json")):
    raise FileNotFoundError("Run: python tests/scripts/zoomy_core/swe/run_inclined_plane.py")

results = {}
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json") and fname != "manifest.json":
        with open(os.path.join(DATA_DIR, fname)) as f:
            r = json.load(f)
            results[fname.replace(".json", "")] = r

leg = {r["level"]: r for k, r in results.items() if "Legendre" in k and "error" not in r}
cheb = {r["level"]: r for k, r in results.items() if "Chebyshev" in k and "error" not in r}

# %% [markdown]
# ## Summary table

# %%
print("| Basis | L | build | solve | steady_t | Linf error | alpha_3 |")
print("|-------|---|-------|-------|----------|-----------|---------|")
for store, bname in [(leg, "Legendre"), (cheb, "Chebyshev")]:
    for level in sorted(store.keys()):
        r = store[level]
        a3 = r["alphas_final"][3] if len(r["alphas_final"]) > 3 else "—"
        a3_str = f"{a3:.2e}" if isinstance(a3, float) else a3
        print(f"| {bname} | {level} | {r['build_time']:.1f}s | {r['solve_time']:.1f}s | "
              f"{r['steady_state_time']} | {r['profile_error_linf']:.2e} | {a3_str} |")

# %% [markdown]
# ## Velocity profiles: numerical vs analytical

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

for row, (store, bname) in enumerate([(leg, "Legendre"), (cheb, "Chebyshev")]):
    for col, level in enumerate([0, 1, 2, 3]):
        ax = axes[row, col]
        if level in store:
            r = store[level]
            ax.plot(r["profile_u_analytical"], r["profile_zeta"], "k-", linewidth=2, label="analytical")
            ax.plot(r["profile_u_numerical"], r["profile_zeta"], "r--", linewidth=1.5, label="numerical")
            ax.set_title(f"{bname} L{level}\nLinf={r['profile_error_linf']:.2e}")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("u(zeta)"); ax.set_ylabel("zeta")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle("Inclined plane: velocity profile convergence", fontsize=13)
plt.savefig(os.path.join(DATA_DIR, "profiles.png"), dpi=150)
print("Saved: profiles.png")

# %% [markdown]
# ## Time evolution of moments

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

for row, (store, bname) in enumerate([(leg, "Legendre"), (cheb, "Chebyshev")]):
    for col, level in enumerate([0, 1, 2, 3]):
        ax = axes[row, col]
        if level not in store:
            continue
        r = store[level]
        t = r["time_points"]
        hist = r["alpha_history"]
        n_mom = level + 1
        for k in range(n_mom):
            vals = [h[k] if k < len(h) else 0 for h in hist]
            ax.plot(t, vals, label=f"alpha_{k}")
        if r.get("steady_state_time"):
            ax.axvline(r["steady_state_time"], color="gray", linestyle=":", label=f"steady @ {r['steady_state_time']:.1f}s")
        ax.set_title(f"{bname} L{level}"); ax.set_xlabel("t"); ax.set_ylabel("alpha_k")
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

fig.suptitle("Moment evolution to steady state", fontsize=13)
plt.savefig(os.path.join(DATA_DIR, "evolution.png"), dpi=150)
print("Saved: evolution.png")

# %% [markdown]
# ## Profile error convergence

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
for store, bname, marker in [(leg, "Legendre", "o"), (cheb, "Chebyshev", "s")]:
    levels = sorted(store.keys())
    errors = [store[l]["profile_error_linf"] for l in levels]
    ax.semilogy(levels, errors, f"{marker}-", label=bname, linewidth=1.5)

ax.set_xlabel("Level"); ax.set_ylabel("Linf error vs analytical")
ax.set_title("Profile error convergence"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xticks(range(4))
plt.savefig(os.path.join(DATA_DIR, "error_convergence.png"), dpi=150)
print("Saved: error_convergence.png")
