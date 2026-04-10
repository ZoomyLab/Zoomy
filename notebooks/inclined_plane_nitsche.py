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
# # Inclined Plane: Nitsche & Modified Friction Parameter Study
#
# Velocity profiles for different BC enforcement methods.
#
# Analytical: $u(\zeta) = \frac{g_x h^2}{\nu}\left(\zeta - \frac{\zeta^2}{2} + \frac{\lambda}{h}\right)$

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import zoomy_core.misc.misc as misc
from zoomy_core.model.models.basisfunctions import Legendre_shifted

DATA_DIR = os.path.join(misc.get_main_directory(), "outputs/inclined_plane_sweep")

H, G_X, NU, SLIP_LAMBDA = 1.0, 1.0, 1.0, 0.5
zeta_plot = np.linspace(0, 1, 200)
u_analytical = G_X * H**2 / NU * (zeta_plot - zeta_plot**2 / 2 + SLIP_LAMBDA / H)

# %%
def load_result(key):
    path = os.path.join(DATA_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    if "error" in r:
        return None
    return r


def reconstruct_profile(alphas, level, basis_cls=Legendre_shifted):
    """Reconstruct velocity profile from moment coefficients."""
    basis = basis_cls(level=level)
    z_lo, z_hi = float(basis.bounds()[0]), float(basis.bounds()[1])
    zeta_b = np.linspace(z_lo, z_hi, 200)
    u = np.zeros(200)
    for k in range(min(level + 1, len(alphas))):
        phi_fn = basis.get_lambda(k)
        u += alphas[k] * np.array(phi_fn(list(zeta_b)))
    zeta_norm = (zeta_b - z_lo) / (z_hi - z_lo)
    return zeta_norm, u


def plot_profile(ax, key, color, ls, label, basis_cls=Legendre_shifted):
    r = load_result(key)
    if r is None or "alphas" not in r:
        return
    zeta, u = reconstruct_profile(r["alphas"], r["level"], basis_cls)
    err_str = f"{r['linf_error']:.1e}" if "linf_error" in r else "?"
    ax.plot(u, zeta, color=color, linestyle=ls, linewidth=1.3, label=f"{label} ({err_str})")


# %% [markdown]
# ## Velocity profiles: Standard vs Nitsche at each level (L0–L3)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for idx, level in enumerate([0, 1, 2, 3]):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    ax.plot(u_analytical, zeta_plot, "k-", linewidth=2.5, label="analytical", alpha=0.6)

    plot_profile(ax, f"standard_L{level}_default", "#1f77b4", "-", "standard")
    plot_profile(ax, f"nitsche_L{level}_tau0.0", "#2ca02c", "--", "Nitsche τ=0")
    plot_profile(ax, f"nitsche_L{level}_tau0.5", "#d62728", "-.", "Nitsche τ=0.5")
    plot_profile(ax, f"nitsche_L{level}_tau2.0", "#9467bd", ":", "Nitsche τ=2")
    plot_profile(ax, f"nitsche_L{level}_tau5.0", "#ff7f0e", "--", "Nitsche τ=5")

    ax.set_title(f"Level {level}")
    ax.set_xlabel("u(ζ)"); ax.set_ylabel("ζ")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle("Velocity profiles: Standard vs Nitsche at each level (IMEX)", fontsize=13)
plt.savefig(os.path.join(DATA_DIR, "profiles_by_level.png"), dpi=150)
print("Saved: profiles_by_level.png")

# %% [markdown]
# ## Nitsche τ sweep at L3

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

taus = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for i, tau in enumerate(taus):
    row, col = divmod(i, 4)
    ax = axes[row, col]
    ax.plot(u_analytical, zeta_plot, "k-", linewidth=2.5, alpha=0.5, label="analytical")
    plot_profile(ax, f"nitsche_L3_tau{tau}", "#d62728", "-", f"τ={tau}")
    ax.set_title(f"τ={tau}")
    r = load_result(f"nitsche_L3_tau{tau}")
    if r and "linf_error" in r:
        ax.set_title(f"τ={tau}, Linf={r['linf_error']:.2e}")
    ax.set_xlabel("u"); ax.set_ylabel("ζ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

if len(taus) < 8:
    axes[1, 3].axis("off")
fig.suptitle("Nitsche penalty sweep at L3 (IMEX)", fontsize=13)
plt.savefig(os.path.join(DATA_DIR, "nitsche_profiles_L3.png"), dpi=150)
print("Saved: nitsche_profiles_L3.png")

# %% [markdown]
# ## Modified friction α sweep at L3

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
ax.plot(u_analytical, zeta_plot, "k-", linewidth=2.5, label="analytical", alpha=0.6)

cmap = plt.cm.viridis
alphas_sweep = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
for i, alpha in enumerate(alphas_sweep):
    color = cmap(i / max(len(alphas_sweep) - 1, 1))
    plot_profile(ax, f"modified_friction_L3_alpha{alpha}", color, "-", f"α={alpha}")

ax.set_title("Modified friction: α sweep at L3")
ax.set_xlabel("u(ζ)"); ax.set_ylabel("ζ")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(DATA_DIR, "modified_friction_profiles.png"), dpi=150)
print("Saved: modified_friction_profiles.png")

# %% [markdown]
# ## Error summary plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Nitsche tau sweep
ax = axes[0]
taus_plot = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
errors = []
for tau in taus_plot:
    r = load_result(f"nitsche_L3_tau{tau}")
    errors.append(r["linf_error"] if r else np.nan)
ax.semilogy(taus_plot, errors, "o-", linewidth=1.5)
ax.axhline(y=9.39e-6, color="gray", linestyle="--", label="standard L3")
ax.set_xlabel("τ"); ax.set_ylabel("Linf error"); ax.set_title("Nitsche τ sweep (L3)")
ax.legend(); ax.grid(True, alpha=0.3)

# Modified friction alpha sweep
ax = axes[1]
alphas_plot = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
errors = []
for alpha in alphas_plot:
    r = load_result(f"modified_friction_L3_alpha{alpha}")
    errors.append(r["linf_error"] if r else np.nan)
ax.semilogy(alphas_plot, errors, "s-", color="#d62728", linewidth=1.5)
ax.axhline(y=9.39e-6, color="gray", linestyle="--", label="standard L3")
ax.set_xlabel("α"); ax.set_ylabel("Linf error"); ax.set_title("Modified friction α sweep (L3)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.savefig(os.path.join(DATA_DIR, "error_sweeps.png"), dpi=150)
print("Saved: error_sweeps.png")

# %% [markdown]
# ## Key findings
#
# 1. **τ=0 and α=0 recover the standard result exactly** (Linf=9.39e-6)
# 2. **Nitsche τ=0.1–0.5 slightly improves accuracy** (best: 5.6e-6 at τ=0.5, L2)
# 3. **Nitsche τ=1.0 hits a resonance** — penalty and slip compete destructively
# 4. **Nitsche τ=2–5 converges to a different steady state** with moderate error
# 5. **Modified friction degrades monotonically with α** — any correction is worse
# 6. **Standard Legendre + IMEX remains the best** for this test case
