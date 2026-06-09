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
# # 002 — Software architecture
#
# Two new packages, one unified entry point for analysis (`PDESystem`).
#
# ## 2.1 `zoomy_core.model.derivation` — model build blocks
#
# | Module | Responsibility |
# |---|---|
# | `coords.py` | Default coordinate symbols `(t, x, ξ, g)` and canonical state functions `h(t,x), b(x)`. |
# | `basis.py` | `shifted_legendre_basis(n_max, ξ)`; `polynomial_integrate` (Poly-based, no `.doit()`). |
# | `flow.py` | `FlowSetup` → `HydrostaticFlow`, `NonHydrostaticFlow` (σ-coord NS, stresses retained). |
# | `ansatz.py` | `PolynomialAnsatz(M, N_w, N_p, basis)`. |
# | `projection.py` | `GalerkinProjection(flow, ansatz, w_mode)`: methods `project_continuity(j)`, `project_x_momentum(j)`, `project_z_momentum(j)`. |
# | `closures.py` | `kbc_bottom_solve_w_N`, `surface_bc_solve_p_N`. |
#
# Models compose these:
#
# ```python
# from zoomy_core.model.derivation import (
#     HydrostaticFlow, PolynomialAnsatz, GalerkinProjection,
# )
# flow   = HydrostaticFlow.with_defaults()
# ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
#                            M=L, N_w=-1, N_p=-1)
# proj   = GalerkinProjection(flow=flow, ansatz=ansatz, w_mode='from_continuity')
# eqs    = [proj.project_continuity(0)] + [proj.project_x_momentum(j) for j in range(L+1)]
# ```
#
# That's the entire SME builder (modulo the ∂_t h substitution post-step).
#
# ## 2.2 `zoomy_core.analysis` — unified analysis library
#
# Single representation: `PDESystem(equations, fields, time, space)`.
# Differential and algebraic equations mix freely.

# %%
from zoomy_core.analysis import PDESystem
help(PDESystem)

# %% [markdown]
# ### Available routines
#
# | Function | Purpose |
# |---|---|
# | `linearise(system, base_state)` | $q \to q_0 + \varepsilon\, \delta q$, return $O(\varepsilon)$ system. |
# | `plane_wave_dispersion(linsys)` | $\delta q \to \hat q\, e^{i(kx-\omega t)}$, solve $\det M(\omega, k) = 0$. |
# | `extract_quasilinear_pencil(linsys)` | Extract $(M_t, [M_{x,a}], M_0)$ matrices. |
# | `generalised_eigenvalues(M_x, M_t)` | Symbolic. |
# | `sample_generalised_eigenvalues(M_x, M_t, samples)` | Numeric (scipy). |
# | `symbolic_eigenvalues_at(system, base)` | One-shot symbolic eigenvalues. |
# | `is_hyperbolic_at(M_x, M_t, sample)` | Boolean + eigenvalues. |
# | `sample_hyperbolicity(M_x, M_t, ranges)` | Random scan over a parameter cube. |
# | `reduce_singular_pencil(M_x, M_t, fields, M_0)` | Eliminate algebraic-constraint rows. |
# | `plot_dispersion(result, k_range, ...)` | $\omega(k)$ or $C(k)$ curves. |
# | `plot_hyperbolic_region_2d(M_x, M_t, axis_a, axis_b, fixed_subs)` | K&T-style 2D scan. |

# %% [markdown]
# ### Example: shallow water dispersion in 5 lines

# %%
import sympy as sp
from zoomy_core.analysis import PDESystem, linearise, plane_wave_dispersion

t, x = sp.symbols("t x", real=True)
g = sp.Symbol("g", positive=True)
H = sp.Symbol("H", positive=True)
h = sp.Function("h")(t, x); u = sp.Function("u")(t, x)
sys_swe = PDESystem(
    equations=[sp.Derivative(h, t) + sp.Derivative(h*u, x),
               sp.Derivative(h*u, t) + sp.Derivative(h*u**2 + g*h**2/2, x)],
    fields=[h, u], time=t, space=[x],
)
disp = plane_wave_dispersion(linearise(sys_swe, {h: H, u: 0}), simplify=True)
disp["solutions"]

# %% [markdown]
# Returns $\omega = \pm \sqrt{gH}\, k$ — the shallow-water wave speed.
