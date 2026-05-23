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
# # 001 — Modeling and derivation
#
# A unified Galerkin-projection pipeline for shallow-water-family
# models: **SWE, SME, ML-SWE, VAM**, and (future) **ML-VAM**.
#
# Every model is a composition of three shared building blocks
# (`zoomy_core.derivation`):
#
# 1. **Flow setup**: σ-coordinate Navier–Stokes (with stresses retained).
#    Two specialisations:
#    * `HydrostaticFlow` — drops the z-momentum equation and fixes the
#      non-hydrostatic pressure remainder to zero.
#    * `NonHydrostaticFlow` — keeps z-momentum and the
#      non-hydrostatic pressure split.
# 2. **Polynomial ansatz** in σ:
#
#    $$
#    u(t,x,\xi) = \sum_{i=0}^{M} u_i(t,x)\, \varphi_i(\xi),\quad
#    w(t,x,\xi) = \sum_{i=0}^{N_w} w_i(t,x)\, \varphi_i(\xi),\quad
#    p(t,x,\xi) = \sum_{i=0}^{N_p} p_i(t,x)\, \varphi_i(\xi).
#    $$
#
#    The $\varphi_i$ are **shifted Legendre polynomials** on $[0,1]$ in
#    paper convention $\varphi_i(0)=1$, $\varphi_i(1)=(-1)^i$,
#    $\int_0^1 \varphi_i^2\,d\xi = 1/(2i+1)$.
# 3. **Galerkin projection** $\int_0^1 \varphi_j(\xi)\cdot(\text{eq})\,d\xi$
#    for $j=0,1,\dots$.  Two modes for the σ-vertical velocity $\omega$:
#    * `'state'` — $w$ is an independent state polynomial (VAM).
#    * `'from_continuity'` — $w$ is depth-integrated from continuity (SME).
#
# The σ-vertical velocity is
# $$
# \omega(t,x,\xi) = w - \partial_t(\xi h + b) - u\,\partial_x(\xi h + b)
# $$
# with the **kinematic boundary conditions** $\omega|_{\xi=0}=\omega|_{\xi=1}=0$
# (no flow through the bottom or surface).
#
# Specific models:
#
# | Model | Flow setup | Ansatz | Projection | Closures |
# |---|---|---|---|---|
# | SWE = SME L=0 | hydrostatic | $M=0$, no $w$, no $p$ | x-mom $j=0$ | — |
# | SME L=$L$ | hydrostatic | $M=L$, no $w$, no $p$ | x-mom $j=0,\dots,L$ | — |
# | VAM $(M,N)$ | non-hydro | $M$, $N_w=N_p=N$ | x-mom $j=0..M$, z-mom $j=0..N-1$, cont $j=1..N$ | KBC@$\xi=0$ for $w_N$; surface BC for $p_N$ |
# | ML-SWE | hydrostatic | layer-Heaviside basis | per-layer | KBC at every interface |

# %% [markdown]
# ## 1.1 Reproducing K&T 2019: SME at level $L$
#
# Let's build SME at level 1 with the derivation library and verify the
# x-momentum equation matches K&T 2019 eq (4.14) row 3:
#
# $$
# \partial_t(h\,s) + \partial_x(2 h u_m s) - u_m\,\partial_x(h s) = 0
# $$
# where $s = u_1$ and $u_m = u_0$.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(), "..", "tutorials", "sme"))
import sympy as sp
from sme_builder import build_sme_pde_system

system, h, u_coeffs = build_sme_pde_system(level=1)
print(f"State fields: {system.fields}")
print(f"Number of equations: {system.n_equations()}")
print()
print("x-momentum j=1 equation (= 0):")
sp.pprint(system.equations[2])

# %% [markdown]
# ### Literal comparison vs K&T 2019 eq (4.14) row 3

# %%
t = system.time
x = system.space[0]
g = sp.Symbol("g", positive=True)
u_0, u_1 = u_coeffs

ref = sp.Derivative(h*u_1, t) + sp.Derivative(2*h*u_0*u_1, x) - u_0*sp.Derivative(h*u_1, x)

# Apply ∂_t h substitution from cont j=0 to BOTH so we can compare.
dt_h_atom = sp.Derivative(h, t)
dt_h_rhs = -sp.Derivative(h*u_0, x)
def apply_dt_h(expr):
    prev = None
    cur = sp.expand(expr.doit())
    while prev != cur:
        prev = cur
        cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}).doit())
    return cur

ref_sub = apply_dt_h(ref)
my_sub = sp.expand(3 * system.equations[2])    # times mu_1=1/3 to compare on common ground
diff = sp.simplify(my_sub - ref_sub)
print(f"3·(my x-mom j=1) − K&T row 3 (after ∂_t h subst) = {diff}")
print("✓ MATCH" if diff == 0 else "✗ MISMATCH")

# %% [markdown]
# ## 1.2 Reproducing Escalante 2024: VAM $(M,N)$
#
# Same pipeline, but with `NonHydrostaticFlow` and a polynomial $w, p$.
# The closure structure is automatic from the projection counting:
#
# * **Evolution** ($M+N+2$ equations): cont $j=0$, x-mom $j=0..M$, z-mom $j=0..N-1$.
# * **Closures** ($N+2$ equations): cont $j=1..N$ (constraints), KBC at $\xi=0$
#   ($w_N$), surface BC at $\xi=1$ ($p_N$).
#
# Total: $M + 2N + 4$ conditions for the $M + 2N + 4$ unknowns.

# %%
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(), "..", "tutorials", "vam"))
from vam_builder import build_vam_pde_system

for M, N in [(1, 2), (2, 3)]:
    system, u_fields, w_fields, p_fields = build_vam_pde_system(M, N)
    print(f"VAM(M={M}, N={N}): {system.n_equations()} equations × {system.n_fields()} fields "
          f"(u: {len(u_fields)}, w: {len(w_fields)}, p: {len(p_fields)})")

# %% [markdown]
# ## 1.3 Multilayer SWE (Aguillon, Hörnschemeyer, Sainte-Marie 2026)
#
# Different basis: per-layer indicator functions $\mathbf{1}_{\alpha}(z)$
# instead of polynomials in $\xi$.  Same projection machinery, just a
# different $\varphi$.  See `tutorials/multilayer/aguillon2026_derivation.py`
# for the implementation; the equations match Aguillon et al. eq (5)
# literally for $N = 2, 3, 5, 7$ layers (each layer has continuity,
# x-momentum, and a passive tracer equation; mass-exchange terms
# $G_{\alpha+1/2}$ come from the kinematic BC at every interface).
