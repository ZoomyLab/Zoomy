# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Multi-Layer SWE and ML-SME (level 1) — Symbolic Derivation
#
# This notebook walks through the symbolic construction of a **piecewise
# polynomial basis** on $\zeta \in [0, 1]$ and how it is used to derive the
# Multi-Layer Shallow Water Equations (ML-SWE, $n_\text{order}=0$) and the
# Multi-Layer Shallow Moment Equations (ML-SME, $n_\text{order}=1$).
#
# The existing `SMEInviscid(level=L, n_layers=N)` Model is also shown for
# comparison — its `.describe()` renders the full derivation graph and the
# resulting symbolic equations.
#
# **Contents**
#
# 1. A piecewise Legendre basis, built with SymPy directly
# 2. Per-layer orthogonality — the correct rescaling
# 3. Derivatives of the basis — where Dirac deltas appear
# 4. The 4-rule classifier for distributional products
# 5. ML-SWE ($L=0$, $N=2$) derivation via existing `SMEInviscid`
# 6. ML-SME ($L=1$, $N=2$) derivation via existing `SMEInviscid`
# 7. Notes on what the current code does vs what a true piecewise basis would do

# %% [markdown]
# ## 1. Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Symbol, Heaviside, DiracDelta, Piecewise, Rational, legendre
from IPython.display import display, Markdown

from zoomy_core.model.models.sme_model import SMEInviscid
from zoomy_core.model.initial_conditions import UserFunction
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts

# %% [markdown]
# ## 2. Piecewise Legendre Basis — Symbolic Construction
#
# For $N$ layers on $\zeta \in [0, 1]$, let the layer interfaces be
# $\zeta_k = k/N$ for $k = 0, \ldots, N$. In each layer, we use the shifted
# Legendre polynomials $P_j^\text{shift}$ (orthogonal on $[0,1]$), but with a
# **local coordinate** that rescales each layer to $[0, 1]$:
#
# $$s_k(\zeta) = N \cdot (\zeta - \zeta_{k-1}), \quad s_k \in [0, 1] \text{ for } \zeta \in [\zeta_{k-1}, \zeta_k].$$
#
# The flat basis is indexed by a pair $(k, j)$:
#
# $$\phi_{k,j}(\zeta) = P_j^\text{shift}(s_k(\zeta)) \cdot W_k(\zeta), \quad W_k(\zeta) = H(\zeta - \zeta_{k-1}) - H(\zeta - \zeta_k),$$
#
# where $W_k$ is the indicator of layer $k$. Each $\phi_{k,j}$ is zero outside
# layer $k$.

# %%
zeta = Symbol("zeta", real=True)


def shifted_legendre(j, s):
    """Shifted Legendre P_j on [0,1] (standard sign convention in Zoomy)."""
    return legendre(j, 2 * s - 1) * (-1) ** j


def layer_basis(k, j, N):
    """Piecewise Legendre basis function: layer k, polynomial index j, N total layers."""
    zk_m = Rational(k, N)       # lower interface
    zk_p = Rational(k + 1, N)   # upper interface
    s = N * (zeta - zk_m)        # local coordinate in [0, 1]
    P = shifted_legendre(j, s)
    W = Heaviside(zeta - zk_m) - Heaviside(zeta - zk_p)
    return sp.expand(P * W)


# %% [markdown]
# ### Show a level-1 piecewise basis with 2 layers

# %%
N_LAYERS = 2
N_ORDER = 1

for k in range(N_LAYERS):
    for j in range(N_ORDER + 1):
        phi = layer_basis(k, j, N_LAYERS)
        display(Markdown(f"**$\\phi_{{{k},{j}}}(\\zeta) = $**"))
        display(phi)

# %% [markdown]
# ### Plot the basis functions

# %%
fig, ax = plt.subplots(figsize=(9, 5))
z_vals = np.linspace(0, 1, 400)
for k in range(N_LAYERS):
    for j in range(N_ORDER + 1):
        phi = layer_basis(k, j, N_LAYERS)
        f = sp.lambdify(zeta, phi, "numpy")
        try:
            y_vals = np.array([float(f(z)) for z in z_vals])
            ax.plot(z_vals, y_vals, lw=2, label=rf"$\phi_{{{k},{j}}}$")
        except Exception as e:
            print(f"Warning: could not plot phi_{{{k},{j}}}: {e}")
for k in range(1, N_LAYERS):
    ax.axvline(k / N_LAYERS, color="k", ls=":", alpha=0.4)
ax.set_xlabel(r"$\zeta$")
ax.set_ylabel(r"$\phi_{k,j}(\zeta)$")
ax.set_title(f"Piecewise Legendre basis — $N={N_LAYERS}$ layers, order $\\leq {N_ORDER}$")
ax.legend(ncol=N_LAYERS)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Per-Layer Orthogonality
#
# Because each layer uses its own local coordinate $s_k = N(\zeta - \zeta_{k-1})$,
# shifted Legendre is orthogonal on each layer:
#
# $$\int_{\zeta_{k-1}}^{\zeta_k} \phi_{k,i}(\zeta)\,\phi_{k,j}(\zeta)\,d\zeta
# = \frac{1}{N} \int_0^1 P_i(s) P_j(s) \, ds = \frac{1}{N} \cdot \frac{\delta_{ij}}{2i+1}.$$
#
# Cross-layer products are zero because the indicators $W_k$ have disjoint
# supports.

# %% [markdown]
# ### Numerical verification

# %%
print(f"Mass matrix M[(k,i),(l,j)] = ∫ φ_{{k,i}} φ_{{l,j}} dζ  (N={N_LAYERS}, L={N_ORDER})")
n_total = N_LAYERS * (N_ORDER + 1)
M = np.zeros((n_total, n_total))
for k in range(N_LAYERS):
    for i in range(N_ORDER + 1):
        row = k * (N_ORDER + 1) + i
        for lay in range(N_LAYERS):
            for j in range(N_ORDER + 1):
                col = lay * (N_ORDER + 1) + j
                integrand = layer_basis(k, i, N_LAYERS) * layer_basis(lay, j, N_LAYERS)
                # Integrate per-layer to avoid the Heaviside integration cost
                val = sp.Integer(0)
                for kk in range(N_LAYERS):
                    a, b = Rational(kk, N_LAYERS), Rational(kk + 1, N_LAYERS)
                    # within layer kk, indicators resolve to 1 iff kk == k and kk == lay
                    if kk == k == lay:
                        s = N_LAYERS * (zeta - a)
                        poly = shifted_legendre(i, s) * shifted_legendre(j, s)
                        val += sp.integrate(poly, (zeta, a, b))
                M[row, col] = float(val)

np.set_printoptions(precision=4, suppress=True)
print(M)
print()
print(f"Expected diagonal entries (k=l, i=j=0): 1/{N_LAYERS} = {1/N_LAYERS}")
print(f"Expected diagonal entries (k=l, i=j=1): 1/(3·{N_LAYERS}) = {1/(3*N_LAYERS):.4f}")
print("→ Block-diagonal with diagonal values 1/N · 1/(2i+1) as predicted.")

# %% [markdown]
# ## 4. Derivatives — where Dirac deltas appear
#
# $\partial_\zeta \phi_{k,j} = \partial_\zeta [P_j(s_k) \cdot W_k]
# = N \cdot P_j'(s_k) \cdot W_k \;+\; P_j(s_k) \cdot \partial_\zeta W_k.$
#
# The last term is the "interface" contribution:
# $\partial_\zeta W_k = \delta(\zeta - \zeta_{k-1}) - \delta(\zeta - \zeta_k)$.

# %%
phi_0_1 = layer_basis(0, 1, N_LAYERS)
dphi_0_1 = sp.diff(phi_0_1, zeta)
display(Markdown(r"**$\phi_{0,1}(\zeta) = $**"))
display(phi_0_1)
display(Markdown(r"**$\partial_\zeta \phi_{0,1}(\zeta) = $**"))
display(dphi_0_1)
print()
print("Note the DiracDelta terms at ζ=0 and ζ=1/N — these are interface jumps.")

# %% [markdown]
# ## 5. The 4-rule classifier for distributional products
#
# When projecting a symbolic equation onto this basis, integrals of products
# of derivatives appear. With piecewise bases, they split into three kinds:
#
# | Rule | Product | Handling |
# |:--:|---|---|
# | 1 | smooth × smooth | integrate per-layer |
# | 2 | smooth × $\delta(\zeta-\zeta_0)$ where smooth is continuous at $\zeta_0$ | sift: $f(\zeta_0)$ |
# | 3 | $\delta(\zeta-\zeta_0) \cdot \delta(\zeta-\zeta_0)$ | **ill-defined** — must tag and resolve |
# | 4 | $\delta(\zeta-\zeta_0) \cdot [\text{discontinuous at } \zeta_0]$ | ambiguous — must tag and resolve |
#
# **Rule 3** is the blocker for ML-SME with viscous terms — the matrix
# $D_{il} = \int \phi_i' \phi_l' \, d\zeta$ produces $\delta^2$ when both
# derivatives have Dirac components at the same interface.
#
# **Rule 4** arises when $u$ itself is discontinuous at an interface (piecewise
# velocity), and we multiply by $\partial_\zeta u$ which has a Dirac at the
# same interface. This happens in nonconservative convection terms.
#
# Below is a small classifier that implements the detection symbolically.

# %%
def classify_term(term, zeta):
    """Classify a product term of the integrand.

    Returns one of: 'smooth', 'sift', 'dirac_squared', 'discontinuous_at_dirac'.
    """
    diracs = list(term.atoms(sp.DiracDelta))
    if not diracs:
        return "smooth"

    from collections import Counter
    positions = Counter(sp.simplify(d.args[0]) for d in diracs)

    # Rule 3: δ² at same point
    if any(v >= 2 for v in positions.values()):
        return "dirac_squared"

    # Rule 4: Heaviside jump at same ζ as a Dirac
    other = term / sp.Mul(*diracs)
    heavisides = list(other.atoms(sp.Heaviside))
    for h in heavisides:
        h_arg = sp.simplify(h.args[0])
        for d_arg in positions:
            if sp.simplify(h_arg - d_arg) == 0:
                return "discontinuous_at_dirac"
            if sp.simplify(h_arg + d_arg) == 0:
                return "discontinuous_at_dirac"

    return "sift"


# %% [markdown]
# ### Examples of the classifier in action

# %%
# smooth × smooth
ex1 = sp.cos(zeta) * zeta**2
# smooth × Dirac
ex2 = sp.cos(zeta) * DiracDelta(zeta - Rational(1, 2))
# Dirac × Dirac (same point)
ex3 = DiracDelta(zeta - Rational(1, 2)) ** 2
# Dirac × Heaviside at same point
ex4 = DiracDelta(zeta - Rational(1, 2)) * Heaviside(zeta - Rational(1, 2))

for name, term in [("smooth·smooth", ex1), ("smooth·δ", ex2),
                   ("δ·δ same point", ex3), ("δ·H same point", ex4)]:
    print(f"{name:<20} → classify = {classify_term(term, zeta)}")

# %% [markdown]
# ## 6. ML-SWE — existing `SMEInviscid(level=0, n_layers=2)`
#
# The current implementation of `n_layers` in `DerivedModel` uses the same
# smooth Legendre basis on $[0,1]$ for every layer, with a uniform weight
# $w_k = 1/N$. This is NOT the same as the piecewise basis above — it's more
# like "$N$ copies of SWE sharing the same $h$ and pressure field".
#
# Shown here for completeness — the `.describe()` output lets you see the
# full derivation chain from INS → hydrostatic → depth integration → BCs.

# %%
model_ml_swe = SMEInviscid(level=0, n_layers=2)
print(f"n_variables = {model_ml_swe.n_variables}")
print(f"variables: {list(model_ml_swe.variables.keys())}")

# %%
display(model_ml_swe.describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ## 7. ML-SME level 1 — existing `SMEInviscid(level=1, n_layers=2)`
#
# At level 1 with 2 "layers" we have 2 moments per layer × 2 layers = 4
# velocity moments. Note that the equations are derived via Legendre
# projection — the symbolic output shows the final depth-integrated moment
# equations. A true piecewise basis would replace the Legendre polynomials
# within each layer with the rescaled local Legendre (as shown in Section 2).

# %%
model_ml_sme = SMEInviscid(level=1, n_layers=2)
print(f"n_variables = {model_ml_sme.n_variables}")
print(f"variables: {list(model_ml_sme.variables.keys())}")

# %%
display(model_ml_sme.describe(
    header=True, derivation="mermaid", assumptions=True,
    final_equation=True, strip_args=True,
))

# %% [markdown]
# ## 8. What the current code does vs what a true piecewise basis would do
#
# | Stage | Current `n_layers` in DerivedModel | True piecewise basis (Section 2) |
# |---|---|---|
# | Basis on $[0,1]$ | Single smooth Legendre (level $L$) | Per-layer shifted Legendre × indicator |
# | # moments total | $N \cdot (L+1)$ per horizontal dimension | Same: $N \cdot (L+1)$ |
# | Inter-layer coupling in advection | None — layers independent (product of weight $1/N$) | None (indicators disjoint) — same structure |
# | Viscous (newtonian) | Uses a single $D$ matrix, per-layer copies with weight $1/N$ — *not physically correct for true multi-layer* | $\partial_\zeta$ produces Dirac terms → $\delta^2$ when squared → needs SIPG/Nitsche or IBP |
# | Interface Riemann | Not applicable (no interfaces) | Would require per-interface jump handling in the numerics |
#
# So at the **inviscid level**, the current code and a true piecewise basis
# are structurally similar — both give $N$ (almost) independent layer systems
# coupled only through $h$ and pressure. The genuine difference matters for
# viscous coupling between layers.

# %% [markdown]
# ## 9. Phase 1 — proposed implementation
#
# To make a proper piecewise basis available in the active pipeline:
#
# 1. Extend `Basisfunction` with a `bounds=(a, b)` parameter and rescale
#    `eval` / `weight` accordingly (preserves orthogonality on subintervals).
# 2. Add a `PiecewiseBasis(base_cls, n_order, n_layers)` wrapper that holds
#    a list of per-layer base instances (each with its own `bounds`).
# 3. Extend `SymbolicIntegrator.integrate()` to loop over `basis.layer_bases`
#    when present, summing per-layer integrals.
# 4. Port the `classify_term` logic above into the integrator, so Dirac
#    products are either sifted (safe cases) or tagged as `InterfaceTerm`
#    objects carrying metadata (location, reason, residual factor).
# 5. Add `.apply(ResolveAsAverage())`, `.apply(ResolveBySIPG(sigma=...))`,
#    etc. operations that consume tagged `InterfaceTerm`s and produce
#    concrete symbolic expressions — the user picks the physics once per
#    model.
# 6. `DerivedModel` stops reading `n_layers` directly; it reads it from the
#    basis instead.
#
# After (1)–(4), ML-SWE and inviscid ML-SME derive cleanly. Viscous
# ML-SME requires (5) for the Dirac-squared terms. No numerics change.

# %% [markdown]
# ## 10. A quick sanity test: per-layer integration
#
# Verify that our classifier + per-layer integration give the expected
# result for a sift-safe Dirac product.

# %%
# ∫₀¹ cos(ζ) · δ(ζ - 1/2) dζ = cos(1/2)
f = sp.cos(zeta) * DiracDelta(zeta - Rational(1, 2))
tag = classify_term(f, zeta)
print(f"term: cos(ζ)·δ(ζ-1/2)  → classify = '{tag}'")
assert tag == "sift"

# Manual sifting: evaluate smooth part at the Dirac location
smooth = f / DiracDelta(zeta - Rational(1, 2))
sifted = smooth.subs(zeta, Rational(1, 2))
print(f"sifted value at ζ=1/2: {sifted} = {float(sifted):.4f}  (exact: cos(1/2) = {float(sp.cos(Rational(1,2))):.4f})")

# %%
# Tag a problematic case — δ · H at same point
g = DiracDelta(zeta - Rational(1, 2)) * Heaviside(zeta - Rational(1, 2))
tag = classify_term(g, zeta)
print(f"term: δ(ζ-1/2)·H(ζ-1/2) → classify = '{tag}'  (would need ResolveAsAverage or similar)")
