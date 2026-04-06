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
# # Building an SME Model with Slip Closure: Step-by-Step
#
# This walkthrough shows the **complete three-phase derivation** of
# Shallow Moment Equations (SME) with Newtonian viscosity and Navier-slip closure,
# displaying the equations at every step so you can see what changes.
#
# | Phase | What happens | You see |
# |-------|-------------|---------|
# | **Phase 1** | Start from 3D INS, apply material + hydrostatic | Simplified momentum PDE |
# | **Phase 2** | Map to $\zeta$-space, Galerkin projection (abstract) | PDE with matrix symbols $M, A, D, \Phi$ |
# | **Phase 3** | Choose basis + level, compute matrices, apply $M^{-1}$ | Concrete flux, source, NC for FVM solver |

# %%
import sympy as sp
from sympy import Symbol, Derivative, Rational, S, sqrt, Matrix, latex, Function, Eq
sp.init_printing()

from IPython.display import display, Math, Latex

def show(*exprs, labels=None):
    """Display one or more sympy expressions with optional labels."""
    for i, e in enumerate(exprs):
        label = labels[i] if labels and i < len(labels) else None
        if hasattr(e, '_repr_latex_'):
            if label:
                display(Math(label + r" \quad " + e._repr_latex_().strip("$")))
            else:
                display(e)
        elif isinstance(e, sp.Basic):
            if label:
                display(Math(label + sp.latex(e)))
            else:
                display(Math(sp.latex(e)))
        else:
            print(e)

def show_eq(lhs_label, rhs, suffix=""):
    """Display: lhs_label = rhs  (rhs is a sympy expression)."""
    display(Math(lhs_label + " = " + sp.latex(rhs) + suffix))

# %% [markdown]
# ---
# ## Phase 1: From Full 3D INS to Basis-Independent Equations
#
# ### Step 1.1: Create the state space
#
# `StateSpace(dimension=2)` means the **xz plane** (1D horizontal shallow water).
# This creates all the shared symbols: coordinates, velocities, pressure, stress tensor.

# %%
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, IBPResult,
    materials, assumptions, Newtonian, Inviscid,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
)

state = StateSpace(dimension=2)
print(state)
print(f"  Stress components: {list(state.tau.keys())}  (no y-components in 2D)")

# %% [markdown]
# ### Step 1.2: Write down the full 3D INS
#
# `FullINS` builds the continuity and momentum equations.
# No assumptions have been made yet -- these are the **exact** incompressible Navier-Stokes.

# %%
ins = FullINS(state)

display(Math(r"\textbf{Continuity:} \quad " + sp.latex(ins.continuity.expr) + " = 0"))
display(Math(r"\textbf{x-momentum:} \quad " + sp.latex(ins.x_momentum.expr) + " = 0"))
display(Math(r"\textbf{z-momentum:} \quad " + sp.latex(ins.z_momentum.expr) + " = 0"))

# %% [markdown]
# The x-momentum contains:
# - **Inertia**: $\partial_t u + \partial_x(u^2) + \partial_z(uw)$
# - **Pressure**: $\frac{1}{\rho}\partial_x p$
# - **Stress divergence**: $-\frac{1}{\rho}(\partial_x \tau_{xx} + \partial_z \tau_{xz})$
#
# The stress symbols ($\tau_{xx}, \tau_{xz}$, ...) are **abstract** -- no constitutive law yet.

# %% [markdown]
# ### Step 1.3: Apply Newtonian material model
#
# The **Newtonian** material substitutes $\tau_{ij} = \mu(\partial_i u_j + \partial_j u_i)$
# with $\mu = \rho\nu$.

# %%
newton = Newtonian(state)

print(f"Newtonian: {len(newton)} substitution rules")
display(newton)

# %%
xm_after_material = ins.x_momentum.apply(newton)

print("x-momentum AFTER Newtonian material:")
display(Math(sp.latex(xm_after_material.expr) + " = 0"))
print(f"  tau symbols gone? {not any(xm_after_material.has(state.tau[k]) for k in state.tau)}")
print(f"  nu introduced?    {xm_after_material.has(newton.nu)}")

# %% [markdown]
# ### Step 1.4: Apply hydrostatic pressure
#
# Replace $p$ with $p_{\text{atm}} + \rho g (\eta - z)$.
# This eliminates the z-momentum equation.

# %%
hydro = HydrostaticPressure(state)

print("Hydrostatic pressure:")
display(hydro)

# %%
xm_after_hydro = xm_after_material.apply(hydro)

print("x-momentum AFTER hydrostatic:")
display(Math(sp.latex(xm_after_hydro.expr) + " = 0"))
print(f"  p removed?   {not xm_after_hydro.has(state.p)}")
print(f"  g present?   {xm_after_hydro.has(state.g)}")

# %% [markdown]
# ### Step 1.5: Inspect boundary conditions

# %%
kbc_bot = KinematicBCBottom(state)
kbc_surf = KinematicBCSurface(state)

print("Kinematic BC at bottom (z = b):")
display(kbc_bot)

print("Kinematic BC at surface (z = eta):")
display(kbc_surf)

# %% [markdown]
# ### Step 1.6: Run the full Phase 1 derivation
#
# `derive_shallow_moments()` automates: material, hydrostatic, depth integration
# with IBP, kinematic BCs, stress BCs, and term tagging.

# %%
from zoomy_core.model.models.model_derivation import (
    derive_shallow_moments, PreProjectedEquations, TaggedTerm,
)

pre = derive_shallow_moments(state, material=Newtonian(state))
print(pre.summary())

# %%
print("Continuity terms:")
for t in pre.continuity:
    print(f"  [{t.role}] {t.origin}")
    display(t.expr)

print("\nx-Momentum terms:")
for t in pre.x_momentum:
    print(f"  [{t.role}] {t.origin}")
    display(t.expr)

# %% [markdown]
# **Key**: the terms still contain $u(t,x,z)$, $w(t,x,z)$ -- no basis chosen yet.

# %% [markdown]
# ---
# ## Phase 2: Abstract $\zeta$-Space Projection
#
# `project_to_zeta()` performs the formal Galerkin projection without choosing a
# specific basis or level.  The result uses abstract matrix symbols:
#
# | Symbol | Definition | Appears in |
# |--------|-----------|------------|
# | $M_{lk}$ | $\int_0^1 \varphi_l \varphi_k\, d\zeta$ | Temporal terms |
# | $A_{lij}$ | $\int_0^1 \varphi_l \varphi_i \varphi_j\, d\zeta$ | Advective flux |
# | $D_{lk}$ | $\int_0^1 \varphi'_l \varphi'_k\, d\zeta$ | Viscous source |
# | $B_{lij}$ | $\int_0^1 \varphi'_l \psi_j \varphi_i\, d\zeta$ | Vertical advection |
# | $\Phi_l$ | $(M \cdot c)_l$ | Pressure flux, topography NC |
# | $\varphi^b_l$ | $\varphi_l(0)$ | Slip source |

# %%
from zoomy_core.model.models.zeta_projection import project_to_zeta

zeta = project_to_zeta(pre)
print(zeta.summary())

# %%
print("Each term and its LaTeX:")
for eq_name, terms in zeta.all_equations().items():
    print(f"\n=== {eq_name} ===")
    for t in terms:
        print(f"  [{t.role}] {t.origin}  deps={t.matrix_deps}")
        display(t)  # uses _repr_latex_

# %% [markdown]
# ### The abstract PDE system
#
# The full projected equation (before $M^{-1}$), with terms grouped by role:

# %%
# Display each momentum term as rendered math
from IPython.display import HTML

lines = []
lines.append(r"<h4>Continuity</h4>")
cont_tex = " + ".join(t.latex_str for t in zeta.continuity)
lines.append(f"$${cont_tex} = 0$$")

lines.append(r"<h4>x-momentum, mode $l$ (raw, before $M^{-1}$)</h4>")
by_role = {"temporal": [], "flux": [], "nonconservative": [], "source": []}
for t in zeta.x_momentum:
    by_role[t.role].append(t.latex_str)

role_labels = {"temporal": "temporal", "flux": "flux", "nonconservative": "NC", "source": "source"}
mom_parts = []
for role in ("temporal", "flux", "nonconservative", "source"):
    if by_role[role]:
        inner = " + ".join(by_role[role])
        mom_parts.append(r"\underbrace{" + inner + r"}_{\text{" + role_labels[role] + "}}")
mom_tex = " + ".join(mom_parts) + " = 0"
lines.append(f"$${mom_tex}$$")

lines.append(r"<h4>Abstract basis matrices</h4>")
lines.append(r"$$M_{lk} = \int_0^1 \varphi_l\,\varphi_k\, d\zeta$$")
lines.append(r"$$A_{lij} = \int_0^1 \varphi_l\,\varphi_i\,\varphi_j\, d\zeta$$")
lines.append(r"$$D_{lk} = \int_0^1 \varphi'_l\,\varphi'_k\, d\zeta$$")
lines.append(r"$$B_{lij} = \int_0^1 \varphi'_l\,\psi_j\,\varphi_i\, d\zeta \quad (\psi_j = \int_0^\zeta \varphi_j\,d\zeta')$$")
lines.append(r"$$\Phi_l = (M \cdot c)_l = \int_0^1 \varphi_l\, d\zeta$$")
lines.append(r"$$\varphi^b_l = \varphi_l(0)$$")

display(HTML("\n".join(lines)))

# %% [markdown]
# ---
# ## Phase 3: Choose a Basis, Get Concrete Equations
#
# Now we pick a **specific basis** and **level** (number of modes beyond the mean).
# The abstract matrices become real numbers, and we get a runnable `Model`.
#
# Basis matrices are **cached** by `(basis_name, level)` -- switching between
# Newtonian and inviscid models at the same level reuses the same matrices.

# %%
from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
from zoomy_core.model.models.basisfunctions import (
    Legendre_shifted, SplineBasis, Chebyshevu_shifted, GalerkinBasis,
)

# %% [markdown]
# ### 3a. Legendre basis, level 2
#
# Legendre on $[0,1]$: $\varphi_k(\zeta) = P_k(2\zeta - 1)(-1)^k$
#
# - **Orthogonal**: $M$ is diagonal
# - $\varphi_0 = 1$: mean velocity mode
# - 3 modes total at level 2: mean + 2 shape corrections

# %%
clear_matrix_cache()
leg = ProjectedModel(zeta, basis_type=Legendre_shifted, level=2, eigenvalue_mode="numerical")

print(f"Model: {leg.n_variables} variables, {leg.dimension}D")
print(f"Variables: {list(leg.variables.keys())}")
print(f"  q2 = h*alpha_0 (mean velocity moment)")
print(f"  q3 = h*alpha_1 (1st correction)")
print(f"  q4 = h*alpha_2 (2nd correction)")
print(f"Mean coefficients: c = {leg.mean_coefficients()}")

# %%
print("Mass matrix M:")
display(leg.mass_matrix())

print("M^{-1}:")
display(leg.mass_matrix_inverse())

# %% [markdown]
# Since $M$ is diagonal for Legendre, $M^{-1}$ is trivially $\text{diag}(1/M_{kk})$.

# %% [markdown]
# ### 3b. Concrete flux (advection + mass)

# %%
F = leg.flux()
print("Advective flux F(Q) -- x-direction:")
for i in range(leg.n_variables):
    expr = sp.simplify(F[i, 0])
    if expr != 0:
        show_eq(f"F_{i}", expr)

# %% [markdown]
# ### 3c. Hydrostatic pressure (separated for well-balanced schemes)

# %%
P = leg.hydrostatic_pressure()
print("Hydrostatic pressure flux:")
for i in range(leg.n_variables):
    val = P[i, 0]
    if val != 0:
        show_eq(f"P_{i}", val)

# %% [markdown]
# ### 3d. Non-conservative matrix (topography + vertical coupling)

# %%
NC = leg.nonconservative_matrix()
print("Non-conservative matrix Bx (non-zero entries):")
for r in range(leg.n_variables):
    for c in range(leg.n_variables):
        val = NC[r, c, 0]
        if val != 0:
            show_eq(f"B_{{{r},{c}}}", sp.simplify(val))

# %% [markdown]
# ### 3e. Source terms: Newtonian viscosity + Navier-slip
#
# This is where the **slip closure** lives.  Two source contributions:
#
# 1. **Newtonian viscosity**: from IBP of $\partial^2 u / \partial \zeta^2$
#    $$S_l^{\text{visc}} = -\frac{\nu}{h} \sum_i \alpha_i D_{il}$$
#
# 2. **Navier-slip**: boundary term at $\zeta = 0$
#    $$S_l^{\text{slip}} = -\frac{1}{\lambda\rho} u_b \varphi_l(0), \quad
#    u_b = \sum_i \alpha_i \varphi_i(0)$$
#
# Both are resolved via $M^{-1}$: $S_k = \sum_l M^{-1}_{kl} S_l^{\text{raw}}$

# %%
visc = leg.newtonian()
slip = leg.slip()

print("Newtonian viscous source:")
for i in range(leg.n_variables):
    val = visc[i]
    if val != 0:
        show_eq(f"S_{{\\text{{visc}},{i}}}", sp.simplify(val))

# %%
print("Navier-slip source:")
for i in range(leg.n_variables):
    val = slip[i]
    if val != 0:
        show_eq(f"S_{{\\text{{slip}},{i}}}", sp.simplify(val))

# %% [markdown]
# **Key observations for Legendre L2:**
#
# - **Viscosity** is proportional to $\nu / h$ and couples modes via the $D$ matrix.
#
# - **Slip** is proportional to $1/(\lambda\rho)$ and couples through boundary
#   values $\varphi_k(0)$.  For Legendre, $\varphi_k(0) = (-1)^k = 1$ for all $k$
#   (shifted), so all modes contribute equally to the bottom velocity.
#
# - Together, viscosity + slip define the velocity profile shape.

# %% [markdown]
# ### 3f. Complete inclined plane source (gravity + viscosity + slip)

# %%
from zoomy_core.misc.misc import ZArray

p = leg.parameters
h_sym = leg.variables[1]
n_mom = leg.level + 1

phi_int = leg._phi_int
raw_grav = [p.g * p.ez * h_sym * phi_int[l] for l in range(n_mom)]
S_grav = ZArray.zeros(leg.n_variables)
for k in range(n_mom):
    S_grav[2 + k] = leg._apply_Minv(raw_grav, k)

print("Gravity source (projected):")
for i in range(leg.n_variables):
    val = S_grav[i]
    if val != 0:
        show_eq(f"S_{{\\text{{grav}},{i}}}", sp.simplify(val))

print("\nTotal source S = gravity + viscosity + slip:")
for i in range(leg.n_variables):
    total = sp.simplify(S_grav[i] + visc[i] + slip[i])
    if total != 0:
        show_eq(f"S_{i}", total)

# %% [markdown]
# At steady state, $S(Q) = 0$ for each mode.  This determines the equilibrium
# velocity profile:
#
# $$u(\zeta) = \frac{g_x h^2}{\nu}\left(\zeta - \frac{\zeta^2}{2} + \frac{\lambda}{h}\right)$$
#
# The Legendre basis at level 2 recovers this **exactly** (parabolic profile).

# %% [markdown]
# ---
# ## Comparison: How the basis changes the equations
#
# The **same** `ZetaProjectedEquations` produces different concrete equations
# depending on the basis.  Let's compare.

# %%
print("=" * 70)
print("BASIS COMPARISON: Mass matrix + Slip source at level 2")
print("=" * 70)

for name, cls in [("Legendre", Legendre_shifted), ("SplineBasis", SplineBasis),
                   ("Chebyshev U", Chebyshevu_shifted)]:
    m = ProjectedModel(zeta, basis_type=cls, level=2, eigenvalue_mode="numerical")

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  c_k = {m.mean_coefficients()}")
    print(f"  phi_k(0) = {[m._phib[k] for k in range(3)]}")

    print(f"\n  Mass matrix M:")
    display(m.mass_matrix())

    print(f"\n  M^{{-1}}:")
    display(m.mass_matrix_inverse())

    s = m.slip()
    print(f"\n  Slip source (with M^{{-1}} applied):")
    for i in range(m.n_variables):
        val = s[i]
        if val != 0:
            show_eq(f"  S_{{\\text{{slip}},{i}}}", sp.simplify(val))

# %% [markdown]
# **Legendre**: $\varphi_k(0) = 1$ for all $k$, so all modes see slip equally.
# $M^{-1}$ is diagonal, no redistribution.
#
# **SplineBasis**: only $\varphi_0$ peaks at the bottom ($\varphi_0(0) = 1$, others $= 0$),
# so slip directly excites only the bottom mode.  But $M^{-1}$ is a **full matrix**
# that redistributes the force across all modes.
#
# **Chebyshev U**: weight $\sqrt{\zeta(1-\zeta)}$ vanishes at boundaries, so boundary
# values in the weighted inner product are killed.

# %% [markdown]
# ---
# ## Summary: The 3-Phase Pipeline
#
# ```
# Phase 1: derive_shallow_moments(state, material=Newtonian(state))
#           |
#           |  INS + material + hydrostatic + BCs + tagging
#           v
#     PreProjectedEquations  (abstract u(t,x,z), no basis)
#           |
# Phase 2: project_to_zeta(pre)
#           |
#           |  Map to zeta in [0,1], formal Galerkin projection
#           |  Write abstract matrices M, A, D, B, Phi, phi_b
#           v
#     ZetaProjectedEquations  (inspectable LaTeX, matrix dependencies)
#           |
# Phase 3: ProjectedModel(zeta, basis_type=Legendre_shifted, level=2)
#           |
#           |  Compute matrices (CACHED by basis+level)
#           |  Apply M^{-1} to all terms
#           v
#     Model with flux(), newtonian(), slip(), ...
#           |
#           v
#     FVM Solver (explicit flux + implicit source via IMEX)
# ```
#
# The slip closure enters as a **source term** in Phase 1 (from IBP boundary at bottom),
# is carried through Phase 2 as an abstract term with `matrix_deps = ('phib',)`,
# and becomes concrete in Phase 3 when the basis boundary values $\varphi_k(0)$ are known.
