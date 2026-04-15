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
# # Three-Phase PDE Model Derivation
#
# Derives shallow water moment equations from the **full 3D Incompressible Navier-Stokes**
# for **any vertical basis**: Legendre, Chebyshev, B-splines, Galerkin (Shen-type).
#
# | Phase | What | Output |
# |-------|------|--------|
# | **Phase 1** | INS + material + hydrostatic + BCs + term tagging | `PreProjectedEquations` |
# | **Phase 2** | Map to $\zeta$-space, formal Galerkin projection with abstract matrices | `ZetaProjectedEquations` |
# | **Phase 3** | Choose basis + level, compute matrices (cached), apply $M^{-1}$ | `ProjectedModel` (a `Model`) |
#
# This notebook walks through all three phases and compares results across bases.

# %%
import sympy as sp
from sympy import Symbol, Derivative, Rational, S, sqrt, Matrix, latex
sp.init_printing()

# %% [markdown]
# ---
# ## Part 1: The INS Generator (building blocks)
#
# Everything starts from the **full 3D INS**. The `StateSpace` holds shared symbols.
# `Expression` objects support `.terms`, `.apply()`, `.ibp()`, `.project()`.
# See the companion notebook `pde_generator_design.py` for details.

# %%
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, IBPResult,
    materials, assumptions, Newtonian, Inviscid,
)

state = StateSpace(dimension=2)
ins = FullINS(state)

print("Full 3D INS equations (1D horizontal):")
for eq in ins.equations:
    print(f"  {eq.name}: {len(eq)} terms")

# %% [markdown]
# ### Material models and assumptions
#
# Materials and assumptions are `Relation` objects (substitution rules).
# They take the `StateSpace`, NOT the equations -- they're independent.

# %%
newton = materials.newtonian(state)
inv = materials.inviscid(state)
hydro = assumptions.hydrostatic_pressure(state)

print("Newtonian stress (6 rules):")
display(newton)

print("\nHydrostatic pressure:")
display(hydro)

# %% [markdown]
# ---
# ## Part 2: Phase 1 -- Basis-Independent Derivation
#
# `derive_shallow_moments()` does the full derivation from INS:
#
# 1. Apply material model (Newtonian by default)
# 2. Apply hydrostatic assumption: simplify z-momentum -> p(z)
# 3. Substitute hydrostatic pressure into x-momentum
# 4. Depth-integrate with IBP on z-derivatives
# 5. Apply kinematic BCs to boundary terms
# 6. Apply stress BCs (Navier-slip at bottom, free at surface)
# 7. Tag each term: `temporal`, `flux`, `nonconservative`, `source`

# %%
from zoomy_core.model.models.model_derivation import (
    derive_shallow_moments, PreProjectedEquations, TaggedTerm,
)

pre = derive_shallow_moments(state, material=Newtonian(state))

print(pre.summary())

# %% [markdown]
# ### Inspecting tagged terms
#
# Each term carries a `role` (how it maps to the `Model` interface)
# and an `origin` (which physical mechanism produced it).

# %%
print("=== Continuity ===")
for t in pre.continuity:
    print(f"  {t.role:20s}  origin={t.origin:20s}  expr={t.expr.expr}")

print("\n=== X-Momentum ===")
for t in pre.x_momentum:
    print(f"  {t.role:20s}  origin={t.origin:20s}  expr={str(t.expr.expr)[:60]}")

# %% [markdown]
# The key insight: **these terms are still abstract** -- they contain `u(t,x,z)`, `w(t,x,z)`,
# not yet expanded into basis coefficients. This is what makes Pass 1 basis-independent.
#
# The tagged roles map directly to `Model` methods:
#
# | Role | Model method | Physical meaning |
# |------|-------------|-----------------|
# | `temporal` | mass matrix M | d(h alpha)/dt |
# | `flux` | `flux()` | d(F)/dx |
# | `nonconservative` | `nonconservative_matrix()` | B(Q) . dQ/dx |
# | `source` | `source()` | algebraic in Q |

# %% [markdown]
# ### Inviscid variant
#
# Switch the material model to see how the tagged terms change.

# %%
pre_inv = derive_shallow_moments(state, material=Inviscid(state))
print(pre_inv.summary())

print("\nCompare x-momentum terms:")
print(f"  Newtonian: {len(pre.x_momentum)} terms")
print(f"  Inviscid:  {len(pre_inv.x_momentum)} terms")

newton_origins = {t.origin for t in pre.x_momentum}
inv_origins = {t.origin for t in pre_inv.x_momentum}
print(f"  Missing in inviscid: {newton_origins - inv_origins}")

# %% [markdown]
# ---
# ## Part 2b: Phase 2 -- Abstract $\zeta$-Space Projection
#
# `project_to_zeta()` takes Phase 1 output and performs the **formal Galerkin projection**
# in normalized $\zeta$-space $[0, 1]$ -- **without choosing a basis or level**.
#
# The projection steps:
# 1. Coordinate transform: $z \to \zeta = (z - b) / H$
# 2. Ansatz substitution: $u(t,x,\zeta) = \sum_k \alpha_k(t,x)\,\varphi_k(\zeta)$
# 3. Galerkin projection: multiply by $\varphi_l(\zeta)$, integrate $\int_0^1 \ldots d\zeta$
# 4. Integration by parts on $\partial/\partial\zeta$ terms
# 5. Apply kinematic BCs at $\zeta = 0$ (bottom) and $\zeta = 1$ (surface)
# 6. Apply stress BCs: free surface at top, Navier-slip at bottom
#
# The result uses **abstract matrix symbols** -- no numerical values yet:
# - $M_{lk}$: mass matrix
# - $A_{lij}$: triple product (advection)
# - $D_{lk}$: stiffness/derivative (viscosity)
# - $B_{lij}$: vertical advection coupling
# - $\Phi_l$: constant projection integral
# - $\varphi_l^b$: boundary values

# %%
from zoomy_core.model.models.zeta_projection import project_to_zeta

zeta = project_to_zeta(pre)
print(zeta.summary())

# %% [markdown]
# ### Abstract equation structure
#
# Each term carries a `role`, `origin`, and `matrix_deps` (which basis matrices it needs).
# The actual matrix values are computed in Phase 3 with a specific basis.

# %%
print("=== Continuity (abstract) ===")
for t in zeta.continuity:
    print(f"  {t.role:20s}  type={t.term_type.value:25s}  deps={t.matrix_deps}")

print("\n=== x-Momentum (abstract, mode l) ===")
for t in zeta.x_momentum:
    print(f"  {t.role:20s}  type={t.term_type.value:25s}  deps={t.matrix_deps}")

# %% [markdown]
# ### LaTeX PDE system (abstract -- before $M^{-1}$)

# %%
print(zeta.latex_system())

# %% [markdown]
# ---
# ## Part 3: Phase 3 -- Basis-Specific Projection (with caching)
#
# `ProjectedModel` takes `ZetaProjectedEquations` (or `PreProjectedEquations`)
# plus a basis + level, and produces a complete `Model`.
#
# **Key improvement**: basis matrices (M, A, D, B, ...) are now **cached** by
# `(basis_name, level)` -- they are model-agnostic. Switching between Newtonian
# and inviscid models reuses the same matrices.
#
# The projection:
# 1. Computes basis matrices via `SymbolicIntegrator` (cached)
# 2. Substitutes ansatz: $u(\zeta) = \sum_k \alpha_k \varphi_k(\zeta)$
# 3. Collects raw projected vectors (with mass matrix M in front)
# 4. Applies $M^{-1}$ once to ALL non-temporal terms
# 5. Outputs `Model`-compatible `flux()`, `source()`, `nonconservative_matrix()`

# %%
from zoomy_core.model.models.projected_model import ProjectedModel
from zoomy_core.model.models.basisfunctions import (
    Legendre_shifted, Chebyshevu_shifted, SplineBasis, GalerkinBasis,
)

# %% [markdown]
# ### 3a. Legendre basis (standard, from ZetaProjectedEquations)
#
# Legendre on [0,1]: $\varphi_k(\zeta) = P_k(2\zeta - 1) \cdot (-1)^k$
#
# - Orthogonal: $M$ is diagonal
# - $\varphi_0 = 1$ (constant), so mean coefficients = $[1, 0, 0, \ldots]$
# - $M^{-1}$ is trivially $\mathrm{diag}(1/M_{kk})$

# %%
leg = ProjectedModel(pre, basis_type=Legendre_shifted, level=2)

print(f"Variables: {list(leg.variables.keys())}")
print(f"n_variables: {leg.n_variables}")
print(f"Mean coefficients: {leg.mean_coefficients()}")

print("\nMass matrix M:")
display(leg.mass_matrix())

print("\nM^{-1}:")
display(leg.mass_matrix_inverse())

# %% [markdown]
# #### Legendre flux

# %%
F_leg = leg.flux()
print("Legendre L2 flux (x-direction):")
for i in range(leg.n_variables):
    print(f"  F[{i}] = {F_leg[i, 0]}")

# %% [markdown]
# #### Legendre hydrostatic pressure (separated for well-balanced schemes)

# %%
P_leg = leg.hydrostatic_pressure()
print("Hydrostatic pressure:")
for i in range(leg.n_variables):
    val = P_leg[i, 0]
    if val != 0:
        print(f"  P[{i}] = {val}")

# %% [markdown]
# #### Legendre non-conservative matrix (topography coupling)

# %%
NC_leg = leg.nonconservative_matrix()
print("Non-conservative Bx (non-zero entries):")
for r in range(leg.n_variables):
    for c in range(leg.n_variables):
        val = NC_leg[r, c, 0]
        if val != 0:
            print(f"  Bx[{r},{c}] = {val}")

# %% [markdown]
# #### Legendre source terms (viscosity + slip friction)

# %%
visc_leg = leg.newtonian()
slip_leg = leg.slip()

print("Newtonian viscous source:")
for i in range(leg.n_variables):
    val = visc_leg[i]
    if val != 0:
        print(f"  S_visc[{i}] = {val}")

print("\nNavier-slip friction:")
for i in range(leg.n_variables):
    val = slip_leg[i]
    if val != 0:
        print(f"  S_slip[{i}] = {val}")

# %% [markdown]
# ### 3b. B-Spline basis
#
# Raw B-splines on [0,1] with hat functions:
# - Partition of unity: $\sum_k B_k = 1$ -> mean coefficients = $[1, 1, 1, \ldots]$
# - **Non-diagonal** mass matrix M
# - $M^{-1}$ is a full matrix -- redistributes contributions across modes

# %%
spl = ProjectedModel(pre, basis_type=SplineBasis, level=2)

print(f"Mean coefficients: {spl.mean_coefficients()}")
print("\nMass matrix M:")
display(spl.mass_matrix())
print("\nM^{-1}:")
display(spl.mass_matrix_inverse())

# %% [markdown]
# #### Spline flux
#
# Compare with Legendre: the mass flux is $h \cdot (\alpha_0 + \alpha_1 + \alpha_2)$
# because all spline coefficients contribute equally to the mean velocity
# (partition of unity).

# %%
F_spl = spl.flux()
print("SplineBasis L2 flux (x-direction):")
for i in range(spl.n_variables):
    expr = sp.simplify(F_spl[i, 0])
    print(f"  F[{i}] = {expr}")

# %% [markdown]
# #### Spline source: M^{-1} redistribution
#
# For splines, slip friction at the bottom ($\zeta = 0$) only directly excites
# $\varphi_0$ (which peaks at the bottom). But $M^{-1}$ redistributes this
# across all modes.

# %%
slip_spl = spl.slip()
print("Spline Navier-slip (with M^{-1} redistribution):")
for i in range(spl.n_variables):
    val = slip_spl[i]
    if val != 0:
        print(f"  S_slip[{i}] = {sp.simplify(val)}")

# %% [markdown]
# ### 3c. Chebyshev U basis (shifted to [0,1])
#
# $\varphi_k(\zeta) = U_k(2\zeta - 1)$ with weight $w(\zeta) = \sqrt{\zeta(1-\zeta)}$.
#
# - Orthogonal: diagonal $M$ (like Legendre)
# - But $M_{kk} = \pi/8$ (not simple rationals)
# - $\varphi_0 = 1$, same SWE limit as Legendre
# - **Caveat**: weight vanishes at boundaries -> boundary terms are killed

# %%
cheb = ProjectedModel(pre, basis_type=Chebyshevu_shifted, level=2)

print(f"Mean coefficients: {cheb.mean_coefficients()}")
print("\nMass matrix M:")
display(cheb.mass_matrix())
print("\nM^{-1}:")
display(cheb.mass_matrix_inverse())

# %% [markdown]
# #### Chebyshev boundary values
#
# The weight $\sqrt{\zeta(1-\zeta)}$ vanishes at $\zeta=0$ and $\zeta=1$.
# This means slip friction and kinematic BC boundary terms are killed
# in the weighted inner product. This is a known limitation of weighted bases.

# %%
cheb_basis = Chebyshevu_shifted(level=2)
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator

si = SymbolicIntegrator(cheb_basis)
mats = si.compute_all_matrices(2)
print("Chebyshev boundary values phi_k(0):")
for k in range(3):
    print(f"  phi_{k}(0) = {mats['phib'][k]}")

# %% [markdown]
# ### 3d. Galerkin (Shen-type) basis
#
# BC-aware basis via recombination of parent polynomials:
# - $\varphi_0 = 1$ (constant, always)
# - $\varphi_k$ (k >= 1): built to satisfy boundary conditions
#
# Supported BCs: `noslip`, `nostress`, `slip`, `free`

# %%
gal_basis = GalerkinBasis(level=2, parent="legendre",
                           bc_bottom="slip", bc_top="nostress",
                           slip_length=0.5)

print(f"Galerkin basis ({gal_basis.name}):")
for k in range(3):
    print(f"  phi_{k} = {gal_basis.get(k)}")

# %%
gal = ProjectedModel(pre, basis_type=GalerkinBasis, level=2)

print("Galerkin M:")
display(gal.mass_matrix())

# %% [markdown]
# ---
# ## Part 4: Comparing Bases
#
# ### Mass matrix structure

# %%
bases = {
    "Legendre": (Legendre_shifted, 2),
    "SplineBasis": (SplineBasis, 2),
    "Chebyshev U": (Chebyshevu_shifted, 2),
}

print("Mass matrix comparison (level 2):")
print("=" * 60)
for name, (cls, lvl) in bases.items():
    m = ProjectedModel(pre, basis_type=cls, level=lvl)
    M = m.mass_matrix()
    is_diag = M == sp.diag(*[M[i, i] for i in range(3)])
    print(f"\n{name} (diagonal={is_diag}):")
    display(M)

# %% [markdown]
# ### Mean coefficients
#
# The depth-averaged velocity is $\bar{u} = \sum_k c_k \alpha_k$.

# %%
print("Mean coefficients (level 2):")
for name, (cls, lvl) in bases.items():
    m = ProjectedModel(pre, basis_type=cls, level=lvl)
    c = m.mean_coefficients()
    print(f"  {name:15s}: c = {c}")

# %% [markdown]
# ### Flux structure at level 0 (SWE limit)
#
# At level 0, bases with $\varphi_0 = 1$ (Legendre, Chebyshev) recover
# the standard shallow water equations exactly:
# - Mass: $F_1 = h u$
# - Momentum: $F_2 = h u^2$
# - Pressure: $P_2 = g e_z h^2 / 2$
#
# **Note**: SplineBasis L0 uses $B_0 = 1 - \zeta$ (linear hat), not a constant.
# It needs L1+ (two hats) to represent uniform flow. The factors differ by
# $A_{000}/M_{00}$ which for a linear hat gives $3/4$ instead of $1$.

# %%
print("Level 0 flux comparison (SWE limit):")
print("=" * 60)
for name, cls in [("Legendre", Legendre_shifted), ("SplineBasis", SplineBasis)]:
    m0 = ProjectedModel(pre, basis_type=cls, level=0)
    F = m0.flux()
    P = m0.hydrostatic_pressure()
    print(f"\n{name} (phi_0 = {cls(level=0).get(0)}):")
    print(f"  Mass flux:     F[1] = {F[1, 0]}")
    print(f"  Mom. flux:     F[2] = {F[2, 0]}")
    print(f"  Pressure:      P[2] = {P[2, 0]}")

# %% [markdown]
# ---
# ## Part 5: Building a Complete Model for Simulation
#
# To run a simulation, we need:
# 1. A `ProjectedModel` with source terms (gravity + viscosity + friction)
# 2. A `NumericalModel` wrapper for regularization
# 3. A solver (explicit or IMEX)
#
# Here's how to build an inclined plane model:

# %%
from zoomy_core.misc.misc import ZArray

class InclinedPlaneProjected(ProjectedModel):
    """Inclined plane: gravity (g*ez) + Newtonian viscosity + Navier-slip."""

    def source(self):
        p = self.parameters
        h = self.variables[1]
        S = ZArray.zeros(self.n_variables)
        n_mom = self.level + 1
        # Gravity source: g*ez*h projected via Galerkin integral
        # raw[l] = g*ez*h * integral(phi_l) = g*ez*h * (M @ c_mean)[l]
        phi_int = self._phi_int
        raw_grav = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
        for k in range(n_mom):
            S[2 + k] = self._apply_Minv(raw_grav, k)
        # Viscosity + slip friction
        visc = self.newtonian()
        slip = self.slip()
        for i in range(self.n_variables):
            S[i] = S[i] + visc[i] + slip[i]
        return S

# %% [markdown]
# ### Inspect the inclined plane source

# %%
ip_leg = InclinedPlaneProjected(pre, basis_type=Legendre_shifted, level=2,
                                 eigenvalue_mode="numerical")
S_leg = ip_leg.source()

print("Inclined plane Legendre L2 source:")
for i in range(ip_leg.n_variables):
    val = S_leg[i]
    if val != 0:
        print(f"  S[{i}] = {sp.simplify(val)}")

# %% [markdown]
# The gravity term $g \cdot e_z \cdot h$ enters only the mean-velocity mode (index 2)
# for Legendre because $c = [1, 0, 0]$. Viscosity acts on modes 1+ (proportional to
# $\nu / h^2$), and slip friction couples all modes through $\varphi_k(0)$ boundary values.

# %% [markdown]
# ### SplineBasis inclined plane
#
# For splines, gravity is distributed across ALL modes because $c = [1, 1, 1]$
# and then $M^{-1}$ redistributes further.

# %%
ip_spl = InclinedPlaneProjected(pre, basis_type=SplineBasis, level=2,
                                 eigenvalue_mode="numerical")
S_spl = ip_spl.source()

print("Inclined plane SplineBasis L2 source:")
for i in range(ip_spl.n_variables):
    val = S_spl[i]
    if val != 0:
        print(f"  S[{i}] = {sp.simplify(val)}")

# %% [markdown]
# ---
# ## Part 6: Architecture Summary
#
# ```
# Phase 1 (basis-independent)
# ---------------------------
# FullINS(state)
#   |
#   +-- .apply(material)     -> Newtonian / Inviscid stress
#   +-- .apply(hydrostatic)  -> p = rho*g*(eta - z)
#   |
#   v
# derive_shallow_moments()
#   |
#   +-- depth integration with IBP on d/dz terms
#   +-- kinematic BCs on boundary terms
#   +-- stress BCs (slip at bottom, free at top)
#   +-- tag each term: temporal / flux / NC / source
#   |
#   v
# PreProjectedEquations  (reusable for ANY basis)
#
#
# Phase 2 (abstract zeta-space projection)
# -----------------------------------------
# PreProjectedEquations
#   |
#   +-- coordinate transform z -> zeta = (z-b)/H
#   +-- formal Galerkin projection with abstract phi_k(zeta)
#   +-- IBP on d/dzeta terms
#   +-- kinematic + stress BCs
#   +-- tag each term with matrix dependencies
#   |
#   v
# ZetaProjectedEquations  (inspectable LaTeX, abstract matrices)
#   |
#   +-- latex_system()  -> displayable PDE with M, A, D, B, Phi, phi_b
#   +-- summary()       -> text overview of terms and dependencies
#
#
# Phase 3 (basis-specific, cached)
# --------------------------------
# ZetaProjectedEquations + Basis + Level
#   |
#   +-- compute M, A, D, B, phib matrices via SymbolicIntegrator
#   +-- CACHED by (basis_name, level) -- model-agnostic!
#   +-- apply M^{-1} to all non-temporal terms
#   |
#   v
# ProjectedModel(Model)
#   |
#   +-- flux()                    -> advection + mass flux
#   +-- hydrostatic_pressure()    -> g*h^2/2 (separated for well-balanced)
#   +-- nonconservative_matrix()  -> topography + vertical coupling
#   +-- source() (overridable)    -> gravity + viscosity + friction
#   +-- eigenvalues()             -> symbolic or numerical
#   |
#   v
# NumericalModel(ProjectedModel)  -> regularized for simulation
#   |
#   v
# Solver (explicit / IMEX)
# ```
#
# The **same** `ZetaProjectedEquations` feeds into different bases.
# Matrix computation is cached -- switching bases at the same level is instant.
# Switching between Newtonian and inviscid reuses the same matrices.
