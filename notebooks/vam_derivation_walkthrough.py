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
# # Deriving Shallow Water Models from the INS
#
# This notebook derives both the **SME** (hydrostatic shallow moment equations)
# and the **VAM** (non-hydrostatic vorticity-assisted model) from the full 3D
# Incompressible Navier-Stokes, step by step.
#
# Every equation emerges from careful application of:
# 1. **Depth integration** with variable limits $[b, b+H]$
# 2. **Leibniz rule** (moving boundary correction)
# 3. **Fundamental theorem** (for $\partial/\partial z$ terms)
# 4. **Kinematic boundary conditions** at surface and bottom
# 5. **Galerkin projection** onto a polynomial basis
#
# | Model | Assumption | Keeps z-momentum? | Pressure |
# |-------|-----------|-------------------|----------|
# | **SME** | Hydrostatic | No (eliminated) | $p = \rho g(\eta - z)$ |
# | **VAM** | Non-hydrostatic | Yes | Unknown (Poisson) |

# %%
import sympy as sp
from sympy import Symbol, Function, Derivative, Integral, symbols, simplify, expand, Rational
from IPython.display import display, Math, HTML
sp.init_printing()

def show_eq(lhs, rhs, suffix=""):
    display(Math(lhs + " = " + sp.latex(rhs) + suffix))

# %% [markdown]
# ---
# ## Part 1: The Full 3D INS
#
# Everything starts from the incompressible Navier-Stokes in the $xz$ plane.

# %%
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, DepthIntegralResult,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
    Newtonian, Inviscid,
)

state = StateSpace(dimension=2)
ins = FullINS(state)

print("Full 3D INS (xz plane):")
display(Math(r"\text{Continuity: }\quad " + sp.latex(ins.continuity.expr) + " = 0"))
display(Math(r"\text{x-momentum: }\quad " + sp.latex(ins.x_momentum.expr) + " = 0"))
display(Math(r"\text{z-momentum: }\quad " + sp.latex(ins.z_momentum.expr) + " = 0"))

# %% [markdown]
# ### Symbols and geometry
#
# - Bottom: $b(t,x)$, depth: $H(t,x)$, surface: $\eta = b + H$
# - Velocities: $u(t,x,z)$, $w(t,x,z)$
# - Pressure: $p(t,x,z)$, stress: $\tau_{ij}(t,x,z)$

# %%
z = state.z
b, H, eta = state.b, state.H, state.eta

print(f"Bottom: b = {b}")
print(f"Depth:  H = {H}")
print(f"Surface: eta = b + H = {eta}")

# %% [markdown]
# ### Kinematic boundary conditions

# %%
kbc_b = KinematicBCBottom(state)
kbc_s = KinematicBCSurface(state)

print("Kinematic BC at bottom (z = b):")
display(kbc_b)
print("Kinematic BC at surface (z = eta):")
display(kbc_s)

# %% [markdown]
# ---
# ## Part 2: Deriving the Mass Conservation (both SME and VAM)
#
# Start from $\nabla \cdot \mathbf{u} = 0$:
# $$\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z} = 0$$
#
# **Step 1**: Integrate over depth $[b, b+H]$
#
# **Step 2**: For $\partial u/\partial x$ → **Leibniz rule** (swap derivative and integral):
# $$\int_b^\eta \frac{\partial u}{\partial x}\,dz
# = \frac{\partial}{\partial x}\!\int_b^\eta u\,dz
# \;-\; u(\eta)\frac{\partial\eta}{\partial x}
# \;+\; u(b)\frac{\partial b}{\partial x}$$
#
# **Step 3**: For $\partial w/\partial z$ → **fundamental theorem**:
# $$\int_b^\eta \frac{\partial w}{\partial z}\,dz = w(\eta) - w(b)$$
#
# **Step 4**: Apply **kinematic BCs** to boundary terms → they simplify to $\partial H/\partial t$.

# %%
print("Term-by-term depth integration of continuity:")
print()

results = []
for i, term in enumerate(ins.continuity):
    r = term.depth_integrate(b, eta, z)
    results.append(r)
    if isinstance(r, DepthIntegralResult):
        method = "Leibniz" if r.volume.expr != 0 else "fund. theorem"
        print(f"  Term {i}: {term.expr}  →  [{method}]")
        print(f"    volume:   {r.volume.expr}")
        print(f"    bnd(eta): {r.boundary_upper.expr}")
        print(f"    bnd(b):   {r.boundary_lower.expr}")
        print()

# %%
# Collect ALL boundary terms, then apply kinematic BCs
total_volume = Expression(sp.Integer(0))
total_boundary = Expression(sp.Integer(0))

for r in results:
    if isinstance(r, DepthIntegralResult):
        total_volume = total_volume + r.volume
        total_boundary = total_boundary + r.boundary_upper + r.boundary_lower

print("Boundary terms before BCs:")
display(Math(sp.latex(total_boundary.expr)))

boundary_with_bcs = total_boundary.apply(kbc_s).apply(kbc_b)

print("\nBoundary terms after kinematic BCs:")
display(Math(sp.latex(simplify(boundary_with_bcs.expr))))

# %%
print("Final mass conservation:")
total = total_volume + boundary_with_bcs
display(Math(sp.latex(total.expr) + " = 0"))

print("\nWhich is:")
display(Math(r"\frac{\partial H}{\partial t} + \frac{\partial}{\partial x}\!\int_b^\eta u\,dz = 0"))

# %% [markdown]
# This is the **depth-integrated mass conservation** — identical for both SME and VAM.
# No assumptions about pressure or vertical momentum were needed.
#
# After Galerkin projection ($u = \sum \alpha_k \varphi_k$), the integral becomes
# $h \sum c_k \alpha_k$ and we get:
# $$\frac{\partial h}{\partial t} + \frac{\partial(h\bar{u})}{\partial x} = 0$$

# %% [markdown]
# ---
# ## Part 3: SME — Hydrostatic Shallow Moment Equations
#
# The SME makes two key assumptions:
# 1. **Hydrostatic pressure**: $p = p_{\text{atm}} + \rho g (\eta - z)$
# 2. **Eliminate z-momentum** (it becomes the hydrostatic balance)
#
# Only the **x-momentum** equation survives.

# %% [markdown]
# ### Step 3.1: Apply Newtonian material + hydrostatic assumption

# %%
newton = Newtonian(state)
hydro = HydrostaticPressure(state)

print("Newtonian stress:")
display(newton)

print("\nHydrostatic pressure:")
display(hydro)

# %%
xm = ins.x_momentum.apply(newton).apply(hydro)

print("x-momentum after Newtonian + hydrostatic:")
display(Math(sp.latex(xm.expr) + " = 0"))
print(f"\nHas p? {xm.has(state.p)}  (eliminated by hydrostatic)")
print(f"Has g? {xm.has(state.g)}  (pressure gradient → gravity)")

# %% [markdown]
# ### Step 3.2: Depth-integrate the x-momentum
#
# Each term in the x-momentum equation gets depth-integrated:
# - $\partial u/\partial t$ → Leibniz (with respect to $t$)
# - $\partial(u^2)/\partial x$ → Leibniz
# - $\partial(uw)/\partial z$ → fundamental theorem → kinematic BCs
# - $g\partial\eta/\partial x$ → direct (no derivative in $z$)
# - Viscous terms → IBP in $z$ → slip BC at bottom, free surface at top

# %%
print("x-momentum terms:")
for i, term in enumerate(xm):
    print(f"  [{i}] {term.expr}")

# %% [markdown]
# ### Step 3.3: Galerkin projection → ProjectedModel
#
# After depth integration and applying the ansatz $u(\zeta) = \sum \alpha_k \varphi_k(\zeta)$,
# each term maps to a specific part of the FVM solver:
#
# | Term | Role | FVM component |
# |------|------|--------------|
# | $\partial_t(h\alpha_k)$ | temporal | mass matrix $M$ |
# | $\partial_x(h A_{kij}\alpha_i\alpha_j)$ | flux | $F(Q)$ |
# | $g h \Phi_k \partial_x b$ | nonconservative | $B(Q)\partial_x Q$ |
# | $\nu/h \cdot D_{ki}\alpha_i$ | source | $S(Q)$ |
# | $-u_b/(λρ) \cdot \varphi_k(0)$ | source | $S(Q)$ |

# %%
from zoomy_core.model.models.model_derivation import derive_shallow_moments
from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
from zoomy_core.model.models.basisfunctions import Legendre_shifted

pre = derive_shallow_moments(state, material=Newtonian(state))
print(pre.summary())

# %%
clear_matrix_cache()
sme = ProjectedModel(pre, basis_type=Legendre_shifted, level=2, eigenvalue_mode='numerical')

print(f"SME L2: {sme.n_variables} variables: {list(sme.variables.keys())}")
print(f"Mass matrix M:")
display(sme.mass_matrix())

F = sme.flux()
print("\nFlux:")
for i in range(sme.n_variables):
    val = sp.simplify(F[i, 0])
    if val != 0:
        show_eq(f"F_{i}", val)

P = sme.hydrostatic_pressure()
print("\nHydrostatic pressure:")
for i in range(sme.n_variables):
    val = P[i, 0]
    if val != 0:
        show_eq(f"P_{i}", val)

# %% [markdown]
# ---
# ## Part 4: VAM — Non-Hydrostatic (Galerkin Projection)
#
# The VAM does **not** apply the hydrostatic assumption. Instead:
# - Keep **z-momentum** equation → vertical velocity moments $hw_k$
# - Keep **pressure as unknown** → solved via Poisson constraint
# - Apply inviscid material (no stress terms for simplicity)

# %% [markdown]
# ### Step 4.1: Depth-integrate x-momentum (non-hydrostatic)
#
# Same as SME but **without** substituting hydrostatic pressure.
# The pressure gradient $\partial p/\partial x$ stays as an unknown.

# %%
inv = Inviscid(state)
xm_nh = ins.x_momentum.apply(inv)

print("x-momentum (inviscid, non-hydrostatic):")
display(Math(sp.latex(xm_nh.expr) + " = 0"))
print(f"  Has p? {xm_nh.has(state.p)}  (kept as unknown)")

# %% [markdown]
# ### Step 4.2: Depth-integrate z-momentum
#
# This is the **new** equation in the VAM. The z-momentum contains:
# - $\partial w/\partial t$ → temporal (w-velocity moments)
# - $\partial(uw)/\partial x$ → cross-momentum flux
# - $\partial(w^2)/\partial z$ → fundamental theorem + kinematic BCs
# - $g$ → gravity source
# - $\partial p/\partial z$ → pressure coupling (IBP → algebraic)

# %%
zm = ins.z_momentum.apply(inv)
print("z-momentum (inviscid):")
display(Math(sp.latex(zm.expr) + " = 0"))

# %% [markdown]
# ### Step 4.3: Depth-integrate the pressure gradient $\partial p/\partial z$
#
# This is the key difference from SME. After IBP in $\zeta$-space:
# $$\int_0^1 \varphi_l \frac{\partial p}{\partial\zeta}\,d\zeta
# = [p\,\varphi_l]_0^1 - \int_0^1 p \frac{d\varphi_l}{d\zeta}\,d\zeta$$
#
# With $p(\eta) = 0$ (atmospheric gauge) and $p(b) = \sum \pi_k \varphi_k(0)$,
# this gives algebraic coupling between pressure modes and w-velocity modes.

# %% [markdown]
# ### Step 4.4: Build the VAM model

# %%
from zoomy_core.model.models.vam_derivation import derive_vam_moments
from zoomy_core.model.models.vam_projected_model import VAMProjectedHyperbolic, VAMProjectedPoisson

vam_pre = derive_vam_moments(state, material=inv)
print(vam_pre.summary())

# %%
clear_matrix_cache()
hyp = VAMProjectedHyperbolic(vam_pre, basis_type=Legendre_shifted, level=1,
                              eigenvalue_mode='symbolic')

vn = list(hyp.variables.keys())
print(f"VAM L1: {hyp.n_variables} variables: {vn}")
print(f"Aux: {[str(hyp.aux_variables[i]) for i in range(hyp.n_aux_variables)]}")

print(f"\nMass matrix:")
display(hyp.mass_matrix())

# %%
F = hyp.flux()
print("Flux:")
for i in range(hyp.n_variables):
    val = sp.simplify(F[i, 0])
    if val != 0:
        show_eq(f"F_{{{vn[i]}}}", val)

# %%
ev = hyp.eigenvalues()
print("Wave speeds:")
for i in range(hyp.n_variables):
    if ev[i] != 0:
        show_eq(f"\\lambda_{{{vn[i]}}}", ev[i])

# %%
R = hyp.source_implicit()
print("Pressure source (implicit, from dp/dx and dp/dz):")
for i in range(hyp.n_variables):
    val = R[i]
    if val != 0:
        show_eq(f"\\tau_{{{vn[i]}}}", sp.simplify(val))

# %% [markdown]
# **Note on the z-momentum pressure:**
# - Mode 0: $-2p_1$ (from IBP of $\partial p/\partial\zeta$ against $\varphi_0$)
# - Mode 1: **0** (Galerkin projection vanishes by symmetry)
#
# The hardcoded VAM uses a vorticity-assisted projection that gives $6(p_0-p_1)$
# for mode 1 — a different (but also valid) reduced model.

# %% [markdown]
# ---
# ## Part 5: Poisson Constraints for Pressure
#
# The pressure is determined by enforcing constraints from the splitting approach:
# 1. Compute velocity **predictor**: $U^* = U^n - \Delta t \cdot \text{Flux}(U^n)$
# 2. **Correct** with pressure: $U^{n+1} = U^* - \Delta t \cdot \tau(P)$
# 3. Enforce **constraints** $I_1 = 0$, $I_2 = 0$ on $U^{n+1}$
#
# This produces a Poisson-type system for $hp_0, hp_1$.

# %%
poi = VAMProjectedPoisson(hyp)

I1, I2 = poi.get_constraints()
I1_clean, _ = poi.make_derivative_symbols(I1)
I2_clean, _ = poi.make_derivative_symbols(I2)

print("I1 (continuity constraint):")
display(I1_clean)

print("\nI2 (vorticity constraint):")
display(I2_clean)

# %%
R0, R1 = poi.get_residual_equations()
R0_clean, _ = poi.make_derivative_symbols(R0)
R1_clean, _ = poi.make_derivative_symbols(R1)

print("Poisson residual (R[hp0] = I1 + I2):")
display(R0_clean)
print("\nPoisson residual (R[hp1] = I1 - I2):")
display(R1_clean)

# %% [markdown]
# ---
# ## Part 6: Attempting to Solve — The Symbol Mismatch Bug
#
# The model builds correctly. But the numerical solve fails because
# `NumericalModel` creates its **own SymPy symbols** that don't match
# the analytical model's symbols (different `positive` assumptions).

# %%
import numpy as np
from zoomy_core.model.numerical_model import NumericalModel
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC

nv = hyp.n_variables
def ic(x, _nv=nv):
    Q = np.zeros(_nv)
    b_val = 0.2 * np.exp(-x[0]**2)
    h0 = max(0.34 - b_val, 0.015) if x[0] < 1 else 0.015
    Q[0] = h0
    if x[0] < 1:
        Q[1] = h0 * 0.11197 / 0.34
    Q[-1] = b_val
    return Q

bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
num = NumericalModel(hyp, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))

print("The symbol mismatch:")
print(f"  Analytical h: positive={hyp.variables[0].is_positive}, id={id(hyp.variables[0])}")
print(f"  Numerical h:  positive={num.variables[0].is_positive}, id={id(num.variables[0])}")
print(f"  Same object? {hyp.variables[0] is num.variables[0]}")
print(f"  Equal?       {hyp.variables[0] == num.variables[0]}")

# %%
import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
from zoomy_core.fvm.solver_pressurized_imex import PressurizedIMEXSolver
from zoomy_core.misc.misc import Zstruct
import zoomy_core.fvm.timestepping as timestepping
import os

mesh = petscMesh.Mesh.create_1d(domain=(-1.5, 15.0), n_inner_cells=50)
os.makedirs("outputs/vam_test", exist_ok=True)
settings = Zstruct(output=Zstruct(directory="outputs/vam_test", filename="tmp",
                                   snapshots=2, clean_directory=False))

class PressurizedSolver(_GeneratedModelFluxMixin, PressurizedIMEXSolver):
    pass

solver = PressurizedSolver(time_end=0.1, settings=settings,
                            compute_dt=timestepping.adaptive(CFL=0.1), min_dt=1e-8)
object.__setattr__(solver, "source_mode", "local")
object.__setattr__(solver, "implicit_maxiter", 5)
object.__setattr__(solver, "implicit_tol", 1e-8)
object.__setattr__(solver, "pressure_n_vars", 2)
object.__setattr__(solver, "pressure_maxiter", 3)
object.__setattr__(solver, "pressure_tol", 1e-6)

for pn, pv in [("g", 9.81)]:
    pk = list(num.parameters.keys())
    if pn in pk:
        num.parameter_values[pk.index(pn)] = pv

print("\nAttempting to solve VAM on bump test...")
try:
    Q, Qaux = solver.solve(mesh, num, write_output=False)
    print("SUCCESS")
    for i, name in enumerate(vn):
        print(f"  {name}: [{Q[i,:].min():.6f}, {Q[i,:].max():.6f}]")
except TypeError as e:
    print(f"EXPECTED ERROR: {e}")
    print()
    print("Root cause: NumericalModel duplicates symbols with different")
    print("assumptions (h without positive=True). The compiled flux function")
    print("can't evaluate because the analytical model's symbols (in the flux")
    print("expression) don't match the numerical model's symbols (passed to lambdify).")
    print()
    print("Fix: NumericalModel must reuse the analytical model's symbols directly,")
    print("using opaque functions (clamp_positive, conditional) instead of")
    print("creating new symbols with different SymPy assumptions.")

# %% [markdown]
# ---
# ## Summary
#
# | Step | SME | VAM |
# |------|-----|-----|
# | Start | Full 3D INS | Full 3D INS |
# | Material | Newtonian | Inviscid |
# | Pressure | Hydrostatic ($p = \rho g(\eta-z)$) | Unknown (Poisson) |
# | z-momentum | Eliminated | Kept → $hw_k$ moments |
# | Depth integration | Leibniz + fund. theorem + BCs | Same |
# | Projection | Galerkin onto $\varphi_k(\zeta)$ | Same (+ w and p basis) |
# | Result | `ProjectedModel` with flux, NC, source | `VAMProjectedHyperbolic` + `VAMProjectedPoisson` |
# | Solver | Standard IMEX | Predictor-Poisson-Corrector (blocked by symbol bug) |
#
# **What works end-to-end:**
# - SME: INS → derive → project → solve (inclined plane, dam break, 1D/2D) ✓
# - VAM: INS → derive → project → model builds at arbitrary level ✓
#
# **What's blocked:**
# - VAM numerical solve: `NumericalModel` symbol mismatch
# - Fix: refactor `NumericalModel` to reuse analytical symbols + opaque functions
