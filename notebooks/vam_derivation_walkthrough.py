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
# # VAM Derivation from INS: Step-by-Step
#
# This notebook derives the **Vorticity-Assisted Model (VAM)** — a non-hydrostatic
# shallow water model — from the full 3D Incompressible Navier-Stokes equations
# using the 3-phase pipeline.
#
# Unlike the SME (which eliminates z-momentum via hydrostatic assumption),
# the VAM keeps **both u and w velocity moments** and treats **pressure as an unknown**
# solved via a Poisson constraint.
#
# | Phase | What | Output |
# |-------|------|--------|
# | **Phase 1** | INS + inviscid, keep z-momentum, keep pressure | `VAMPreProjectedEquations` |
# | **Phase 2** | Abstract $\zeta$-projection with u, w, p matrix symbols | `VAMZetaProjectedEquations` |
# | **Phase 3** | Choose basis + level → concrete flux, NC, pressure source | `VAMProjectedHyperbolic` |
# | **Poisson** | Extract pressure constraints I1, I2 from splitting | `VAMProjectedPoisson` |
# | **Solve** | Predictor-Poisson-Corrector IMEX | ← **fails due to symbol mismatch** |

# %%
import sympy as sp
from sympy import Symbol, Derivative, Rational, S, sqrt, latex
from IPython.display import display, Math
sp.init_printing()

def show_eq(lhs, rhs):
    display(Math(lhs + " = " + sp.latex(rhs)))

# %% [markdown]
# ---
# ## Phase 1: Full 3D INS → Non-Hydrostatic Tagged Terms
#
# Start from the full INS. Apply inviscid material ($\tau = 0$).
# **Do NOT apply hydrostatic assumption** — keep z-momentum and pressure.

# %%
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Inviscid,
)

state = StateSpace(dimension=2)  # 2D = xz plane
ins = FullINS(state)
inv = Inviscid(state)

print("Full 3D INS (xz plane, inviscid):")
display(Math(r"\text{Continuity: } " + sp.latex(ins.continuity.expr) + " = 0"))
display(Math(r"\text{x-momentum: } " + sp.latex(ins.x_momentum.apply(inv).expr) + " = 0"))
display(Math(r"\text{z-momentum: } " + sp.latex(ins.z_momentum.apply(inv).expr) + " = 0"))

print("\nNote: pressure p(t,x,z) is NOT eliminated (non-hydrostatic)")

# %%
from zoomy_core.model.models.vam_derivation import derive_vam_moments

vam = derive_vam_moments(state, material=inv)
print(vam.summary())

# %%
print("Tagged terms:")
for eq_name, terms in vam.all_equations().items():
    print(f"\n  {eq_name}:")
    for t in terms:
        print(f"    [{t.role:15s}] {t.origin:30s}  {t.expr.expr}")

# %% [markdown]
# Key difference from SME: we have **z_momentum** terms (temporal, advection, gravity)
# and **pressure_x** / **pressure_z** terms kept as unknowns.

# %% [markdown]
# ---
# ## Phase 2: Abstract $\zeta$-Space Projection
#
# The Galerkin projection uses abstract matrix symbols. The VAM projection has:
# - **x-momentum**: same as SME (M, A, B, $\Phi$)
# - **z-momentum**: same matrices but with w-velocity indices ($\gamma_k$)
# - **Cross-momentum**: $A_{lij}\alpha_i\gamma_j$ (u × w coupling)
# - **Pressure**: $\partial(h\pi_k)/\partial x$ and IBP of $\partial p/\partial\zeta$

# %%
from zoomy_core.model.models.vam_zeta_projection import project_vam_to_zeta

zeta = project_vam_to_zeta(vam)
print(zeta.summary())

# %%
for eq_name, terms in zeta.all_equations().items():
    print(f"\n=== {eq_name} ===")
    for t in terms:
        print(f"  [{t.role:15s}] deps={t.matrix_deps}")
        display(t)

# %% [markdown]
# ---
# ## Phase 3: Concrete Model (Legendre L1)
#
# Choose Legendre basis at level 1: $\varphi_0 = 1$, $\varphi_1 = 1 - 2\zeta$.
#
# State vector: $Q = [h, hu_0, hu_1, hw_0, hw_1, b]$ (6 variables).
# The w-closure mode $hw_2$ is an auxiliary variable from the continuity constraint.

# %%
from zoomy_core.model.models.vam_projected_model import VAMProjectedHyperbolic, VAMProjectedPoisson
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.projected_model import clear_matrix_cache

clear_matrix_cache()
hyp = VAMProjectedHyperbolic(vam, basis_type=Legendre_shifted, level=1,
                              eigenvalue_mode='symbolic')

print(f"Variables ({hyp.n_variables}): {list(hyp.variables.keys())}")
print(f"Aux variables ({hyp.n_aux_variables}): {[str(hyp.aux_variables[i]) for i in range(hyp.n_aux_variables)]}")
print(f"\nMass matrix M:")
display(hyp.mass_matrix())

# %% [markdown]
# ### Flux

# %%
F = hyp.flux()
print("Flux F(Q):")
for i, name in enumerate(hyp.variables.keys()):
    val = sp.simplify(F[i, 0])
    if val != 0:
        show_eq(f"F_{{{name}}}", val)

# %% [markdown]
# Note: $F_{hw_1}$ includes the **closure term** $(2/5) \cdot hu_1 \cdot hw_2 / h$
# where $hw_2$ is the auxiliary w-mode from the continuity constraint.

# %% [markdown]
# ### Eigenvalues

# %%
ev = hyp.eigenvalues()
print("Wave speeds:")
for i, name in enumerate(hyp.variables.keys()):
    if ev[i] != 0:
        show_eq(f"\\lambda_{{{name}}}", ev[i])

# %% [markdown]
# ### Pressure Source (from Galerkin projection of $\nabla p$)
#
# The pressure enters as an **implicit source** via auxiliary variables $hp_0, hp_1$:
#
# - **x-momentum**: from $\int \varphi_l \frac{\partial p}{\partial x} dz$ (Leibniz rule)
# - **z-momentum**: from $\int \varphi_l \frac{\partial p}{\partial z} dz$ (IBP in $\zeta$)

# %%
R = hyp.source_implicit()
print("Pressure source (implicit):")
for i, name in enumerate(hyp.variables.keys()):
    val = R[i]
    if val != 0:
        show_eq(f"\\tau_{{{name}}}", sp.simplify(val))

# %% [markdown]
# **Galerkin result at L1:**
# - z-momentum mode 0: $-2p_1$ (from IBP: boundary terms cancel in volume, only $[p\varphi_0]_0^1 = -2p_1$)
# - z-momentum mode 1: **0** (the Galerkin projection of $\partial p/\partial\zeta$ against $\varphi_1$ vanishes by symmetry)
#
# Note: The hardcoded VAM uses a vorticity-assisted projection that gives $6(p_0 - p_1)$
# for z-momentum mode 1. The Galerkin version is a different (but valid) reduced model.

# %% [markdown]
# ---
# ## Poisson Constraints
#
# The pressure Poisson system comes from the **splitting approach**:
# 1. Compute velocity predictor $U^* = U^n - \Delta t \cdot \text{Flux}(U^n)$
# 2. Apply pressure correction: $U^{n+1} = U^* - \Delta t \cdot \tau(P)$
# 3. Enforce constraints $I_1 = 0$, $I_2 = 0$ on $U^{n+1}$
#
# This produces a Poisson-type system for $hp_0, hp_1$.

# %%
poi = VAMProjectedPoisson(hyp)

I1, I2 = poi.get_constraints()
R0, R1 = poi.get_residual_equations()

I1_clean, subs1 = poi.make_derivative_symbols(I1)
I2_clean, subs2 = poi.make_derivative_symbols(I2)

print("Constraint I1 (continuity):")
display(I1_clean)

print("\nConstraint I2 (vorticity):")
display(I2_clean)

print("\nResidual equations:")
R0_clean, _ = poi.make_derivative_symbols(R0)
R1_clean, _ = poi.make_derivative_symbols(R1)
print("R[hp0] = I1 + I2:")
display(R0_clean)
print("R[hp1] = I1 - I2:")
display(R1_clean)

# %% [markdown]
# ---
# ## Attempting to Solve: The Symbol Mismatch Bug
#
# The VAM model builds correctly (flux, NC, eigenvalues, pressure source, Poisson).
# But when we try to **run it numerically** with the IMEX solver, it fails.
#
# The root cause: `NumericalModel` creates its **own SymPy symbols** that are
# different objects from the analytical model's symbols. When `sp.lambdify`
# compiles the flux expression, the analytical model's `h` (with `positive=True`)
# doesn't match the `NumericalModel`'s `h` (without assumptions).
#
# This means the compiled numpy function has **unbound symbols** → can't convert to float.

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
        Q[1] = h0 * 0.11197 / 0.34  # hu0
    Q[-1] = b_val  # bathymetry
    return Q

bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
num = NumericalModel(hyp, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))

# Show the symbol mismatch
print("=== THE SYMBOL MISMATCH ===\n")
print(f"Analytical model h: {hyp.variables[0]}")
print(f"  assumptions: positive={hyp.variables[0].is_positive}")
print()
print(f"NumericalModel h:   {num.variables[0]}")
print(f"  assumptions: positive={num.variables[0].is_positive}")
print()
print(f"Same object?  {hyp.variables[0] is num.variables[0]}")
print(f"Equal?        {hyp.variables[0] == num.variables[0]}")
print()
print("The flux expression uses the ANALYTICAL h (positive=True).")
print("The lambdify compilation receives NUMERICAL h (no assumptions).")
print("These are different SymPy Symbols → unbound → float conversion fails.")

# %%
# Demonstrate the actual error
import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
from zoomy_core.fvm.solver_pressurized_imex import PressurizedIMEXSolver
from zoomy_core.misc.misc import Zstruct
import zoomy_core.fvm.timestepping as timestepping
import os

mesh = petscMesh.Mesh.create_1d(domain=(0, 1), n_inner_cells=10)
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

print("Attempting to solve...")
try:
    Q, Qaux = solver.solve(mesh, num, write_output=False)
    print("SUCCESS (unexpected)")
except TypeError as e:
    print(f"EXPECTED ERROR: {e}")
    print()
    print("This error occurs because:")
    print("1. NumericalModel creates its own symbols (h, hu0, ...) ")
    print("2. These are DIFFERENT SymPy objects from the analytical model's symbols")
    print("3. The compiled flux function (via sp.lambdify) receives NumericalModel's symbols")
    print("4. But the flux expression contains the ANALYTICAL model's symbols")
    print("5. The unmatched symbols can't be converted to float → TypeError")
    print()
    print("FIX NEEDED: NumericalModel should NOT create its own symbols.")
    print("It should reuse the analytical model's symbols directly and only")
    print("add opaque numerical functions (clamp_positive, conditional) on top.")

# %% [markdown]
# ---
# ## Summary
#
# **What works:**
# - Full 3-phase derivation: INS → VAM Phase 1/2/3
# - Flux (including hw2 closure), NC (gravity/topography), eigenvalues: all correct at L1
# - Pressure source from Galerkin projection of dp/dx and dp/dz
# - Poisson constraints I1, I2 derived symbolically from splitting
# - Builds at arbitrary level (L1-L4 tested)
#
# **What's blocked:**
# - Numerical solve fails due to **symbol mismatch** between `NumericalModel` and analytical model
# - Fix: refactor `NumericalModel` to not duplicate symbols
#
# **Galerkin vs VAM difference:**
# - z-momentum mode 1 pressure: Galerkin gives 0, hardcoded VAM gives $6(p_0-p_1)$
# - The hardcoded VAM uses vorticity-assisted projection (different equations from standard Galerkin)
# - Both are valid reduced models
