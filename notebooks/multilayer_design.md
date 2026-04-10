# Multi-Layer Solver: Design Decisions

## Status: In Progress

## 1. Strategy

### Phase A: Single-layer SWE against Basilisk (proof of pipeline)
Run the `GeneratedShallowModel(n_layers=1, level=0)` through the existing solver and compare against Basilisk `swe_dambreak` and `swe_bump1d`.

This validates:
- GeneratedShallowModel plugs into the solver correctly
- The symbolic→NumPy compilation works
- The Basilisk comparison infrastructure works

### Phase B: Multi-layer interface flux design
Design and implement vertical coupling between layers.

### Phase C: Multi-layer simulation against Basilisk
Compare against `ml_lock_exchange` or `ml_wind_driven`.

### Phase D: JAX port

---

## 2. Interface Flux Design for Multi-Layer

### The problem
For `n_layers > 1`, the vertical derivative of the Heaviside-windowed ansatz produces
DiracDelta terms at internal layer interfaces. These represent the vertical mass and
momentum exchange between layers.

In Basilisk's layered solver, this is the `G_{k+1/2}` term — the vertical transport
velocity at the interface between layer k and k+1.

### Design decision: Source-like coupling, NOT Riemann solver

**Rationale:**
- Vertical exchange is NOT wave propagation — it's a vertical velocity coupling
- In the multi-layer SWE, the interface velocity G_{k+1/2} is determined by the
  incompressibility constraint (continuity equation), not by a Riemann problem
- The existing ShallowMomentsTopo handles moment coupling via the B matrix
  (non-conservative terms), not via a separate Riemann solver
- For consistency with the existing architecture, interface terms should be
  either source terms or non-conservative product terms

**Implementation:**
1. For single-layer (n_layers=1): no interface terms, works like ShallowMomentsTopo
2. For multi-layer (n_layers>1):
   - Intra-layer terms: same structure as single-layer (flux + nonconservative)
   - Inter-layer coupling: added to the nonconservative matrix B
   - Vertical transport G_{k+1/2}: computed from the continuity equation
     (G is NOT an independent variable — it's diagnosed)
   - Viscous inter-layer exchange: added to source terms

### Interface velocity G_{k+1/2}

From the continuity equation for layer k:
  ∂h_k/∂t + ∂(h_k u_k)/∂x = G_{k-1/2} - G_{k+1/2}

The total continuity (sum over all layers) gives:
  ∂H/∂t + ∂(Σ h_k u_k)/∂x = 0  (with G_{-1/2} = G_{N+1/2} = 0)

So G_{k+1/2} is determined by summing the individual continuity equations from bottom to layer k.
This is a diagnostic relationship, not a prognostic one.

For the momentum equation of layer k, the vertical advection term is:
  G_{k+1/2} * u_{k+1/2} - G_{k-1/2} * u_{k-1/2}

where u_{k+1/2} is the velocity at the interface (upwind: u_k if G > 0, u_{k+1} if G < 0).

**Key insight:** In the single-layer shallow moments model (ShallowMomentsTopo), the vertical
coupling is already encoded in the B matrix (basis function derivative products). For multi-layer,
the B matrix handles intra-layer coupling, and the inter-layer G terms are additional source terms.

---

## 3. Model Integration with Existing Solver

### Approach
The `GeneratedShallowModel` already inherits from `Model` and has:
- `flux()`, `hydrostatic_pressure()`, `nonconservative_matrix()`, `source()`, `eigenvalues()`

To run it through the solver, I need:
1. Wrap it as a `StructuredDerivativeModel` (for the DerivativeAwareSolver)
2. Or use it directly with `HyperbolicSolver` + appropriate Numerics

### Decision: Use StructuredDerivativeModel wrapper

I'll create a `NumericalGeneratedShallowModel` that:
- Inherits from `StructuredDerivativeModel`
- Uses `GeneratedShallowModel` internally for the equations
- Adds `hinv` as a user aux variable
- Provides field_map for the Numerics/Rusanov infrastructure

### State vector layout
For 1D, n_layers=1, level=0:
  Q = [b, h, hu]
  Qaux = [hinv]

This is identical to `SWEBeachTopoModel` from the tutorial.

For multi-layer (n_layers=2, level=0):
  Q = [b, h, hu_0, hu_1]
  Qaux = [hinv]

---

## 4. Mesh

### 1D: use Mesh.create_1d()
For dam break: domain (-5, 5), 500 cells
For bump: domain (-0.5, 0.5), 500 cells

### 2D: create with gmsh
Simple rectangular domain. Will create if needed.

---

## 5. Basilisk Comparison

### Method
1. Load Basilisk VTK output using `tests/common/basilisk_loader.py`
2. Interpolate onto our mesh
3. Compare at matching time snapshots
4. Report L1, L2, Linf norms

### Test cases
- **swe_dambreak**: classical Riemann problem, analytical solution exists
- **swe_bump1d**: Gaussian bump on flat bed, tests wave propagation
- **swe_parabola**: has analytical solution for convergence study

---

## 6. Questions / Open Items

1. **Eigenvalue computation for multi-layer:** The quasilinear matrix for 2+ layers
   produces quartic+ characteristic polynomials. SymPy struggles with these.
   **Decision:** Use numerical eigenvalue computation at runtime for multi-layer.
   For single-layer, keep symbolic eigenvalues.

2. **Layer-to-layer viscous coupling:** The viscous stress at layer interfaces
   needs the velocity gradient across the interface. This is a finite difference
   approximation: (u_{k+1} - u_k) / (Δz_{k} + Δz_{k+1}) * 2.
   **Decision:** Add this as a source term, not IBP.

3. **Wet/dry treatment for multi-layer:** Each layer can independently go dry.
   **Decision:** For now, use a global eps threshold. Revisit later.
