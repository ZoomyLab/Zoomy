# Session Summary: Symbolic Numerics + Pipeline Refactor

**Date**: 2026-04-03 to 2026-04-07
**Branch**: develop (pushed)

---

## What was accomplished

### 1. Three-Phase INS Pipeline (SME)
- **Phase 1**: `derive_shallow_moments()` → `PreProjectedEquations`
- **Phase 2**: `project_to_zeta()` → `ZetaProjectedEquations` (abstract matrix symbols)
- **Phase 3**: `ProjectedModel` → concrete model with cached basis matrices
- Verified: inclined plane L0-L4, all bases (Legendre, Chebyshev, Spline)

### 2. SymbolicIntegrator Optimizations
- Symmetry exploitation: A fully symmetric (63-72% fewer integrals)
- Chebyshev: analytical orthogonal integration (exact pi/8, no nsimplify garbage)
- Legendre: fast antiderivative path (100x vs sp_integrate)
- Spline: fast knot-span antiderivative
- Timeout on sympy.integrate (30s default)

### 3. VAM Derivation from INS
- `vam_derivation.py`: non-hydrostatic Phase 1
- `vam_zeta_projection.py`: abstract Phase 2 with u/w/p terms
- `vam_projected_model.py`: Phase 3 with flux, NC, eigenvalues, pressure source
- Verified at L1-L4 Legendre (19 tests pass)

### 4. Expression Enhancements
- `depth_integrate()`: Leibniz rule + fundamental theorem per term
- `map_with_bcs()`: depth-integrate full equation + apply kinematic BCs → clean dH/dt
- `classify()`: auto-classify terms (temporal, convective, diffusive, source)
- `project_onto_basis()`: substitute basis expansion, evaluate integrals → matrix products
- `derivation_mermaid()`: mermaid diagram of derivation history
- `describe()`: markdown/LaTeX/text output with strip_args option
- `latex(strip_args=True)`: u(t,x,z) → u in display

### 5. DerivedSystem (Model Caching)
- `sme()` / `vam()`: factory functions for pre-derived systems
- `.with_material()`: add viscosity post-derivation
- `.with_basis()`: project onto any basis
- `.save()` / `.load()`: pickle for reuse

### 6. Symbolic Numerics (No NumericalModel)
- `symbolic_numerics.py`: regularize model without duplicating symbols
- SME solver works directly with ProjectedModel (Linf=1.2e-05)
- VAM solver works directly with VAMProjectedHyperbolic (no NaN)
- No more NumericalModel symbol mismatch bug

### 7. Pressurized IMEX Solver
- `solver_pressurized_imex.py`: predictor-Poisson-corrector
- Runs on VAM bump test (hyperbolic part correct)
- Pressure Poisson needs proper spatial derivative computation

### 8. Comprehensive Testing
- Inclined plane: all bases × BC modes × Nitsche levels
- 2D unstructured mesh: SME dam break on triangular mesh
- 48 INS tests + 19 VAM tests pass

---

## What remains

### Priority 1: Pressure Poisson Solver
The `_solve_pressure_poisson` in PressurizedIMEXSolver returns zeros because
the constraint evaluation doesn't properly compute spatial derivatives of P.
Need to wire `compute_derivatives(mesh, ...)` for the pressure field.

### Priority 2: Multi-layer Basis
Redesign: `n_layers` should be a piecewise basis definition, not a model parameter.
The ProjectedModel should take a basis instance (which may be piecewise for multi-layer).

### Priority 3: Dimension Naming
dimension=2 means 2D physical space (xz), not mesh dimension.
Update all references for clarity.

### Priority 4: Legacy Cleanup
Old models moved to legacy/ but some code still references them.
The NumericalModel wrapper is deprecated but still used in some test scripts.

---

## Key files

| File | Status |
|------|--------|
| `ins_generator.py` | Expression with depth_integrate, map, classify, project, describe |
| `symbolic_integrator.py` | Symmetry, Chebyshev orthogonal, Legendre fast path |
| `projected_model.py` | Matrix caching by (basis, level) |
| `vam_projected_model.py` | VAM with flux, NC, eigenvalues, pressure source |
| `derived_system.py` | DerivedSystem, sme(), vam() factories |
| `symbolic_numerics.py` | regularize_model() — no NumericalModel wrapper |
| `solver_pressurized_imex.py` | Predictor-Poisson-Corrector IMEX |
