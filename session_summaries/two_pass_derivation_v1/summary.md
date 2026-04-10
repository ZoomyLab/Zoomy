# Session Summary: Two-Pass PDE Derivation v1

**Date**: 2026-04-03
**Branch**: develop
**Context**: Continuation of multi-session effort to build automated PDE generator that derives layer-averaged/moment-projected shallow water equations from 3D INS.

---

## What was accomplished this session

### 1. Tutorial notebook (`notebooks/two_pass_derivation.py`)

Created a jupytext notebook showcasing the two-pass derivation pipeline:

- **Pass 1** (`model_derivation.py`): `derive_shallow_moments(state, material)` produces `PreProjectedEquations` with tagged terms (temporal, flux, NC, source). Shows Newtonian vs Inviscid material switching.
- **Pass 2** (`projected_model.py`): `ProjectedModel(pre, basis_type, level)` takes PreProjectedEquations + basis, computes basis matrices via `SymbolicIntegrator`, applies M^{-1}, produces `Model`-compatible class.
- Compares Legendre, SplineBasis, Chebyshev U, and GalerkinBasis at level 2
- Shows mass matrix structure (diagonal vs non-diagonal), mean coefficients, flux/pressure/NC/source terms
- Demonstrates building an `InclinedPlaneProjected` model with gravity + viscosity + slip friction

### 2. Bug fix: Raw Galerkin projection weights (`projected_model.py`)

**The bug**: When computing hydrostatic pressure, topography NC, and gravity source, the code used `c_mean[l]` as the raw Galerkin projection vector before applying `M^{-1}`. This is wrong for non-diagonal M.

**Why it was invisible for Legendre**: For diagonal M, `M^{-1} @ c_mean = c_mean` (since c_mean = [1,0,...] and M[0,0]=1 for Legendre). Pure coincidence.

**What went wrong for splines**: `M^{-1} @ c_mean = [6, 0, 6]` instead of `[1, 1, 1]`. This made gravity 6x too large, causing Linf=2.0 on the inclined plane.

**The fix**: Added `_phi_int[l] = (M @ c_mean)_l = integral(phi_l, domain)` as a precomputed quantity. This is the correct raw Galerkin projection of a constant. Now `M^{-1} @ _phi_int = c_mean` for all bases.

Changed in 3 places:
- `hydrostatic_pressure()`: `raw_p` uses `phi_int[l]` instead of `c_mean[l]`
- `nonconservative_matrix()`: `raw_topo` uses `phi_int[l]` instead of `c_mean[l]`
- 2D topography section: same fix

### 3. Inclined plane experiment (`tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py`)

Ran Legendre vs SplineBasis at levels 1-4 with IMEX solver.

**Results after fix**:

| Basis | L1 | L2 | L3 | L4 |
|-------|------|------|------|------|
| Legendre | 1.25e-01 | **1.05e-05** | 9.56e-06 | 9.03e-06 |
| SplineBasis | 1.25e-01 | 3.13e-02 | 1.39e-02 | 7.82e-03 |

- Legendre L2: exact parabolic profile recovery (spectral accuracy)
- SplineBasis: O(h^2) convergence for linear hat functions (expected)
- L1: identical for both (2 DOF, both approximate parabola as linear)

### 4. Pre-existing test failure

`test_generated_model.py::TestNonconservativeVerification::test_nc_1d_level1` fails with `NC[3,3,0] mismatch: diff=-2*q2/q1`. This is in the OLD `GeneratedShallowModel` (not `ProjectedModel`) and was not introduced by this session's changes.

---

## Files modified this session

- **`library/zoomy_core/zoomy_core/model/models/projected_model.py`** -- added `_phi_int`, fixed pressure/NC raw vectors
- **`notebooks/two_pass_derivation.py`** -- NEW tutorial notebook
- **`tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py`** -- NEW inclined plane test with ProjectedModel

---

## What was interrupted

The session ended while running `test_generated_model.py` (completed in background, 12 passed / 1 failed pre-existing). The inclined plane sweep completed successfully. All deliverables are done.

---

## Context from previous sessions (carried forward)

The full history spans multiple sessions covering:
1. INS generator (`ins_generator.py`): `StateSpace`, `FullINS`, `Expression`, `IBPResult`, materials, assumptions
2. Symbolic integrator (`symbolic_integrator.py`): unified integration with strategy dispatch
3. Basis functions (`basisfunctions.py`): Legendre, Chebyshev, SplineBasis, GalerkinBasis, `mean_coefficients()`
4. Numerical model (`numerical_model.py`): regularization, opaque functions, wet/dry treatment
5. Solver infrastructure: `GeneratedModelSolver`, IMEX, opaque `max_wavespeed`
6. Two-pass architecture: Pass 1 (basis-independent) -> Pass 2 (basis-specific with M^{-1})

---

## Next task: Three-phase architecture redesign

The user has requested the following redesign before continuing computation. This is the verbatim task:

> 1. In the state space, the dimension is wrong. If we have xz, it is dimension=2. The notation for dimension in the projected version is misleading.
>
> 2. I do not want to even need to write assumptions for tau_xy if the state space is dimension 2, then we should naturally only have tau_xx, tau_xz, tau_zx and tau_zz, the rest should never be there and needs definition.
>
> 3. I think we need 3 phases:
>    - Phase 2 should be where we take the simplified INS into the zeta-space of free surface flows, but yet without dimensional reduction. This means, we apply the kinematic boundary conditions, and we, formally, without yet calculating out the integrals, project the equation (we do not need to know how many basis functions we have, we just compute with respect to a general c(zeta) phi_k(zeta) and define it later). This is important because we can build in the integration by parts here and build in the boundary conditions on an abstract level and see the resulting tex of the PDE. Also, sorting the terms into the flux, dflux, nc-flux, source, ... can be done at this level already.
>    - Lastly, we have part 3, where we assume a basis (including all the multi-layer stuff) and have the compute heavy stuff. The stuff before was hopefully light (check that, otherwise we need to cache this). So we cache the matrices of part 3 in a smart way, maybe by e.g. hash so we can make sure we can separate different basis functions. Then, we also separate according to the different projection terms. However, they are model agnostic, too, as e.g. the int c phi_i phi_j phi_k will be there regardless of the assumptions. So we should not separate this by model assumption, only by integration operation. The final model after integration should however, be saved (symbolic operators).

### Concrete sub-tasks for next session:

1. **Fix StateSpace dimension semantics**: xz = dimension 2, xyz = dimension 3. The "horizontal dimension" concept (1D vs 2D shallow water) should be a separate attribute.
2. **Fix stress tensor in StateSpace**: Only create tau components that exist for the given dimension (e.g., dimension=2 -> tau_xx, tau_xz, tau_zx, tau_zz only). No phantom tau_xy, tau_yy etc.
3. **Redesign to 3-phase architecture**:
   - **Phase 1**: INS + material + hydrostatic (same as current Pass 1, light)
   - **Phase 2** (NEW): Map to zeta-space, apply kinematic BCs, IBP, project onto abstract `c(zeta) phi_k(zeta)` without specifying basis or level. Tag terms into flux/NC/source. Output: symbolic PDE in zeta-space with abstract test functions. Should be displayable as LaTeX.
   - **Phase 3**: Choose basis + level, evaluate integrals, compute matrices, apply M^{-1}. Cache matrices by integration operation (not by model assumption). Cache final symbolic model.
4. **Caching strategy**: Basis matrices keyed by (basis_name, level, integration_type), not by model. Final assembled model cached separately.
