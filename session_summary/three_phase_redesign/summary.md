# Session Summary: Three-Phase Redesign

**Date**: 2026-04-03
**Branch**: develop (uncommitted changes)

---

## What was accomplished

### Previous session (carried forward)

1. **Tutorial notebook** (`notebooks/two_pass_derivation.py`) — showcases two-pass derivation pipeline
2. **Bug fix in `projected_model.py`** — `_phi_int` fix for raw Galerkin projection weights (M @ c_mean vs c_mean)
3. **Inclined plane experiment** (`tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py`) — Legendre exact at L2, SplineBasis converging at O(h^2)

### This session

4. **StateSpace dimension semantics fixed** (`ins_generator.py`):
   - `dimension=2` now means xz plane (was `dimension=1`)
   - `dimension=3` means xyz space (was `dimension=2`)
   - Added `state.horizontal_dim` property (= dim - 1)
   - Added `state.has_y` property
   - `dimension=1` now raises ValueError

5. **Stress tensor fixed** (`ins_generator.py`):
   - `dimension=2` (xz): only creates `tau_xx, tau_xz, tau_zx, tau_zz` (4 components)
   - `dimension=3` (xyz): creates all 9 components
   - No more phantom `tau_xy`, `tau_yy` etc. for 2D case

6. **All downstream code updated** to use `state.has_y` instead of `state.dim > 1`:
   - `FullINS`: continuity, x/y/z_momentum, stress_divergence, equations
   - `Newtonian`: only builds stress rules for existing tau components
   - `KinematicBCBottom`, `KinematicBCSurface`: boundary condition checks
   - `derive_shallow_moments()`, `_tag_momentum()`: term tagging
   - `ProjectedModel.__init__`: uses `pre.horizontal_dim` for Model.dimension and n_vars

7. **Files updated with `StateSpace(dimension=2)`** (was `dimension=1`):
   - `projected_model.py` (docstring)
   - `model_derivation.py` (docstring)
   - `notebooks/two_pass_derivation.py`
   - `notebooks/pde_generator_design.py`
   - `tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py`

8. **Verified**: Legendre L2 flux/pressure match previous results after dimension change.

---

## What still needs updating (dimension=1 → dimension=2)

Two files still have `StateSpace(dimension=1)`:
- `notebooks/two_pass_derivation.ipynb` (the compiled ipynb, regenerate from .py)
- `tests/unit/zoomy_core/test_ins_generator.py` — **must be updated**

---

## What remains to do (three-phase redesign)

### Completed
- [x] Fix StateSpace dimension semantics (dim=2 for xz, dim=3 for xyz)
- [x] Fix stress tensor components (only create what exists)

### Not started

**Phase 2: Abstract zeta-space projection** (the main new piece):
- Map simplified INS into normalized zeta-space [0,1] of free surface flows
- Apply kinematic BCs and IBP at the abstract level
- Project onto abstract `c(zeta) * phi_k(zeta)` without choosing basis or level
- Tag terms into flux/NC/source
- Output displayable LaTeX PDE
- This should be lightweight (no numerical integration)

**Phase 3: Basis-specific integration with caching**:
- Takes Phase 2 output + basis + level
- Compute basis matrices (M, A, D, B, phib) via SymbolicIntegrator
- Cache matrices by `(basis_name, level, integration_type)`, NOT by model assumption
- The matrix integrals (e.g., triple product) are model-agnostic
- Apply M^{-1}, produce final Model
- Cache final symbolic model separately

**Update notebook and tests**:
- Update `two_pass_derivation.py` to show 3-phase pipeline
- Update `test_ins_generator.py` for `dimension=2`
- Verify inclined plane still works end-to-end

---

## User's verbatim task description

> 1. in the state space, the dimension is wrong. if we have xz, it is dimension=2. The notation for dimension in the projected version is misleading.
> 2. I do not want to even need to write assumptions for tau_xy if the state space is dimension 2, then we should naturally only have tau_xx, tau_xz, tau_zx and tau_zz, the rest should never be there and needs definition.
> 3. I think we need 3 phases, phase 2 should be where we take the simplified INS into the zeta-space of free surface flows, but yet without dimensional reduction. This means, we apply the kinematic boundary conditions, and we, formally, without yet calculating out the integrals, project the equation (we do not need to know how many basis functions we have, we just compute with respect to a general c(zeta) phi_k(zeta) and define it later. this is important because we can build in the integration by parts here and build in the boundary conditions on an abstract level and see the resulting tex of the PDE. Also, sorting the terms into the flux, dflux, nc-flux, source, .. can be done at this level already. lastly, we have part 3, where we assume a basis (including all the multi-layer stuff) and have the compute heavy stuff. the stuff before was hopefully light (check that, otherwise we need to cache this). so we cache the matrices of part 3 in a smart way, maybe by e.g. hash so we can make sure we can separate different basis functions. then, we also separate according to the different projection terms. However, they are model agnostic, too, as e.g. the int c phi_i phi_j phi_k will be there regardless of the assumptions. so we should not separate this by model assumption, only by integration operation. the final model after integration should however, be saved (symbolic operators).

---

## Key files

| File | Status |
|------|--------|
| `library/zoomy_core/zoomy_core/model/models/ins_generator.py` | Modified (dimension + tau fix) |
| `library/zoomy_core/zoomy_core/model/models/model_derivation.py` | Modified (has_y, horizontal_dim) |
| `library/zoomy_core/zoomy_core/model/models/projected_model.py` | Modified (phi_int fix + horizontal_dim) |
| `library/zoomy_core/zoomy_core/model/models/symbolic_integrator.py` | Unchanged |
| `library/zoomy_core/zoomy_core/model/models/basisfunctions.py` | Unchanged |
| `notebooks/two_pass_derivation.py` | Modified (dimension=2) |
| `notebooks/pde_generator_design.py` | Modified (dimension=2) |
| `tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py` | Modified (dimension=2) |
| `tests/unit/zoomy_core/test_ins_generator.py` | **NEEDS UPDATE** (still dimension=1) |
