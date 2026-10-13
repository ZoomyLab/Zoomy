# Non-dimensional shallow scaling: design (agreed 2026-10-13, not implemented)

Goal: replace the hand-made truncations of the moment-model derivation (`moment_scaling`'s in-plane
drop, the hydrostatic step's dropped vertical inertia, `small_slope_scaling`, the `nu d_e d_e w` drop
of `Stress.full`) by one generic, order-based truncation that works on the general stress tensor
before any material law, and a closure that is checked against that scaling when it is inserted.

## Pipeline (physical coordinates, before the sigma map and the projection)

```python
m.add_equation(Mass(m)); m.add_equation(Momentum(m))   # full 3-D balance, sigma = -p I + tau, all tau_ij opaque
m.apply(FreeSurfaceNonDim())                            # 1  scales + groups (thesis defaults)
m.apply(ShallowScaling(horizontal=0, vertical=0))       # 2  drop beyond the chosen eps order, closure-free
m.apply(InsertClosure(closures))                        # 3  material law in the SCALED frame + consistency check
m.apply(ShallowScaling(horizontal=0, vertical=0))       # 4  same order: drops the closure's own sub-leading terms
m.apply(Dimensionalize())                               # 5  back to physical variables
m.apply(PDETransformation({z: (zeta, Eq(z, b + h*zeta))}))   # sigma map, then the projection
```

1. **`NonDimensionalize(scales, groups)`**: the generic operator. `scales` maps every coordinate,
   field, parameter and each of the stress components to a scale; `groups` are the user's target
   dimensionless numbers (definitions). Every symbol declares its dimension (L, T, M); a
   Buckingham-Pi check (null space of the dimension matrix) verifies that the groups are
   dimensionless, independent and complete, so every parameter can be written through scales and
   groups. The operator never chooses groups (the choice is not unique); on request it suggests a basis.
   **`FreeSurfaceNonDim`** subclasses it with the thesis choices (`20_methodology.md`):
   eq. (scaling-variables) x, y ~ L; z, b, s ~ H; t ~ L/U; u, v ~ U; w ~ eps U; eq. (pressure-scaling)
   p = rho g (s - z) + q, q ~ rho U^2; eq. (sme-stress-scaling), the closure-free anisotropic ansatz
   after Kowalski 2019: tau_xz, tau_yz ~ S, tau_xx, tau_xy, tau_yy, tau_zz ~ eps S; groups eq. (fr-de):
   eps = H / L, Fr, De (the stress group with S).
2. **`ShallowScaling(order per equation, regime)`**: expands in eps and drops terms beyond the order;
   `regime` gives the orders of the groups (default Fr, De = O(1); Marche 2007 eq. (5.1) for the
   friction coefficients). Horizontal and vertical momentum keep separate orders: SME-type (vertical
   O(1): hydrostatic) vs VAM-type (vertical keeps eps^2).
3. **`InsertClosure`**: the material law (Newtonian, Bingham, k-epsilon, ...) is substituted in the
   scaled variables, tau_ij = scale_ij * tau_hat_ij(u_hat, w_hat, ...). **Consistency check:** every
   tau_hat_ij the closure produces must be O(1) in eps (no negative power of eps), i.e. the closure must
   be compatible with the ansatz of step 1; the ratio of the closure's own scale to S fixes the stress
   group (Newtonian: S = rho nu U / H gives the thesis' 1 / De). A component larger than assumed (e.g.
   an in-plane stress of order S) is an error naming the component, not a silent truncation.
   Boundary traction closures (bed friction, Navier slip, stress-free surface) go through the same
   check with the regime's orders.
4. The second truncation at the same order removes the closure's sub-leading terms consistently, e.g.
   Newtonian tau_hat_xz = (d_z u_hat + eps^2 d_x w_hat) / De keeps only d_z u_hat at order 0, without
   anyone naming that term.
5. **`Dimensionalize`**: the inverse substitution; the result is the dimensional model the
   sigma map and the projection take as now.

## Acceptance

- Thesis regime, order 0: SME (levels 0-3), VAM, ML-SME, ML-VAM with Newtonian / Navier slip /
  stress-free / Manning / k-epsilon derive byte-identically to today's models (which replaces the hand
  rules one by one).
- Next order with Newtonian: equals today's `Stress.full`; the `nu d_e d_e w` term of the vertical row
  is then kept or dropped by its order, not by hand.
- The Pi check rejects an incomplete or dependent group set with a message naming the missing group.

## Priority

User priority 3 (after: QR codes end to end; backend containers + matrix + thesis-case reproduction).
