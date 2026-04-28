# DAE / IMEX RK references (for future implementation)

When implementing the direct DAE solver for VAM (no Chorin pressure splitting),
**pin the algorithm to one of these references** — copy the Butcher tableau /
algorithm verbatim and cite the DOI in the docstring.

## Foundational

- **Kennedy & Carpenter (2003)** — foundational additive Runge–Kutta (ARK)
  IMEX schemes for convection–diffusion–reaction equations.
  DOI: [10.1016/S0168-9274(02)00138-1](https://doi.org/10.1016/S0168-9274(02)00138-1)
  Provides ARK2, ARK3, ARK4, ARK5 schemes (3rd–5th order).

- **Pareschi & Russo (2005)** — IMEX RK for hyperbolic systems with stiff
  relaxation; introduces the SSP-explicit + L-stable-DIRK split.
  DOI: [10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
  Asymptotic-preserving in the zero-relaxation limit.

## Most directly applicable to VAM

- **Gardner, Guerra, Hamon, Reynolds, Ullrich, Woodward (2018)** — "Implicit–
  explicit (IMEX) Runge–Kutta methods for non-hydrostatic atmospheric models"
  (Geosci. Model Dev.).
  DOI: [10.5194/gmd-11-1497-2018](https://doi.org/10.5194/gmd-11-1497-2018)
  Recommended pair-up: **ARS343** and **ARK324** for our use case (their best
  performers). They use Newton–Krylov with GMRES for the implicit stage solve.
  Same physical structure as VAM: stiff acoustic / pressure waves treated
  implicitly, advective transport explicit.

## Software references (algorithm transcription targets)

- **SUNDIALS / IDA** — variable-order, variable-coefficient BDF in
  fixed-leading-coefficient form, Newton with direct or Krylov (GMRES,
  BiCGStab, TFQMR) linear solvers.
  https://computing.llnl.gov/projects/sundials/ida
  Useful as a sanity-check reference for BDF-style DAE integration.

## Recommendations for implementation order

1. Start with **Pareschi–Russo IMEX-SSP3** (their 3rd-order SSP-explicit +
   3-stage DIRK). Simplest of the three, well-documented, self-contained.
2. Use it on a 1D toy DAE to verify (pendulum, simple acoustic constraint).
3. Verify on VAM(1, 2) against the existing SplittingSolver.
4. Only then consider the more advanced **Gardner ARS343** for performance.
5. **Always cite the DOI** in the docstring of the time-integrator class so
   the implementation is auditable.

## Algorithm cheat sheet (Pareschi–Russo IMEX-SSP3)

Split the ODE/DAE: `du/dt = f(u) + g(u)` with `f` non-stiff (advection) and
`g` stiff (constraint / pressure / source).

For an s-stage IMEX scheme with tableaus `(A, b, c)` explicit and
`(A_tilde, b_tilde, c_tilde)` implicit:

```
for i = 1..s:
    K_i = u^n + Δt * Σ_{j<i} A[i,j] * f(K^*_j)
                + Δt * Σ_{j≤i} A_tilde[i,j] * g(K^*_j)
    [solve for K^*_i implicitly using Newton]
u^{n+1} = u^n + Δt * Σ_i b[i] * f(K^*_i)
              + Δt * Σ_i b_tilde[i] * g(K^*_i)
```

The implicit step at stage `i` is a nonlinear solve (Newton) where the
constraint `G(u) = 0` enters the residual. For our DAE with singular `M_t`,
the algebraic-row of the residual is the constraint itself, evaluated at the
stage-implicit value.
