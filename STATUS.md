# Phase 1 — overnight status

Latest push: `symbolic-rework@1f0bc7c4`.

## Tonight's work in 30 seconds

1. **Generic VAM at (M, N)** — `escalante2024_generic.py` derives eq (4)–(5) for any
   degree of u, w, p; `escalante2024_poisson_generic.py` derives the N×N Poisson
   reduction; `escalante2024_dispersion.py` reproduces eq (8) of the paper at (1, 2)
   and computes the next-order C²/(gH) at (2, 3) (a fresh result).
2. **Unified analysis library `zoomy_core.analysis`** — `PDESystem(equations, fields,
   time, space)` is the only model representation it touches; `linearise`,
   `plane_wave_dispersion`, `extract_quasilinear_pencil`, `sample_hyperbolicity` work
   on any `PDESystem`.  The same library handles SWE, SME, VAM (with constraints, via
   the rank-deficient pencil), 2D SWE, and DAE-style toy systems.  9 unit tests.
3. **SME hyperbolicity sampler** — `sme_hyperbolicity.py` recovers K&T 2019's L=2
   loss-of-hyperbolicity in extreme regimes; L=0, L=1 are 100% hyperbolic at every
   range tested (matches `u_0 ± √(gH + u_1²)` formula).
4. **VAM hyperbolicity** — `escalante2024_hyperbolicity.py` exercises the
   constraint-pencil path (M_t rank-deficient by exactly the number of algebraic
   closures) and reports 100% hyperbolic at the rest-vicinity sample range.

Things to discuss tomorrow:

- The dispersion analysis works in the **principal-symbol limit** (`k → ∞`); for
  finite k, the M_0 coupling adds a `1/(ik) M_0` term to the pencil that the current
  `sample_hyperbolicity` ignores.  For VAM at non-rest states M_0 may matter; let's
  decide if we want a finite-k sampling mode.
- The analysis library is purely sympy/scipy; it doesn't go through `BaseModel`.  If
  you want existing BaseModel-based models to plug in, we should write a thin
  `adapt_basemodel(model) -> PDESystem` adapter (~30 lines).
- 2D plane-wave dispersion takes `axis=0` or `axis=1` to pick a propagation
  direction; verified on 2D SWE.  For full dispersion-curve plotting we'd want a
  ``(k_x, k_y) → ω`` interface.

## Four literature matches

Pipeline reproduces published equations exactly across SWE + SME + multilayer SWE + VAM:

```bash
# SWE / SME up to level 2 — Kowalski & Torrilhon 2019 eqs (4.13), (4.14), (4.17)
python tutorials/sme/kt2019_verification.py --level 0   # ✓ MATCH
python tutorials/sme/kt2019_verification.py --level 1   # ✓ MATCH (×2)
python tutorials/sme/kt2019_verification.py --level 2   # ✓ MATCH (×3)

# Multilayer SWE — Aguillon, Hörnschemeyer, Sainte-Marie 2026 eq (5)
python tutorials/multilayer/aguillon2026_derivation.py --N 2   # ✓ MATCH (×6)
python tutorials/multilayer/aguillon2026_derivation.py --N 3   # ✓ MATCH (×9)
python tutorials/multilayer/aguillon2026_derivation.py --N 5   # ✓ MATCH (×15)
python tutorials/multilayer/aguillon2026_derivation.py --N 7   # ✓ MATCH (×21)

# VAM (non-hydrostatic) — Escalante, Morales de Luna, Cantero-Chinchilla,
# Castro-Orgaz 2024 eq (4)-(5) and the Poisson reduction eq (15)
python tutorials/vam/escalante2024_derivation.py        # ✓ MATCH (×6: continuity j=0/j=1, x-mom j=0, z-mom j=0, I_2 from KBCs, w_2 closure)
python tutorials/vam/escalante2024_poisson.py           # ✓ Poisson form verified (eq 15)
```

All four of your requested cases — **SWE, SME Lvl 2, multilayer SWE, VAM** — work and
match the literature you indicated.

## What the pipelines actually verify

* **K&T 2019**: full eq (4.13)/(4.14)/(4.17) including the `Q ∂_x V` non-conservative
  matrix at level 2.  Reference equations transcribed literally from the paper, then
  compared against pipeline output via expand-derivatives + continuity-substitution
  to a canonical normal form.
* **Aguillon 2026 eq (5)**: per-layer continuity, x-momentum, and passive tracer.
  Reference equations transcribed literally from the paper.  Compared via
  `expand_derivatives` + `subst({∂_t z_b: 0, G_{1/2}: 0, G_{N+1/2}: 0})` to canonical
  normal form.
* **Escalante 2024 VAM eq (4)–(5)**: the non-hydrostatic VAM model, derived in
  σ-coordinates with the polynomial ansatz `u ∈ P_1[ξ]`, `w, p ∈ P_2[ξ]`.  Galerkin
  projection against `φ_0` and `φ_1` (shifted Legendre on `[0, 1]`) gives the 6
  evolution equations; the 3 closure constraints come from the KBCs at the surface
  and the bottom (after eliminating `∂_t h` via j=0 continuity).  No hydrostatic
  assumption is made — z-momentum is kept and projected on the same basis.
* **Escalante 2024 VAM eq (15)** (the Poisson reduction): the splitting U^(k) =
  U^(k̃) − Δt T(U^(k), P^(k), ∂_x P^(k), ∂_x b) is substituted into the constraints
  I_1, I_2.  The script verifies that the result is **strictly linear** in
  (`p_0, p_1, ∂_x p_0, ∂_x p_1, ∂_xx p_0, ∂_xx p_1`) — i.e. the same Poisson-like
  2×2 system the paper claims, with leading coefficients `a_1 = b_1 = -Δt h`,
  `a_4 = -Δt h/3`, `b_4 = 0`.

The K&T matches go through the bug-3 closure on the symbolic-primitive layer (the
Wronskian asymmetry from a held `Derivative(Integral, t)` atom that step 9 missed —
fixed by an explicit fixpoint loop using `product_rule_forward`,
`distribute_derivative_over_add`, `subst(dt_h_relation)`, then re-projecting).

## Analytical-analysis library — `zoomy_core.analysis`

A new package (`library/zoomy_core/zoomy_core/analysis/`) wraps a single unified
representation `PDESystem(equations, fields, time, space)` and provides:

| Module | Responsibility |
|---|---|
| `pde_system.py` | `PDESystem` dataclass — list of sympy LHS expressions = 0 |
| `linearisation.py` | `linearise(system, base_state)` — generic O(ε) linearisation |
| `plane_wave.py`    | `plane_wave_dispersion` — `det M(ω, k) = 0` solver |
| `pencil.py`        | `extract_quasilinear_pencil` `(M_t, M_xa, M_0)`; symbolic + numerical generalised eigenvalues |
| `hyperbolicity.py` | `sample_hyperbolicity` — random-sample states, return fraction with all-real eigenvalues |

Ground rule: the package never reads model-specific attributes; every model
(SWE, SME, VAM, ML-SWE, future ML-VAM, …) becomes a `PDESystem` and
plugs into the same routines.  9 unit tests in
`library/zoomy_core/tests/analysis/test_analysis.py`.

**Verified results across the family** (everything in `tutorials/`):

| Model | Tutorial | Result |
|---|---|---|
| VAM dispersion (1,2) | `tutorials/vam/escalante2024_dispersion.py` | C²/(gH) = 12(H²k² + 12)/(H⁴k⁴ + 60 H²k² + 144) ✓ matches Escalante 2024 eq (8) |
| VAM dispersion (2,3) | same                                          | C²/(gH) = 24(H⁴k⁴ + 70 H²k² + 600) / (H⁶k⁶ + 264 H⁴k⁴ + 6480 H²k² + 14400) — fresh result |
| SME dispersion       | `tutorials/sme/sme_dispersion.py`            | C² = gH at rest for L = 0, 1, 2 ✓ |
| SWE eigenvalues      | (analysis test) | u_0 ± √(gH) ✓ |
| SME L=1 eigenvalues  | (analysis test) | u_0, u_0 ± √(gH + u_1²) ✓ matches K&T 2019 |
| SME L=2 hyperbolicity| `tutorials/sme/sme_hyperbolicity.py`         | 100% hyperbolic at typical ranges; 93.6% with `--U-range 5 --H-min 0.5 --g 1` (recovers K&T's loss-of-hyperbolicity regime) |
| VAM (1,2) hyperbolicity | `tutorials/vam/escalante2024_hyperbolicity.py` | M_t rank 5/9 (4 constraints), 100% hyperbolic, 3 finite eigenvalues per sample matching the dispersion polynomial degree |

## Files added

| Path | Purpose |
|---|---|
| `library/zoomy_core/zoomy_core/symbolic/` | New principled-primitive package (sp_safe, errors, auto_eval_guard, primitives_*, canonicalise) |
| `library/zoomy_core/tests/symbolic/` | 55 unit tests, all passing |
| `tutorials/sme/kt2019_verification.py` | K&T 2019 literature compare for SWE + SME-1 + SME-2 |
| `tutorials/sme/bug3_closure_via_primitives.py` | Bug-3 closure proof of concept (kept for reference) |
| `tutorials/sme/slim_walkthrough_primitives.py` | Full primitive-only port of slim_walkthrough (early draft, has equation-management wrinkles) |
| `tutorials/multilayer/aguillon2026_derivation.py` | Multilayer SWE literature verification |
| `tutorials/vam/escalante2024_derivation.py` | VAM eq (4)–(5) derivation (no hydrostatic; σ-coord Galerkin against shifted Legendre on [0,1]) |
| `tutorials/vam/escalante2024_generic.py` | VAM derivation generalised to arbitrary (M, N) — verified at (1,2) and (2,3) |
| `tutorials/vam/escalante2024_poisson.py` | VAM splitting → eq (15) Poisson form (linearity in P verified, leading coefficients extracted) |
| `tutorials/vam/escalante2024_poisson_generic.py` | Poisson splitting at arbitrary (M, N) — verified at (1,2), (2,3), (3,4) |
| `tutorials/vam/escalante2024_dispersion.py` | VAM dispersion via `zoomy_core.analysis` — matches paper eq (8); fresh result at (2,3) |
| `tutorials/vam/escalante2024_hyperbolicity.py` | VAM hyperbolicity sampler (DAE pencil) |
| `tutorials/sme/sme_builder.py` | Generic SME PDESystem builder for arbitrary level |
| `tutorials/sme/sme_dispersion.py` | SME dispersion at any level (matches gH at rest) |
| `tutorials/sme/sme_hyperbolicity.py` | SME hyperbolicity sampler (recovers K&T's L=2 loss-of-hyperbolicity) |
| `library/zoomy_core/zoomy_core/analysis/` | New unified analysis library (PDESystem, linearise, plane_wave, pencil, hyperbolicity) |
| `library/zoomy_core/tests/analysis/` | 9 unit tests for the analysis library |
| `tutorials/sme/slim_walkthrough.py` | Updated — step 14 closes bug 3 via primitives |

## Open follow-ups (in priority order)

1. **ML-SME — combining Aguillon multilayer + K&T per-layer Galerkin** (the natural next
   step you mentioned: the Heaviside-type basis composes with per-layer Legendre).  Math
   sketch:

       u(t,x,z) = Σ_α 1_α · (u_α + Σ_k α_{k,α} φ_k(ζ_α)),   ζ_α = (z − z_{α-1/2})/h_α

   The level-0-per-layer reduces to Aguillon eq (5), so we have a verifiable ground
   truth there.  Levels k≥1 are the new ML-SME closures.  Will attempt overnight; if
   I get a working L=1, will commit; otherwise will document the design.

2. **Phase 1b — full migration of slim_walkthrough.py to primitives**.  Steps 1–13
   currently still use legacy `IntegralTransform`/`ProjectBasisIntegrals`/etc.  Step 14
   uses the new primitive layer and closes bug 3.  Full migration drops the legacy
   composite ops entirely.

3. **Phase 1c — hard-break migration of remaining notebooks** (`symbolic_walkthrough.py`,
   `performance_walkthrough.py`, `vam_zeta_projection.py`, `zeta_projection.py`,
   `projected_model.py`, `derived_model.py`).  Each replaces legacy composite ops with
   primitive recipes; CI grep guard added.

## Notes on the multilayer derivation

The current multilayer derivation works in physical-z coords without the per-layer ζ_α
affine map you mentioned (each layer mapping to [0,1]).  For level-0-per-layer
(constant-in-z within a layer) this is fine — the integrals all have closed-form
constant or polynomial integrands.  The affine map will matter when extending to
level-1+ per layer (the ML-SME case), where the basis evaluation needs canonical
ζ ∈ [0, 1].

I deliberately did **not** substitute `∂_t h_α` via continuity in the derivation, per
your instruction.  The current form keeps `∂_t h_α` explicit, matching Aguillon's
presentation.

## What's NOT done

* Multi-layer SME at level ≥ 1 (architectural sketch landed at
  `tutorials/multilayer/ml_sme_prototype.py` — L=0 works, L≥1 raises
  `NotImplementedError` with a 6-step recipe in the docstring; the next-step
  extension once you decide the framework looks right).
* Full slim_walkthrough.py migration to primitives — partial (step 14 only).
* Other notebooks (`symbolic_walkthrough.py`, `performance_walkthrough.py`,
  `vam_zeta_projection.py`, `zeta_projection.py`, `projected_model.py`) untouched.

The legacy single-layer pipeline at steps 1–13 of slim_walkthrough.py still works fine;
the primitive-layer step 14 is what closes bug 3.  Nothing has regressed.

## ML-SME architectural notes

The framework you sketched composes cleanly: the Heaviside indicator selects each
layer's integration interval [z_{α-1/2}, z_{α+1/2}], and within each layer the
per-layer Legendre basis φ_k((z - z_{α-1/2})/h_α) expands the velocity profile.
Mass exchanges G_{α±1/2} arise naturally from the kinematic BC at every interface
(eq (5) of Aguillon).  At L=0 this collapses to MLSWE.

The six steps to wire up L=1 (already in the prototype's docstring):
1. Apply `Multiply(phi_k(zeta_alpha))` to the layer's momentum.
2. `product_rule_inverse` per term to expose ∂_v(φ·f).
3. `leibniz_general` / `fundamental_theorem` per term over the layer.
4. `subst(kbc)` at both interfaces (yields G_{α±1/2}).
5. `affine_change_of_variable` z → ζ·h_α + z_{α-1/2}, ζ ∈ [0, 1].
6. `function_expand(ansatz)` + `project_basis_integrand` + bug-3 fixpoint
   (`distribute_derivative_over_add` + `subst(dt_h_relation)` per layer).

All six primitives exist in `zoomy_core.symbolic`.  The remaining work is the
per-layer driver that composes them — paralleling
`tutorials/sme/kt2019_verification.py` but with layer-α bounds and the per-layer
∂_t h_α from layer-α continuity (instead of the single-layer ∂_t h from K&T).
There's a subtle question about whether the layer-α continuity
``∂_t h_α + ∂_x(h_α u_α) = G_{α+1/2} - G_{α-1/2}`` should be substituted into
``α_{k,α}·∂_t h_α`` residuals — the paper's instruction was *not* to substitute
``∂_t h_α`` via continuity for the L=0 case, so by analogy I'd leave the L≥1
residuals symbolic too until you decide.
