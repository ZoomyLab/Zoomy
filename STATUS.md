# Phase 1 — overnight status

Pushed to `symbolic-rework@ffa0c325`.

## Three literature matches

Pipeline reproduces published equations exactly across SWE + SME + multilayer SWE:

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
```

All three of your requested cases — **SWE, SME Lvl 2, multilayer SWE** — work and match
the literature you indicated.

## What the pipelines actually verify

* **K&T 2019**: full eq (4.13)/(4.14)/(4.17) including the `Q ∂_x V` non-conservative
  matrix at level 2.  Reference equations transcribed literally from the paper, then
  compared against pipeline output via expand-derivatives + continuity-substitution
  to a canonical normal form.
* **Aguillon 2026 eq (5)**: per-layer continuity, x-momentum, and passive tracer.
  Reference equations transcribed literally from the paper.  Compared via
  `expand_derivatives` + `subst({∂_t z_b: 0, G_{1/2}: 0, G_{N+1/2}: 0})` to canonical
  normal form.

The K&T matches go through the bug-3 closure on the symbolic-primitive layer (the
Wronskian asymmetry from a held `Derivative(Integral, t)` atom that step 9 missed —
fixed by an explicit fixpoint loop using `product_rule_forward`,
`distribute_derivative_over_add`, `subst(dt_h_relation)`, then re-projecting).

## Files added

| Path | Purpose |
|---|---|
| `library/zoomy_core/zoomy_core/symbolic/` | New principled-primitive package (sp_safe, errors, auto_eval_guard, primitives_*, canonicalise) |
| `library/zoomy_core/tests/symbolic/` | 55 unit tests, all passing |
| `tutorials/sme/kt2019_verification.py` | K&T 2019 literature compare for SWE + SME-1 + SME-2 |
| `tutorials/sme/bug3_closure_via_primitives.py` | Bug-3 closure proof of concept (kept for reference) |
| `tutorials/sme/slim_walkthrough_primitives.py` | Full primitive-only port of slim_walkthrough (early draft, has equation-management wrinkles) |
| `tutorials/multilayer/aguillon2026_derivation.py` | Multilayer SWE literature verification |
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

* Multi-layer SME (per-layer Legendre + Heaviside outer basis) — attempting next.
* Full slim_walkthrough.py migration to primitives — partial (step 14 only).
* Other notebooks (`symbolic_walkthrough.py`, etc.) untouched.

The legacy single-layer pipeline at steps 1–13 of slim_walkthrough.py still works fine;
the primitive-layer step 14 is what closes bug 3.  Nothing has regressed.
