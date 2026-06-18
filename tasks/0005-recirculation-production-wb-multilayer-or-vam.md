# 0005 — Production WB for the recirculating (reversing) BBSM13 equilibrium

**What:** a well-balancing that holds the **reversing** BBSM13 moving equilibrium
without the high-moment Gibbs order ceiling that limits `projected_bernoulli`.

**Where:** `library/zoomy_core/zoomy_core/model/models/` — either
`ml_sme.py`/`ml_swe.py` (multilayer) or `vam.py` (non-hydrostatic).

**How:** the obstruction is a velocity **jump** at the turning depth that a single
smooth Legendre basis can only represent `~L^{-1/2}` (Gibbs). Two escapes, both
verified in this thread:
- **Distinct-density multilayer** — split at the dividing streamline so each layer
  is single-signed and smooth; the interface becomes a physical (admissible)
  vortex sheet carried by two separate bases → no Gibbs. `MLSME`/`MLSWE` exist but
  are **same-density** (depth-fraction, mass-exchange) — needs **per-layer density**
  (reduced gravity) so the piezometric potential jumps at the interface.
- **VAM** (`vam.py`) carries an independent non-hydrostatic pressure DOF — the
  "pressure as a free variable" route. Heavier/different model.

**Why:** single-layer hydrostatic SME structurally cannot hold a sliding interface;
`projected_bernoulli` order-1 *degrades* at L7–8 on β=1, and Gauss/quadrature does
not help (it's representation, not integration).

**Learned:** the jump `2√(2gε)` is INTRINSIC-vertical (persists as Δσ→0), its size
∝ the per-face bed step ∝ Δx (so it vanishes under mesh refinement but caps the
moment hierarchy at fixed grid). For one fluid, hydrostatic continuity pins all
"layers" to one surface potential `gη*` → the multi-phase split alone gains no DOF
unless the densities differ. This is research-grade.
