# 0004 — DG1 wet/dry shoreline: exact balance + positivity

**What:** Make the Firedrake DG(1) SWE scheme preserve lake-at-rest and
positivity across the moving wet/dry shoreline. Right now partially-wet
cells at the waterline inject spurious momentum (|u| 0.2–1.9 m/s on the
dry-hump rest test) and the cell mean of h dips negative at the front.

**Where:**
- `library/zoomy_firedrake/zoomy_firedrake/firedrake_solver.py` —
  `_apply_positivity_scaling` (Zhang–Shu θ-scaling, currently nodal only)
  and the hydrostatic-reconstruction path in the Riemann solver.
- Test: `thesis/notebooks/malpasset/dg_bump_suite.py` with
  `BUMP_B_MAX=2.5` (dry hump) — 7–40 s/case.

**How (known approach):** implement the Xing–Zhang–Shu partially-wet-cell
treatment (Xing, Zhang & Shu 2010, *Adv. Water Resour.* 33:1476):
redefine the cell-bottom over the wet sub-region so the still-water
reconstruction stays flat, and apply the PP scaling to the *reconstructed*
depth at the quadrature/limiter nodes, not the raw nodal values. Pair
with `PositiveNonconservativeRusanov` (HLL is out of scope of the
Xing–Zhang cell-mean positivity proof — it gave h̄<0 at fronts).

**Why:** DG(1) is wanted for less-diffusive Malpasset results, but it is
not paper-grade until the shoreline preserves equilibrium. DG(0) already
does this exactly and is production-ready.

**Learned:**
- Mass conservation HIDES the defect (broken DG1 baseline had
  ΔV/V0~1e-13 while min h = −32.6 m). Always pair a mass plot with an
  unfiltered min-h plot — `malpasset_mass_minh_audit.py`.
- On the dry hump, ofdg is *worse* than the vertex limiter (1.9 vs
  0.18 m/s) — the oscillation damper perturbs steep partially-wet cells.
- `library/zoomy_firedrake` is owner-less; change it via a §7 intent in
  the `malpasset-firedrake` hub thread.
