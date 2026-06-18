# 0006 — Chase the secondary ~1e-9 WB floor at high moment level (no-backflow, Gauss)

**What:** with Gauss quadrature the no-backflow WB error tracks the projection
floor through L6 but then bottoms at ~1e-9 for L7–8 (vs floor ~1e-13). Find and
lower that secondary, non-quadrature floor.

**Where:**
- `library/zoomy_core/zoomy_core/fvm/bernoulli_wb.py` — the η* Newton (its
  tolerance / iteration cap in `_reconstruct_projected_bernoulli`).
- `library/zoomy_core/zoomy_core/fvm/solver_numpy.py` — time integration (RK/CFL),
  Rusanov dissipation.

**How:** isolate by perturbing one knob at a time on the no-backflow case (Gauss,
L8, t=10): (a) tighten the η* Newton residual tol; (b) shrink CFL / raise RK order;
(c) reduce Rusanov dissipation on the (near-equilibrium) faces. Whichever lowers
L8 below 1e-9 is the cause. Reproduce with `bbsm13_quadrature_study.py` extended to
sweep the knob.

**Why:** to let L7–8 reach the projection floor; currently L8 = 8.4e-9 (still 380×
below the old trapezoid stall). Diminishing returns — 1e-9 is already excellent —
so low priority unless deep-L convergence is needed for a figure/claim.

**Learned:** it is NOT quadrature (Gauss is spectral and removes the dominant
1e-6 trapezoid floor). It only appears *after* the quadrature floor is gone, so it
was invisible with the default trapezoid. Likely time-integration/Newton-tol.
