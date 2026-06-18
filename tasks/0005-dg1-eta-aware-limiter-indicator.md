# 0005 — DG1 limiter: η-aware indicator (limit h+b, not h)

**What:** Stop the slope limiters (vertex, ofdg) from perturbing
lake-at-rest. On the WET bump rest test they inject ~1e-1 m/s spurious
velocity even though the no-limiter DG1 run is now exact (0.0).

**Where:** `library/zoomy_firedrake/zoomy_firedrake/firedrake_solver.py`
— `_apply_slope_limiter` (and the ofdg jump indicator in
`_apply_ofdg_damping`). Test: `thesis/notebooks/malpasset/dg_bump_suite.py`
(wet, `BUMP_B_MAX=1.5`): rows `dg1_vertex` 6.6e-2, `dg1_ofdg` 1.1e-1.

**How (clear):** the limiters key on raw per-component jumps, but the
bathymetry `b` and depth `h` legitimately jump across faces where the
bed is curved — limiting them there fabricates a slope. Limit the free
surface `η = h + b` (reconstruct η, limit, recover h = η − b) instead of
`h`; or exclude the bathymetry-coupled jump from the indicator. The
solver already has `limiter_exclude_indices` (used for the stationary
`b` row) — extend that idea to an η-based depth indicator.

**Why:** a well-balanced base scheme (done — commit `8194733`) is
undone by a non-well-balanced limiter. Both must respect still water.

**Learned:**
- This is the same disease class as the IP-DG penalty bug (0004/8194733):
  anything that acts on the raw `b`/`h` jump rather than on η breaks
  equilibrium. The unweighted diffusion penalty was the O(1) version;
  the limiter is the ~1e-1 version.
- DG0 is unaffected (no in-cell slope to limit).
