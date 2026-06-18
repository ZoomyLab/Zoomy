# 0010 — Re-assess the JAX VAM pressure-BC change (zoomy_jax `56eff9a`) after the split fix

**What:** Decide whether zoomy_jax `56eff9a` (apply pressure-mode BCs in the
elliptic solve — evaluate P boundary-face values via `rt_bc` instead of
`u_bf=None` / Neumann-zero) is needed, and keep or revert it.

**Where:**
- `library/zoomy_jax/zoomy_jax/fvm/solver_chorin_vam_jax.py` —
  `_bc_boundary_face_values` (added, ≈ line 282), the `bf=…` wiring in
  `setup_simulation`, and `_refresh_pressure_aux` (passes `u_bf` from the BC,
  ≈ line 403). This is the whole of `56eff9a`.

**How:** Once task **0009** lands (predictor is mass-conservative and the outflow
stops over-filling), re-run `steffler_vam_jax.py` both with and without
`56eff9a` and compare the outflow depth and the elliptic residual. If the depth
pins at 0.061 either way, the change is inert → revert it (keep the hot path
simple). If it measurably improves the outflow-depth enforcement through the
pressure step, keep it and add a one-line note to the case README.

**Why:** `56eff9a` was committed as a candidate fix for the over-fill but did NOT
fix it — the over-fill is the predictor mass leak (task 0009), not the pressure
boundary condition. It should not masquerade as the fix. It may still be the
*correct* treatment of the pressure BC, but that has to be judged on its own
merits after 0009, not assumed.

**Learned:** Disabling the corrector did not help the over-fill either; the leak
is upstream in the predictor. The pressure modes on the bend are physically tiny
(near-hydrostatic), so the pressure-BC change has little leverage on the depth —
another reason to suspect it is inert for this case.
