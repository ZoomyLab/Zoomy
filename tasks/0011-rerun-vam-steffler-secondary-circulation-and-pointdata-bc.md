# 0011 — Re-run the VAM steffler bend + point-data BC interpolation for plots (blocked on 0009)

**What:** Two coupled follow-ups, both blocked on task 0009:
1. Re-run the VAM (non-hydrostatic) dim=3 Ghamry & Steffler 270° bend once the
   predictor conserves mass, confirm the outflow pins at 0.061 m, and produce the
   secondary-circulation cross-section plot (compare to SME and to Fig 7).
2. Build a `interpolate_point_data_with_bcs` post-processing helper that lifts
   cell-centred data to point data **respecting the boundary conditions**, and
   re-use it for (a) the 3-D reconstruction and (b) a standalone 2-D enricher.

**Where:**
- `thesis/notebooks/steffler_jax/steffler_vam_jax.py` — the VAM Chorin driver
  (predictor → GMRES pressure → corrector via `ChorinSplitVAMSolverJax`).
- `thesis/notebooks/steffler_jax/cross_section.py` — secondary-flow plot; assumes
  the SME 6-row state, needs adapting for VAM's 10-row state
  `[b,h,q_x_0,q_x_1,q_y_0,q_y_1,r_0,r_1,P_0,P_1]`.
- The new helper belongs in a post-processing module reused by the 3-D generator.

**How:**
1. After 0009: `JAX_PLATFORMS=cpu python steffler_vam_jax.py --t-end <developed>`
   and check `h` at the outflow line. Expect ≈0.061 m, P-modes active but small.
2. `interpolate_point_data_with_bcs(cell_values, mesh, bcs)`: for each boundary
   face use the BC's `face_value`/`face_state` (e.g. `Wall` reflects normal
   momentum → enforces no-penetration at the wall) when averaging cell centres to
   vertices, instead of a naive cell-to-point average. Post-processing only.

**Why:** This is the actual deliverable — the secondary circulation on the bend.
It cannot be trusted until the mass leak (0009) is fixed (the over-filled depth
distorts the velocity field). The point-data helper fixes the *plot* artefacts
the user flagged: striped (non-continuous) colouring, streamlines that ignore the
wall no-penetration, and streamlines that appear to bump into the bed at the
90°/270° stations.

**Learned:**
- VAM run pattern (NOT the SME `HyperbolicSolver`):
  `sm.initial_conditions=…; split = model.chorin_split(dt, system_model=sm);
  ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press,
  split.SM_corr).setup_simulation(mesh)` then `run_jit_steps` chunks.
- VAM `w` is algebraic from the `r` moments + the bed kinematic top mode
  (`vam_flow` in `steffler_vam_jax.py`); no `interpolate_to_3d` needed for VAM.
- The secondary-flow magnitude is also sensitive to the bulk-stress→moment-source
  closure (see task 0005/0006 family and the `STEFFLER_VAM_3D_CLOSURES.md` note),
  separate from the mass-conservation bug.
