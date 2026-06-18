# 0006 — Malpasset DG1 guideline run on the fixed scheme

**What:** Produce the Malpasset DG(1) full-domain t=100 result on the
SIPG-fixed scheme (commit `8194733`) — GIF + mass/min-h audit. The
previous guideline GIF (2026-06-12 07:47) is PRE-fix; the post-fix run
was started but killed at session close, so no post-fix Malpasset DG1
visual exists yet.

**Where / how:** config in
`tutorials/firedrake/malpasset_viscous_v2.py`; run
`MalpassetSWE`, `limiter="ofdg"`, `positivity_limiter=True`,
`riemann_solver_cls=PositiveNonconservativeRusanov`, `CFL=0.4`,
env `MALPASSET_EV_GATE=0`, `snapshots=60`, `time_end=100`. ≈80 min in
the `containers/zoomy_firedrake/zoomy_firedrake_dev.sif` apptainer
(`apptainer exec --bind … --env ZOOMY_ROOT=…,OMP_NUM_THREADS=1`). Then:
- GIF: `thesis/notebooks/malpasset/render_malpasset_gif.py
  outputs/firedrake_viscous_v2_dg1_guideline <out.gif> "DG1 guideline"`
  (run with the host `zoomy` micromamba python — the SIF has no pyvista).
- Audit: `thesis/notebooks/malpasset/malpasset_mass_minh_audit.py
  outputs/firedrake_viscous_v2_dg1_guideline dg1_guideline`.

**Why:** close the DG1 deliverable loop and give the user a current-scheme
visual. BUT note this still rides on the open shoreline issue (0004) —
expect residual front artifacts until that lands; this run is the
"where are we now" snapshot, not the final paper figure.

**Learned:**
- `clean_directory=True` wipes the output dir at startup, so a killed
  run leaves NO frames — don't rely on a half-finished run's VTUs.
- The renderer globs must exclude stale MPI-rank files
  (`simulation_0_3.vtu`); use `re.fullmatch(r"simulation_\d+\.vtu", …)`
  (already fixed in `render_malpasset_gif.py`).
- Run the long sim in the SIF, render/audit on the host env (pyvista).
