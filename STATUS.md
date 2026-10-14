# STATUS: SME-VOF live coupling on the symbolic solvers (branch dev/smevof-live)

Goal: GUI sessions `coupling-live-0`, `coupling-triple-live-0` run LIVE on the symbolic
solvers (SME participant: JAX symsolver in the JAX image; VOF participant: OpenFOAM 13
incompressibleVoF + `swePreciceCoupling` FO in the OpenFOAM image; preCICE across the two
containers), skip removed in `tests/backend_e2e`; old-vs-new metrics; then the thesis cases
`sme_vof_coupling`, `sme_coupling_jax` migrated (thesis worktree `~/git/wt-thsmevof`, branch
`mig/sme-vof`).

Worktrees: `~/git/wt-smevof` (superrepo + submodule worktrees zoomy_core, zoomy_jax,
zoomy_foam, zoomy_gui, zoomy_amrex, all on `dev/smevof-live`), `~/git/wt-thsmevof`.
Scratch: `$SCRATCH/smevof/` (session scratchpad).

## Facts established
- Today the live sessions run `thesis/notebooks/coupling/cases/sme_vof{,_triple}/run.py` IN
  PLACE in a host checkout (old zoomyFoam `bin/zoomyFoam_L1w` + FO
  `libswePreciceCoupling.so`), H_R = 0.10 (generate.py hard-coded), T = 4, window 2e-3,
  characteristic ghost + frozen mass row (rp), lambda_s 0.5, nu 1e-3 (generate.py defaults;
  settings.json values are not used).
- Only Fr* = 0.752 (H_R = 0.04) is in use -> the ported sessions run H_R = 0.04.
- Old image `~/git/Zoomy/containers/zoomy_openfoam/zoomy_openfoam.sif` (Jul 14) has preCICE
  3.4.1; the current images (OpenFOAM, JAX) have 3.4.0.
- The FO includes the SME model header (Model.H / NumericsKernels.H from
  `zoomy_foam/create_model.py`, old printers, still working on current core).

## Done
- (none yet)

## Next
- old reference run at H_R = 0.04 (scratch `smevof/ref0`)
