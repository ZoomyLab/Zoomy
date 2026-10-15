# dev/amrex-amr-gpu (wt-amrexfix) — status

Worktrees: superrepo + zoomy_amrex, zoomy_core on `dev/amrex-amr-gpu` (from integ/20261014);
zoomy_jax, zoomy_foam detached at integ's commits (unchanged).  Not pushed, not merged.
Details: ANALYSIS.md section 55.  Matrix store: main's store copied, the touched cases rerun
(`docs/dev/symbolic_solver/matrix/store`, untracked).  Image: `containers/zoomy_amrex/zoomy_amrex.sif`
of this worktree = the published image + libnvjitlink-12-6 (untracked; the recipes are fixed).

## 1. C17 `dry` — done (core 8f56c4c, amrex 55a6b3a)
- The reflux made h < 0: the fine level's stage-2 halo held the coarse state after its whole step.
- Composite stages: all levels advance stage by stage, reflux + average-down per stage; MOOD on
  the composite.  dry: mass 0, h_min 0.  Lie's interface lag gone as a by-product (0.59 -> 0.38).
## 2. C17 long-run hang — done (core 63cdf15, amrex a100cf2)
- h refluxed, q not -> garbage velocity in thin cells -> dt collapse (base 64: 9556 steps vs 386).
- amr_keep: a refluxed cell keeps its velocity.  New C17 row `dry_long` (0.94 of the fine steps).
## 3. GPU — done (amrex 9c133a5; superrepo: image recipes)
- 3 boxes / 2 GPUs: no-sync march raced AMReX's unfused per-box streams -> garbage pressure rows,
  NaN FlexGMRES.  One GPU stream.  CUDA_VISIBLE_DEVICES passed into the image.  Image lacked
  libnvJitLink (no hypre-CUDA link).  hypre GPU smoother = CPU's: 38 -> 20 iterations, but
  5-10x slower per solve (open: Chebyshev 23 its at 0.8x).
## 4. AMR implicit time lag — Lie fixed by (1); IMEX-SSP2 analysed, plan in ANALYSIS 55d
