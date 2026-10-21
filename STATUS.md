# dev/vam-default status (2026-10-21)

- zoomy_core (dev/vam-default): 7cfad77 SME / ML-SME on the pipeline (byte-identical), InsertClosure
  takes shear-only bulk / unscaled bed closures; 865d192 VAM default = order-consistent (Fr, De = O(1)),
  VAMWeakViscosity = thesis VAM, VAM hand path deleted, CACHE_VERSION v10; 92b8372 ML-VAM traction
  default shared (mean) + VerticalTraction.function.
- superrepo (dev/vam-default): pointer, docs guide, gate48, MATRIX_ONLY filter, ANALYSIS 63.
- thesis (mig/vam-default, ~/git/wt-thvamdefault): 347d9931 mechanical VAM -> VAMWeakViscosity.
- Tests: core 505 pass; goldens m11-m13/S02/N02/P01 unchanged (VAMWeakViscosity), new m17; jax: only
  timing-gate flakes (as main); QR 27 pass; gate48 0; matrix C8/C10/C15/C16/C31 (partial) 0 FAIL.
- Open: SME Stress.full (options in ANALYSIS 63); ML-VAM not on the pipeline (shallow = weak regime);
  matrix rows not rerun: C16/C31 foam+amrex, C35, C37, mpi4/gpu; m05 large golden fails on main too.
