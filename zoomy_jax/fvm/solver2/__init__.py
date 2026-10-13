"""`solver2` — the v6 unified march, jax reference implementation.

Side-by-side with `solver_jax.py`; NOTHING is deleted and nothing in the
production path is modified (standing user ruling).  This package is the
reference implementation of
`proposals/2026-07-20-solver-flowcharts-phase1.md` sections v5 + v6: the
Procedure IR that core will later emit must reproduce these semantics.

What is REUSED UNCHANGED from the existing jax solver
-----------------------------------------------------
* `zoomy_jax.transformation.jax_runtime.JaxRuntime` — every kernel:
  `numerical_flux`, `numerical_fluctuations`, `eigenvalues`, `source`,
  `boundary_conditions`, `nonconservative_matrix`, `update_variables`,
  `update_aux_variables`.  Not one line of physics is re-derived here.
* `zoomy_jax.fvm.reconstruction_jax` — `ConstantReconstruction`,
  `LSQMUSCLReconstructionJAX`, `PositivityPreservingLSQMUSCLJAX`,
  `EtaWellBalancedLSQMUSCLJAX` and the limiter/positivity machinery inside
  them, selected by the INHERITED `HyperbolicSolver._build_reconstruction`.
* `zoomy_jax.mesh.mesh` — `convert_mesh_to_jax`, `lsq_gradient_per_field`.
* `HyperbolicSolver.setup_simulation` (model coercion to NSM, `dt_max`
  default fill, `ensure_lsq_mesh`, periodic-BC remap, `initialize`,
  `create_runtime`, initial aux + BC sweep) and
  `HyperbolicSolver._walk_derivative_aux` (the LSQ derivative-aux walk).
* `zoomy_core.fvm.timestepping` semantics for the CFL denominator
  (re-expressed in `blocks.reduce_dt` so the two stored per-face bounds
  feed it directly).

What is NEW here
----------------
The SHELL: the block decomposition, the v6 dt-at-step-head ordering, the
stored face arrays, the fused troubled detection, the Shu-Osher stage form,
the whole-step order-1 MOOD redo, and the fatal dt guard.

Signature deviations from the design text (all deliberate, all minor)
---------------------------------------------------------------------
1. `dt_pass` / `halo_bc` / `reconstruct` / `mood_resolve` take `Ops`
   (and `halo_bc` additionally `p`) explicitly.  The design lists `Ops`
   only on `flux_pass`; the kernels these blocks call (`eigenvalues`,
   `bc_face`, `reconstruct`) live in `Ops`, and the BC kernel binds the
   parameter vector.  Nothing else changed.
2. `reconstruct` also takes `bf_values` from `halo_bc`.  On an
   unstructured mesh with no ghost CELLS the boundary state IS the per-face
   value `halo_bc` synthesises, and the limiter bounds need it.
3. `gather_update` takes `h_index` so the fused troubled predicate can do
   PAD (`h < 0`) as well as CAD (non-finite).
4. `mood_resolve` takes the step's own `stage_loop` so the sanctioned
   whole-step order-1 redo owns no scheme logic of its own.
5. `a_stage` is the Shu-Osher `beta`; the `alpha` convex combination is
   applied by the stage loop (amendment 14's "Shu-Osher stage form").
"""

from zoomy_jax.fvm.solver2.blocks import (  # noqa: F401
    assert_dt_admissible, dt_pass, flag_troubled, flux_pass, gather_update,
    halo_bc, mood_resolve, reconstruct, reduce_dt, update_aux,
    update_variables,
)
from zoomy_jax.fvm.solver2.context import (  # noqa: F401
    TABLEAU_EULER, TABLEAU_SSPRK2, MeshRT, Ops, build_operators, prepare_mesh,
)
from zoomy_jax.fvm.solver2.march import MarchSolver, describe_nsm  # noqa: F401
from zoomy_jax.fvm.solver2.state import (  # noqa: F401
    MarchState, proceed, should_write,
)

__all__ = [
    "MarchSolver", "MarchState", "MeshRT", "Ops",
    "TABLEAU_EULER", "TABLEAU_SSPRK2",
    "assert_dt_admissible", "build_operators", "describe_nsm", "dt_pass",
    "flag_troubled", "flux_pass", "gather_update", "halo_bc", "mood_resolve",
    "prepare_mesh", "proceed", "reconstruct", "reduce_dt", "should_write",
    "update_aux", "update_variables",
]
