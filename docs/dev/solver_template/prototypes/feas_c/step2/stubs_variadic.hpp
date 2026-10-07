// Arity-agnostic stubs: isolates the ORCHESTRATION defects (undeclared
// results, redeclaration, const writes, pointer arithmetic) from arity ones.
#pragma once
#define PORTABLE_FN
#define VSTUB(n) template <class... A> void n(A&&...) {}
VSTUB(solver_prepare_mesh) VSTUB(solver_build_operators) VSTUB(solver_initialize_state)
VSTUB(solver_halo_bc) VSTUB(solver_mood_resolve) VSTUB(solver_implicit_source)
VSTUB(solver_implicit_diffusion) VSTUB(solver_write_fields) VSTUB(solver_adapt_mesh)
VSTUB(solver_reduce_min) VSTUB(solver_coupling_init) VSTUB(solver_coupling_read)
VSTUB(solver_coupling_write) VSTUB(solver_coupling_advance) VSTUB(solver_coupling_checkpoint)
VSTUB(solver_coupling_face_fixup) VSTUB(solver_coupling_finalize) VSTUB(solver_dt_pass)
VSTUB(solver_reduce_dt) VSTUB(solver_apply_dt_floor) VSTUB(solver_assert_dt_admissible)
VSTUB(solver_assert_march_progress) VSTUB(solver_stage_base) VSTUB(solver_clear_troubled)
VSTUB(solver_merge_troubled) VSTUB(solver_reconstruct) VSTUB(solver_cell_ncp)
VSTUB(solver_no_cell_term) VSTUB(solver_flux_pass) VSTUB(solver_gather_update)
VSTUB(solver_update_variables) VSTUB(solver_update_aux)
