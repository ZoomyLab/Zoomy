// Strict stub header: one C++ declaration per REQUIRED_PROCEDURES entry,
// following the CONTRACT in zoomy_core/solver/external.py:
//   * entries with declared ``args``  -> exactly those args (ExternalProcedure.args)
//   * doc-only entries                -> the "(in...) -> (out...)" doc signature,
//                                        outputs as trailing written params.
// Scalar results are T& (the printer passes results as trailing args).
#pragma once
#define PORTABLE_FN
// -- declared args
void solver_prepare_mesh();
void solver_build_operators();
void solver_initialize_state();
void solver_halo_bc(const double* Q, const double* Qaux, double time);
void solver_mood_resolve(int* troubled, const double* Q0, double* Q_cand,
                         const double* Qaux, const double* p, double time, double dt);
void solver_implicit_source(double* Q_cand, const double* Qaux, const double* p, double dt);
void solver_implicit_diffusion(double* Q_cand, const double* Qaux, const double* p, double dt);
void solver_write_fields(const double* Q, const double* Qaux, double time, int i_snapshot);
void solver_adapt_mesh();
double solver_reduce_min(const double* lam_f, int n_faces);
void solver_coupling_init();
void solver_coupling_read(double time, double dt);
void solver_coupling_write(const double* Q, double time);
void solver_coupling_advance(double dt);
void solver_coupling_checkpoint();
void solver_coupling_face_fixup(double* Fface, double* Dp, double* Dm, int n_faces);
void solver_coupling_finalize();
// -- doc-only signatures
void solver_dt_pass(const double* Q, const double* Qaux, const double* p, double c_eps_h,
                    double* lam_lo_f, double* lam_hi_f);
void solver_reduce_dt(const double* lam_lo_f, const double* lam_hi_f, const double* inradius_f,
                      int n_faces, double c_cfl, int c_cfl_dimension, int c_cfl_degree_factor,
                      double c_dt_max, double clamp, double& dt);
void solver_apply_dt_floor(double dt, double c_dt_floor, double& dt_out);
void solver_assert_dt_admissible(double dt, const double* lam_lo_f, const double* lam_hi_f,
                                 const double* inradius_f, double time, int iteration,
                                 double& dt_out);
void solver_assert_march_progress(double time, int iteration, double dt, double t_end);
void solver_stage_base(const double* Q, double* Q0);
void solver_clear_troubled(const double* Q, int* troubled);
void solver_merge_troubled(const int* troubled, const int* troubled_stage, int* troubled_out);
void solver_reconstruct(const double* Q, const double* Qaux, const double* bf_values, int o1,
                        double* Q_L, double* Q_R, double* lim_grad);
void solver_cell_ncp(const double* Q, const double* Qaux, const double* p,
                     const double* lim_grad, double* cell_term);
void solver_no_cell_term(const double* Q, double* cell_term);
void solver_flux_pass(const double* Q, const double* Qaux, const double* p, double t_stage,
                      const double* Q_L, const double* Q_R, double* Fface, double* Dp, double* Dm);
void solver_gather_update(const double* Q0, const double* Fface, const double* Dp,
                          const double* Dm, double dt, double beta, const double* cell_term,
                          const double* Qaux, const double* p, double t_stage,
                          double* Q_cand, int* troubled);
void solver_update_variables(const double* Q, const double* Qaux, const double* p,
                             double time, double dt, double* Q_out);
void solver_update_aux(const double* Q, const double* Qaux, const double* p,
                       double time, double dt, double* Qaux_out);
