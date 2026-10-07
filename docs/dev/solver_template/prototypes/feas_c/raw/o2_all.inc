// order      : 2
// tableau    : ((0.0, 1.0), (0.5, 1.0))
// flags      : coupled=True, dt_floor=False, implicit_diffusion=True, implicit_source=True, interior_ncp=True, mood=True, write_output=True
// constants  :
// c_cfl                    = 0.9          <- user law (pure safety factor, passed in, never adjusted)
// c_cfl_degree_factor      = 1            <- 2k+1 for reconstruction degree k=0
// c_cfl_dimension          = 1            <- MESH dimension (user law)
// c_dt_floor               = None         <- OFF (user law: no dt floor, the guard is FATAL)
// c_dt_max                 = 1.0          <- caller override
// c_eps_h                  = 1e-14        <- absent (no eigenvalues slot / no depth state 'h'); fell back to constants.EPS_H_FALLBACK
// c_kp_eps                 = 1e-08        <- zoomy_core.systemmodel.operations._DEFAULT_WET_DRY_EPS
// c_mood_h_bound           = 0.0          <- user law: PAD is h < 0, never floored/clipped
// c_mood_require_finite    = 1            <- CAD half of the troubled predicate
// c_write_eps              = 1e-14        <- constants.WRITE_EPS (drift-free write gate)
    // Drift-free write gate: due = time + c_write_eps >= i_snapshot * write_interval; the stamp is recomputed, never accumulated. Writes ``do_write`` and the advanced ``i_snapshot``.
    PORTABLE_FN static inline void solver_should_write(
        const double time,
        const double dt,
        const int i_snapshot,
        const double write_interval)
    {
        double c_write_eps = 1.0e-14;
        int do_write = 0;
        if (i_snapshot*write_interval <= c_write_eps + time) {
            do_write = 1;
        } else {
            do_write = 0;
        }
        if (i_snapshot*write_interval <= c_write_eps + time) {
            i_snapshot = i_snapshot + 1;
        }
    }

    // One step of the v6 march at a FROZEN dt (2 Shu-Osher stage(s), reconstruction order 2).  dt is an argument: the step head owns it and no stage recomputes it. MOOD / implicit source / implicit diffusion are BUILD-time branches.
    PORTABLE_FN static inline void solver_hyperbolic_step(
        double* Q,
        double* Qaux,
        const double* p,
        const double time,
        const int iteration,
        const double dt)
    {
        solver_stage_base(Q, Q0);
        solver_clear_troubled(Q, troubled);
        int o1 = 0;
        double c_alpha_0 = 0.0;
        double c_beta_0 = 1.0;
        double time_stage = time;
        solver_halo_bc(Q, Qaux, p, time_stage, Q, Qaux, bf_values);
        solver_reconstruct(Q, Qaux, bf_values, o1, Q_L, Q_R, lim_grad);
        solver_cell_ncp(Q, Qaux, p, lim_grad, cell_term);
        solver_flux_pass(Q, Qaux, p, time_stage, Q_L, Q_R, Fface, Dp, Dm);
        solver_coupling_face_fixup(Fface, Dp, Dm, n_faces, Fface, Dp, Dm);
        solver_gather_update(Q, Fface, Dp, Dm, dt, c_beta_0, cell_term, Qaux, p, time_stage, Q_cand, troubled_stage);
        solver_merge_troubled(troubled, troubled_stage, troubled);
        Q = Q0*c_alpha_0 + Q_cand*(1 - c_alpha_0);
        double c_alpha_1 = 0.5;
        double c_beta_1 = 1.0;
        double time_stage = dt + time;
        solver_halo_bc(Q, Qaux, p, time_stage, Q, Qaux, bf_values);
        solver_reconstruct(Q, Qaux, bf_values, o1, Q_L, Q_R, lim_grad);
        solver_cell_ncp(Q, Qaux, p, lim_grad, cell_term);
        solver_flux_pass(Q, Qaux, p, time_stage, Q_L, Q_R, Fface, Dp, Dm);
        solver_coupling_face_fixup(Fface, Dp, Dm, n_faces, Fface, Dp, Dm);
        solver_gather_update(Q, Fface, Dp, Dm, dt, c_beta_1, cell_term, Qaux, p, time_stage, Q_cand, troubled_stage);
        solver_merge_troubled(troubled, troubled_stage, troubled);
        Q = Q0*c_alpha_1 + Q_cand*(1 - c_alpha_1);
        double c_mood_h_bound = 0.0;
        int c_mood_require_finite = 1;
        solver_mood_resolve(troubled, Q0, Q, Qaux, p, time, dt, c_mood_h_bound, c_mood_require_finite, Q);
        solver_implicit_source(Q, Qaux, p, dt, Q);
        solver_implicit_diffusion(Q, Qaux, p, dt, Q);
        time = dt + time;
        iteration = iteration + 1;
        solver_update_variables(Q, Qaux, p, time, dt, Q);
        solver_update_aux(Q, Qaux, p, time, dt, Qaux);
    }

    // The v6 march.  Step head: dt_pass -> reduce_dt -> FATAL admissibility guard; then one step at that FROZEN dt; then the progress guard and the drift-free write cadence.  No dt-halving, no retry, no CFL reduction.
    PORTABLE_FN static inline void solver_march(
        double* Q,
        double* Qaux,
        const double* p,
        const double time,
        const int iteration,
        const int i_snapshot,
        const double t_end,
        const double dt_window,
        const double write_interval,
        const double* inradius_f,
        const int n_faces)
    {
        while (time < t_end) {
            double c_eps_h = 1.0e-14;
            solver_dt_pass(Q, Qaux, p, c_eps_h, lam_lo_f, lam_hi_f);
            double c_cfl = 0.90000000000000002;
            int c_cfl_dimension = 1;
            int c_cfl_degree_factor = 1;
            double c_dt_max = 1.0;
            solver_reduce_dt(lam_lo_f, lam_hi_f, inradius_f, n_faces, c_cfl, c_cfl_dimension, c_cfl_degree_factor, c_dt_max, dt_window, dt);
            solver_assert_dt_admissible(dt, lam_lo_f, lam_hi_f, inradius_f, time, iteration, dt);
            solver_hyperbolic_step(Q, Qaux, p, time, iteration, dt, Q, Qaux, time, iteration, troubled);
            solver_assert_march_progress(time, iteration, dt, t_end);
            solver_should_write(time, dt, i_snapshot, write_interval, do_write, i_snapshot);
            solver_write_fields(Q, Qaux, time, i_snapshot, do_write);
        }
    }
