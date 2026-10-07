// The 32 REQUIRED_PROCEDURES bodies, "dmplex-like" plain-array backend.
// Trivial 1-D SWE on [b, h, q_0] (layout from the EMITTED model_layout.hpp):
// Rusanov face flux, minmod MUSCL at order 2, wall ghosts, path-conservative
// bed fluctuation (zero on the flat-bed dam break).  Feasibility stubs only.
#include <algorithm>
#include <limits>

#include "zoomy_vec_backend.hpp"

namespace {
inline int NC(const Ctx& c) { return c.mesh.n_cells; }
inline int NF(const Ctx& c) { return c.mesh.n_faces; }
inline double minmod(double a, double b) {
    return (a * b <= 0.0) ? 0.0 : (std::fabs(a) < std::fabs(b) ? a : b);
}
// cell value with wall ghosts from bf_values (side 0 = left, 1 = right)
inline double cellv(const Ctx& c, const CellState& Q, const BCValues& bf, int v, int i) {
    if (i < 0) return bf[v * 2 + 0];
    if (i >= NC(c)) return bf[v * 2 + 1];
    return Q[v * NC(c) + i];
}
inline void wavespeeds(double g, double h, double q, double eps_h, double& lo, double& hi) {
    const double hh = std::max(eps_h, h);
    const double u = q / hh, c = std::sqrt(g * hh);
    lo = u - c; hi = u + c;
}
}  // namespace

// ── setup / lifecycle ────────────────────────────────────────────────────
void solver_prepare_mesh(Ctx& ctx) {
    Mesh& m = ctx.mesh;
    m.n_faces = m.n_cells + 1;
    m.dx = m.length / m.n_cells;
    m.inradius_f.assign(m.n_faces, 0.5 * m.dx);
    m.cell_volumes.assign(m.n_cells, m.dx);
}
void solver_build_operators(Ctx& ctx) { ctx.order = MARCH_ORDER; }
void solver_initialize_state(Ctx& ctx) {
    State& S = ctx.state;
    const int nc = NC(ctx);
    zoomy_alloc_cell_state(ctx, S.Q);
    zoomy_alloc_cell_aux(ctx, S.Qaux);
    S.p.assign(PARAM_VALUES, PARAM_VALUES + N_PARAMS);
    for (int i = 0; i < nc; ++i) {
        const double x = (i + 0.5) * ctx.mesh.dx;
        S.Q[IDX_b * nc + i] = 0.0;
        S.Q[IDX_h * nc + i] = x < 0.5 * ctx.mesh.length ? 2.0 : 1.0;
        S.Q[IDX_q_0 * nc + i] = 0.0;
    }
    S.time = 0.0; S.iteration = 0; S.i_snapshot = 0;
}
void solver_adapt_mesh(Ctx&) {}
void solver_coupling_init(Ctx&) {}
void solver_coupling_checkpoint(Ctx&) {}
void solver_coupling_finalize(Ctx&) {}
void solver_coupling_read(Ctx&, const double, const double) {}
void solver_coupling_write(Ctx&, const CellState&, const double) {}
void solver_coupling_advance(Ctx&, const double) {}
// STRICT no-op without a peer (D8)
void solver_coupling_face_fixup(Ctx&, FaceState&, FaceState&, FaceState&, const int) {}

double solver_reduce_min(Ctx&, const FaceScalar& lam_f, const int n_faces) {
    double m = std::numeric_limits<double>::infinity();
    for (int f = 0; f < n_faces; ++f) m = std::min(m, lam_f[f]);
    return m;
}

// ── step head ────────────────────────────────────────────────────────────
void solver_dt_pass(Ctx& ctx, const CellState& Q, const CellAux&, const Params& p,
                    const double c_eps_h, FaceScalar& lam_lo_f, FaceScalar& lam_hi_f) {
    // NB runs BEFORE any halo_bc of this step: boundary faces use the inner
    // cell only (a wall ghost has the same |u|+c).  See report: head-of-step
    // halo dependency.
    const int nc = NC(ctx);
    const double g = p[PAR_g];
    for (int f = 0; f < NF(ctx); ++f) {
        const int L = std::max(f - 1, 0), R = std::min(f, nc - 1);
        double lo1, hi1, lo2, hi2;
        wavespeeds(g, Q[IDX_h * nc + L], Q[IDX_q_0 * nc + L], c_eps_h, lo1, hi1);
        wavespeeds(g, Q[IDX_h * nc + R], Q[IDX_q_0 * nc + R], c_eps_h, lo2, hi2);
        lam_lo_f[f] = std::min(lo1, lo2);
        lam_hi_f[f] = std::max(hi1, hi2);
    }
}
void solver_reduce_dt(Ctx& ctx, const FaceScalar& lo, const FaceScalar& hi,
                      const FaceScalar& inradius_f, const int n_faces, const double c_cfl,
                      const int c_cfl_dimension, const int c_cfl_degree_factor,
                      const double c_dt_max, const double clamp, double& dt) {
    (void)ctx;
    double d = std::numeric_limits<double>::infinity();
    for (int f = 0; f < n_faces; ++f) {
        const double lam = std::max(std::fabs(lo[f]), std::fabs(hi[f]));
        if (lam > 0.0)   // wave-free face drops out (REQ-190)
            d = std::min(d, c_cfl * 2.0 * inradius_f[f] /
                                (c_cfl_dimension * c_cfl_degree_factor * lam));
    }
    dt = std::min(std::min(d, c_dt_max), clamp);
}
void solver_apply_dt_floor(Ctx&, double& dt, const double c_dt_floor) { dt = std::max(dt, c_dt_floor); }
void solver_assert_dt_admissible(Ctx&, double& dt, const FaceScalar& lo, const FaceScalar& hi,
                                 const FaceScalar&, const double time, const int iteration) {
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        std::fprintf(stderr, "FATAL dt=%g at t=%g iter=%d (lam0=[%g,%g])\n", dt, time,
                     iteration, lo[0], hi[0]);
        std::abort();
    }
}
void solver_assert_march_progress(Ctx& ctx, const double time, const int iteration,
                                  const double dt, const double t_end) {
    if (iteration > ctx.max_steps) {
        std::fprintf(stderr, "FATAL no progress: iter=%d t=%g dt=%g t_end=%g\n", iteration,
                     time, dt, t_end);
        std::abort();
    }
}

// ── stage ────────────────────────────────────────────────────────────────
void solver_stage_base(Ctx&, const CellState& Q, CellState& Q0) {
    std::copy(Q.begin(), Q.end(), Q0.begin());
}
void solver_clear_troubled(Ctx&, const CellState&, CellMask& troubled) {
    std::fill(troubled.begin(), troubled.end(), 0);
}
void solver_merge_troubled(Ctx&, CellMask& troubled, const CellMask& ts) {
    for (std::size_t i = 0; i < troubled.size(); ++i) troubled[i] = troubled[i] || ts[i];
}
void solver_halo_bc(Ctx& ctx, CellState& Q, CellAux&, const Params&, const double,
                    BCValues& bf) {
    const int nc = NC(ctx);
    for (int v = 0; v < N_VARS; ++v) {
        const double s = (v == IDX_q_0) ? -1.0 : 1.0;   // wall
        bf[v * 2 + 0] = s * Q[v * nc + 0];
        bf[v * 2 + 1] = s * Q[v * nc + nc - 1];
    }
}
void solver_reconstruct(Ctx& ctx, const CellState& Q, const CellAux&, const BCValues& bf,
                        const int o1, FaceState& Q_L, FaceState& Q_R, CellGrad& lim_grad) {
    const int nc = NC(ctx), nf = NF(ctx);
    const double dx = ctx.mesh.dx;
    const bool second = ctx.order >= 2 && !o1;
    for (int v = 0; v < N_VARS; ++v)
        for (int i = 0; i < nc; ++i)
            lim_grad[v * nc + i] =
                second ? minmod((cellv(ctx, Q, bf, v, i) - cellv(ctx, Q, bf, v, i - 1)) / dx,
                                (cellv(ctx, Q, bf, v, i + 1) - cellv(ctx, Q, bf, v, i)) / dx)
                       : 0.0;
    for (int v = 0; v < N_VARS; ++v)
        for (int f = 0; f < nf; ++f) {
            Q_L[v * nf + f] = (f - 1 >= 0)
                ? Q[v * nc + f - 1] + 0.5 * dx * lim_grad[v * nc + f - 1] : bf[v * 2 + 0];
            Q_R[v * nf + f] = (f < nc)
                ? Q[v * nc + f] - 0.5 * dx * lim_grad[v * nc + f] : bf[v * 2 + 1];
        }
}
void solver_cell_ncp(Ctx& ctx, const CellState& Q, const CellAux&, const Params& p,
                     const CellGrad& lim_grad, CellState& cell_term) {
    const int nc = NC(ctx);
    std::fill(cell_term.begin(), cell_term.end(), 0.0);
    for (int i = 0; i < nc; ++i)   // -g h b_x : the NSM's nonconservative row q_0
        cell_term[IDX_q_0 * nc + i] = -p[PAR_g] * Q[IDX_h * nc + i] * lim_grad[IDX_b * nc + i];
}
void solver_no_cell_term(Ctx&, const CellState&, CellState& cell_term) {
    std::fill(cell_term.begin(), cell_term.end(), 0.0);
}
void solver_flux_pass(Ctx& ctx, const CellState&, const CellAux&, const Params& p,
                      const double, const FaceState& Q_L, const FaceState& Q_R,
                      FaceState& F, FaceState& Dp, FaceState& Dm) {
    const int nf = NF(ctx);
    const double g = p[PAR_g];
    for (int f = 0; f < nf; ++f) {
        const double hL = Q_L[IDX_h * nf + f], qL = Q_L[IDX_q_0 * nf + f], bL = Q_L[IDX_b * nf + f];
        const double hR = Q_R[IDX_h * nf + f], qR = Q_R[IDX_q_0 * nf + f], bR = Q_R[IDX_b * nf + f];
        const double uL = qL / hL, uR = qR / hR;
        const double s = std::max(std::fabs(uL) + std::sqrt(g * hL), std::fabs(uR) + std::sqrt(g * hR));
        F[IDX_b * nf + f] = 0.0;
        F[IDX_h * nf + f] = 0.5 * (qL + qR) - 0.5 * s * (hR - hL);
        F[IDX_q_0 * nf + f] = 0.5 * (qL * uL + 0.5 * g * hL * hL + qR * uR + 0.5 * g * hR * hR)
                              - 0.5 * s * (qR - qL);
        for (int v = 0; v < N_VARS; ++v) { Dp[v * nf + f] = 0.0; Dm[v * nf + f] = 0.0; }
        const double d = 0.5 * g * 0.5 * (hL + hR) * (bR - bL);
        Dp[IDX_q_0 * nf + f] = d; Dm[IDX_q_0 * nf + f] = d;
    }
}
void solver_gather_update(Ctx& ctx, const CellState& Q, const FaceState& F, const FaceState& Dp,
                          const FaceState& Dm, const double dt, const double beta,
                          const CellState& cell_term, const CellAux&, const Params&,
                          const double, CellState& Q_cand, CellMask& troubled_stage) {
    // NB the detector bound c_mood_h_bound is NOT passed here (it is emitted
    // only inside the MOOD branch, after the stages) -> the 0 below is a
    // hard-coded constant the user law forbids.  Reported.
    const int nc = NC(ctx), nf = NF(ctx);
    const double r = beta * dt / ctx.mesh.dx;
    for (int v = 0; v < N_VARS; ++v)
        for (int i = 0; i < nc; ++i)
            Q_cand[v * nc + i] = Q[v * nc + i]
                - r * (F[v * nf + i + 1] - F[v * nf + i] + Dm[v * nf + i + 1] + Dp[v * nf + i])
                + beta * dt * cell_term[v * nc + i];
    for (int i = 0; i < nc; ++i) {
        bool bad = Q_cand[IDX_h * nc + i] < 0.0;
        for (int v = 0; v < N_VARS; ++v) bad = bad || !std::isfinite(Q_cand[v * nc + i]);
        troubled_stage[i] = bad;
    }
}
void solver_mood_resolve(Ctx&, const CellMask& troubled, const CellState&, CellState&,
                         const CellAux&, const Params&, const double time, const double,
                         const double, const int) {
    int n = 0;
    for (int t : troubled) n += t;
    if (n) {
        std::fprintf(stderr, "MOOD stub: %d troubled cells at t=%g; the whole-step O1 "
                             "redo is not implemented in the feasibility stubs\n", n, time);
        std::abort();
    }
}
void solver_implicit_source(Ctx&, CellState&, const CellAux&, const Params&, const double) {}
void solver_implicit_diffusion(Ctx&, CellState&, const CellAux&, const Params&, const double) {}
void solver_update_variables(Ctx&, CellState&, const CellAux&, const Params&, const double,
                             const double) {}
void solver_update_aux(Ctx& ctx, const CellState& Q, CellAux& Qaux, const Params&,
                       const double, const double) {
    // NB c_kp_eps is resolved by core but never emitted/passed -> 1/h here.
    const int nc = NC(ctx);
    const double dx = ctx.mesh.dx;
    for (int i = 0; i < nc; ++i) {
        const int L = std::max(i - 1, 0), R = std::min(i + 1, nc - 1);
        const double w = (R - L) * dx;
        Qaux[AUX_dq0dx * nc + i] = (Q[IDX_q_0 * nc + R] - Q[IDX_q_0 * nc + L]) / w;
        Qaux[AUX_dhdx * nc + i] = (Q[IDX_h * nc + R] - Q[IDX_h * nc + L]) / w;
        Qaux[AUX_dbdx * nc + i] = (Q[IDX_b * nc + R] - Q[IDX_b * nc + L]) / w;
        Qaux[AUX_hinv * nc + i] = 1.0 / Q[IDX_h * nc + i];
    }
}
void solver_write_fields(Ctx& ctx, const CellState& Q, const CellAux&, const double time,
                         const int i_snapshot, const int do_write) {
    if (!do_write) return;
    const int nc = NC(ctx);
    double mass = 0.0, mom = 0.0;
    for (int i = 0; i < nc; ++i) {
        mass += Q[IDX_h * nc + i] * ctx.mesh.cell_volumes[i];
        mom += Q[IDX_q_0 * nc + i] * ctx.mesh.cell_volumes[i];
    }
    std::printf("  snap %3d  t=%.10f  mass=%.15f  momentum=%+.6e\n", i_snapshot, time, mass, mom);
}
