// Trivial block bodies: 1-D periodic linear advection, p = (a, u_min).
// Same arithmetic, in the same order, as zoomy_core.solver.template.toy
// .advection_bodies(order=2, limiter="none") -- the gate compares bitwise.
#include "zoomy_template_vec.hpp"

namespace {
inline int NC(const Ctx& c) { return c.n_cells; }
inline int NF(const Ctx& c) { return c.n_faces; }
}  // namespace

void halo_bc(Ctx& ctx, const CellState& Q, const CellAux& Qaux, const Params&, const double,
             const Geom&, BCState& bc_Q, BCAux& bc_aux) {
    const int nc = NC(ctx);
    for (int v = 0; v < ctx.nv; ++v) {
        bc_Q[v * 2 + 0] = Q[v * nc + nc - 1];
        bc_Q[v * 2 + 1] = Q[v * nc + 0];
    }
    for (int v = 0; v < ctx.naux; ++v) {
        bc_aux[v * 2 + 0] = Qaux[v * nc + nc - 1];
        bc_aux[v * 2 + 1] = Qaux[v * nc + 0];
    }
}

void reconstruct(Ctx& ctx, const CellState& Q, const CellAux& Qaux, const BCState& bc_Q,
                 const BCAux& bc_aux, const Params&, const Geom&, const CellMask& o1,
                 FaceState& Q_L, FaceState& Q_R, FaceAux& Qaux_L, FaceAux& Qaux_R,
                 CellGrad& grad) {
    const int nc = NC(ctx), nf = NF(ctx);
    const double dx = 1.0 / nc;
    std::vector<double> left(nc), right(nc);
    for (int v = 0; v < ctx.nv; ++v) {
        for (int i = 0; i < nc; ++i) {
            const double em = (i == 0) ? bc_Q[v * 2 + 0] : Q[v * nc + i - 1];
            const double ep = (i == nc - 1) ? bc_Q[v * 2 + 1] : Q[v * nc + i + 1];
            const double q = Q[v * nc + i];
            const double dl = (q - em) / dx;
            const double dr = (ep - q) / dx;
            double g = 0.5 * (dl + dr);
            if (o1[i]) g = 0.0;
            grad[v * nc + i] = g;
            left[i] = q + 0.5 * dx * g;
            right[i] = q - 0.5 * dx * g;
        }
        Q_L[v * nf + 0] = left[nc - 1];
        for (int f = 1; f < nf; ++f) Q_L[v * nf + f] = left[f - 1];
        for (int f = 0; f < nc; ++f) Q_R[v * nf + f] = right[f];
        Q_R[v * nf + nc] = right[0];
    }
    for (int v = 0; v < ctx.naux; ++v) {
        Qaux_L[v * nf + 0] = bc_aux[v * 2 + 0];
        for (int f = 1; f < nf; ++f) Qaux_L[v * nf + f] = Qaux[v * nc + f - 1];
        for (int f = 0; f < nc; ++f) Qaux_R[v * nf + f] = Qaux[v * nc + f];
        Qaux_R[v * nf + nc] = bc_aux[v * 2 + 1];
    }
}

void flux_pass(Ctx& ctx, const FaceState& Q_L, const FaceState& Q_R, const FaceAux&,
               const FaceAux&, const Params& p, const double, const Geom&, FaceState& F,
               FaceState& Dp, FaceState& Dm) {
    const double a = p[0];
    for (std::size_t k = 0; k < F.size(); ++k) {
        F[k] = (a > 0) ? a * Q_L[k] : a * Q_R[k];
        Dp[k] = 0.0;
        Dm[k] = 0.0;
    }
    (void)ctx;
}

void gather_update(Ctx& ctx, const CellState& Q, const FaceState& F, const FaceState& Dp,
                   const FaceState& Dm, const CellState& cell_term, const CellAux&,
                   const Params& p, const double, const double dt, const Geom&,
                   CellState& Q_cand, CellMask& troubled) {
    const int nc = NC(ctx), nf = NF(ctx);
    const double dx = 1.0 / nc;
    ++ctx.stage_calls;
    for (int v = 0; v < ctx.nv; ++v)
        for (int i = 0; i < nc; ++i)
            Q_cand[v * nc + i] = Q[v * nc + i]
                - dt / dx * (F[v * nf + i + 1] - F[v * nf + i] + Dm[v * nf + i + 1] + Dp[v * nf + i])
                + dt * cell_term[v * nc + i];
    for (int i = 0; i < nc; ++i) {
        bool bad = Q_cand[i] < p[1];
        for (int v = 0; v < ctx.nv; ++v) bad = bad || !std::isfinite(Q_cand[v * nc + i]);
        troubled[i] = bad;
    }
}

void compute_dt(Ctx& ctx, const CellState&, const CellAux&, const Params& p, const Geom&,
                const double cfl, const int dimension, const int degree_factor,
                const double dt_max, double& dt) {
    const double dx = 1.0 / NC(ctx);
    const double lam = std::fabs(p[0]);
    dt = cfl * 2.0 * (0.5 * dx) / (dimension * degree_factor * lam);
    dt = std::min(dt, dt_max);
}

void update_aux(Ctx& ctx, const CellState& Q, CellAux& Qaux, const Params&, const Geom&,
                const double) {
    const int nc = NC(ctx);
    const double dx = 1.0 / nc;
    for (int v = 0; v < ctx.nv; ++v)
        for (int i = 0; i < nc; ++i) {
            const double qp = Q[v * nc + (i + 1) % nc];
            const double qm = Q[v * nc + (i - 1 + nc) % nc];
            Qaux[v * nc + i] = (qp - qm) / (2.0 * dx);
        }
}

void update_variables(Ctx&, CellState&, const CellAux&, const Params&, const Geom&,
                      const double, const double) {}

void write_fields(Ctx& ctx, const CellState& Q, const CellAux&, const double time,
                  const int i_snapshot) {
    if (!ctx.out) return;
    std::fprintf(ctx.out, "snap %d %.17g", i_snapshot, time);
    for (double q : Q) std::fprintf(ctx.out, " %.17g", q);
    std::fprintf(ctx.out, "\n");
}
