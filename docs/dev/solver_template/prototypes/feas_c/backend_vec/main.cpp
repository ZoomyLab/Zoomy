// Driver: set up the plain-array backend and call the EMITTED solver_march.
#include <cstdio>
#include <cstring>

#include "zoomy_vec_backend.hpp"
#include "march_procs.inc"   // EMITTED: should_write / hyperbolic_step / march

int main(int argc, char** argv) {
    Ctx ctx;
    ctx.mesh.n_cells = 100;
    ctx.mesh.length = 1.0;
    ctx.max_steps = 1000;
    solver_prepare_mesh(ctx);
    solver_build_operators(ctx);
    allocate_march_workspace(ctx);   // EMITTED from the typed declaration table
    solver_initialize_state(ctx);
    State& S = ctx.state;

    const double dt_max = 1.0 / 1024.0;   // == emitted c_dt_max (2^-10, exact)
    const double t_end = 10 * dt_max;     // -> exactly 10 steps
    double mass0 = 0.0;
    for (int i = 0; i < ctx.mesh.n_cells; ++i)
        mass0 += S.Q[IDX_h * ctx.mesh.n_cells + i] * ctx.mesh.cell_volumes[i];
    std::printf("order %d: dam break h=2|1, %d cells, t_end=%g, mass0=%.15f\n",
                MARCH_ORDER, ctx.mesh.n_cells, t_end, mass0);

    solver_march(ctx, S.Q, S.Qaux, S.p, S.time, S.iteration, S.i_snapshot, t_end,
                 /*dt_window=*/t_end /* one coupling window = whole run; 0 trips the FATAL dt guard */, /*write_interval=*/0.0, ctx.mesh.inradius_f,
                 ctx.mesh.n_faces);

    double mass = 0.0;
    for (int i = 0; i < ctx.mesh.n_cells; ++i)
        mass += S.Q[IDX_h * ctx.mesh.n_cells + i] * ctx.mesh.cell_volumes[i];
    std::printf("done: iterations=%d time=%.10f i_snapshot=%d mass=%.15f drift=%.3e\n",
                S.iteration, S.time, S.i_snapshot, mass, mass - mass0);
    if (argc > 1) {
        FILE* fp = std::fopen(argv[1], "w");
        for (int i = 0; i < ctx.mesh.n_cells; ++i)
            std::fprintf(fp, "%.17g %.17g\n", S.Q[IDX_h * ctx.mesh.n_cells + i],
                         S.Q[IDX_q_0 * ctx.mesh.n_cells + i]);
        std::fclose(fp);
    }
    return 0;
}
