// Driver: read the initial state, run the EMITTED march, write the result.
//   main <input.txt> <output.txt>
// input:  n_cells, then Q (n_cells values), then p (a, u_min), then time_end
#include "zoomy_template_vec.hpp"
#include "template_procs.inc"   // EMITTED procedures

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::FILE* in = std::fopen(argv[1], "r");
    Ctx ctx;
    int n = 0;
    if (std::fscanf(in, "%d", &n) != 1) return 2;
    ctx.n_cells = n;
    ctx.n_faces = n + 1;
    allocate_workspace(ctx);                // EMITTED allocator (F4)
    CellState Q(n);
    CellAux Qaux(n);
    Params p(2);
    for (int i = 0; i < n; ++i) if (std::fscanf(in, "%lf", &Q[i]) != 1) return 2;
    for (int i = 0; i < n; ++i) if (std::fscanf(in, "%lf", &Qaux[i]) != 1) return 2;
    for (int i = 0; i < 2; ++i) if (std::fscanf(in, "%lf", &p[i]) != 1) return 2;
    double time_end = 0.0;
    if (std::fscanf(in, "%lf", &time_end) != 1) return 2;
    std::fclose(in);
    ctx.out = std::fopen(argv[2], "w");
    Geom geom{n, 1.0 / n};
    int i_snapshot = 0, iteration = 0;
    double time = 0.0;
    march(ctx, Q, Qaux, p, geom, 0.0, time_end, i_snapshot, time, iteration);
    std::fprintf(ctx.out, "final %d %d %.17g %ld", iteration, i_snapshot, time, ctx.stage_calls);
    for (double q : Q) std::fprintf(ctx.out, " %.17g", q);
    std::fprintf(ctx.out, "\n");
    std::fclose(ctx.out);
    return 0;
}
