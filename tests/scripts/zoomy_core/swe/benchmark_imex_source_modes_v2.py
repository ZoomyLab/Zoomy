import time

import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver

from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from tutorials.swe.simple_swe_v2 import make_model as make_model_swe


def _single_run(name, model, mesh, time_end, cfl, source_mode, use_analytic_chain_jvp):
    solver = IMEXSourceSolver(time_end=time_end, compute_dt=timestepping.adaptive(CFL=cfl))
    object.__setattr__(solver, "source_mode", source_mode)
    object.__setattr__(solver, "implicit_maxiter", 6)
    object.__setattr__(solver, "gmres_maxiter", 30)
    object.__setattr__(solver, "use_analytic_chain_jvp", use_analytic_chain_jvp)
    Q, _Qaux = solver.solve(mesh, model, write_output=False)
    stats = solver.last_stats
    n = mesh.n_inner_cells
    h = Q[0, :n]
    finite_ratio = np.isfinite(Q[:, :n]).sum() / Q[:, :n].size
    return stats, h, finite_ratio


def run_case(
    name,
    model,
    mesh,
    time_end,
    cfl,
    source_mode="auto",
    use_analytic_chain_jvp=True,
    repeats=3,
):
    # Warmup
    _single_run(name, model, mesh, time_end, cfl, source_mode, use_analytic_chain_jvp)

    runs = []
    for _ in range(repeats):
        stats, h, finite_ratio = _single_run(
            name, model, mesh, time_end, cfl, source_mode, use_analytic_chain_jvp
        )
        runs.append((stats, h, finite_ratio))

    total_times = [r[0].total_time_s for r in runs]
    init_times = [r[0].init_time_s for r in runs]
    runtime_times = [r[0].runtime_only_s for r in runs]
    implicit_times = [r[0].implicit_time_s for r in runs]
    steps = [r[0].n_steps for r in runs]
    calls = [r[0].implicit_calls for r in runs]
    finite = [r[2] for r in runs]
    h_last = runs[-1][1]
    stats = runs[-1][0]
    print("=" * 80)
    print(f"{name}")
    print(f"source_mode_used: {stats.source_mode}")
    print(f"use_analytic_chain_jvp: {use_analytic_chain_jvp}")
    print(f"steps(avg): {np.mean(steps):.1f}")
    print(f"total_time_s(avg): {np.mean(total_times):.4f}")
    print(f"init_time_s(avg): {np.mean(init_times):.4f}")
    print(f"runtime_only_s(avg): {np.mean(runtime_times):.4f}")
    print(f"implicit_time_s(avg): {np.mean(implicit_times):.4f}")
    print(f"implicit_calls(avg): {np.mean(calls):.1f}")
    print(f"h_range(last): [{float(np.nanmin(h_last)):.6f}, {float(np.nanmax(h_last)):.6f}]")
    print(f"finite_ratio(avg): {np.mean(finite):.3f}")
    print("=" * 80)
    return {
        "name": name,
        "wall": float(np.mean(total_times)),
        "init": float(np.mean(init_times)),
        "runtime": float(np.mean(runtime_times)),
        "stats": stats,
        "finite_ratio": float(np.mean(finite)),
    }


def run():
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=220, lsq_degree=2)
    time_end = 0.08
    cfl = 0.5

    swe = make_model_swe()                 # local source
    gn_cls = ClassicalGreenNaghdi1D()      # derivative-coupled source

    results = []
    results.append(run_case("SWE (fast local source)", swe, mesh, time_end, cfl, source_mode="auto", repeats=3))
    results.append(run_case("GN-classical (global implicit source, analytic Jv)", gn_cls, mesh, time_end, cfl, source_mode="auto", use_analytic_chain_jvp=True, repeats=3))
    results.append(run_case("GN-classical (global implicit source, FD-Jv)", gn_cls, mesh, time_end, cfl, source_mode="auto", use_analytic_chain_jvp=False, repeats=3))

    print("Summary (sorted by total time):")
    for r in sorted(results, key=lambda x: x["wall"]):
        print(
            f"- {r['name']}: total={r['wall']:.4f}s, init={r['init']:.4f}s, runtime={r['runtime']:.4f}s"
        )


if __name__ == "__main__":
    run()
