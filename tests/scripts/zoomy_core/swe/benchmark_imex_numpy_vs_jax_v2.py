import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver

from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D


def _run_once(label, solver_cls, model, mesh, time_end, cfl, jv_backend):
    solver = solver_cls(time_end=time_end, compute_dt=timestepping.adaptive(CFL=cfl))
    object.__setattr__(solver, "source_mode", "auto")
    object.__setattr__(solver, "jv_backend", jv_backend)
    object.__setattr__(solver, "implicit_maxiter", 6)
    object.__setattr__(solver, "gmres_maxiter", 30)
    object.__setattr__(solver, "gmres_tol", 1e-7)
    object.__setattr__(solver, "fd_eps", 1e-7)
    Q, _Qaux = solver.solve(mesh, model, write_output=False)
    stats = solver.last_stats
    n = mesh.n_inner_cells
    finite_ratio = float(np.isfinite(Q[:, :n]).sum() / Q[:, :n].size)
    return Q[:, :n], stats, finite_ratio


def _run_case(label, solver_cls, model, mesh, time_end, cfl, jv_backend, repeats=2):
    # Warmup
    _run_once(label, solver_cls, model, mesh, time_end, cfl, jv_backend)
    runs = [
        _run_once(label, solver_cls, model, mesh, time_end, cfl, jv_backend)
        for _ in range(repeats)
    ]
    Q_last = runs[-1][0]
    total = np.mean([r[1].total_time_s for r in runs])
    init = np.mean([r[1].init_time_s for r in runs])
    runtime = np.mean([r[1].runtime_only_s for r in runs])
    implicit = np.mean([r[1].implicit_time_s for r in runs])
    steps = np.mean([r[1].n_steps for r in runs])
    calls = np.mean([r[1].implicit_calls for r in runs])
    finite = np.mean([r[2] for r in runs])
    print("=" * 90)
    print(f"{label}")
    print(f"jv_backend={jv_backend}")
    print(f"steps(avg): {steps:.1f}, implicit_calls(avg): {calls:.1f}")
    print(f"total(avg): {total:.4f}s, init(avg): {init:.4f}s, runtime(avg): {runtime:.4f}s")
    print(f"implicit(avg): {implicit:.4f}s, finite_ratio(avg): {finite:.3f}")
    print("=" * 90)
    return {
        "label": label,
        "jv_backend": jv_backend,
        "Q": Q_last,
        "wall": float(total),
        "init": float(init),
        "runtime": float(runtime),
        "implicit": float(implicit),
        "finite_ratio": float(finite),
    }


def _state_diff(A, B):
    d = B - A
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))


def run():
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=160, lsq_degree=2)
    time_end = 0.06
    cfl = 0.5

    variants = [
        ("NumPy IMEX", IMEXSourceSolver, "analytic"),
        ("NumPy IMEX", IMEXSourceSolver, "fd"),
    ]
    try:
        from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax

        variants.extend(
            [
                ("JAX IMEX", IMEXSourceSolverJax, "analytic"),
                ("JAX IMEX", IMEXSourceSolverJax, "fd"),
                ("JAX IMEX", IMEXSourceSolverJax, "ad"),
            ]
        )
    except Exception as exc:
        print(f"JAX IMEX variants skipped: {repr(exc)}")

    results = []
    for family, solver_cls, jv_backend in variants:
        model = ClassicalGreenNaghdi1D()
        label = f"{family} | GN-classical global source"
        results.append(
            _run_case(label, solver_cls, model, mesh, time_end, cfl, jv_backend, repeats=2)
        )

    # Use NumPy+analytic as reference for cross-backend comparison.
    ref = next(r for r in results if r["label"].startswith("NumPy") and r["jv_backend"] == "analytic")
    print("\nState differences vs reference (NumPy analytic):")
    for r in results:
        l2, linf = _state_diff(ref["Q"], r["Q"])
        print(
            f"- {r['label']} [{r['jv_backend']}]: "
            f"L2={l2:.3e}, Linf={linf:.3e}, total={r['wall']:.4f}s"
        )

    print("\nSorted by total wall time:")
    for r in sorted(results, key=lambda x: x["wall"]):
        print(
            f"- {r['label']} [{r['jv_backend']}]: "
            f"total={r['wall']:.4f}s, implicit={r['implicit']:.4f}s"
        )


if __name__ == "__main__":
    run()
