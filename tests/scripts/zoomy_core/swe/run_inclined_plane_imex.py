"""
Inclined plane convergence with IMEX solver (implicit source).
Standard Legendre, Galerkin-Legendre, Galerkin-Chebyshev, levels 1-4.

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane_imex.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_imex"
MAX_LEVEL = 4
BUILD_TIMEOUT = 900


def run_single(args):
    basis_parent, level, use_imex = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.models.basisfunctions import GalerkinBasis, Legendre_shifted
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.model.derivative_workflow import DerivativeAwareSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0

    if basis_parent == "standard_legendre":
        basis_cls = Legendre_shifted
    else:
        class _Basis(GalerkinBasis):
            name = f"Galerkin_{basis_parent}"
            def __init__(self, level=0, **kw):
                super().__init__(level=level, parent=basis_parent,
                                 bc_bottom="slip", bc_top="nostress",
                                 slip_length=SLIP_LAMBDA, **kw)
        basis_cls = _Basis

    class InclinedPlane(GeneratedShallowModel):
        parameters = {
            "g": (9.81, "positive"), "eps": (1e-6, "positive"), "ez": (1.0, "positive"),
            "rho": (RHO, "positive"), "lamda": (SLIP_LAMBDA, "positive"),
            "nu": (NU, "positive"), "g_x": (G_X, "positive"),
        }
        def source(self):
            p = self.parameters
            S = ZArray.zeros(self.n_variables)
            S[2] = p.g_x * self.variables[1]
            visc = self.newtonian(); slip = self.slip()
            for i in range(self.n_variables):
                S[i] = S[i] + visc[i] + slip[i]
            return S

    result = {"basis": basis_parent, "level": level, "imex": use_imex}

    t0 = time.time()
    try:
        a = InclinedPlane(n_layers=1, level=level, dimension=1,
                          basis_type=basis_cls, eigenvalue_mode="numerical")
        result["build_time"] = time.time() - t0
        result["n_vars"] = a.n_variables
    except Exception as e:
        result["build_time"] = time.time() - t0
        result["error"] = f"build: {str(e)[:200]}"
        return result

    nv = a.n_variables
    bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
    def ic(x, _nv=nv):
        Q = np.zeros(_nv); Q[1] = H; return Q

    try:
        num = NumericalModel(a, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))
        for pn, pv in [("lamda", SLIP_LAMBDA), ("nu", NU), ("g_x", G_X), ("rho", RHO)]:
            num.parameter_values[list(num.parameters.keys()).index(pn)] = pv
    except Exception as e:
        result["error"] = f"NumericalModel: {str(e)[:200]}"
        return result

    mesh = BaseMesh.create_1d(domain=(0., 1.), n_inner_cells=5)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=2, clean_directory=False))

    if use_imex:
        class IMEXGeneratedSolver(_GeneratedModelFluxMixin, IMEXSourceSolver):
            pass
        solver = IMEXGeneratedSolver(time_end=10.0, settings=settings,
                                      compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6)
        object.__setattr__(solver, "source_mode", "local")
        object.__setattr__(solver, "implicit_maxiter", 20)
        object.__setattr__(solver, "implicit_tol", 1e-10)
    else:
        from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
        solver = GeneratedModelSolver(time_end=10.0, settings=settings,
                                       compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6)

    t1 = time.time()
    try:
        Q, _ = solver.solve(mesh, num, write_output=False)
        result["solve_time"] = time.time() - t1
    except Exception as e:
        result["solve_time"] = time.time() - t1
        result["error"] = f"solve: {str(e)[:200]}"
        return result

    h = Q[1, 0]
    alphas = [float(Q[2 + k, 0]) / max(h, 1e-10) for k in range(level + 1)]
    result["alphas"] = alphas

    basis_obj = a.basisfunctions
    z_lo, z_hi = float(basis_obj.bounds()[0]), float(basis_obj.bounds()[1])
    zeta_b = np.linspace(z_lo, z_hi, 200)
    u_num = np.zeros(200)
    for k in range(level + 1):
        phi_fn = basis_obj.get_lambda(k)
        u_num += alphas[k] * np.array(phi_fn(list(zeta_b)))
    zeta_norm = (zeta_b - z_lo) / (z_hi - z_lo)
    u_analytical = G_X * H**2 / NU * (zeta_norm - zeta_norm**2 / 2 + SLIP_LAMBDA / H)

    result["linf_error"] = float(np.max(np.abs(u_num - u_analytical)))
    result["profile_zeta"] = zeta_norm.tolist()
    result["profile_u_numerical"] = u_num.tolist()
    result["profile_u_analytical"] = u_analytical.tolist()
    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = []
    for level in range(1, MAX_LEVEL + 1):
        for bp in ["standard_legendre", "legendre", "chebyshev"]:
            configs.append((bp, level, False))  # explicit
            configs.append((bp, level, True))   # IMEX

    print(f"Inclined plane: explicit vs IMEX, levels 1-{MAX_LEVEL}")
    print(f"3 bases x {MAX_LEVEL} levels x 2 solvers = {len(configs)} configs")
    print()

    all_results = {}
    # Run level by level, parallel within each level
    for level in range(1, MAX_LEVEL + 1):
        level_configs = [(bp, level, imex) for bp, l, imex in configs if l == level]

        with ProcessPoolExecutor(max_workers=min(len(level_configs), 6)) as ex:
            futures = {}
            for cfg in level_configs:
                tag = "imex" if cfg[2] else "expl"
                key = f"{cfg[0]}_L{cfg[1]}_{tag}"
                futures[key] = (ex.submit(run_single, cfg), cfg)

            for key, (future, cfg) in sorted(futures.items()):
                try:
                    r = future.result(timeout=BUILD_TIMEOUT)
                    all_results[key] = r
                    with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                        json.dump(r, f)
                    if "error" in r:
                        print(f"  {key}: FAILED — {r['error'][:50]}")
                    else:
                        print(f"  {key}: Linf={r['linf_error']:.2e} build={r['build_time']:.1f}s solve={r['solve_time']:.1f}s")
                except TimeoutError:
                    print(f"  {key}: TIMEOUT")
                except Exception as e:
                    print(f"  {key}: ERROR — {str(e)[:50]}")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {"status": "error" if "error" in v else "ok",
                        "linf_error": v.get("linf_error"),
                        "build_time": v.get("build_time", -1),
                        "solve_time": v.get("solve_time", -1)}
                   for k, v in all_results.items()}, f, indent=2)

    # Print summary table
    print()
    print("| Basis | L | Explicit Linf | IMEX Linf | Expl time | IMEX time |")
    print("|-------|---|--------------|-----------|-----------|-----------|")
    for level in range(1, MAX_LEVEL + 1):
        for bp in ["standard_legendre", "legendre", "chebyshev"]:
            ek = f"{bp}_L{level}_expl"
            ik = f"{bp}_L{level}_imex"
            re = all_results.get(ek, {})
            ri = all_results.get(ik, {})
            e_err = f"{re['linf_error']:.2e}" if "linf_error" in re else "FAIL"
            i_err = f"{ri['linf_error']:.2e}" if "linf_error" in ri else "FAIL"
            e_time = f"{re.get('solve_time',0):.1f}s" if "error" not in re else "—"
            i_time = f"{ri.get('solve_time',0):.1f}s" if "error" not in ri else "—"
            print(f"| {bp} | {level} | {e_err} | {i_err} | {e_time} | {i_time} |")

    print(f"\nResults saved to {OUTPUT_DIR}/")


def main():
    return run()


if __name__ == "__main__":
    run()
