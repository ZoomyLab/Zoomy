"""
Inclined plane convergence test using ProjectedModel (two-pass derivation).

Compares Legendre vs SplineBasis at levels 1-4 with IMEX solver.
Analytical solution: u(zeta) = g_x * h^2 / nu * (zeta - zeta^2/2 + lambda/h)

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane_projected.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_projected"
MAX_LEVEL = 4
BUILD_TIMEOUT = 900

H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    basis_name, level, use_imex = args

    from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel
    from zoomy_core.model.models.basisfunctions import Legendre_shifted, SplineBasis
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.model.derivative_workflow import DerivativeAwareSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.mesh.mesh as petscMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    basis_map = {
        "legendre": Legendre_shifted,
        "spline": SplineBasis,
    }
    basis_cls = basis_map[basis_name]

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Newtonian(state))

    class InclinedPlaneProjected(ProjectedModel):
        def source(self):
            p = self.parameters
            h = self.variables[1]
            S = ZArray.zeros(self.n_variables)
            n_mom = self.level + 1
            phi_int = self._phi_int
            raw_grav = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                S[2 + k] = self._apply_Minv(raw_grav, k)
            visc = self.newtonian()
            slip = self.slip()
            for i in range(self.n_variables):
                S[i] = S[i] + visc[i] + slip[i]
            return S

    result = {"basis": basis_name, "level": level, "imex": use_imex}

    t0 = time.time()
    try:
        a = InclinedPlaneProjected(
            pre, basis_type=basis_cls, level=level,
            n_layers=1, eigenvalue_mode="numerical",
        )
        result["build_time"] = time.time() - t0
        result["n_vars"] = a.n_variables
    except Exception as e:
        result["build_time"] = time.time() - t0
        result["error"] = f"build: {str(e)[:200]}"
        return result

    nv = a.n_variables
    bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = H
        return Q

    try:
        num = NumericalModel(a, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))
        param_keys = list(num.parameters.keys())
        for pn, pv in [("lamda", SLIP_LAMBDA), ("nu", NU), ("rho", RHO), ("ez", G_X)]:
            if pn in param_keys:
                num.parameter_values[param_keys.index(pn)] = pv
        # g=1.0 so that g*ez = g_x = 1.0
        if "g" in param_keys:
            num.parameter_values[param_keys.index("g")] = 1.0
    except Exception as e:
        result["error"] = f"NumericalModel: {str(e)[:200]}"
        return result

    mesh = petscMesh.Mesh.create_1d(domain=(0., 1.), n_inner_cells=5)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=2, clean_directory=False))

    if use_imex:
        class IMEXGeneratedSolver(_GeneratedModelFluxMixin, IMEXSourceSolver):
            pass
        solver = IMEXGeneratedSolver(
            time_end=10.0, settings=settings,
            compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6,
        )
        object.__setattr__(solver, "source_mode", "local")
        object.__setattr__(solver, "implicit_maxiter", 20)
        object.__setattr__(solver, "implicit_tol", 1e-10)
    else:
        from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
        solver = GeneratedModelSolver(
            time_end=10.0, settings=settings,
            compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6,
        )

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
        for bp in ["legendre", "spline"]:
            configs.append((bp, level, True))   # IMEX only

    print(f"Inclined plane (ProjectedModel): IMEX, levels 1-{MAX_LEVEL}")
    print(f"2 bases x {MAX_LEVEL} levels = {len(configs)} configs")
    print()

    all_results = {}
    for level in range(1, MAX_LEVEL + 1):
        level_configs = [c for c in configs if c[1] == level]

        with ProcessPoolExecutor(max_workers=len(level_configs)) as ex:
            futures = {}
            for cfg in level_configs:
                key = f"{cfg[0]}_L{cfg[1]}_imex"
                futures[key] = (ex.submit(run_single, cfg), cfg)

            for key, (future, cfg) in sorted(futures.items()):
                try:
                    r = future.result(timeout=BUILD_TIMEOUT)
                    all_results[key] = r
                    with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                        json.dump(r, f)
                    if "error" in r:
                        print(f"  {key}: FAILED -- {r['error'][:80]}")
                    else:
                        print(f"  {key}: Linf={r['linf_error']:.2e} build={r['build_time']:.1f}s solve={r['solve_time']:.1f}s")
                except TimeoutError:
                    print(f"  {key}: TIMEOUT")
                except Exception as e:
                    print(f"  {key}: ERROR -- {str(e)[:80]}")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {"status": "error" if "error" in v else "ok",
                        "linf_error": v.get("linf_error"),
                        "build_time": v.get("build_time", -1),
                        "solve_time": v.get("solve_time", -1)}
                   for k, v in all_results.items()}, f, indent=2)

    print()
    print("| Basis | Level | Linf Error | Build (s) | Solve (s) |")
    print("|-------|-------|-----------|-----------|-----------|")
    for level in range(1, MAX_LEVEL + 1):
        for bp in ["legendre", "spline"]:
            key = f"{bp}_L{level}_imex"
            r = all_results.get(key, {})
            err = f"{r['linf_error']:.2e}" if "linf_error" in r else "FAIL"
            bt = f"{r.get('build_time',0):.1f}" if "error" not in r else "--"
            st = f"{r.get('solve_time',0):.1f}" if "error" not in r else "--"
            print(f"| {bp:10s} | {level} | {err:>10s} | {bt:>9s} | {st:>9s} |")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
