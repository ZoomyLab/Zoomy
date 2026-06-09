"""
Inclined plane spectral convergence: Galerkin-Legendre vs Galerkin-Chebyshev
with IMEX solver, levels 1-8.

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane_galerkin_imex.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_galerkin_imex"
MAX_LEVEL = 8
BUILD_TIMEOUT = 900


def run_single(args):
    basis_parent, level = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.derivation.basisfunctions import GalerkinBasis
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0

    class _Basis(GalerkinBasis):
        name = f"Galerkin_{basis_parent}"
        def __init__(self, level=0, **kw):
            super().__init__(level=level, parent=basis_parent,
                             bc_bottom="slip", bc_top="nostress",
                             slip_length=SLIP_LAMBDA, **kw)

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

    result = {"basis": basis_parent, "level": level}

    t0 = time.time()
    try:
        a = InclinedPlane(n_layers=1, level=level, dimension=1,
                          basis_type=_Basis, eigenvalue_mode="numerical")
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

    class IMEXGeneratedSolver(_GeneratedModelFluxMixin, IMEXSourceSolver):
        pass
    solver = IMEXGeneratedSolver(time_end=10.0, settings=settings,
                                  compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6)
    object.__setattr__(solver, "source_mode", "local")
    object.__setattr__(solver, "implicit_maxiter", 20)
    object.__setattr__(solver, "implicit_tol", 1e-10)

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
    result["l2_error"] = float(np.sqrt(np.mean((u_num - u_analytical)**2)))
    result["profile_zeta"] = zeta_norm.tolist()
    result["profile_u_numerical"] = u_num.tolist()
    result["profile_u_analytical"] = u_analytical.tolist()
    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Galerkin IMEX convergence: levels 1-{MAX_LEVEL}")
    print()

    all_results = {}
    failed_bases = set()

    for level in range(1, MAX_LEVEL + 1):
        configs = []
        for bp in ["legendre", "chebyshev"]:
            if bp not in failed_bases:
                configs.append((bp, level))

        if not configs:
            break

        with ProcessPoolExecutor(max_workers=len(configs)) as ex:
            futures = {}
            for cfg in configs:
                key = f"{cfg[0]}_L{cfg[1]}"
                futures[key] = (ex.submit(run_single, cfg), cfg)

            for key, (future, cfg) in sorted(futures.items()):
                try:
                    r = future.result(timeout=BUILD_TIMEOUT)
                    all_results[key] = r
                    with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                        json.dump(r, f)
                    if "error" in r:
                        print(f"  {key}: FAILED — {r['error'][:50]}")
                        failed_bases.add(cfg[0])
                    else:
                        print(f"  {key}: Linf={r['linf_error']:.2e} L2={r['l2_error']:.2e} "
                              f"build={r['build_time']:.1f}s solve={r['solve_time']:.1f}s")
                except TimeoutError:
                    print(f"  {key}: TIMEOUT")
                    all_results[key] = {"basis": cfg[0], "level": cfg[1], "error": "TIMEOUT"}
                    failed_bases.add(cfg[0])
                except Exception as e:
                    print(f"  {key}: ERROR — {str(e)[:50]}")
                    failed_bases.add(cfg[0])

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {"status": "error" if "error" in v else "ok",
                        "linf_error": v.get("linf_error"),
                        "l2_error": v.get("l2_error"),
                        "build_time": v.get("build_time", -1),
                        "solve_time": v.get("solve_time", -1)}
                   for k, v in all_results.items()}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def main():
    return run()


if __name__ == "__main__":
    run()
