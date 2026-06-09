"""
Inclined plane: parameter sweep for BC enforcement methods.

1. Nitsche penalty: sweep tau = [0.1, 1, 10, 100, 1000]
   Penalizes slip residual: du/dz(0) - u(0)/lambda
2. Modified friction: sweep alpha = [0.01, 0.1, 0.5, 1.0, 2.0]
   gamma_bar = 1/(lambda + alpha * h/(2*nu))
3. Spline basis (source-only slip): levels 1-4
4. Spline basis (Galerkin BC): levels 1-4

All at level 2 for parameter sweeps, then best params at levels 1-4.
All use IMEX solver.
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_sweep"
BUILD_TIMEOUT = 600
H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    method, level, extra_params = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.derivation.basisfunctions import Legendre_shifted, GalerkinBasis, Basisfunction
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    import sympy as sp
    from sympy import Symbol, diff, Rational

    basis_cls = Legendre_shifted

    # Source term varies by method
    _method = method
    _extra = extra_params

    class InclinedPlane(GeneratedShallowModel):
        parameters = {
            "g": (9.81, "positive"), "eps": (1e-6, "positive"), "ez": (1.0, "positive"),
            "rho": (RHO, "positive"), "lamda": (SLIP_LAMBDA, "positive"),
            "nu": (NU, "positive"), "g_x": (G_X, "positive"),
        }

        def source(self_inner):
            p = self_inner.parameters
            n_vars = self_inner.n_variables
            n_mom = self_inner.level + 1
            b, h_var, mu, mv, hinv = self_inner.get_primitives()
            M = self_inner.basismatrices.M
            D = self_inner.basismatrices.D
            phib = self_inner.basismatrices.phib

            S = ZArray.zeros(n_vars)
            S[2] = p.g_x * self_inner.variables[1]

            # Viscous diffusion always present
            visc = self_inner.newtonian()
            for i in range(n_vars):
                S[i] = S[i] + visc[i]

            # Always include standard slip as baseline
            slip = self_inner.slip()
            for i in range(n_vars):
                S[i] = S[i] + slip[i]

            if _method == "modified_friction":
                # ADDITIVE correction: replace slip with modified version
                # First undo the standard slip, then add modified
                for i in range(n_vars):
                    S[i] = S[i] - slip[i]
                alpha_param = _extra.get("alpha", 1.0)
                h_sym = self_inner.variables[1]
                gamma_bar = sp.Integer(1) / (p.lamda + sp.Rational(alpha_param).limit_denominator(1000) * h_sym / (2 * p.nu))
                u_b = sum(mu[0][i] * phib[i] for i in range(n_mom))
                for k in range(n_mom):
                    S[2 + k] = S[2 + k] - gamma_bar * u_b * phib[k] / M[k, k]

            elif _method == "nitsche":
                # ADDITIVE penalty on top of standard slip
                tau = _extra.get("tau", 10.0)
                h_sym = self_inner.variables[1]
                z_sym = Symbol("z")
                dphi_at_0 = [float(diff(self_inner.basisfunctions.get(k), z_sym).subs(z_sym, 0)) for k in range(n_mom)]
                u_b = sum(mu[0][i] * phib[i] for i in range(n_mom))
                du_dz_b = sum(mu[0][i] * dphi_at_0[i] / h_sym for i in range(n_mom))
                slip_residual = du_dz_b - u_b / p.lamda
                for k in range(n_mom):
                    S[2 + k] = S[2 + k] - tau * slip_residual * phib[k] / M[k, k]

            return S

    result = {"method": method, "level": level, "params": extra_params}

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
    class IMEX(_GeneratedModelFluxMixin, IMEXSourceSolver): pass
    solver = IMEX(time_end=10.0, settings=settings,
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
    zeta = np.linspace(0, 1, 200)
    u_num = np.zeros(200)
    for k in range(level + 1):
        phi_fn = basis_obj.get_lambda(k)
        u_num += alphas[k] * np.array(phi_fn(list(zeta)))
    u_analytical = G_X * H**2 / NU * (zeta - zeta**2 / 2 + SLIP_LAMBDA / H)
    result["linf_error"] = float(np.max(np.abs(u_num - u_analytical)))
    result["l2_error"] = float(np.sqrt(np.mean((u_num - u_analytical)**2)))
    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = []

    # Standard baseline at each level
    for level in range(1, 5):
        configs.append(("standard", level, {}))

    # Trivial test: tau=0 and alpha=0 must match standard
    configs.append(("nitsche", 3, {"tau": 0.0}))
    configs.append(("modified_friction", 3, {"alpha": 0.0}))

    # Modified friction: sweep alpha at L3
    for alpha in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]:
        configs.append(("modified_friction", 3, {"alpha": alpha}))

    # Nitsche: sweep tau at L3
    for tau in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        configs.append(("nitsche", 3, {"tau": tau}))

    # Best params at levels 1-4
    for level in range(1, 5):
        configs.append(("modified_friction", level, {"alpha": 0.01}))
        configs.append(("nitsche", level, {"tau": 2.0}))

    print(f"Parameter sweep: {len(configs)} configs")
    print()

    all_results = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as ex:
        futures = {}
        for cfg in configs:
            param_str = "_".join(f"{k}{v}" for k, v in cfg[2].items()) if cfg[2] else "default"
            key = f"{cfg[0]}_L{cfg[1]}_{param_str}"
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
                    print(f"  {key}: Linf={r['linf_error']:.2e}")
            except TimeoutError:
                print(f"  {key}: TIMEOUT")
            except Exception as e:
                print(f"  {key}: ERROR — {str(e)[:50]}")

    # Print summary tables
    print()
    print("=== Modified friction: alpha sweep at L3 ===")
    print("| alpha | Linf | L2 |")
    print("|-------|------|-----|")
    for alpha in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]:
        key = f"modified_friction_L3_alpha{alpha}"
        r = all_results.get(key, {})
        if "linf_error" in r:
            print(f"| {alpha} | {r['linf_error']:.2e} | {r['l2_error']:.2e} |")
        else:
            print(f"| {alpha} | FAIL | — |")

    print()
    print("=== Nitsche penalty: tau sweep at L3 ===")
    print("| tau | Linf | L2 |")
    print("|-----|------|-----|")
    for tau in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        key = f"nitsche_L3_tau{tau}"
        r = all_results.get(key, {})
        if "linf_error" in r:
            print(f"| {tau} | {r['linf_error']:.2e} | {r['l2_error']:.2e} |")
        else:
            print(f"| {tau} | FAIL | — |")

    print()
    print("=== Level convergence (best params) ===")
    print("| Method | L1 | L2 | L3 | L4 |")
    print("|--------|-----|-----|-----|-----|")
    for method, param_str in [("standard", "default"),
                               ("modified_friction", "alpha0.0"),
                               ("modified_friction", "alpha0.01"),
                               ("nitsche", "tau0.0"),
                               ("nitsche", "tau2.0")]:
        row = f"| {method} |"
        for level in range(1, 5):
            key = f"{method}_L{level}_{param_str}"
            r = all_results.get(key, {})
            if "linf_error" in r:
                row += f" {r['linf_error']:.2e} |"
            else:
                row += " FAIL |"
        print(row)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {"status": "error" if "error" in v else "ok",
                        "linf_error": v.get("linf_error"),
                        "l2_error": v.get("l2_error")}
                   for k, v in all_results.items()}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def main():
    return run()


if __name__ == "__main__":
    run()
