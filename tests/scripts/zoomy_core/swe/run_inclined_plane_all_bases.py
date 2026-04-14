"""
Inclined plane convergence: all bases x all BC methods, levels 1-4, parallel.

Bases:
  1. Legendre (standard)
  2. Chebyshev U (shifted to [0,1])
  3. SplineBasis (B-spline hats)
  4. GalerkinBasis (Legendre parent, slip+nostress BCs built into basis)
  5. Nitsche penalty (Legendre + additive penalty tau=2.0)

Analytical solution:
    u(zeta) = g_x h^2 / nu * (zeta - zeta^2/2 + lambda/h)

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane_all_bases.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_all_bases"
MAX_LEVEL = 4
BUILD_TIMEOUT = 300  # 5 min per run max

H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    method, level, extra_params = args

    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Build exceeded {BUILD_TIMEOUT}s")
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(BUILD_TIMEOUT)

    try:
        return _run_single_impl(method, level, extra_params)
    finally:
        signal.alarm(0)


def _run_single_impl(method, level, extra_params):

    from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel
    from zoomy_core.model.models.basisfunctions import (
        Legendre_shifted, SplineBasis, Chebyshevu_shifted, GalerkinBasis,
    )
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    import sympy as sp
    from sympy import Symbol, diff, Rational, S

    # GalerkinBasis needs numeric slip_length to avoid symbolic explosion
    class GalerkinSlip(GalerkinBasis):
        """GalerkinBasis with slip BC at bottom, nostress at top, numeric slip_length."""
        def __init__(self, level=0, **kwargs):
            super().__init__(
                level=level, parent="legendre",
                bc_bottom="slip", bc_top="nostress",
                slip_length=SLIP_LAMBDA, **kwargs,
            )

    basis_map = {
        "legendre":  Legendre_shifted,
        "chebyshev": Chebyshevu_shifted,
        "spline":    SplineBasis,
        "galerkin":  GalerkinSlip,
        "nitsche":   Legendre_shifted,
    }
    basis_cls = basis_map[method]
    _method = method
    _extra = extra_params

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Newtonian(state))

    class InclinedPlaneProjected(ProjectedModel):
        def source(self_inner):
            p = self_inner.parameters
            h_var = self_inner.variables[1]
            n_vars = self_inner.n_variables
            n_mom = self_inner.level + 1
            phi_int = self_inner._phi_int
            phib = self_inner._phib

            S = ZArray.zeros(n_vars)

            # -- Gravity source --
            raw_grav = [p.g * p.ez * h_var * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                S[2 + k] = self_inner._apply_Minv(raw_grav, k)

            # -- Viscous diffusion (always) --
            visc = self_inner.newtonian()
            for i in range(n_vars):
                S[i] = S[i] + visc[i]

            # -- Slip source --
            slip = self_inner.slip()
            for i in range(n_vars):
                S[i] = S[i] + slip[i]

            # -- Nitsche additive penalty --
            if _method == "nitsche":
                tau = _extra.get("tau", 0.5)
                z_sym = Symbol("z")
                dphi_at_0 = [
                    float(diff(self_inner.basisfunctions.get(k), z_sym).subs(z_sym, 0))
                    for k in range(n_mom)
                ]
                b, h_sym, mu, mv, hinv = self_inner.get_primitives()
                u_b = sum(mu[0][i] * phib[i] for i in range(n_mom))
                du_dz_b = sum(mu[0][i] * dphi_at_0[i] * hinv for i in range(n_mom))
                slip_residual = du_dz_b - u_b / p.lamda
                M = self_inner._M
                for k in range(n_mom):
                    S[2 + k] = S[2 + k] - tau * slip_residual * phib[k] / M[k, k]

            return S

    result = {"method": method, "level": level, "params": extra_params}

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
        result["error"] = f"build: {str(e)[:300]}"
        return result

    nv = a.n_variables
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])
    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = H
        return Q

    try:
        num = NumericalModel(
            a, boundary_conditions=bcs,
            initial_conditions=IC.UserFunction(ic),
        )
        param_keys = list(num.parameters.keys())
        for pn, pv in [("lamda", SLIP_LAMBDA), ("nu", NU), ("rho", RHO), ("ez", G_X)]:
            if pn in param_keys:
                num.parameter_values[param_keys.index(pn)] = pv
        if "g" in param_keys:
            num.parameter_values[param_keys.index("g")] = 1.0
    except Exception as e:
        result["error"] = f"NumericalModel: {str(e)[:300]}"
        return result

    mesh = BaseMesh.create_1d(domain=(0., 1.), n_inner_cells=5)
    settings = Zstruct(
        output=Zstruct(
            directory=OUTPUT_DIR, filename="tmp",
            snapshots=2, clean_directory=False,
        ),
    )

    class IMEX(_GeneratedModelFluxMixin, IMEXSourceSolver):
        pass

    solver = IMEX(
        time_end=10.0, settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6,
    )
    object.__setattr__(solver, "source_mode", "local")
    object.__setattr__(solver, "implicit_maxiter", 20)
    object.__setattr__(solver, "implicit_tol", 1e-10)

    t1 = time.time()
    try:
        Q, _ = solver.solve(mesh, num, write_output=False)
        result["solve_time"] = time.time() - t1
    except Exception as e:
        result["solve_time"] = time.time() - t1
        result["error"] = f"solve: {str(e)[:300]}"
        return result

    # -- Extract solution and compute error --
    h_val = Q[1, 0]
    alphas = [float(Q[2 + k, 0]) / max(h_val, 1e-10) for k in range(level + 1)]
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
    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    methods = [
        ("legendre",  {}),
        ("chebyshev", {}),
        ("spline",    {}),
        ("galerkin",  {}),
        ("nitsche",   {"tau": 0.5}),   # tau=0.5 is sweet spot (prior finding)
    ]

    configs = []
    for level in range(1, MAX_LEVEL + 1):
        for method, params in methods:
            configs.append((method, level, params))

    print(f"Inclined plane convergence: {len(methods)} methods x L1-L{MAX_LEVEL}")
    print(f"  Total: {len(configs)} runs (parallel per level)")
    print()

    all_results = {}

    # Run level by level to keep memory bounded, but parallelize within each level
    for level in range(1, MAX_LEVEL + 1):
        level_configs = [c for c in configs if c[1] == level]
        print(f"--- Level {level} ({len(level_configs)} runs) ---")

        with ProcessPoolExecutor(max_workers=len(level_configs)) as ex:
            futures = {}
            for cfg in level_configs:
                param_str = "_".join(f"{k}{v}" for k, v in cfg[2].items()) if cfg[2] else "std"
                key = f"{cfg[0]}_L{cfg[1]}_{param_str}"
                futures[key] = (ex.submit(run_single, cfg), cfg)

            for key, (future, cfg) in sorted(futures.items()):
                try:
                    r = future.result(timeout=BUILD_TIMEOUT)
                    all_results[key] = r
                    if "error" in r:
                        status = f"FAIL: {r['error'][:60]}"
                    else:
                        status = (
                            f"Linf={r['linf_error']:.2e}  L2={r['l2_error']:.2e}  "
                            f"build={r['build_time']:.1f}s  solve={r['solve_time']:.1f}s"
                        )
                    print(f"  {key:35s} {status}")
                except TimeoutError:
                    print(f"  {key:35s} TIMEOUT")
                except Exception as e:
                    print(f"  {key:35s} ERROR: {str(e)[:60]}")

    # ---- Summary table ----
    print()
    print("=" * 90)
    print("CONVERGENCE TABLE: L-infinity error")
    print("=" * 90)
    header = f"{'Method':20s}"
    for level in range(1, MAX_LEVEL + 1):
        header += f" | {'L' + str(level):>12s}"
    print(header)
    print("-" * len(header))

    for method, params in methods:
        param_str = "_".join(f"{k}{v}" for k, v in params.items()) if params else "std"
        label = method
        if params:
            label += f" ({', '.join(f'{k}={v}' for k, v in params.items())})"
        row = f"{label:20s}"
        for level in range(1, MAX_LEVEL + 1):
            key = f"{method}_L{level}_{param_str}"
            r = all_results.get(key, {})
            if "linf_error" in r:
                row += f" | {r['linf_error']:>12.4e}"
            else:
                row += f" | {'FAIL':>12s}"
        print(row)

    # ---- Timing table ----
    print()
    print("=" * 90)
    print("TIMING TABLE: build + solve (seconds)")
    print("=" * 90)
    header = f"{'Method':20s}"
    for level in range(1, MAX_LEVEL + 1):
        header += f" | {'L' + str(level):>12s}"
    print(header)
    print("-" * len(header))

    for method, params in methods:
        param_str = "_".join(f"{k}{v}" for k, v in params.items()) if params else "std"
        label = method
        if params:
            label += f" ({', '.join(f'{k}={v}' for k, v in params.items())})"
        row = f"{label:20s}"
        for level in range(1, MAX_LEVEL + 1):
            key = f"{method}_L{level}_{param_str}"
            r = all_results.get(key, {})
            if "build_time" in r and "solve_time" in r and "error" not in r:
                total = r["build_time"] + r["solve_time"]
                row += f" | {total:>10.1f}s "
            else:
                row += f" | {'--':>12s}"
        print(row)

    # ---- Save results ----
    manifest = {}
    for k, v in all_results.items():
        manifest[k] = {
            "status": "error" if "error" in v else "ok",
            "linf_error": v.get("linf_error"),
            "l2_error": v.get("l2_error"),
            "build_time": v.get("build_time", -1),
            "solve_time": v.get("solve_time", -1),
        }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/manifest.json")


if __name__ == "__main__":
    run()
