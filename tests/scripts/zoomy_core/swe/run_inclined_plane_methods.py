"""
Inclined plane: comparison of BC enforcement methods.

Method 1: Standard Legendre + original slip (baseline, works at L2 with IMEX)
Method 2: Standard Legendre + modified friction (Huang et al. 2025)
Method 3: Standard Legendre + Nitsche penalty
Method 4: Galerkin-Legendre (BCs in basis)
Method 5: Spline basis + source-only slip
Method 6: Spline basis + Galerkin BC-built-in

All use IMEX solver. Levels 1-6.

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane_methods.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/inclined_plane_methods"
MAX_LEVEL = 6
BUILD_TIMEOUT = 900

H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    method, level = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.models.basisfunctions import (
        Legendre_shifted, GalerkinBasis, Basisfunction,
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
    from sympy import Symbol, Rational, bspline_basis_set

    # --- Basis selection ---
    if method in ("standard", "modified_friction", "nitsche"):
        basis_cls = Legendre_shifted
    elif method == "galerkin_legendre":
        class _GLeg(GalerkinBasis):
            name = "GalerkinLeg"
            def __init__(self, level=0, **kw):
                super().__init__(level=level, parent="legendre",
                                 bc_bottom="slip", bc_top="nostress",
                                 slip_length=SLIP_LAMBDA, **kw)
        basis_cls = _GLeg
    elif method in ("spline_source", "spline_galerkin"):
        # B-spline basis on [0,1] with uniform knots
        class _SplineBasis(Basisfunction):
            name = "BSpline_uniform"
            def bounds(self):
                return [0, 1]
            def weight(self, z):
                return sp.Integer(1)
            def basis_definition(self, degree=1, knots=None):
                z_sym = Symbol("z")
                n = self.level + 1
                if knots is None:
                    # Uniform knots with clamped ends
                    n_internal = max(n - degree + 1, 1)
                    internal = [Rational(i, n_internal) for i in range(n_internal + 1)]
                    knots = [Rational(0)] * degree + internal + [Rational(1)] * degree
                basis = bspline_basis_set(degree, knots, z_sym)
                # Normalize: first basis = 1 (constant)
                result = [sp.Integer(1)]
                for i, b in enumerate(basis[1:min(n, len(basis))]):
                    result.append(b)
                # Pad if needed
                while len(result) < n:
                    result.append(sp.Integer(0))
                return result[:n]
        if method == "spline_galerkin":
            class _SplineGalerkin(GalerkinBasis):
                name = "SplineGalerkin"
                def __init__(self, level=0, **kw):
                    # Use Legendre parent for now — spline Galerkin needs more work
                    super().__init__(level=level, parent="legendre",
                                     bc_bottom="slip", bc_top="nostress",
                                     slip_length=SLIP_LAMBDA, **kw)
            basis_cls = _SplineGalerkin
        else:
            basis_cls = _SplineBasis
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Model with appropriate source ---
    class InclinedPlane(GeneratedShallowModel):
        parameters = {
            "g": (9.81, "positive"), "eps": (1e-6, "positive"), "ez": (1.0, "positive"),
            "rho": (RHO, "positive"), "lamda": (SLIP_LAMBDA, "positive"),
            "nu": (NU, "positive"), "g_x": (G_X, "positive"),
            "nitsche_tau": (100.0, "positive"),
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

            # Gravity on mean velocity
            S[2] = p.g_x * self_inner.variables[1]

            # Viscous diffusion (always present)
            visc = self_inner.newtonian()
            for i in range(n_vars):
                S[i] = S[i] + visc[i]

            if method == "modified_friction":
                # Huang et al.: gamma_bar = 1 / (lambda + h/(2*nu))
                # S_0 = -gamma_bar * u_mean = -gamma_bar * alpha_0
                h_sym = self_inner.variables[1]
                gamma_bar = sp.Integer(1) / (p.lamda + h_sym / (2 * p.nu))
                alpha_0 = self_inner.variables[2] * hinv
                S[2] = S[2] - gamma_bar * alpha_0
                # Higher moments: no direct friction, only viscous damping

            elif method == "nitsche":
                # Penalty: S_k += -tau/h * (u_b - u_target) * phi_k(0) / M[k,k]
                # For slip: u_target = lambda * du/dz|_bottom
                # Simplification: u_target = 0 for strong penalty
                tau = p.nitsche_tau
                h_sym = self_inner.variables[1]
                u_b = sum(mu[0][i] * phib[i] for i in range(n_mom))
                for k in range(n_mom):
                    S[2 + k] = S[2 + k] - tau / h_sym * u_b * phib[k] / M[k, k]

            else:
                # Standard slip or Galerkin (slip still needed for phi_0)
                slip = self_inner.slip()
                for i in range(n_vars):
                    S[i] = S[i] + slip[i]

            return S

    result = {"method": method, "level": level}

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

    class IMEXSolver(_GeneratedModelFluxMixin, IMEXSourceSolver):
        pass
    solver = IMEXSolver(time_end=10.0, settings=settings,
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

    methods = ["standard", "modified_friction", "nitsche", "galerkin_legendre"]
    # Skip spline for now — needs more work on basis definition

    print(f"Inclined plane: {len(methods)} methods x levels 1-{MAX_LEVEL}")
    print(f"Methods: {methods}")
    print()

    all_results = {}
    failed = set()

    for level in range(1, MAX_LEVEL + 1):
        configs = [(m, level) for m in methods if m not in failed]

        with ProcessPoolExecutor(max_workers=min(len(configs), 6)) as ex:
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
                        print(f"  {key}: FAILED — {r['error'][:60]}")
                    else:
                        print(f"  {key}: Linf={r['linf_error']:.2e} L2={r['l2_error']:.2e} "
                              f"build={r['build_time']:.1f}s solve={r['solve_time']:.1f}s")
                except TimeoutError:
                    print(f"  {key}: TIMEOUT")
                    failed.add(cfg[0])
                except Exception as e:
                    print(f"  {key}: ERROR — {str(e)[:50]}")

    # Summary table
    print()
    print("| Method | L | Linf | L2 | build | solve |")
    print("|--------|---|------|-----|-------|-------|")
    for level in range(1, MAX_LEVEL + 1):
        for m in methods:
            key = f"{m}_L{level}"
            r = all_results.get(key, {})
            if "linf_error" in r:
                print(f"| {m} | {level} | {r['linf_error']:.2e} | {r['l2_error']:.2e} | {r['build_time']:.1f}s | {r['solve_time']:.1f}s |")
            elif "error" in r:
                print(f"| {m} | {level} | FAIL | — | — | — |")

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
