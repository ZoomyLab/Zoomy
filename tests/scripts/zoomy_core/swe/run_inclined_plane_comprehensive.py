"""
Comprehensive inclined plane convergence study.

3 bases x 2 BC modes x 3 Nitsche tau values x levels 1-4.

Bases:       Legendre, Chebyshev U, SplineBasis
BC modes:    (a) default — standard ProjectedModel slip
             (b) galerkin — slip+nostress BCs baked into basis polynomials
Nitsche:     tau=0 (control, recovers standard), tau=0.5 (sweet spot), tau=1.0

Analytical solution:
    u(zeta) = g_x h^2 / nu * (zeta - zeta^2/2 + lambda/h)

Runs level-by-level, parallel within each level.
Results printed incrementally so you can see L1/L2 before L3/L4 finish.
"""

import os
import json
import time
import signal
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

OUTPUT_DIR = "outputs/inclined_plane_comprehensive"
RUN_TIMEOUT = 300   # 5 min per config

H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    config_key, basis_name, bc_mode, tau, level = args

    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(
        TimeoutError("run timed out")))
    signal.alarm(RUN_TIMEOUT)

    try:
        return _run(config_key, basis_name, bc_mode, tau, level)
    except Exception as e:
        return {"key": config_key, "error": str(e)[:300]}
    finally:
        signal.alarm(0)


def _run(config_key, basis_name, bc_mode, tau, level):
    from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
    from zoomy_core.model.models.basisfunctions import (
        Legendre_shifted, SplineBasis, Chebyshevu_shifted, GalerkinBasis,
    )
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.mesh.mesh as petscMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    import sympy as sp
    from sympy import Symbol, diff, Rational

    # --- Select basis class ---
    if bc_mode == "galerkin":
        # Galerkin basis with parent matching the requested polynomial family
        if basis_name == "legendre":
            class Basis(GalerkinBasis):
                name = "Galerkin_legendre_slip_nostress"
                def __init__(self, level=0, **kw):
                    super().__init__(level=level, parent="legendre",
                                    bc_bottom="slip", bc_top="nostress",
                                    slip_length=SLIP_LAMBDA, **kw)
        elif basis_name == "chebyshev":
            class Basis(GalerkinBasis):
                name = "Galerkin_chebyshev_slip_nostress"
                def __init__(self, level=0, **kw):
                    super().__init__(level=level, parent="chebyshev",
                                    bc_bottom="slip", bc_top="nostress",
                                    slip_length=SLIP_LAMBDA, **kw)
        else:
            # Spline + galerkin doesn't make sense (splines aren't polynomials
            # in the GalerkinBasis sense). Skip.
            return {"key": config_key, "error": "spline+galerkin not supported"}
    else:
        basis_map = {
            "legendre": Legendre_shifted,
            "chebyshev": Chebyshevu_shifted,
            "spline": SplineBasis,
        }
        Basis = basis_map[basis_name]

    _tau = tau

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Newtonian(state))

    class InclinedPlane(ProjectedModel):
        def source(self_inner):
            p = self_inner.parameters
            h_var = self_inner.variables[1]
            n_vars = self_inner.n_variables
            n_mom = self_inner.level + 1
            phi_int = self_inner._phi_int
            phib = self_inner._phib

            S = ZArray.zeros(n_vars)

            # Gravity
            raw_grav = [p.g * p.ez * h_var * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                S[2 + k] = self_inner._apply_Minv(raw_grav, k)

            # Viscosity
            visc = self_inner.newtonian()
            for i in range(n_vars):
                S[i] = S[i] + visc[i]

            # Slip
            slip = self_inner.slip()
            for i in range(n_vars):
                S[i] = S[i] + slip[i]

            # Nitsche penalty (additive, tau=0 recovers standard)
            if _tau > 0:
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
                    S[2 + k] = S[2 + k] - _tau * slip_residual * phib[k] / M[k, k]

            return S

    result = {"key": config_key, "basis": basis_name, "bc_mode": bc_mode,
              "tau": tau, "level": level}

    t0 = time.time()
    try:
        clear_matrix_cache()
        a = InclinedPlane(pre, basis_type=Basis, level=level,
                          n_layers=1, eigenvalue_mode="numerical")
        result["build_time"] = time.time() - t0
        result["n_vars"] = a.n_variables
    except Exception as e:
        result["build_time"] = time.time() - t0
        result["error"] = f"build: {str(e)[:200]}"
        return result

    nv = a.n_variables
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])

    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = H
        return Q

    try:
        num = NumericalModel(a, boundary_conditions=bcs,
                             initial_conditions=IC.UserFunction(ic))
        pk = list(num.parameters.keys())
        for pn, pv in [("lamda", SLIP_LAMBDA), ("nu", NU), ("rho", RHO),
                        ("ez", G_X), ("g", 1.0)]:
            if pn in pk:
                num.parameter_values[pk.index(pn)] = pv
    except Exception as e:
        result["error"] = f"num: {str(e)[:200]}"
        return result

    mesh = petscMesh.Mesh.create_1d(domain=(0., 1.), n_inner_cells=5)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=2, clean_directory=False))

    class IMEX(_GeneratedModelFluxMixin, IMEXSourceSolver):
        pass

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

    # Extract error
    h_val = Q[1, 0]
    alphas = [float(Q[2 + k, 0]) / max(h_val, 1e-10) for k in range(level + 1)]

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

    bases = ["legendre", "chebyshev", "spline"]
    bc_modes = ["default", "galerkin"]
    taus = [0.0, 0.5, 1.0]
    max_level = 4

    # Build config list
    configs = []
    for level in range(1, max_level + 1):
        for basis in bases:
            for bc_mode in bc_modes:
                for tau in taus:
                    key = f"{basis}_{bc_mode}_tau{tau}_L{level}"
                    configs.append((key, basis, bc_mode, tau, level))

    print(f"Comprehensive inclined plane: {len(configs)} configs")
    print(f"  {len(bases)} bases x {len(bc_modes)} BC modes x {len(taus)} tau x L1-L{max_level}")
    print()

    all_results = {}

    for level in range(1, max_level + 1):
        level_configs = [c for c in configs if c[4] == level]
        print(f"--- Level {level} ({len(level_configs)} configs) ---")

        with ProcessPoolExecutor(max_workers=min(len(level_configs), 8)) as ex:
            futures = {c[0]: ex.submit(run_single, c) for c in level_configs}
            for key in sorted(futures):
                try:
                    r = futures[key].result(timeout=RUN_TIMEOUT + 60)
                    all_results[key] = r
                    if "error" in r:
                        print(f"  {key:45s} FAIL: {r['error'][:50]}")
                    else:
                        print(f"  {key:45s} Linf={r['linf_error']:.2e}  "
                              f"build={r['build_time']:.1f}s  solve={r['solve_time']:.1f}s")
                except FuturesTimeout:
                    all_results[key] = {"key": key, "error": "TIMEOUT"}
                    print(f"  {key:45s} TIMEOUT")
                except Exception as e:
                    all_results[key] = {"key": key, "error": str(e)[:100]}
                    print(f"  {key:45s} ERROR: {str(e)[:50]}")

        # Print level summary table
        print(f"\n  Level {level} summary (Linf error):")
        header = f"  {'':20s}"
        for tau in taus:
            header += f" | {'tau=' + str(tau):>12s}"
        print(header)
        print("  " + "-" * (22 + 15 * len(taus)))
        for basis in bases:
            for bc_mode in bc_modes:
                label = f"{basis}/{bc_mode}"
                row = f"  {label:20s}"
                for tau in taus:
                    key = f"{basis}_{bc_mode}_tau{tau}_L{level}"
                    r = all_results.get(key, {})
                    if "linf_error" in r:
                        err = r["linf_error"]
                        if err > 1e10:
                            row += f" | {'DIVERGED':>12s}"
                        else:
                            row += f" | {err:>12.2e}"
                    else:
                        row += f" | {'FAIL':>12s}"
                row += f"  ({bc_mode[0]})"
                print(row)
        print()

    # Final convergence table
    print("=" * 90)
    print("CONVERGENCE TABLE (Linf error across levels)")
    print("=" * 90)
    for tau in taus:
        print(f"\n  tau = {tau}")
        header = f"  {'Config':25s}"
        for level in range(1, max_level + 1):
            header += f" | {'L' + str(level):>10s}"
        print(header)
        print("  " + "-" * (27 + 13 * max_level))
        for basis in bases:
            for bc_mode in bc_modes:
                if basis == "spline" and bc_mode == "galerkin":
                    continue
                label = f"{basis}/{bc_mode}"
                row = f"  {label:25s}"
                for level in range(1, max_level + 1):
                    key = f"{basis}_{bc_mode}_tau{tau}_L{level}"
                    r = all_results.get(key, {})
                    if "linf_error" in r:
                        err = r["linf_error"]
                        if err > 1e10:
                            row += f" | {'DIVERGED':>10s}"
                        else:
                            row += f" | {err:>10.2e}"
                    else:
                        row += f" | {'FAIL':>10s}"
                print(row)

    # Save
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "key"}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/manifest.json")


if __name__ == "__main__":
    run()
