"""
Inclined plane: boundary-layer-resolving spline basis.

Tests non-uniform B-spline knots with clustering near z=0, with and without
no-slip enforcement in the basis.

Configurations:
  1. spline_uniform      — standard uniform knots (baseline)
  2. spline_bl_0.1       — boundary layer delta=0.1
  3. spline_bl_0.05      — boundary layer delta=0.05
  4. spline_bl_0.01      — boundary layer delta=0.01
  5. noslip_bl_0.1       — BL spline with phi_k(0)=0 for k>=1 (no-slip in basis)
  6. noslip_bl_0.05      — same, delta=0.05
  7. legendre            — Legendre reference

Levels 1-4, all with IMEX solver.
"""

import os, json, time, signal, numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FT

OUTPUT_DIR = "outputs/inclined_plane_bl"
TIMEOUT = 300
H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0


def run_single(args):
    key, method, level, extra = args
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("timeout")))
    signal.alarm(TIMEOUT)
    try:
        return _run(key, method, level, extra)
    except Exception as e:
        return {"key": key, "error": str(e)[:300]}
    finally:
        signal.alarm(0)


def _run(key, method, level, extra):
    import sys, os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    for sub in ["zoomy_core", "zoomy_jax", "zoomy_tests"]:
        p = os.path.join(root, "library", sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import sympy
    from sympy import Symbol, Rational, bspline_basis_set, diff
    from sympy.abc import z as z_sym
    from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
    from zoomy_core.model.models.basisfunctions import (
        Legendre_shifted, SplineBasis, Basisfunction,
    )
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    # --- Build basis class ---

    if method == "legendre":
        BasisCls = Legendre_shifted
    elif method == "spline_uniform":
        BasisCls = SplineBasis
    elif method.startswith("spline_bl_"):
        delta = extra["delta"]

        class BasisCls(SplineBasis):
            name = f"BL_spline_d{delta}"

            def _make_knots(self):
                n = self.level + 1
                d = Rational(delta).limit_denominator(1000)
                # Non-uniform: one narrow span [0, delta] at bottom,
                # rest distributed from delta to 1
                n_internal = max(n - self._degree, 1)
                # First internal knot is delta, rest evenly spaced from delta to 1
                if n_internal <= 1:
                    internal = [d]
                else:
                    internal = [d]
                    for i in range(1, n_internal):
                        internal.append(d + (Rational(1) - d) * Rational(i, n_internal - 1 + 1))
                    # Make sure we don't duplicate 1
                    internal = [Rational(0)] + internal + [Rational(1)]
                    internal = sorted(set(internal))

                knots = ([Rational(0)] * (self._degree + 1)
                         + internal[1:-1]
                         + [Rational(1)] * (self._degree + 1))
                return knots

    elif method.startswith("noslip_bl_"):
        delta = extra["delta"]

        class BasisCls(Basisfunction):
            name = f"NoSlip_BL_d{delta}"

            def bounds(self):
                return [0, 1]

            def weight(self, z):
                return sympy.Integer(1)

            def _make_knots(self):
                n = self.level + 1
                d = Rational(delta).limit_denominator(1000)
                n_internal = max(n - 1, 1)
                if n_internal <= 1:
                    internal = [d]
                else:
                    internal = [d]
                    for i in range(1, n_internal):
                        internal.append(d + (Rational(1) - d) * Rational(i, n_internal))
                    internal = [Rational(0)] + internal + [Rational(1)]
                    internal = sorted(set(internal))
                knots = ([Rational(0)] * 2 + internal[1:-1] + [Rational(1)] * 2)
                return knots

            def get_knot_spans(self):
                knots = self._make_knots()
                unique = sorted(set(float(k) for k in knots))
                spans = []
                for i in range(len(unique) - 1):
                    if unique[i + 1] > unique[i]:
                        spans.append((Rational(unique[i]).limit_denominator(10000),
                                      Rational(unique[i + 1]).limit_denominator(10000)))
                return spans

            def basis_definition(self):
                z = Symbol("z")
                knots = self._make_knots()
                raw = bspline_basis_set(1, knots, z)
                n = self.level + 1
                raw = list(raw[:n])
                while len(raw) < n:
                    raw.append(sympy.Integer(0))

                # phi_0 = 1 (constant, for mass conservation)
                # phi_k (k>=1) = B_k  (these already have B_k(0)=0 for BL knots)
                # Check: if B_k(0) != 0, subtract B_0 contribution
                B0_at_0 = raw[0].subs(z, 0)
                result = [sympy.Integer(1)]
                for k in range(1, n):
                    bk_at_0 = raw[k].subs(z, 0)
                    if bk_at_0 != 0 and B0_at_0 != 0:
                        result.append(raw[k] - (bk_at_0 / B0_at_0) * raw[0])
                    else:
                        result.append(raw[k])
                return result

            def mean_coefficients(self):
                # phi_0 = 1, so c = [1, 0, 0, ...]
                c = [sympy.Integer(0)] * (self.level + 1)
                c[0] = sympy.Integer(1)
                return c
    else:
        return {"key": key, "error": f"unknown method {method}"}

    # --- Build and solve ---
    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Newtonian(state))

    class InclinedPlane(ProjectedModel):
        def source(self_inner):
            p = self_inner.parameters
            h_var = self_inner.variables[1]
            n_vars = self_inner.n_variables
            n_mom = self_inner.level + 1
            phi_int = self_inner._phi_int
            S = ZArray.zeros(n_vars)
            raw_grav = [p.g * p.ez * h_var * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                S[2 + k] = self_inner._apply_Minv(raw_grav, k)
            visc = self_inner.newtonian()
            slip = self_inner.slip()
            for i in range(n_vars):
                S[i] = S[i] + visc[i] + slip[i]
            return S

    result = {"key": key, "method": method, "level": level}

    t0 = time.time()
    try:
        clear_matrix_cache()
        a = InclinedPlane(pre, basis_type=BasisCls, level=level,
                          n_layers=1, eigenvalue_mode="numerical")
        result["build_time"] = time.time() - t0
    except Exception as e:
        result["error"] = f"build: {str(e)[:200]}"
        result["build_time"] = time.time() - t0
        return result

    nv = a.n_variables
    bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"),
                                  BC.Extrapolation(tag="right")])
    def ic(x, _nv=nv):
        Q = np.zeros(_nv); Q[1] = H; return Q

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

    # Also compute error in boundary layer [0, 0.1] and bulk [0.1, 1]
    bl_mask = zeta_norm <= 0.1
    bulk_mask = zeta_norm > 0.1
    if bl_mask.any():
        result["bl_linf"] = float(np.max(np.abs(u_num[bl_mask] - u_analytical[bl_mask])))
    if bulk_mask.any():
        result["bulk_linf"] = float(np.max(np.abs(u_num[bulk_mask] - u_analytical[bulk_mask])))

    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    methods = [
        ("legendre",       {}),
        ("spline_uniform", {}),
        ("spline_bl_0.1",  {"delta": 0.1}),
        ("spline_bl_0.05", {"delta": 0.05}),
        ("spline_bl_0.01", {"delta": 0.01}),
        ("noslip_bl_0.1",  {"delta": 0.1}),
        ("noslip_bl_0.05", {"delta": 0.05}),
    ]

    configs = []
    for level in range(1, 5):
        for method, extra in methods:
            key = f"{method}_L{level}"
            configs.append((key, method, level, extra))

    print(f"BL spline study: {len(configs)} configs ({len(methods)} methods x L1-L4)")
    print()

    all_results = {}
    for level in range(1, 5):
        lc = [c for c in configs if c[2] == level]
        print(f"--- Level {level} ---")

        with ProcessPoolExecutor(max_workers=min(len(lc), 8)) as ex:
            futures = {c[0]: ex.submit(run_single, c) for c in lc}
            for k in sorted(futures):
                try:
                    r = futures[k].result(timeout=TIMEOUT + 30)
                    all_results[k] = r
                    if "error" in r:
                        print(f"  {k:30s} FAIL: {r['error'][:50]}")
                    else:
                        bl = f"  bl={r.get('bl_linf', 0):.2e}" if "bl_linf" in r else ""
                        print(f"  {k:30s} Linf={r['linf_error']:.2e}{bl}  "
                              f"build={r['build_time']:.1f}s")
                except FT:
                    all_results[k] = {"key": k, "error": "TIMEOUT"}
                    print(f"  {k:30s} TIMEOUT")
                except Exception as e:
                    print(f"  {k:30s} ERROR: {str(e)[:50]}")
        print()

    # Final table
    print("=" * 95)
    print("CONVERGENCE (Linf error)")
    print("=" * 95)
    header = f"{'Method':25s}"
    for level in range(1, 5):
        header += f" | {'L'+str(level):>10s}"
    print(header)
    print("-" * (27 + 13 * 4))
    for method, _ in methods:
        row = f"{method:25s}"
        for level in range(1, 5):
            k = f"{method}_L{level}"
            r = all_results.get(k, {})
            if "linf_error" in r:
                row += f" | {r['linf_error']:>10.2e}"
            else:
                row += f" | {'FAIL':>10s}"
        print(row)

    # BL vs bulk error table
    print()
    print("=" * 95)
    print("BOUNDARY LAYER vs BULK error (Linf, L3)")
    print("=" * 95)
    print(f"{'Method':25s} | {'BL [0,0.1]':>12s} | {'Bulk [0.1,1]':>12s} | {'Total':>12s}")
    print("-" * 70)
    for method, _ in methods:
        k = f"{method}_L3"
        r = all_results.get(k, {})
        if "linf_error" in r:
            bl = r.get("bl_linf", float("nan"))
            bulk = r.get("bulk_linf", float("nan"))
            print(f"{method:25s} | {bl:>12.2e} | {bulk:>12.2e} | {r['linf_error']:>12.2e}")
        else:
            print(f"{method:25s} | {'FAIL':>12s} | {'FAIL':>12s} | {'FAIL':>12s}")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}/manifest.json")


if __name__ == "__main__":
    run()
