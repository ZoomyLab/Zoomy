"""
Hybrid Chebyshev basis: phi_0=1, phi_1=linear, phi_k=U_k for k>=2.

Two variants:
  1. Raw (non-orthogonal): just use {1, z, U_2, U_3, ...} with Chebyshev weight.
     M is full. Compute M^{-1}.
  2. Gram-Schmidt orthogonalized w.r.t. Chebyshev weight sqrt(z(1-z)).
     M is diagonal by construction.

Both span the SAME function space, so the Galerkin projection should be
mathematically identical.  This script verifies that and measures performance.

Also compares against standard Legendre and standard Chebyshev.
"""

import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core", "zoomy_jax", "zoomy_tests"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import time
import numpy as np
import sympy
from sympy import Symbol, Rational, sqrt, pi, integrate, diff, Poly
from sympy.abc import z
from sympy.functions.special.polynomials import chebyshevu

from zoomy_core.model.models.basisfunctions import (
    Basisfunction, Legendre_shifted, Chebyshevu_shifted,
)
from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
from zoomy_core.model.models.model_derivation import derive_shallow_moments
from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
from zoomy_core.misc.misc import ZArray


# ---------------------------------------------------------------------------
# Weighted inner product helper (Chebyshev weight on [0,1])
# ---------------------------------------------------------------------------

# Cache the Chebyshevu_shifted analytical integrator
_cheb_integrator = Chebyshevu_shifted(level=0)

def inner(f, g):
    """<f, g>_w = int_0^1 f*g*sqrt(z*(1-z)) dz, exact."""
    product = sympy.expand(f * g)
    result = _cheb_integrator.analytical_weighted_integral(product, z)
    if result is not None:
        return result
    # Fallback
    return integrate(product * sqrt(z * (1 - z)), (z, 0, 1))


def norm_w(f):
    return sympy.sqrt(inner(f, f))


# ---------------------------------------------------------------------------
# Variant 1: Raw hybrid (non-orthogonal)
# ---------------------------------------------------------------------------

class HybridChebyshev(Basisfunction):
    """
    phi_0 = 1, phi_1 = z, phi_k = U_k(2z-1) for k >= 2.
    Weight = sqrt(z(1-z)).  M is NOT diagonal.
    """
    name = "HybridCheby_raw"

    def bounds(self):
        return [0, 1]

    def weight(self, z):
        return sympy.sqrt(z * (1 - z))

    def basis_definition(self):
        z_sym = Symbol("z")
        basis = [sympy.Integer(1), z_sym]
        for k in range(2, self.level + 1):
            basis.append(chebyshevu(k, 2 * z_sym - 1))
        return basis

    def analytical_weighted_integral(self, poly_expr, var):
        return _cheb_integrator.analytical_weighted_integral(poly_expr, var)

    def mean_coefficients(self):
        c = [sympy.Integer(0)] * (self.level + 1)
        c[0] = sympy.Integer(1)
        return c


# ---------------------------------------------------------------------------
# Variant 2: Gram-Schmidt orthogonalized
# ---------------------------------------------------------------------------

class HybridChebyshevOrtho(Basisfunction):
    """
    Gram-Schmidt orthogonalized basis w.r.t. Chebyshev weight sqrt(z(1-z)).

    Construction:
      1. Start with phi_0 = 1  (kept, not modified)
      2. Take z, orthogonalize against phi_0 → phi_1 = z - 1/2
      3. Take U_2(2z-1), orthogonalize against phi_0, phi_1 → phi_2
      4. Take U_3(2z-1), orthogonalize against phi_0..phi_2 → phi_3
      etc.

    M is diagonal by construction.  phi_0 = 1 is preserved so the
    SWE limit is exact.
    """
    name = "HybridCheby_ortho"

    def bounds(self):
        return [0, 1]

    def weight(self, z):
        return sympy.sqrt(z * (1 - z))

    def basis_definition(self):
        z_sym = Symbol("z")
        # phi_0 = 1, phi_1 comes from z, phi_k comes from U_k(2z-1)
        raw = [sympy.Integer(1), z_sym]
        for k in range(2, self.level + 1):
            raw.append(chebyshevu(k, 2 * z_sym - 1))

        # Gram-Schmidt: phi_0 = 1 is KEPT, rest orthogonalized against all previous
        ortho = []
        for k in range(len(raw)):
            v = raw[k]
            for j in range(len(ortho)):
                proj = inner(raw[k], ortho[j]) / inner(ortho[j], ortho[j])
                v = v - proj * ortho[j]
            v = sympy.simplify(sympy.expand(v))
            ortho.append(v)

        return [sympy.expand(o) for o in ortho]

    def analytical_weighted_integral(self, poly_expr, var):
        return _cheb_integrator.analytical_weighted_integral(poly_expr, var)

    def mean_coefficients(self):
        c = [sympy.Integer(0)] * (self.level + 1)
        c[0] = sympy.Integer(1)
        return c


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Newtonian(state))
    level = 2

    bases = {
        "Legendre":           Legendre_shifted,
        "Chebyshev_std":      Chebyshevu_shifted,
        "HybridCheby_raw":    HybridChebyshev,
        "HybridCheby_ortho":  HybridChebyshevOrtho,
    }

    # --- Step 1: Show basis functions ---
    print("=" * 70)
    print(f"BASIS FUNCTIONS (level {level})")
    print("=" * 70)
    for name, cls in bases.items():
        b = cls(level=level)
        print(f"\n{name}:")
        for k in range(level + 1):
            phi = b.get(k)
            val0 = phi.subs(z, 0) if hasattr(phi, 'subs') else phi
            print(f"  phi_{k} = {phi}")
            print(f"       phi_{k}(0) = {val0}")

    # --- Step 2: Build models and compare ---
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON (level {level})")
    print("=" * 70)

    models = {}
    for name, cls in bases.items():
        clear_matrix_cache()
        t0 = time.time()
        try:
            m = ProjectedModel(pre, basis_type=cls, level=level,
                               eigenvalue_mode="numerical")
            dt = time.time() - t0
            models[name] = m
            print(f"\n{name}: build={dt:.2f}s, n_vars={m.n_variables}")
        except Exception as e:
            print(f"\n{name}: FAILED — {str(e)[:100]}")
            continue

        M = m.mass_matrix()
        is_diag = M == sympy.diag(*[M[i, i] for i in range(level + 1)])
        print(f"  M diagonal? {is_diag}")
        sympy.pprint(M)

        phib = [m._phib[k] for k in range(level + 1)]
        print(f"  phi_k(0) = {phib}")
        print(f"  c_mean   = {m.mean_coefficients()}")

    # --- Step 3: Compare flux/source ---
    print("\n" + "=" * 70)
    print("FLUX AND SOURCE COMPARISON")
    print("=" * 70)

    for name in models:
        m = models[name]
        F = m.flux()
        P = m.hydrostatic_pressure()
        visc = m.newtonian()
        slip = m.slip()

        print(f"\n--- {name} ---")
        print("  Flux (x):")
        for i in range(m.n_variables):
            val = sympy.simplify(F[i, 0])
            if val != 0:
                print(f"    F[{i}] = {val}")

        print("  Pressure:")
        for i in range(m.n_variables):
            val = P[i, 0]
            if val != 0:
                print(f"    P[{i}] = {val}")

        print("  Slip:")
        for i in range(m.n_variables):
            val = slip[i]
            if val != 0:
                print(f"    S_slip[{i}] = {sympy.simplify(val)}")

    # --- Step 4: Check if raw and ortho give same model ---
    if "HybridCheby_raw" in models and "HybridCheby_ortho" in models:
        print("\n" + "=" * 70)
        print("RAW vs ORTHO: are the models identical?")
        print("=" * 70)
        m_raw = models["HybridCheby_raw"]
        m_ort = models["HybridCheby_ortho"]

        # Evaluate at a test point
        nv = m_raw.n_variables
        test_vals = {m_raw.variables[i]: Rational(i + 1, 3) for i in range(nv)}
        test_params = {}
        for p_sym, (p_val, _) in zip(m_raw.parameters.values(),
                                      [(1, ""), (1e-6, ""), (1, ""),
                                       (1000, ""), (Rational(1, 2), ""), (1, "")]):
            test_params[p_sym] = p_val
        subs = {**test_vals, **test_params}

        # The flux/source expressions WILL differ because q_k means different
        # things in different bases.  What should match is the reconstructed
        # velocity profile.  We verify that in the inclined plane test below.
        print("  (Symbolic expressions differ because q_k = h*alpha_k and alpha_k")
        print("   multiplies different basis functions.  The velocity profile is what")
        print("   should match — verified via inclined plane below.)")

    # --- Step 5: Inclined plane convergence ---
    print("\n" + "=" * 70)
    print("INCLINED PLANE CONVERGENCE (L1-L4)")
    print("=" * 70)

    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.mesh.mesh as petscMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    H, G_X, NU, SLIP_LAMBDA, RHO = 1.0, 1.0, 1.0, 0.5, 1.0

    test_bases = {
        "Legendre":          Legendre_shifted,
        "HybridCheby_raw":   HybridChebyshev,
        "HybridCheby_ortho": HybridChebyshevOrtho,
    }

    results = {}
    for lvl in range(1, 5):
        for bname, bcls in test_bases.items():
            key = f"{bname}_L{lvl}"
            clear_matrix_cache()

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

            try:
                t0 = time.time()
                a = InclinedPlane(pre, basis_type=bcls, level=lvl,
                                  n_layers=1, eigenvalue_mode="numerical")
                build_t = time.time() - t0

                nv = a.n_variables
                bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"),
                                              BC.Extrapolation(tag="right")])
                def ic(x, _nv=nv):
                    Q = np.zeros(_nv); Q[1] = H; return Q

                num = NumericalModel(a, boundary_conditions=bcs,
                                     initial_conditions=IC.UserFunction(ic))
                pk = list(num.parameters.keys())
                for pn, pv in [("lamda", SLIP_LAMBDA), ("nu", NU), ("rho", RHO),
                                ("ez", G_X), ("g", 1.0)]:
                    if pn in pk:
                        num.parameter_values[pk.index(pn)] = pv

                mesh = petscMesh.Mesh.create_1d(domain=(0., 1.), n_inner_cells=5)
                settings = Zstruct(output=Zstruct(directory="outputs/hybrid_cheby",
                                                   filename="tmp", snapshots=2,
                                                   clean_directory=False))
                class IMEX(_GeneratedModelFluxMixin, IMEXSourceSolver): pass
                solver = IMEX(time_end=10.0, settings=settings,
                              compute_dt=timestepping.adaptive(CFL=0.45), min_dt=1e-6)
                object.__setattr__(solver, "source_mode", "local")
                object.__setattr__(solver, "implicit_maxiter", 20)
                object.__setattr__(solver, "implicit_tol", 1e-10)

                t1 = time.time()
                Q, _ = solver.solve(mesh, num, write_output=False)
                solve_t = time.time() - t1

                h_val = Q[1, 0]
                alphas = [float(Q[2 + k, 0]) / max(h_val, 1e-10)
                          for k in range(lvl + 1)]
                basis_obj = a.basisfunctions
                z_lo, z_hi = float(basis_obj.bounds()[0]), float(basis_obj.bounds()[1])
                zeta = np.linspace(z_lo, z_hi, 200)
                u_num = np.zeros(200)
                for k in range(lvl + 1):
                    phi_fn = basis_obj.get_lambda(k)
                    u_num += alphas[k] * np.array(phi_fn(list(zeta)))
                zeta_n = (zeta - z_lo) / (z_hi - z_lo)
                u_ana = G_X * H**2 / NU * (zeta_n - zeta_n**2 / 2 + SLIP_LAMBDA / H)

                err = float(np.max(np.abs(u_num - u_ana)))
                results[key] = err
                print(f"  {key:30s} Linf={err:.2e}  build={build_t:.1f}s  solve={solve_t:.1f}s")
            except Exception as e:
                results[key] = None
                print(f"  {key:30s} FAIL: {str(e)[:60]}")

    # Final table
    print(f"\n{'Method':25s}", end="")
    for lvl in range(1, 5):
        print(f" | {'L'+str(lvl):>10s}", end="")
    print()
    print("-" * (27 + 13 * 4))
    for bname in test_bases:
        print(f"{bname:25s}", end="")
        for lvl in range(1, 5):
            key = f"{bname}_L{lvl}"
            err = results.get(key)
            if err is not None:
                print(f" | {err:>10.2e}", end="")
            else:
                print(f" | {'FAIL':>10s}", end="")
        print()


if __name__ == "__main__":
    os.makedirs("outputs/hybrid_cheby", exist_ok=True)
    main()
