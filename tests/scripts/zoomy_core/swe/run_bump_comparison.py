"""
Bump flow test: SME (hydrostatic) L0-L4 + VAM model complexity comparison.

Uses Gaussian bathymetry: b = 0.2 * exp(-x^2)
Initial condition: supercritical flow over bump.

SME runs use the ProjectedModel + IMEX solver (proven pipeline).
VAM comparison is symbolic only (model build + equation count) since
the two-step Poisson solver isn't integrated yet.
"""

import os
import sys
import time
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core", "zoomy_jax", "zoomy_tests"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import sympy as sp
from zoomy_core.model.models.ins_generator import StateSpace, Newtonian, Inviscid
from zoomy_core.model.models.model_derivation import derive_shallow_moments
from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.vam_derivation import derive_vam_moments
from zoomy_core.model.models.vam_projected_model import VAMProjectedHyperbolic
from zoomy_core.misc.misc import ZArray

OUTPUT_DIR = "outputs/bump_comparison"


def run_sme_bump(level):
    """Run the hydrostatic SME on the bump test at given level."""
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import _GeneratedModelFluxMixin
    from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
    from zoomy_core.misc.misc import Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.mesh.mesh as petscMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state, material=Inviscid(state))

    class BumpModel(ProjectedModel):
        def source(self_inner):
            p = self_inner.parameters
            h = self_inner.variables[1]
            n_vars = self_inner.n_variables
            n_mom = self_inner.level + 1
            phi_int = self_inner._phi_int
            S = ZArray.zeros(n_vars)
            # Gravity source (inclined component = 0 for flat, but pressure/topo via NC)
            return S

    clear_matrix_cache()
    t0 = time.time()
    model = BumpModel(pre, basis_type=Legendre_shifted, level=level,
                       eigenvalue_mode="numerical")
    build_time = time.time() - t0

    nv = model.n_variables

    # IC: supercritical flow over Gaussian bump
    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        b_val = 0.2 * np.exp(-x[0]**2)
        h0 = max(0.34 - b_val, 0.015) if x[0] < 1 else max(0.015, 0.015)
        Q[0] = b_val                  # bathymetry
        Q[1] = h0                      # water depth
        if x[0] < 1:
            Q[2] = h0 * 0.11197 / 0.34  # hu0 = h * u0
        return Q

    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    try:
        num = NumericalModel(model, boundary_conditions=bcs,
                             initial_conditions=IC.UserFunction(ic))
        pk = list(num.parameters.keys())
        if "g" in pk:
            num.parameter_values[pk.index("g")] = 9.81
    except Exception as e:
        return {"level": level, "error": f"NumericalModel: {e}", "build_time": build_time}

    mesh = petscMesh.Mesh.create_1d(domain=(-1.5, 15.0), n_inner_cells=200)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=2, clean_directory=False))

    class IMEX(_GeneratedModelFluxMixin, IMEXSourceSolver):
        pass

    solver = IMEX(time_end=0.5, settings=settings,
                  compute_dt=timestepping.adaptive(CFL=0.3), min_dt=1e-8)
    object.__setattr__(solver, "source_mode", "local")
    object.__setattr__(solver, "implicit_maxiter", 10)
    object.__setattr__(solver, "implicit_tol", 1e-8)

    t1 = time.time()
    try:
        Q, _ = solver.solve(mesh, num, write_output=False)
        solve_time = time.time() - t1
        h_profile = Q[1, :]
        eta_profile = Q[0, :] + Q[1, :]  # b + h = surface elevation
        return {
            "level": level, "n_vars": nv, "build_time": build_time,
            "solve_time": solve_time, "h_max": float(np.max(h_profile)),
            "eta_max": float(np.max(eta_profile)),
            "h_min": float(np.min(h_profile)),
        }
    except Exception as e:
        return {"level": level, "error": f"solve: {str(e)[:200]}", "build_time": build_time,
                "solve_time": time.time() - t1}


def vam_model_complexity(level):
    """Build a Galerkin VAM model and report complexity."""
    state = StateSpace(dimension=2)
    vam = derive_vam_moments(state, material=Inviscid(state))

    clear_matrix_cache()
    t0 = time.time()
    hyp = VAMProjectedHyperbolic(vam, basis_type=Legendre_shifted, level=level,
                                  eigenvalue_mode='numerical')
    build_time = time.time() - t0

    nv = hyp.n_variables
    F = hyp.flux()
    NC = hyp.nonconservative_matrix()
    R = hyp.source_implicit()
    ev = hyp.eigenvalues()

    flux_nz = sum(1 for i in range(nv) if sp.simplify(F[i, 0]) != 0)
    nc_nz = sum(1 for r in range(nv) for c in range(nv) if NC[r, c, 0] != 0)
    src_nz = sum(1 for i in range(nv) if R[i] != 0)
    ev_nz = sum(1 for i in range(nv) if ev[i] != 0)

    return {
        "level": level, "n_vars": nv, "build_time": build_time,
        "flux_nonzero": flux_nz, "nc_nonzero": nc_nz,
        "source_nonzero": src_nz, "eigenvalues_nonzero": ev_nz,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Part 1: VAM model complexity ----
    print("=" * 60)
    print("VAM MODEL COMPLEXITY (Galerkin, Legendre)")
    print("=" * 60)
    print(f"{'Level':>5s} {'Vars':>5s} {'Build':>8s} {'Flux':>5s} {'NC':>5s} {'Src':>5s} {'EV':>4s}")
    print("-" * 45)

    for level in range(1, 5):
        try:
            info = vam_model_complexity(level)
            print(f"  L{level}  {info['n_vars']:5d} {info['build_time']:7.2f}s "
                  f"{info['flux_nonzero']:5d} {info['nc_nonzero']:5d} "
                  f"{info['source_nonzero']:5d} {info['eigenvalues_nonzero']:4d}")
        except Exception as e:
            print(f"  L{level}  FAILED: {e}")

    # ---- Part 2: SME bump test ----
    print()
    print("=" * 60)
    print("SME BUMP TEST (hydrostatic, Legendre, t=0.5s)")
    print("=" * 60)
    print(f"{'Level':>5s} {'Vars':>5s} {'Build':>8s} {'Solve':>8s} {'h_max':>8s} {'h_min':>8s} {'eta_max':>8s}")
    print("-" * 60)

    for level in range(0, 5):
        r = run_sme_bump(level)
        if "error" in r:
            print(f"  L{level}  FAILED: {r['error'][:60]}")
        else:
            print(f"  L{level}  {r['n_vars']:5d} {r['build_time']:7.2f}s "
                  f"{r['solve_time']:7.2f}s {r['h_max']:8.5f} {r['h_min']:8.5f} "
                  f"{r.get('eta_max', 0):8.5f}")


if __name__ == "__main__":
    main()
