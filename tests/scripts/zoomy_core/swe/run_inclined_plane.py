"""
Inclined plane rundown with Navier-slip: spectral accuracy test.

Analytical solution: u(zeta) = g_x*h^2/nu * (zeta - zeta^2/2 + lambda/h)
This is quadratic in zeta, so level 2 should capture it exactly.

Runs Legendre and Chebyshev at levels 0-3, measures:
1. Velocity profile error vs analytical
2. Whether alpha_3 = 0 at level 3
3. Time to reach steady state

Usage:
    python tests/scripts/zoomy_core/swe/run_inclined_plane.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

OUTPUT_DIR = "outputs/inclined_plane"


def analytical_profile(zeta, h, g_x, nu, lam):
    """Exact steady-state velocity profile."""
    return g_x * h**2 / nu * (zeta - zeta**2 / 2 + lam / h)


def analytical_moments_legendre(h, g_x, nu, lam):
    """Exact Legendre moment coefficients (orthogonal weight=1)."""
    return {
        0: g_x * h * (h + 3 * lam) / (3 * nu),
        1: -g_x * h**2 / (4 * nu),
        2: -g_x * h**2 / (12 * nu),
        3: 0.0,
    }


def run_single(args):
    """Run one (basis, level) config."""
    basis_name, level, weight_mode = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.models.basisfunctions import Legendre_shifted, Chebyshevu_shifted
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    import sympy as sp

    # Physical parameters (t_char = h^2/nu = 1s, rho=1 for clean units)
    H = 1.0
    G_X = 1.0
    NU = 1.0
    SLIP_LAMBDA = 0.5
    RHO = 1.0
    TIME_END = 10.0
    N_CELLS = 5
    CFL = 0.45

    basis_cls = Legendre_shifted if basis_name == "Legendre" else Chebyshevu_shifted

    class InclinedPlaneModel(GeneratedShallowModel):
        """SWE on inclined plane: gravity source + viscous + slip."""
        parameters = {
            "g": (9.81, "positive"),
            "eps": (1e-6, "positive"),
            "ez": (1.0, "positive"),
            "rho": (RHO, "positive"),
            "lamda": (SLIP_LAMBDA, "positive"),
            "nu": (NU, "positive"),
            "g_x": (G_X, "positive"),
        }

        def source(self):
            p = self.parameters
            n_vars = self.n_variables
            n_mom = self.level + 1
            b, h, mu, mv, hinv = self.get_primitives()
            M = self.basismatrices.M

            S = ZArray.zeros(n_vars)

            # Gravity source: g_x * h * integral(phi_k) / M[k,k]
            # For phi_0=1: integral = 1. For higher k: integral=0 (orthogonal to constant)
            # So only S[2] (alpha_0 equation) gets gravity
            S[2] = p.g_x * self.variables[1]

            # Viscous diffusion
            visc = self.newtonian()
            for i in range(n_vars):
                S[i] = S[i] + visc[i]

            # Bottom slip
            slip = self.slip()
            for i in range(n_vars):
                S[i] = S[i] + slip[i]

            return S

    result = {"basis": basis_name, "level": level}

    t0 = time.time()
    try:
        a = InclinedPlaneModel(n_layers=1, level=level, dimension=1, basis_type=basis_cls,
                               eigenvalue_mode="numerical", weight_mode=weight_mode)
        result["build_time"] = time.time() - t0
        result["n_vars"] = a.n_variables
    except Exception as e:
        result["build_time"] = time.time() - t0
        result["error"] = f"build: {str(e)[:200]}"
        return result

    nv = a.n_variables
    # Periodic BCs
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    # IC: uniform h, zero velocity
    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = H
        return Q

    try:
        num = NumericalModel(a, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))
        # Set parameters
        for pname, pval in [("lamda", SLIP_LAMBDA), ("nu", NU), ("g_x", G_X), ("rho", RHO)]:
            idx = list(num.parameters.keys()).index(pname)
            num.parameter_values[idx] = pval
    except Exception as e:
        result["error"] = f"NumericalModel: {str(e)[:200]}"
        return result

    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=N_CELLS)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=100, clean_directory=False))
    solver = GeneratedModelSolver(time_end=TIME_END, settings=settings,
                                   compute_dt=timestepping.adaptive(CFL=CFL), min_dt=1e-6)

    # Track evolution by running in chunks
    n_chunks = 20
    dt_chunk = TIME_END / n_chunks
    Q_current = np.empty((nv, mesh.n_cells))
    Q_current = num.initial_conditions.apply(mesh.cell_centers, Q_current)

    alpha_history = []
    time_points = [0.0]
    n = mesh.n_inner_cells

    # Record initial
    h_val = Q_current[1, 0]
    alphas_init = [float(Q_current[2 + k, 0]) / max(h_val, 1e-10) for k in range(level + 1)]
    alpha_history.append(alphas_init)

    t_solve_start = time.time()
    steady_state_time = None
    try:
        for chunk in range(n_chunks):
            t_end_chunk = (chunk + 1) * dt_chunk
            solver_chunk = GeneratedModelSolver(
                time_end=t_end_chunk, settings=settings,
                compute_dt=timestepping.adaptive(CFL=CFL), min_dt=1e-6)
            Q_new, _ = solver_chunk.solve(mesh, num, write_output=False)
            Q_current = Q_new

            h_val = Q_current[1, 0]
            alphas = [float(Q_current[2 + k, 0]) / max(h_val, 1e-10) for k in range(level + 1)]
            alpha_history.append(alphas)
            time_points.append(t_end_chunk)

            # Check steady state (compare with previous)
            if len(alpha_history) >= 2 and steady_state_time is None:
                diff = max(abs(alphas[k] - alpha_history[-2][k]) for k in range(level + 1))
                if diff < 1e-6 and chunk > 2:
                    steady_state_time = t_end_chunk

        result["solve_time"] = time.time() - t_solve_start
    except Exception as e:
        result["solve_time"] = time.time() - t_solve_start
        result["error"] = f"solve: {str(e)[:200]}"
        return result

    # Final state
    h_final = Q_current[1, 0]
    alphas_final = [float(Q_current[2 + k, 0]) / max(h_final, 1e-10) for k in range(level + 1)]
    result["alphas_final"] = alphas_final
    result["h_final"] = float(h_final)
    result["steady_state_time"] = steady_state_time
    result["alpha_history"] = alpha_history
    result["time_points"] = time_points

    # Analytical comparison
    analytical_alphas = analytical_moments_legendre(H, G_X, NU, SLIP_LAMBDA)
    result["analytical_alphas"] = {str(k): v for k, v in analytical_alphas.items()}

    # Velocity profile comparison
    basis_obj = basis_cls(level=level)
    z_lo, z_hi = float(basis_obj.bounds()[0]), float(basis_obj.bounds()[1])
    zeta_pts = np.linspace(z_lo, z_hi, 200)
    zeta_norm = (zeta_pts - z_lo) / (z_hi - z_lo)

    u_numerical = np.zeros(200)
    for k in range(level + 1):
        phi_fn = basis_obj.get_lambda(k)
        u_numerical += alphas_final[k] * np.array(phi_fn(list(zeta_pts)))

    u_analytical = analytical_profile(zeta_norm, H, G_X, NU, SLIP_LAMBDA)

    result["profile_zeta"] = zeta_norm.tolist()
    result["profile_u_numerical"] = u_numerical.tolist()
    result["profile_u_analytical"] = u_analytical.tolist()
    result["profile_error_linf"] = float(np.max(np.abs(u_numerical - u_analytical)))
    result["profile_error_l2"] = float(np.sqrt(np.mean((u_numerical - u_analytical)**2)))

    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = []
    for level in range(4):
        configs.append(("Legendre", level, "orthogonal"))
        configs.append(("Chebyshev", level, "physical"))

    print(f"Inclined plane rundown: levels 0-3, Legendre + Chebyshev")
    print(f"Parameters: g_x=0.1, nu=0.01, lambda=0.5, h=1.0, t_end=10.0")
    print()

    results = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as ex:
        future_map = {}
        for cfg in configs:
            key = f"{cfg[0]}_L{cfg[1]}"
            future_map[key] = (ex.submit(run_single, cfg), cfg)

        for key, (future, cfg) in sorted(future_map.items()):
            try:
                r = future.result(timeout=600)
                results[key] = r
                with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                    json.dump(r, f)

                if "error" in r:
                    print(f"  {key}: FAILED — {r['error'][:50]}")
                else:
                    print(f"  {key}: build={r['build_time']:.1f}s solve={r['solve_time']:.1f}s "
                          f"steady_t={r['steady_state_time']} "
                          f"Linf_err={r['profile_error_linf']:.2e} "
                          f"alphas={[f'{a:.4f}' for a in r['alphas_final']]}")
            except Exception as e:
                print(f"  {key}: ERROR — {str(e)[:50]}")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({k: {"status": "error" if "error" in v else "ok"} for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def main():
    return run()


if __name__ == "__main__":
    run()
