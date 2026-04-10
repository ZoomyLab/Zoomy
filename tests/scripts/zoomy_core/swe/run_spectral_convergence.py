"""
Spectral convergence study: Legendre vs Chebyshev, levels 0-10.
Runs level by level (low to high), parallelizing within each level.
Each config has a 15-minute timeout. Results saved to JSON.

Usage:
    python tests/scripts/zoomy_core/swe/run_spectral_convergence.py
"""

import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

OUTPUT_DIR = "outputs/spectral_convergence"
MAX_LEVEL = 10
BUILD_TIMEOUT = 900


def run_single(args):
    """Run one (basis, level, weight_mode) config in a subprocess."""
    basis_name, level, weight_mode = args

    from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
    from zoomy_core.model.models.basisfunctions import Legendre_shifted, Chebyshevu_shifted
    from zoomy_core.model.numerical_model import NumericalModel
    from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
    from zoomy_core.misc.misc import ZArray, Zstruct
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.mesh.mesh as petscMesh
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    DOMAIN = (-5.0, 5.0)
    N_CELLS = 30
    TIME_END = 0.3
    CFL = 0.45
    SLIP_LAMBDA = 0.5
    H_LEFT, H_RIGHT = 1.0, 0.5

    basis_cls = Legendre_shifted if basis_name == "Legendre" else Chebyshevu_shifted

    class WithSlip(GeneratedShallowModel):
        def source(self):
            return ZArray(self.slip())

    result = {"basis": basis_name, "level": level, "weight_mode": weight_mode}

    t0 = time.time()
    try:
        a = WithSlip(n_layers=1, level=level, dimension=1, basis_type=basis_cls,
                     eigenvalue_mode="numerical", weight_mode=weight_mode)
        a.parameter_defaults_map["lamda"] = SLIP_LAMBDA
        a.parameter_values[list(a.parameters.keys()).index("lamda")] = SLIP_LAMBDA
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
        Q[1] = H_LEFT if float(x[0]) < 0 else H_RIGHT
        return Q

    try:
        num = NumericalModel(a, boundary_conditions=bcs, initial_conditions=IC.UserFunction(ic))
        num.parameter_values[list(num.parameters.keys()).index("lamda")] = SLIP_LAMBDA
    except Exception as e:
        result["error"] = f"NumericalModel: {str(e)[:200]}"
        return result

    mesh = petscMesh.Mesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    settings = Zstruct(output=Zstruct(directory=OUTPUT_DIR, filename="tmp",
                                       snapshots=2, clean_directory=False))
    solver = GeneratedModelSolver(time_end=TIME_END, settings=settings,
                                   compute_dt=timestepping.adaptive(CFL=CFL), min_dt=1e-6)
    t1 = time.time()
    try:
        Q, _ = solver.solve(mesh, num, write_output=False)
        result["solve_time"] = time.time() - t1
    except Exception as e:
        result["solve_time"] = time.time() - t1
        result["error"] = f"solve: {str(e)[:200]}"
        return result

    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n].tolist()
    h = Q[1, :n].tolist()
    Q_all = Q[:, :n].tolist()
    result["x"] = x
    result["h"] = h
    result["Q"] = Q_all
    result["h_min"] = float(min(h))
    result["h_max"] = float(max(h))
    result["h_positive"] = all(v >= 0 for v in h)

    basis_obj = basis_cls(level=level)
    profiles = {}
    for x_pos in [-1.0, 1.0]:
        i_cell = int(np.argmin(np.abs(np.array(x) - x_pos)))
        h_c = max(h[i_cell], 1e-10)
        n_mom = level + 1
        alphas = [Q_all[2 + k][i_cell] / h_c for k in range(n_mom)]
        z_lo, z_hi = float(basis_obj.bounds()[0]), float(basis_obj.bounds()[1])
        zeta_pts = np.linspace(z_lo, z_hi, 200).tolist()
        u_profile = np.zeros(200)
        for k in range(n_mom):
            phi_fn = basis_obj.get_lambda(k)
            u_profile += alphas[k] * np.array(phi_fn(zeta_pts))
        zeta_norm = ((np.array(zeta_pts) - z_lo) / (z_hi - z_lo)).tolist()
        profiles[str(x_pos)] = {"zeta": zeta_norm, "u": u_profile.tolist()}
    result["profiles"] = profiles

    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    failed_bases = set()

    print(f"Spectral convergence: levels 0-{MAX_LEVEL}, timeout={BUILD_TIMEOUT}s")
    print(f"{'Basis':<12s} {'L':>2s} {'n_vars':>6s} {'build':>7s} {'solve':>7s} {'h_min':>8s} {'h_max':>8s}")
    print("-" * 60)

    for level in range(MAX_LEVEL + 1):
        configs = []
        if "Legendre" not in failed_bases:
            configs.append(("Legendre", level, "orthogonal"))
        if "Chebyshev" not in failed_bases:
            configs.append(("Chebyshev", level, "physical"))

        if not configs:
            print(f"  Level {level}: all bases failed at lower levels, skipping.")
            break

        with ProcessPoolExecutor(max_workers=len(configs)) as ex:
            futures = {}
            for cfg in configs:
                key = f"{cfg[0]}_L{level}"
                futures[key] = (ex.submit(run_single, cfg), cfg)

            for key, (future, cfg) in futures.items():
                try:
                    r = future.result(timeout=BUILD_TIMEOUT)
                    all_results[key] = r

                    # Save immediately
                    with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                        json.dump(r, f)

                    if "error" in r:
                        print(f"  {cfg[0]:<12s} {level:>2d} {'—':>6s} {r.get('build_time',0):>6.1f}s "
                              f"{'—':>7s} {'—':>8s} {'—':>8s} FAIL: {r['error'][:30]}")
                        failed_bases.add(cfg[0])
                    else:
                        print(f"  {cfg[0]:<12s} {level:>2d} {r['n_vars']:>6d} {r['build_time']:>6.1f}s "
                              f"{r['solve_time']:>6.1f}s {r['h_min']:>8.4f} {r['h_max']:>8.4f}")
                except TimeoutError:
                    print(f"  {cfg[0]:<12s} {level:>2d} {'—':>6s} TIMEOUT ({BUILD_TIMEOUT}s)")
                    all_results[key] = {"basis": cfg[0], "level": level, "error": "TIMEOUT"}
                    with open(os.path.join(OUTPUT_DIR, f"{key}.json"), "w") as f:
                        json.dump(all_results[key], f)
                    failed_bases.add(cfg[0])
                    future.cancel()
                except Exception as e:
                    print(f"  {cfg[0]:<12s} {level:>2d} {'—':>6s} ERROR: {str(e)[:40]}")
                    all_results[key] = {"basis": cfg[0], "level": level, "error": str(e)[:200]}
                    failed_bases.add(cfg[0])

    # Save manifest
    manifest = {}
    for key, r in all_results.items():
        manifest[key] = {
            "status": "error" if "error" in r else "ok",
            "error": r.get("error", ""),
            "build_time": r.get("build_time", -1),
            "solve_time": r.get("solve_time", -1),
            "n_vars": r.get("n_vars", -1),
            "h_min": r.get("h_min"),
            "h_max": r.get("h_max"),
        }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def main():
    return run()


if __name__ == "__main__":
    run()
