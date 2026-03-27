import os

import matplotlib.pyplot as plt
import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.misc.misc import Zstruct

from tutorials.swe.beach_runup_swe_vs_gn_classical_v2 import (
    ClassicalGNBeachTopoModel,
    SWEBeachTopoModel,
    SymbolicNumericsBeachIMEXSolver,
    SymbolicNumericsBeachSolver,
    infer_required_lsq_degree,
    make_bathymetry,
    make_initial_eta,
    shoreline_position,
)


def _build_models(b_func, eta_func, u0_func, boundary_conditions):
    def make_ic_swe():
        def ic_q(x):
            X = float(x[0])
            b = float(b_func(X))
            eta = float(eta_func(X))
            h = max(eta - b, 1e-8)
            u0 = float(u0_func(X, h))
            Q = np.zeros(3, dtype=float)
            Q[0] = b
            Q[1] = h
            Q[2] = h * u0
            return Q

        def ic_aux(x):
            X = float(x[0])
            b = float(b_func(X))
            eta = float(eta_func(X))
            h = max(eta - b, 1e-8)
            return np.array([1.0 / max(h, 1e-8)], dtype=float)

        return SWEBeachTopoModel(
            boundary_conditions=boundary_conditions,
            initial_conditions=IC.UserFunction(ic_q),
            aux_initial_conditions=IC.UserFunction(ic_aux),
        )

    def make_ic_gn():
        def ic_q(x):
            X = float(x[0])
            b = float(b_func(X))
            eta = float(eta_func(X))
            h = max(eta - b, 1e-8)
            u0 = float(u0_func(X, h))
            Q = np.zeros(3, dtype=float)
            Q[0] = b
            Q[1] = h
            Q[2] = u0
            return Q

        def ic_aux(x):
            X = float(x[0])
            b = float(b_func(X))
            eta = float(eta_func(X))
            h = max(eta - b, 1e-8)
            return np.array([1.0 / max(h, 1e-8)], dtype=float)

        return ClassicalGNBeachTopoModel(
            boundary_conditions=boundary_conditions,
            initial_conditions=IC.UserFunction(ic_q),
            aux_initial_conditions=IC.UserFunction(ic_aux),
        )

    return make_ic_swe(), make_ic_gn()


def _solve_model(model, mesh, time_end, cfl):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/benchmark_suite_tmp",
            filename="tmp",
            snapshots=2,
            clean_directory=False,
        )
    )

    if isinstance(model, ClassicalGNBeachTopoModel):
        solver = SymbolicNumericsBeachIMEXSolver(
            time_end=time_end,
            settings=settings,
            compute_dt=timestepping.adaptive(CFL=cfl),
        )
        object.__setattr__(solver, "source_mode", "auto")
        object.__setattr__(solver, "implicit_maxiter", 8)
        object.__setattr__(solver, "gmres_maxiter", 40)
        object.__setattr__(solver, "use_analytic_chain_jvp", True)
    else:
        solver = SymbolicNumericsBeachSolver(
            time_end=time_end,
            settings=settings,
            compute_dt=timestepping.adaptive(CFL=cfl),
        )

    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    QN, _ = solver.solve(mesh, model, write_output=False)
    return Q0, QN


def _diagnostics(name, x, Q0, QN, h_thr=2e-3):
    n = x.shape[0]
    h0 = Q0[1, :n]
    hN = QN[1, :n]
    x_peak0 = float(x[np.argmax(h0)])
    x_peakN = float(x[np.argmax(hN)])
    x_shore0 = shoreline_position(x, h0, threshold=h_thr)
    x_shoreN = shoreline_position(x, hN, threshold=h_thr)
    finite_ratio = float(np.isfinite(QN[:, :n]).sum() / QN[:, :n].size)
    print(
        f"{name}: finite={finite_ratio:.3f}, h_min={float(hN.min()):.3e}, h_max={float(hN.max()):.3e}, "
        f"peak_shift={x_peakN - x_peak0:.3f}, shore_shift={x_shoreN - x_shore0:.3f}"
    )


def run():
    cases = []

    # 1) Flat bottom, no wet/dry front (fully wet).
    cases.append(
        dict(
            label="flat_nowetdry",
            domain=(0.0, 40.0),
            n_cells=120,
            time_end=4.0,
            cfl=0.25,
            b_func=lambda x: 0.0,
            eta_func=lambda x: 1.0 + 0.03 * np.exp(-((x - 8.0) ** 2) / 4.0),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions(
                [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
            ),
        )
    )

    # 2) Flat bottom, with wet/dry front (dry outside compact wet patch).
    cases.append(
        dict(
            label="flat_wetdry",
            domain=(0.0, 40.0),
            n_cells=120,
            time_end=4.0,
            cfl=0.2,
            b_func=lambda x: 0.0,
            eta_func=lambda x: max(0.0, 0.20 * np.exp(-((x - 6.0) ** 2) / 3.0)),
            u0_func=lambda x, h: 0.8 if h > 5e-3 else 0.0,
            bcs=BC.BoundaryConditions(
                [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
            ),
        )
    )

    # 3) Current beach runup case.
    cases.append(
        dict(
            label="beach_current",
            domain=(0.0, 70.0),
            n_cells=120,
            time_end=6.0,
            cfl=0.2,
            b_func=lambda x: make_bathymetry(x),
            eta_func=lambda x: make_initial_eta(x),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions(
                [
                    BC.Extrapolation(tag="left"),
                    BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
                ]
            ),
        )
    )

    os.makedirs("outputs/tutorial_beach_runup_v2", exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)

    for i, cfg in enumerate(cases):
        swe, gn = _build_models(
            b_func=cfg["b_func"],
            eta_func=cfg["eta_func"],
            u0_func=cfg["u0_func"],
            boundary_conditions=cfg["bcs"],
        )
        lsq_degree = infer_required_lsq_degree([swe, gn], minimum_degree=1)
        mesh = petscMesh.Mesh.create_1d(
            domain=cfg["domain"],
            n_inner_cells=cfg["n_cells"],
            lsq_degree=lsq_degree,
        )
        n = mesh.n_inner_cells
        x = mesh.cell_centers[0, :n]

        Q0_swe, QN_swe = _solve_model(swe, mesh, cfg["time_end"], cfg["cfl"])
        Q0_gn, QN_gn = _solve_model(gn, mesh, cfg["time_end"], cfg["cfl"])

        print(f"\nCase {i+1}: {cfg['label']} (lsq_degree={lsq_degree})")
        _diagnostics("  SWE", x, Q0_swe, QN_swe)
        _diagnostics("  GN ", x, Q0_gn, QN_gn)

        b = Q0_swe[0, :n]
        eta0 = b + Q0_swe[1, :n]
        eta_swe = b + QN_swe[1, :n]
        eta_gn = b + QN_gn[1, :n]
        u_swe = QN_swe[2, :n] / np.maximum(QN_swe[1, :n], 1e-10)
        u_gn = QN_gn[2, :n]

        axes[i, 0].plot(x, b, "k-", linewidth=1.0, label="b")
        axes[i, 0].plot(x, eta0, "k--", linewidth=1.0, label="eta0")
        axes[i, 0].plot(x, eta_swe, label="eta SWE")
        axes[i, 0].plot(x, eta_gn, label="eta GN")
        axes[i, 0].set_title(f"{cfg['label']} - free surface")
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].plot(x, u_swe, label="u SWE")
        axes[i, 1].plot(x, u_gn, label="u GN")
        axes[i, 1].set_title(f"{cfg['label']} - velocity")
        axes[i, 1].legend(fontsize=8)

    out_path = "outputs/tutorial_beach_runup_v2/benchmark_suite_symbolic_numerics_v2.png"
    fig.savefig(out_path, dpi=150)
    print("\nsaved:", out_path)


if __name__ == "__main__":
    run()
