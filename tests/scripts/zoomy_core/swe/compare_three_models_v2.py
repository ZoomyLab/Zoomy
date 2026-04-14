import os

import matplotlib.pyplot as plt
import numpy as np

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver

from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from tutorials.swe.simple_gn_v2 import make_model as make_model_gn_minimal
from tutorials.swe.simple_swe_v2 import make_model as make_model_swe


def make_model_gn_classical():
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic_func(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = 1.0 + 0.01 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)
        Q[1] = 0.0
        return Q

    return ClassicalGreenNaghdi1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
    )


def solve_model(model, mesh, time_end, cfl):
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    solver = DerivativeAwareSolver(
        time_end=time_end,
        settings=Zstruct(output=Zstruct(directory="outputs/tmp", filename="tmp", snapshots=2)),
        compute_dt=timestepping.adaptive(CFL=cfl),
    )
    Qnew, _ = solver.solve(mesh, model, write_output=False)
    return Q0, Qnew


def run():
    # NOTE:
    # This script compares 3 models on a common flat-bottom pulse case.
    # A true beach runup benchmark requires explicit bathymetry source terms
    # and wetting/drying treatment, which are not part of these minimal models.
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=350, lsq_degree=2)
    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n]

    models = [
        ("SWE", make_model_swe()),
        ("GN-minimal", make_model_gn_minimal()),
        ("GN-classical", make_model_gn_classical()),
    ]

    results = []
    per_model_settings = {
        "SWE": (0.06, 0.6),
        "GN-minimal": (0.06, 0.5),
        "GN-classical": (0.01, 0.2),
    }

    for name, model in models:
        time_end, cfl = per_model_settings[name]
        Q0, QN = solve_model(model, mesh, time_end=time_end, cfl=cfl)
        if name == "GN-classical":
            h0 = Q0[0, :n]
            u0 = Q0[1, :n]
            hN = QN[0, :n]
            uN = QN[1, :n]
        else:
            h0 = Q0[0, :n]
            u0 = Q0[1, :n] / np.maximum(h0, 1e-14)
            hN = QN[0, :n]
            uN = QN[1, :n] / np.maximum(hN, 1e-14)
        results.append((name, h0, u0, hN, uN))
        print(f"{name}: h range [{float(hN.min()):.6f}, {float(hN.max()):.6f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Plot initial reference from first model (all are same profile definition type)
    axes[0].plot(x, results[0][1], "k--", linewidth=1.0, label="initial h")
    axes[1].plot(x, results[0][2], "k--", linewidth=1.0, label="initial u")

    for name, _h0, _u0, hN, uN in results:
        axes[0].plot(x, hN, label=f"{name} final")
        axes[1].plot(x, uN, label=f"{name} final")

    axes[0].set_title("Final water height comparison")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].set_title("Final mean velocity comparison")
    axes[1].set_xlabel("x")
    axes[1].legend()

    out_path = "outputs/tutorial_compare_v2/compare_three_models_h_u.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved comparison plot:", out_path)
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    run()
