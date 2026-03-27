import os

import matplotlib.pyplot as plt
import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver

from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D


def make_model():
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic_func(x):
        Q = np.zeros(2, dtype=float)
        # Primitive variables: Q = [h, u]
        Q[0] = 1.0 + 0.01 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)
        Q[1] = 0.0
        return Q

    return ClassicalGreenNaghdi1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
    )


def run():
    model = make_model()
    model.print_model_functions(function_names=["flux", "source"])

    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=300, lsq_degree=2)
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)

    settings = Zstruct(
        output=Zstruct(
            directory="outputs/tutorial_gn_classical_v2",
            filename="gn_classical_v2",
            snapshots=2,
            clean_directory=True,
        )
    )
    solver = DerivativeAwareSolver(
        time_end=0.01,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.2),
    )
    Qnew, Qaux = solver.solve(mesh, model, write_output=False)

    print("final h range:", float(Qnew[0].min()), float(Qnew[0].max()))
    print("final u range:", float(Qnew[1].min()), float(Qnew[1].max()))
    print("Qaux shape:", Qaux.shape)

    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n]
    h0 = Q0[0, :n]
    u0 = Q0[1, :n]
    hN = Qnew[0, :n]
    uN = Qnew[1, :n]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, h0, label="h(t=0)")
    axes[0].plot(x, hN, label="h(t_end)")
    axes[0].set_title("Classical GN: water height h")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].plot(x, u0, label="u(t=0)")
    axes[1].plot(x, uN, label="u(t_end)")
    axes[1].set_title("Classical GN: mean velocity u")
    axes[1].set_xlabel("x")
    axes[1].legend()

    plot_path = "outputs/tutorial_gn_classical_v2/gn_classical_v2_h_u.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    print("saved plot:", plot_path)
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    run()
