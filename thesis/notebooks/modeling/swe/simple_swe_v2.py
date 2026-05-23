import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import Matrix

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import BaseMesh
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import (
    DerivativeAwareSolver,
    StructuredDerivativeModel,
)


class SWEModelV2(StructuredDerivativeModel):
    """SWE with named field access, no derivative buffer entries."""

    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h = self.Q.h
        hu = self.Q.hu
        g = self.params.g

        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu**2 / h + 0.5 * g * h**2
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


def make_model():
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic_func(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = 1.0 + 0.1 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)
        Q[1] = 0.0
        return Q

    model = SWEModelV2(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
    )
    return model


def run():
    model = make_model()
    model.print_model_functions(function_names=["flux", "source"])

    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=300)
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/tutorial_swe_v2",
            filename="swe_v2",
            snapshots=2,
            clean_directory=True,
        )
    )
    solver = DerivativeAwareSolver(
        time_end=0.25,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
    )
    Qnew, Qaux = solver.solve(mesh, model, write_output=False)
    print("final h range:", float(Qnew[0].min()), float(Qnew[0].max()))
    print("Qaux shape:", Qaux.shape)

    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n]
    h0 = Q0[0, :n]
    u0 = Q0[1, :n] / np.maximum(h0, 1e-14)
    hN = Qnew[0, :n]
    uN = Qnew[1, :n] / np.maximum(hN, 1e-14)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, h0, label="h(t=0)")
    axes[0].plot(x, hN, label="h(t_end)")
    axes[0].set_title("Water height h")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].plot(x, u0, label="u(t=0)")
    axes[1].plot(x, uN, label="u(t_end)")
    axes[1].set_title("Mean velocity u")
    axes[1].set_xlabel("x")
    axes[1].legend()

    plot_path = "outputs/tutorial_swe_v2/swe_v2_h_u.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    print("saved plot:", plot_path)
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    run()
