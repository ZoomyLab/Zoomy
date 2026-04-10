"""
Multi-layer shallow water test: GeneratedShallowModel (n_layers=2, level=0)
through the GeneratedModelSolver.

State: Q = [b, h, hu0, hu1]  (4 variables)
  b   = bed topography (passive)
  h   = total water depth
  hu0 = h * u_layer0  (bottom layer momentum)
  hu1 = h * u_layer1  (top layer momentum)

Flux (from GeneratedShallowModel with n_layers=2, level=0):
  F[0,0] = 0
  F[1,0] = hu0/2 + hu1/2
  F[2,0] = hu0^2 / (2*h)          i.e. hu0^2 * hinv / 2
  F[3,0] = hu1^2 / (2*h)          i.e. hu1^2 * hinv / 2

Hydrostatic pressure:
  P[2,0] = g * ez * h^2 / 2

Nonconservative matrix (bed-slope coupling):
  A[2, 0, 0] = g * ez * h
"""

import os
import numpy as np
from sympy import Matrix

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel


class NumericalMultilayerSWE(StructuredDerivativeModel):
    """
    Two-layer SWE from GeneratedShallowModel(n_layers=2, level=0).
    State: Q = [b, h, hu0, hu1], Aux: [hinv]
    """

    dimension = 1
    variables = ["b", "h", "hu0", "hu1"]
    user_aux_variables = ["hinv"]
    parameters = {"g": (9.81, "positive"), "eps": (1e-8, "positive"), "ez": (1.0, "positive")}
    numerics_scaled_q_indices = [2, 3]

    def flux(self):
        h = self.Q.h
        hu0 = self.Q.hu0
        hu1 = self.Q.hu1
        hinv = self.A.hinv
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = 0
        F[1, 0] = hu0 / 2 + hu1 / 2
        F[2, 0] = hu0 * hu0 * hinv / 2
        F[3, 0] = hu1 * hu1 * hinv / 2
        return ZArray(F)

    def hydrostatic_pressure(self):
        h = self.Q.h
        g = self.params.g
        ez = self.params.ez
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = g * ez * h * h / 2
        return ZArray(P)

    def nonconservative_matrix(self):
        h = self.Q.h
        g = self.params.g
        ez = self.params.ez
        A = ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        A[2, 0, 0] = g * ez * h
        return A

    def source(self):
        return ZArray.zeros(self.n_variables)


def make_shear_flow_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    def ic_func(x):
        X = float(x[0])
        Q = np.zeros(4, dtype=float)
        Q[0] = 0.0
        Q[1] = 1.0
        Q[2] = 1.0 * (-0.5)
        Q[3] = 1.0 * 0.5
        return Q

    def ic_aux(x):
        h = 1.0
        return np.array([1.0 / max(h, 1e-14)], dtype=float)

    return NumericalMultilayerSWE(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


def make_lock_exchange_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    def ic_func(x):
        X = float(x[0])
        Q = np.zeros(4, dtype=float)
        Q[0] = 0.0
        Q[1] = 1.0
        if X < 5.0:
            Q[2] = 1.0 * 0.5
            Q[3] = 1.0 * (-0.5)
        else:
            Q[2] = 1.0 * (-0.5)
            Q[3] = 1.0 * 0.5
        return Q

    def ic_aux(x):
        h = 1.0
        return np.array([1.0 / max(h, 1e-14)], dtype=float)

    return NumericalMultilayerSWE(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


def solve_case(model, domain, n_cells, time_end, cfl=0.45, g=9.81):
    model.parameter_values[0] = g
    model.parameter_values[2] = 1.0

    mesh = BaseMesh.create_1d(domain=domain, n_inner_cells=n_cells)

    settings = Zstruct(
        output=Zstruct(
            directory="outputs/generated_multilayer_test",
            filename="gen_ml",
            snapshots=2,
            clean_directory=False,
        )
    )

    solver = GeneratedModelSolver(
        time_end=time_end,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=cfl),
    )

    Q, Qaux = solver.solve(mesh, model, write_output=False)
    return mesh, Q, Qaux


def report_diagnostics(name, mesh, Q):
    n = mesh.n_inner_cells
    h = Q[1, :n]
    hu0 = Q[2, :n]
    hu1 = Q[3, :n]
    hinv = 1.0 / np.maximum(h, 1e-14)
    u0 = hu0 * hinv
    u1 = hu1 * hinv
    dx = mesh.cell_volumes[:n]
    mass = np.sum(h * dx)
    print(f"  h   range: [{h.min():.6f}, {h.max():.6f}]")
    print(f"  u0  range: [{u0.min():.6f}, {u0.max():.6f}]")
    print(f"  u1  range: [{u1.min():.6f}, {u1.max():.6f}]")
    print(f"  Mass (h*dx): {mass:.6f}")
    h_positive = np.all(h > -1e-10)
    h_finite = np.all(np.isfinite(h))
    u_finite = np.all(np.isfinite(u0)) and np.all(np.isfinite(u1))
    print(f"  h positive: {h_positive}, all finite: {h_finite and u_finite}")
    return h_positive and h_finite and u_finite


def run():
    os.makedirs("outputs/generated_multilayer_test", exist_ok=True)

    all_passed = True

    print("=" * 60)
    print("Test 1: Shear flow (u_bottom=-0.5, u_top=0.5)")
    print("=" * 60)
    model_shear = make_shear_flow_model()
    mesh_shear, Q_shear, _ = solve_case(
        model_shear, domain=(0.0, 10.0), n_cells=200,
        time_end=1.0, cfl=0.45, g=9.81,
    )
    ok = report_diagnostics("shear", mesh_shear, Q_shear)
    all_passed = all_passed and ok

    print()
    print("=" * 60)
    print("Test 2: Lock exchange (velocity flip at x=5)")
    print("=" * 60)
    model_lock = make_lock_exchange_model()
    mesh_lock, Q_lock, _ = solve_case(
        model_lock, domain=(0.0, 10.0), n_cells=200,
        time_end=1.0, cfl=0.45, g=9.81,
    )
    ok = report_diagnostics("lock_exchange", mesh_lock, Q_lock)
    all_passed = all_passed and ok

    print()
    if all_passed:
        print("All multi-layer tests PASSED.")
    else:
        print("Some tests FAILED.")

    return all_passed


def main():
    return run()


if __name__ == "__main__":
    run()
