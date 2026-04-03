"""
End-to-end test: GeneratedShallowModel (n_layers=1, level=0) through
the GeneratedModelSolver, compared against Basilisk SWE benchmarks.
"""

import os
import numpy as np
from sympy import Matrix

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel


class NumericalSWEGenerated(StructuredDerivativeModel):
    """
    SWE from the GeneratedShallowModel equations.
    State: Q = [b, h, hu], Aux: [hinv]
    """

    dimension = 1
    variables = ["b", "h", "hu"]
    user_aux_variables = ["hinv"]
    parameters = {"g": (9.81, "positive"), "eps": (1e-8, "positive"), "ez": (1.0, "positive")}
    numerics_scaled_q_indices = [2]

    def flux(self):
        h = self.Q.h
        hu = self.Q.hu
        hinv = self.A.hinv
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = 0
        F[1, 0] = hu
        F[2, 0] = hu * hu * hinv
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


def make_dambreak_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    def ic_func(x):
        X = float(x[0])
        Q = np.zeros(3, dtype=float)
        Q[0] = 0.0
        Q[1] = 1.0 if X < 0.0 else 0.1
        Q[2] = 0.0
        return Q

    def ic_aux(x):
        X = float(x[0])
        h = 1.0 if X < 0.0 else 0.1
        return np.array([1.0 / max(h, 1e-14)], dtype=float)

    return NumericalSWEGenerated(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


def make_bump_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    def ic_func(x):
        X = float(x[0])
        Q = np.zeros(3, dtype=float)
        Q[0] = 0.0
        Q[1] = 0.1 + np.exp(-200.0 * X * X)
        Q[2] = 0.0
        return Q

    def ic_aux(x):
        X = float(x[0])
        h = 0.1 + np.exp(-200.0 * X * X)
        return np.array([1.0 / max(h, 1e-14)], dtype=float)

    return NumericalSWEGenerated(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


def solve_case(model, domain, n_cells, time_end, cfl=0.9, g=1.0):
    model.parameter_values[0] = g
    model.parameter_values[2] = 1.0

    mesh = petscMesh.Mesh.create_1d(domain=domain, n_inner_cells=n_cells)

    settings = Zstruct(
        output=Zstruct(
            directory="outputs/generated_swe_test",
            filename="gen_swe",
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


def compare_with_basilisk(case_name, field_name, mesh, Q, field_index, time_index=-1):
    from tests.common.basilisk_loader import load_vtk_series, compare_fields

    try:
        times, x_refs, vals = load_vtk_series(case_name, field_name)
    except FileNotFoundError:
        print(f"  [SKIP] Basilisk data not found for {case_name}")
        return None

    x_ref = x_refs[time_index]
    v_ref = vals[time_index]

    n = mesh.n_inner_cells
    x_zoomy = mesh.cell_centers[0, :n]
    v_zoomy = Q[field_index, :n]

    metrics = compare_fields(x_ref, v_ref, x_zoomy, v_zoomy, rtol=0.1, atol=1e-3)
    return metrics


def run():
    os.makedirs("outputs/generated_swe_test", exist_ok=True)

    print("=" * 60)
    print("Test 1: Dam Break (g=1, flat bed)")
    print("=" * 60)
    model_db = make_dambreak_model()
    mesh_db, Q_db, _ = solve_case(
        model_db, domain=(-5.0, 5.0), n_cells=500,
        time_end=4.0, cfl=0.9, g=1.0,
    )
    n = mesh_db.n_inner_cells
    h = Q_db[1, :n]
    print(f"  h range: [{h.min():.4f}, {h.max():.4f}]")
    print(f"  Mass: {np.sum(h * mesh_db.cell_volumes[:n]):.6f}")

    metrics = compare_with_basilisk("swe_dambreak", "Water_Depth", mesh_db, Q_db, 1)
    if metrics:
        print(f"  vs Basilisk: L1={metrics['l1']:.4e}, L2={metrics['l2']:.4e}, "
              f"Linf={metrics['linf']:.4e}, passed={metrics['passed']}")

    print()
    print("=" * 60)
    print("Test 2: Gaussian Bump (g=1, flat bed)")
    print("=" * 60)
    model_bump = make_bump_model()
    mesh_bump, Q_bump, _ = solve_case(
        model_bump, domain=(-5.0, 5.0), n_cells=500,
        time_end=2.5, cfl=0.9, g=1.0,
    )
    n = mesh_bump.n_inner_cells
    h = Q_bump[1, :n]
    print(f"  h range: [{h.min():.4f}, {h.max():.4f}]")

    metrics = compare_with_basilisk("swe_bump1d", "Water_Depth", mesh_bump, Q_bump, 1)
    if metrics:
        print(f"  vs Basilisk: L1={metrics['l1']:.4e}, L2={metrics['l2']:.4e}, "
              f"Linf={metrics['linf']:.4e}, passed={metrics['passed']}")

    print("\nDone.")


def main():
    return run()


if __name__ == "__main__":
    run()
