import numpy as np

import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.jvp_numpy import analytic_source_jvp, fd_jvp
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from tutorials.swe.simple_gn_v2 import make_model as make_model_gn_minimal


def _init_state(symbolic_model, mesh):
    Q = np.empty((symbolic_model.n_variables, mesh.n_cells), dtype=float)
    if hasattr(symbolic_model, "initial_conditions") and symbolic_model.initial_conditions is not None:
        Q = symbolic_model.initial_conditions.apply(mesh.cell_centers, Q)
    else:
        Q[0, :] = 1.0 + 0.02 * np.exp(-((mesh.cell_centers[0] - 5.0) ** 2) / 0.8)
        for i in range(1, symbolic_model.n_variables):
            Q[i, :] = 0.0
    return Q


def _compare_for_model(name, symbolic_model, mesh, dt=1e-2, eps=1e-7):
    runtime_model = NumpyRuntimeModel(symbolic_model)
    solver = DerivativeAwareSolver(time_end=0.0)

    Q = _init_state(symbolic_model, mesh)
    Qold = np.array(Q, copy=True)
    Qaux = np.zeros((symbolic_model.n_aux_variables, mesh.n_cells), dtype=float)
    Qaux = solver.update_qaux(Q, Qaux, Qold, Qaux, mesh, runtime_model, symbolic_model.parameter_values, 0.0, dt)

    rng = np.random.default_rng(4)
    V = rng.normal(size=Q.shape)
    V /= max(np.linalg.norm(V), 1e-14)

    parameters = np.asarray(symbolic_model.parameter_values)

    def residual_fn(Qin):
        Qin_aux = solver.update_qaux(
            Qin,
            Qaux,
            Qold,
            Qaux,
            mesh,
            runtime_model,
            parameters,
            0.0,
            dt,
        )
        return runtime_model.source(Qin, Qin_aux, parameters)

    jv_fd = fd_jvp(residual_fn, Q, V, eps=eps)
    jv_analytic_lagged = analytic_source_jvp(
        runtime_model,
        symbolic_model,
        Q,
        Qaux,
        V,
        mesh,
        dt,
        include_chain_rule=False,
    )
    jv_analytic_full = analytic_source_jvp(
        runtime_model,
        symbolic_model,
        Q,
        Qaux,
        V,
        mesh,
        dt,
        include_chain_rule=True,
    )

    def rel_err(a, b):
        num = np.linalg.norm(a - b)
        den = max(np.linalg.norm(b), 1e-14)
        return num / den

    err_lagged = rel_err(jv_analytic_lagged, jv_fd)
    err_full = rel_err(jv_analytic_full, jv_fd)

    print("=" * 80)
    print(f"Model: {name}")
    print(f"n_variables={symbolic_model.n_variables}, n_aux_variables={symbolic_model.n_aux_variables}")
    print(f"FD-JVP eps={eps}, dt={dt}")
    print(f"relative error (analytic lagged vs FD): {err_lagged:.3e}")
    print(f"relative error (analytic full chain vs FD): {err_full:.3e}")
    print("-" * 80)
    print("||Jv_fd||:", float(np.linalg.norm(jv_fd)))
    print("||Jv_analytic_lagged||:", float(np.linalg.norm(jv_analytic_lagged)))
    print("||Jv_analytic_full||:", float(np.linalg.norm(jv_analytic_full)))
    print("=" * 80)


def run():
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=120, lsq_degree=2)
    _compare_for_model("GN-minimal", make_model_gn_minimal(), mesh, dt=1e-2, eps=1e-7)
    _compare_for_model("GN-classical", ClassicalGreenNaghdi1D(), mesh, dt=1e-2, eps=1e-7)


if __name__ == "__main__":
    run()
