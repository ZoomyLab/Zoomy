import os

import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.symbolic_numerics_v2 import (
    NonconservativeRusanov,
    PositiveNonconservativeRusanov,
    QuasilinearRusanov,
)
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver

from tutorials.swe.beach_runup_swe_vs_gn_classical_v2 import SWEBeachTopoModel, make_bathymetry


def make_model():
    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(tag="left"),
            BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
        ]
    )

    def eta_func(x):
        return 2.8 + 0.03 * np.exp(-((x - 10.0) ** 2) / 5.0)

    def ic_q(x):
        X = float(x[0])
        b = make_bathymetry(X)
        eta = eta_func(X)
        h = max(eta - b, 1e-8)
        Q = np.zeros(3, dtype=float)
        Q[0] = b
        Q[1] = h
        Q[2] = 0.0
        return Q

    def ic_aux(x):
        X = float(x[0])
        b = make_bathymetry(X)
        eta = eta_func(X)
        h = max(eta - b, 1e-8)
        return np.array([1.0 / max(h, 1e-8)], dtype=float)

    return SWEBeachTopoModel(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_q),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


class _SWEOperatorMixin:
    numerics_cls = PositiveNonconservativeRusanov

    def _field_map_from_symbolic_model(self, symbolic_model):
        var_keys = list(symbolic_model.variables.keys())
        aux_keys = list(symbolic_model.aux_variables.keys())
        h_idx = var_keys.index("h")
        b_idx = var_keys.index("b")
        hinv_idx = aux_keys.index("hinv") if "hinv" in aux_keys else 0
        return {
            "h": {"container": "q", "index": h_idx},
            "b": {"container": "q", "index": b_idx},
            "hinv": {"container": "qaux", "index": hinv_idx},
        }

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        out = super().update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
        symbolic_model = model.model if hasattr(model, "model") else model
        var_keys = list(symbolic_model.variables.keys())
        aux_keys = list(symbolic_model.aux_variables.keys())
        if "hinv" in aux_keys and "h" in var_keys:
            i_hinv = aux_keys.index("hinv")
            i_h = var_keys.index("h")
            eps = 1e-12
            if hasattr(symbolic_model.parameters, "contains") and symbolic_model.parameters.contains("eps"):
                eps_i = list(symbolic_model.parameters.keys()).index("eps")
                eps = float(parameters[eps_i])
            out[i_hinv, :] = 1.0 / np.maximum(Q[i_h, :], eps)
        return out

    def get_flux_operator(self, mesh, model):
        symbolic_model = model.model if hasattr(model, "model") else model
        field_map = self._field_map_from_symbolic_model(symbolic_model)
        scaled_q_indices = getattr(symbolic_model, "numerics_scaled_q_indices", None)
        numerics = self.numerics_cls(
            symbolic_model,
            field_map=field_map,
            scaled_q_indices=scaled_q_indices,
        )
        runtime_numerics = numerics.to_runtime_numpy()

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        normals = mesh.face_normals
        face_volumes = mesh.face_volumes
        cell_volumesA = mesh.cell_volumes[iA]
        cell_volumesB = mesh.cell_volumes[iB]
        n_vars = symbolic_model.n_variables

        def flux_operator(dt, Q, Qaux, parameters, dQ):
            dQ = np.zeros_like(dQ)
            for f in range(mesh.n_faces):
                qA = Q[:, iA[f]]
                qB = Q[:, iB[f]]
                qauxA = Qaux[:, iA[f]]
                qauxB = Qaux[:, iB[f]]
                n = normals[:, f]
                fluct = np.asarray(
                    runtime_numerics.numerical_fluctuations(qA, qB, qauxA, qauxB, parameters, n),
                    dtype=float,
                ).reshape(-1)
                num_flux = np.asarray(
                    runtime_numerics.numerical_flux(qA, qB, qauxA, qauxB, parameters, n),
                    dtype=float,
                ).reshape(-1)
                Dp = fluct[:n_vars]
                Dm = fluct[n_vars:]
                dQ[:, iA[f]] -= (num_flux + Dm) * face_volumes[f] / cell_volumesA[f]
                dQ[:, iB[f]] -= (-num_flux + Dp) * face_volumes[f] / cell_volumesB[f]
            return dQ

        return flux_operator


class FluxHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = PositiveNonconservativeRusanov


class FluxNoHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = NonconservativeRusanov


class QuasiNoHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = QuasilinearRusanov


def solve_case(model, mesh, solver_cls, time_end=5.0, cfl=0.22):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/investigate_bathy_fullwet_hr_vs_quasilinear_v2/tmp",
            filename="tmp",
            snapshots=2,
            clean_directory=False,
        )
    )
    solver = solver_cls(
        time_end=time_end,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=cfl),
    )
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    QN, _ = solver.solve(mesh, model, write_output=False)
    return Q0, QN


def make_runtime_numerics(model, numerics_cls):
    symbolic_model = model.model if hasattr(model, "model") else model
    var_keys = list(symbolic_model.variables.keys())
    aux_keys = list(symbolic_model.aux_variables.keys())
    field_map = {
        "h": {"container": "q", "index": var_keys.index("h")},
        "b": {"container": "q", "index": var_keys.index("b")},
        "hinv": {"container": "qaux", "index": aux_keys.index("hinv")},
    }
    scaled_q_indices = getattr(symbolic_model, "numerics_scaled_q_indices", None)
    numerics = numerics_cls(
        symbolic_model,
        field_map=field_map,
        scaled_q_indices=scaled_q_indices,
    )
    return numerics.to_runtime_numpy()


def face_operator_stats(mesh, Q, Qaux, parameters, runtime_numerics):
    iA = mesh.face_cells[0]
    iB = mesh.face_cells[1]
    normals = mesh.face_normals
    n_vars = Q.shape[0]
    nf = np.zeros((mesh.n_faces, n_vars), dtype=float)
    fl = np.zeros((mesh.n_faces, 2 * n_vars), dtype=float)
    for f in range(mesh.n_faces):
        qA = Q[:, iA[f]]
        qB = Q[:, iB[f]]
        qaA = Qaux[:, iA[f]]
        qaB = Qaux[:, iB[f]]
        n = normals[:, f]
        nf[f, :] = np.asarray(
            runtime_numerics.numerical_flux(qA, qB, qaA, qaB, parameters, n), dtype=float
        ).reshape(-1)
        fl[f, :] = np.asarray(
            runtime_numerics.numerical_fluctuations(qA, qB, qaA, qaB, parameters, n),
            dtype=float,
        ).reshape(-1)
    return nf, fl


def diff_stats(A, B, name):
    d = B - A
    print(
        f"{name}: L2={np.sqrt(np.mean(d*d)):.3e}, Linf={np.max(np.abs(d)):.3e}, mean={np.mean(d):.3e}"
    )


def state_stats(Qref, Qtest, label):
    href = Qref[1, :]
    htest = Qtest[1, :]
    etaref = Qref[0, :] + href
    etatest = Qtest[0, :] + htest
    uref = Qref[2, :] / np.maximum(href, 1e-10)
    utest = Qtest[2, :] / np.maximum(htest, 1e-10)
    d_eta = etatest - etaref
    d_u = utest - uref
    d_h = htest - href
    print(
        f"{label}: deta_linf={np.max(np.abs(d_eta)):.3e}, "
        f"du_linf={np.max(np.abs(d_u)):.3e}, dh_linf={np.max(np.abs(d_h)):.3e}"
    )


def run():
    os.makedirs("outputs/investigate_bathy_fullwet_hr_vs_quasilinear_v2", exist_ok=True)

    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 70.0), n_inner_cells=260, lsq_degree=1)
    n = mesh.n_inner_cells
    model = make_model()

    # Build initial Q/Qaux for face-operator diagnostics.
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    Qaux0 = np.empty((model.n_aux_variables, mesh.n_cells), dtype=float)
    Qaux0 = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux0)
    parameters = np.asarray(model.parameter_values, dtype=float)

    rt_flux_nohr = make_runtime_numerics(model, NonconservativeRusanov)
    rt_flux_hr = make_runtime_numerics(model, PositiveNonconservativeRusanov)
    rt_quasi_nohr = make_runtime_numerics(model, QuasilinearRusanov)

    nf_flux_nohr, fl_flux_nohr = face_operator_stats(mesh, Q0, Qaux0, parameters, rt_flux_nohr)
    nf_flux_hr, fl_flux_hr = face_operator_stats(mesh, Q0, Qaux0, parameters, rt_flux_hr)
    nf_quasi_nohr, fl_quasi_nohr = face_operator_stats(mesh, Q0, Qaux0, parameters, rt_quasi_nohr)

    print("\n[Face operator differences at initial state]")
    diff_stats(nf_flux_nohr, nf_flux_hr, "flux noHR -> flux HR: numerical_flux")
    diff_stats(fl_flux_nohr, fl_flux_hr, "flux noHR -> flux HR: numerical_fluctuations")
    diff_stats(
        nf_flux_nohr,
        nf_quasi_nohr,
        "flux noHR -> quasi noHR: numerical_flux (quasi flux is zero)",
    )
    diff_stats(
        fl_flux_nohr,
        fl_quasi_nohr,
        "flux noHR -> quasi noHR: numerical_fluctuations",
    )
    diff_stats(
        fl_flux_hr,
        fl_quasi_nohr,
        "flux HR -> quasi noHR: numerical_fluctuations",
    )

    print("\n[End-state differences at T=5.0]")
    _, Q_flux_hr = solve_case(make_model(), mesh, FluxHRSolver)
    _, Q_flux_nohr = solve_case(make_model(), mesh, FluxNoHRSolver)
    _, Q_quasi_nohr = solve_case(make_model(), mesh, QuasiNoHRSolver)

    Qfhr = Q_flux_hr[:, :n]
    Qfnr = Q_flux_nohr[:, :n]
    Qqnr = Q_quasi_nohr[:, :n]

    state_stats(Qfhr, Qfnr, "flux HR vs flux noHR")
    state_stats(Qfnr, Qqnr, "flux noHR vs quasi noHR")
    state_stats(Qfhr, Qqnr, "flux HR vs quasi noHR")


if __name__ == "__main__":
    run()
