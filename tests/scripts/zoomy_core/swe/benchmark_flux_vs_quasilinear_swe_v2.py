import csv
import os

import numpy as np
import sympy as sp

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.symbolic_numerics_v2 import (
    NonconservativeRusanov,
    PositiveNonconservativeRusanov,
    PositiveQuasilinearRusanov,
    QuasilinearRusanov,
)
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver

from tutorials.swe.beach_runup_swe_vs_gn_classical_v2 import (
    SWEBeachTopoModel,
    make_bathymetry,
)


def _make_swe_model(b_func, eta_func, u0_func, boundary_conditions):
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
                    runtime_numerics.numerical_fluctuations(
                        qA, qB, qauxA, qauxB, parameters, n
                    ),
                    dtype=float,
                ).reshape(-1)
                num_flux = np.asarray(
                    runtime_numerics.numerical_flux(
                        qA, qB, qauxA, qauxB, parameters, n
                    ),
                    dtype=float,
                ).reshape(-1)
                Dp = fluct[:n_vars]
                Dm = fluct[n_vars:]
                dQ[:, iA[f]] -= (num_flux + Dm) * face_volumes[f] / cell_volumesA[f]
                dQ[:, iB[f]] -= (-num_flux + Dp) * face_volumes[f] / cell_volumesB[f]
            return dQ

        return flux_operator


class SWEFluxSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = PositiveNonconservativeRusanov


class SWEQuasilinearSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = PositiveQuasilinearRusanov


class SWEQuasilinearNoHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = QuasilinearRusanov


class WetDryPathPositiveQuasilinearRusanov(PositiveQuasilinearRusanov):
    """
    Experimental path-adjusted quasilinear variant:
    integrate along a path in (eta, b) with h = max(0, eta-b),
    then rebuild hu from interpolated velocity and path depth.
    """

    def _path_state(self, qL, qR, auxL, auxR, xi):
        q_path = ZArray(qL + xi * (qR - qL))
        aux_path = ZArray(auxL + xi * (auxR - auxL))

        bL = self._field_value("b", qL, auxL)
        bR = self._field_value("b", qR, auxR)
        hL = self._field_value("h", qL, auxL)
        hR = self._field_value("h", qR, auxR)
        etaL = hL + bL
        etaR = hR + bR

        b_path = (1 - xi) * bL + xi * bR
        eta_path = (1 - xi) * etaL + xi * etaR
        h_path = sp.Max(0.0, eta_path - b_path)
        eps = self._eps_symbol()

        self._set_field_value("b", q_path, aux_path, b_path)
        self._set_field_value("h", q_path, aux_path, h_path)
        if "hinv" in self._field_map:
            self._set_field_value("hinv", q_path, aux_path, 1 / (h_path + eps))

        # For momentum-like entries, interpolate velocity and rebuild hu=h*u
        # to keep depth-momentum consistency along the path.
        for idx in self._scaled_q_indices:
            uL = qL[idx] / sp.Max(hL, eps)
            uR = qR[idx] / sp.Max(hR, eps)
            u_path = (1 - xi) * uL + xi * uR
            q_path[idx] = h_path * u_path

        return q_path, aux_path

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        n_vars = self.model.n_variables
        dim = len(n)
        dQ = qR - qL
        A_int = ZArray.zeros(n_vars, n_vars)

        for xi, wi in zip(xi_np, wi_np):
            q_path, aux_path = self._path_state(qL, qR, auxL, auxR, xi)
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)
            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i, j, d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n

        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )
        term_advection = A_int @ dQ
        Id = self.get_viscosity_identity_fluctuations()
        term_dissipation = s_max * (Id @ dQ)
        Dp = 0.5 * (term_advection + term_dissipation)
        Dm = 0.5 * (term_advection - term_dissipation)
        return ZArray([Dp, Dm]).flatten()


class SWEQuasilinearWetDryPathSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = WetDryPathPositiveQuasilinearRusanov


class PositiveQuasilinearNoPressureJump(PositiveQuasilinearRusanov):
    """
    Diagnostic variant: keep HR states, but remove the hydrostatic-pressure
    jump correction term from PositiveRusanov.numerical_fluctuations.
    """

    def numerical_fluctuations(self):
        return NonconservativeRusanov.numerical_fluctuations(self)


class SWEQuasilinearHRNoJumpSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = PositiveQuasilinearNoPressureJump


def _solve(model, mesh, time_end, cfl, solver_cls):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/flux_vs_quasilinear_swe_v2/tmp",
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


def _weighted_mass(h, volumes):
    return float(np.sum(h * volumes))


def _compare_solutions(vols, Q_ref, Q_test):
    h_f = Q_ref[1, :]
    h_q = Q_test[1, :]
    eta_f = Q_ref[0, :] + h_f
    eta_q = Q_test[0, :] + h_q
    u_f = Q_ref[2, :] / np.maximum(h_f, 1e-10)
    u_q = Q_test[2, :] / np.maximum(h_q, 1e-10)

    d_h = h_q - h_f
    d_eta = eta_q - eta_f
    d_u = u_q - u_f

    l2 = lambda arr: float(np.sqrt(np.sum(arr * arr) / arr.size))
    linf = lambda arr: float(np.max(np.abs(arr)))

    m_f = _weighted_mass(h_f, vols)
    m_q = _weighted_mass(h_q, vols)

    return {
        "dh_l2": l2(d_h),
        "dh_linf": linf(d_h),
        "deta_l2": l2(d_eta),
        "deta_linf": linf(d_eta),
        "du_l2": l2(d_u),
        "du_linf": linf(d_u),
        "mass_flux": m_f,
        "mass_quasi": m_q,
        "mass_rel_diff": abs(m_q - m_f) / max(abs(m_f), 1e-16),
        "hmin_flux": float(h_f.min()),
        "hmin_quasi": float(h_q.min()),
        "finite_ref": float(np.isfinite(Q_ref).sum() / Q_ref.size),
        "finite_test": float(np.isfinite(Q_test).sum() / Q_test.size),
    }


def run():
    os.makedirs("outputs/flux_vs_quasilinear_swe_v2", exist_ok=True)

    cases = [
        dict(
            label="flat_dynamic_fullwet",
            domain=(0.0, 40.0),
            n_cells=220,
            time_end=4.0,
            cfl=0.22,
            b_func=lambda x: 0.0,
            eta_func=lambda x: 1.0 + 0.03 * np.exp(-((x - 9.0) ** 2) / 5.0),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
        ),
        dict(
            label="bathy_dynamic_fullwet",
            domain=(0.0, 70.0),
            n_cells=260,
            time_end=5.0,
            cfl=0.22,
            b_func=lambda x: make_bathymetry(x),
            eta_func=lambda x: 2.8 + 0.03 * np.exp(-((x - 10.0) ** 2) / 5.0),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions(
                [
                    BC.Extrapolation(tag="left"),
                    BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
                ]
            ),
        ),
        dict(
            label="bathy_dynamic_wetdry",
            domain=(0.0, 70.0),
            n_cells=180,
            # Quasilinear path becomes extremely stiff near advancing dry fronts.
            # Keep a short horizon to compare early-time dynamics only.
            time_end=1.0,
            cfl=0.18,
            b_func=lambda x: make_bathymetry(x),
            eta_func=lambda x: 1.0 + 0.0185 * (1.0 / np.cosh(np.sqrt(3.0 * 0.0185 / 4.0) * (x - 12.0))) ** 2,
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions(
                [
                    BC.Extrapolation(tag="left"),
                    BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
                ]
            ),
        ),
    ]

    rows = []
    for cfg in cases:
        mesh = petscMesh.Mesh.create_1d(
            domain=cfg["domain"],
            n_inner_cells=cfg["n_cells"],
            lsq_degree=1,
        )
        n = mesh.n_inner_cells
        x = mesh.cell_centers[0, :n]
        vols = mesh.cell_volumes[:n]

        model_flux = _make_swe_model(
            cfg["b_func"], cfg["eta_func"], cfg["u0_func"], cfg["bcs"]
        )
        model_quasi_hr = _make_swe_model(
            cfg["b_func"], cfg["eta_func"], cfg["u0_func"], cfg["bcs"]
        )
        model_quasi_nohr = _make_swe_model(
            cfg["b_func"], cfg["eta_func"], cfg["u0_func"], cfg["bcs"]
        )

        _, QN_flux = _solve(model_flux, mesh, cfg["time_end"], cfg["cfl"], SWEFluxSolver)
        _, QN_quasi_hr = _solve(
            model_quasi_hr, mesh, cfg["time_end"], cfg["cfl"], SWEQuasilinearSolver
        )
        _, QN_quasi_nohr = _solve(
            model_quasi_nohr, mesh, cfg["time_end"], cfg["cfl"], SWEQuasilinearNoHRSolver
        )
        model_quasi_wdpath = _make_swe_model(
            cfg["b_func"], cfg["eta_func"], cfg["u0_func"], cfg["bcs"]
        )
        model_quasi_hr_nojump = _make_swe_model(
            cfg["b_func"], cfg["eta_func"], cfg["u0_func"], cfg["bcs"]
        )
        _, QN_quasi_wdpath = _solve(
            model_quasi_wdpath,
            mesh,
            cfg["time_end"],
            cfg["cfl"],
            SWEQuasilinearWetDryPathSolver,
        )
        _, QN_quasi_hr_nojump = _solve(
            model_quasi_hr_nojump,
            mesh,
            cfg["time_end"],
            cfg["cfl"],
            SWEQuasilinearHRNoJumpSolver,
        )
        Qf = QN_flux[:, :n]
        Qq_hr = QN_quasi_hr[:, :n]
        Qq_nohr = QN_quasi_nohr[:, :n]
        Qq_wdpath = QN_quasi_wdpath[:, :n]
        Qq_hr_nojump = QN_quasi_hr_nojump[:, :n]

        m_hr = _compare_solutions(vols, Qf, Qq_hr)
        m_hr["case"] = cfg["label"]
        m_hr["n_cells"] = int(cfg["n_cells"])
        m_hr["variant"] = "quasilinear_hr"
        rows.append(m_hr)

        m_nohr = _compare_solutions(vols, Qf, Qq_nohr)
        m_nohr["case"] = cfg["label"]
        m_nohr["n_cells"] = int(cfg["n_cells"])
        m_nohr["variant"] = "quasilinear_nohr"
        rows.append(m_nohr)

        m_hr_vs_nohr = _compare_solutions(vols, Qq_hr, Qq_nohr)
        m_hr_vs_nohr["case"] = cfg["label"]
        m_hr_vs_nohr["n_cells"] = int(cfg["n_cells"])
        m_hr_vs_nohr["variant"] = "hr_vs_nohr"
        rows.append(m_hr_vs_nohr)

        m_wdpath = _compare_solutions(vols, Qf, Qq_wdpath)
        m_wdpath["case"] = cfg["label"]
        m_wdpath["n_cells"] = int(cfg["n_cells"])
        m_wdpath["variant"] = "quasilinear_hr_wdpath"
        rows.append(m_wdpath)

        m_hr_vs_wdpath = _compare_solutions(vols, Qq_hr, Qq_wdpath)
        m_hr_vs_wdpath["case"] = cfg["label"]
        m_hr_vs_wdpath["n_cells"] = int(cfg["n_cells"])
        m_hr_vs_wdpath["variant"] = "hr_vs_wdpath"
        rows.append(m_hr_vs_wdpath)

        m_hr_nojump = _compare_solutions(vols, Qf, Qq_hr_nojump)
        m_hr_nojump["case"] = cfg["label"]
        m_hr_nojump["n_cells"] = int(cfg["n_cells"])
        m_hr_nojump["variant"] = "quasilinear_hr_nojump"
        rows.append(m_hr_nojump)

        m_hr_vs_nojump = _compare_solutions(vols, Qq_hr, Qq_hr_nojump)
        m_hr_vs_nojump["case"] = cfg["label"]
        m_hr_vs_nojump["n_cells"] = int(cfg["n_cells"])
        m_hr_vs_nojump["variant"] = "hr_vs_hr_nojump"
        rows.append(m_hr_vs_nojump)

        print(f"{cfg['label']}:")
        print(
            f"  flux vs quasi-HR   : deta_linf={m_hr['deta_linf']:.3e}, du_linf={m_hr['du_linf']:.3e}, "
            f"mass_rel_diff={m_hr['mass_rel_diff']:.3e}"
        )
        print(
            f"  flux vs quasi-noHR : deta_linf={m_nohr['deta_linf']:.3e}, du_linf={m_nohr['du_linf']:.3e}, "
            f"mass_rel_diff={m_nohr['mass_rel_diff']:.3e}"
        )
        print(
            f"  quasi-HR vs noHR   : deta_linf={m_hr_vs_nohr['deta_linf']:.3e}, du_linf={m_hr_vs_nohr['du_linf']:.3e}, "
            f"mass_rel_diff={m_hr_vs_nohr['mass_rel_diff']:.3e}"
        )
        print(
            f"  flux vs quasi-wdpath: deta_linf={m_wdpath['deta_linf']:.3e}, du_linf={m_wdpath['du_linf']:.3e}, "
            f"mass_rel_diff={m_wdpath['mass_rel_diff']:.3e}"
        )
        print(
            f"  quasi-HR vs wdpath : deta_linf={m_hr_vs_wdpath['deta_linf']:.3e}, du_linf={m_hr_vs_wdpath['du_linf']:.3e}, "
            f"mass_rel_diff={m_hr_vs_wdpath['mass_rel_diff']:.3e}"
        )
        print(
            f"  flux vs HR-nojump  : deta_linf={m_hr_nojump['deta_linf']:.3e}, du_linf={m_hr_nojump['du_linf']:.3e}, "
            f"mass_rel_diff={m_hr_nojump['mass_rel_diff']:.3e}"
        )
        print(
            f"  quasi-HR vs HR-nojump: deta_linf={m_hr_vs_nojump['deta_linf']:.3e}, du_linf={m_hr_vs_nojump['du_linf']:.3e}, "
            f"mass_rel_diff={m_hr_vs_nojump['mass_rel_diff']:.3e}"
        )

    out_csv = "outputs/flux_vs_quasilinear_swe_v2/metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", out_csv)


if __name__ == "__main__":
    run()
