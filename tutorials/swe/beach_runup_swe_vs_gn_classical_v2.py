import os

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
import zoomy_core.misc.io as io
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver
from zoomy_core.fvm.symbolic_numerics_v2 import PositiveNonconservativeRusanov
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import (
    DerivativeAwareSolver,
    DerivativeAwareSolverMixin,
    DerivativeSpec,
    StructuredDerivativeModel,
)


class SWEBeachTopoModel(StructuredDerivativeModel):
    """
    SWE-like beach model in conservative form:
      Q = [b, h, hu]
    """

    dimension = 1
    variables = ["b", "h", "hu"]
    user_aux_variables = ["hinv"]
    parameters = {"g": (9.81, "positive"), "eps": (1e-8, "positive")}

    # For hydrostatic reconstruction we scale only momentum-like entries.
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
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = 0.5 * g * h * h
        return ZArray(P)

    def nonconservative_matrix(self):
        h = self.Q.h
        g = self.params.g
        A = ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        A[2, 0, 0] = g * h
        return A

    def source(self):
        return ZArray.zeros(self.n_variables)


class ClassicalGNBeachTopoModel(StructuredDerivativeModel):
    """
    Classical GN-like beach model in primitive form:
      Q = [b, h, u]
      h_t + (h u)_x = 0
      u_t + (u^2/2 + g h)_x + g b_x = (h^2/3) * (u_txx + u*u_xxx - u_x*u_xx)
    """

    dimension = 1
    variables = ["b", "h", "u"]
    user_aux_variables = ["hinv"]
    parameters = {
        "g": (9.81, "positive"),
        "eps": (1e-8, "positive"),
    }
    auto_requested_derivatives = False

    # In primitive form, u should not be rescaled by h in reconstruction.
    numerics_scaled_q_indices = []

    def requested_derivatives(self):
        return [
            DerivativeSpec(field="u", axes=("t", "x", "x")),
            DerivativeSpec(field="u", axes=("x",)),
            DerivativeSpec(field="u", axes=("x", "x")),
            DerivativeSpec(field="u", axes=("x", "x", "x")),
        ]

    def flux(self):
        h = self.Q.h
        u = self.Q.u
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = 0
        F[1, 0] = h * u
        F[2, 0] = 0.5 * u * u
        return ZArray(F)

    def hydrostatic_pressure(self):
        h = self.Q.h
        g = self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = g * h
        return ZArray(P)

    def nonconservative_matrix(self):
        g = self.params.g
        A = ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        A[2, 0, 0] = g
        return A

    def source(self):
        h = self.Q.h
        u = self.Q.u
        u_txx = self.D.dtxx(self.Q.u)
        u_x = self.D.dx(self.Q.u)
        u_xx = self.D.dxx(self.Q.u)
        u_xxx = self.D.diff(self.Q.u, ("x", "x", "x"))
        S = ZArray.zeros(self.n_variables)
        S[2] = (h * h * (1.0 / 3.0)) * (u_txx + u * u_xxx - u_x * u_xx)
        return S


class SymbolicNumericsBeachFluxMixin:
    """
    Uses symbolic_numerics_v2 for face fluctuations (with wet/dry reconstruction)
    and keeps the derivative-aware source workflow for GN-like models.
    """

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
        out = DerivativeAwareSolverMixin.update_qaux(
            self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )
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

        numerics = PositiveNonconservativeRusanov(
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
                # Split path-conservative update:
                # - conservative numerical flux contribution
                # - nonconservative fluctuation contribution
                dQ[:, iA[f]] -= (num_flux + Dm) * face_volumes[f] / cell_volumesA[f]
                dQ[:, iB[f]] -= (-num_flux + Dp) * face_volumes[f] / cell_volumesB[f]
            return dQ

        return flux_operator


# Explicit solver for SWE-like (non-stiff source) runs.
class SymbolicNumericsBeachSolver(SymbolicNumericsBeachFluxMixin, DerivativeAwareSolver):
    pass


# IMEX solver for GN-like (stiff derivative-coupled source) runs.
class SymbolicNumericsBeachIMEXSolver(
    SymbolicNumericsBeachFluxMixin, IMEXSourceSolver
):
    pass


def make_bathymetry(x, x_beach_start=25.0, slope=1.0 / 19.85):
    return np.maximum(0.0, slope * (x - x_beach_start))


def make_initial_eta(x, h0=1.0, amp=0.0185, x0=12.0):
    # Solitary-wave-like profile used in canonical beach-runup benchmarks.
    k = np.sqrt(3.0 * amp / (4.0 * h0**3))
    sech = 1.0 / np.cosh(k * (x - x0))
    return h0 + amp * sech * sech


def make_model(model_cls):
    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(tag="left"),
            BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
        ]
    )

    def ic_q(x):
        X = float(x[0])
        b = make_bathymetry(X)
        eta = make_initial_eta(X)
        h = max(eta - b, 1e-8)
        Q = np.zeros(3, dtype=float)
        Q[0] = b
        Q[1] = h
        Q[2] = 0.0
        return Q

    def ic_aux(x):
        X = float(x[0])
        b = make_bathymetry(X)
        eta = make_initial_eta(X)
        h = max(eta - b, 1e-8)
        return np.array([1.0 / max(h, 1e-8)], dtype=float)

    return model_cls(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_q),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


def shoreline_position(x, h, threshold=2e-3):
    wet = np.where(h > threshold)[0]
    if wet.size == 0:
        return float("nan")
    return float(np.max(x[wet]))


def solve_case(model, mesh, time_end, cfl, model_tag):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/tutorial_beach_runup_v2/hdf5",
            filename=f"beach_runup_{model_tag}",
            snapshots=80,
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
    QN, _ = solver.solve(mesh, model, write_output=True)
    h5_path = os.path.join(
        settings.output.directory, f"{settings.output.filename}.h5"
    )
    x_timeline, q_timeline, _qaux_timeline, t_timeline = io.load_timeline_of_fields_from_hdf5(
        h5_path
    )
    return Q0, QN, x_timeline, q_timeline, t_timeline


def infer_required_lsq_degree(models, minimum_degree=1):
    """
    Infer LSQ polynomial degree from declared/inferred derivative demand.
    """
    required = int(minimum_degree)
    for model in models:
        specs = getattr(model, "derivative_specs", [])
        for spec in specs:
            nx = sum(axis == "x" for axis in spec.axes)
            required = max(required, int(nx))
    return required


def compute_runup_metrics(x_shore_time, t_timeline):
    """
    Compute shoreline and runup metrics from saved snapshots.
    """
    x_shore = np.asarray(x_shore_time, dtype=float)

    valid = np.isfinite(x_shore)
    if not np.any(valid):
        return {
            "x_shore_time": x_shore,
            "x_shore_max": float("nan"),
            "t_shore_max": float("nan"),
            "R_num": float("nan"),
        }

    i_max = np.nanargmax(x_shore)
    x_shore_max = float(x_shore[i_max])
    t_shore_max = float(t_timeline[i_max])

    # Vertical excursion relative to initial shoreline level on the same slope.
    x_shore_0 = float(x_shore[0]) if np.isfinite(x_shore[0]) else x_shore_max
    b_0 = float(make_bathymetry(x_shore_0))
    b_at_max = float(make_bathymetry(x_shore_max))
    R_num = b_at_max - b_0

    return {
        "x_shore_time": x_shore,
        "x_shore_max": x_shore_max,
        "t_shore_max": t_shore_max,
        "R_num": R_num,
        "x_shore_0": x_shore_0,
    }


def run():
    swe = make_model(SWEBeachTopoModel)
    gn = make_model(ClassicalGNBeachTopoModel)
    lsq_degree = infer_required_lsq_degree([swe, gn], minimum_degree=1)

    # Synolakis-like canonical setup (nonbreaking solitary-wave runup regime).
    mesh = petscMesh.Mesh.create_1d(
        domain=(0.0, 70.0),
        n_inner_cells=36,
        lsq_degree=lsq_degree,
    )
    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n]

    print(f"inferred lsq_degree from derivative demand: {lsq_degree}")

    Q0_swe, QN_swe, x_swe, q_t_swe, t_swe = solve_case(
        swe, mesh, time_end=2.0, cfl=0.25, model_tag="swe"
    )
    Q0_gn, QN_gn, x_gn, q_t_gn, t_gn = solve_case(
        gn, mesh, time_end=2.0, cfl=0.25, model_tag="gn_classical"
    )

    b = Q0_swe[0, :n]
    h0 = Q0_swe[1, :n]
    hN_swe = QN_swe[1, :n]
    hN_gn = QN_gn[1, :n]
    eta0 = b + h0
    etaN_swe = b + hN_swe
    etaN_gn = b + hN_gn

    h_visc_thr = 2e-3
    uN_swe = np.where(hN_swe > h_visc_thr, QN_swe[2, :n] / np.maximum(hN_swe, 1e-10), np.nan)
    uN_gn = np.where(hN_gn > h_visc_thr, QN_gn[2, :n], np.nan)

    x_shore_swe = shoreline_position(x, hN_swe)
    x_shore_gn = shoreline_position(x, hN_gn)
    print(f"SWE shoreline (h>2e-3): x={x_shore_swe:.4f}")
    print(f"GN-classical shoreline (h>2e-3): x={x_shore_gn:.4f}")
    print(
        f"SWE h-range: [{float(hN_swe.min()):.4e}, {float(hN_swe.max()):.4e}], "
        f"GN h-range: [{float(hN_gn.min()):.4e}, {float(hN_gn.max()):.4e}]"
    )
    # Nonbreaking Synolakis-type estimate (reference order-of-magnitude).
    h0 = 1.0
    amp = 0.0185
    slope = 1.0 / 19.85
    runup_ref = h0 * 2.831 * np.sqrt(1.0 / slope) * (amp / h0) ** 1.25
    print(f"reference nonbreaking runup estimate (Synolakis-type): R~{runup_ref:.4f}")

    x_swe_inner = x_swe[:n]
    x_gn_inner = x_gn[:n]
    q_t_swe_inner = q_t_swe[:, :, :n]
    q_t_gn_inner = q_t_gn[:, :, :n]

    x_shore_swe_t = np.array(
        [shoreline_position(x_swe_inner, q_t_swe_inner[i, 1, :], threshold=2e-3) for i in range(q_t_swe_inner.shape[0])],
        dtype=float,
    )
    x_shore_gn_t = np.array(
        [shoreline_position(x_gn_inner, q_t_gn_inner[i, 1, :], threshold=2e-3) for i in range(q_t_gn_inner.shape[0])],
        dtype=float,
    )
    m_swe = compute_runup_metrics(x_shore_swe_t, t_swe)
    m_gn = compute_runup_metrics(x_shore_gn_t, t_gn)
    print(
        f"SWE max shoreline: x={m_swe['x_shore_max']:.4f} at t={m_swe['t_shore_max']:.4f}, "
        f"R_num={m_swe['R_num']:.4f}, R_num/R_ref={m_swe['R_num']/runup_ref:.4f}"
    )
    print(
        f"GN max shoreline: x={m_gn['x_shore_max']:.4f} at t={m_gn['t_shore_max']:.4f}, "
        f"R_num={m_gn['R_num']:.4f}, R_num/R_ref={m_gn['R_num']/runup_ref:.4f}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(x, b, "k-", linewidth=1.0, label="bathymetry b")
    axes[0].plot(x, eta0, "k--", linewidth=1.0, label="eta initial")
    axes[0].plot(x, etaN_swe, label="eta SWE final")
    axes[0].plot(x, etaN_gn, label="eta GN-classical final")
    axes[0].set_title("Beach runup: free surface comparison")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].plot(x, uN_swe, label="u SWE final")
    axes[1].plot(x, uN_gn, label="u GN-classical final")
    axes[1].set_title("Beach runup: velocity comparison (wet cells only)")
    axes[1].set_xlabel("x")
    axes[1].legend()

    out_path = "outputs/tutorial_beach_runup_v2/beach_runup_swe_vs_gn_classical.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)

    # Additional field-level comparison (raw solution components).
    fig_fields, axes_fields = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes_fields[0].plot(x, Q0_swe[0, :n], "k--", linewidth=1.0, label="b initial")
    axes_fields[0].plot(x, QN_swe[0, :n], label="b SWE final")
    axes_fields[0].plot(x, QN_gn[0, :n], label="b GN final")
    axes_fields[0].set_title("Field Q[0] (bathymetry b)")
    axes_fields[0].set_xlabel("x")
    axes_fields[0].legend()

    axes_fields[1].plot(x, Q0_swe[1, :n], "k--", linewidth=1.0, label="h initial")
    axes_fields[1].plot(x, QN_swe[1, :n], label="h SWE final")
    axes_fields[1].plot(x, QN_gn[1, :n], label="h GN final")
    axes_fields[1].set_title("Field Q[1] (water depth h)")
    axes_fields[1].set_xlabel("x")
    axes_fields[1].legend()

    axes_fields[2].plot(x, Q0_swe[2, :n], "k--", linewidth=1.0, label="Q[2] initial")
    axes_fields[2].plot(x, QN_swe[2, :n], label="Q[2] SWE final (hu)")
    axes_fields[2].plot(x, QN_gn[2, :n], label="Q[2] GN final (u)")
    axes_fields[2].set_title("Field Q[2] comparison (SWE: hu, GN: u)")
    axes_fields[2].set_xlabel("x")
    axes_fields[2].legend()

    out_fields_path = "outputs/tutorial_beach_runup_v2/beach_runup_swe_vs_gn_fields.png"
    fig_fields.savefig(out_fields_path, dpi=150)
    print("saved:", out_fields_path)

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    run()
