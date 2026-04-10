import csv
import os
from typing import Dict, List, Tuple

import numpy as np

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.symbolic_numerics_v2 import (
    NonconservativeRusanov,
    PositiveNonconservativeRusanov,
    QuasilinearRusanov,
)
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.derivative_workflow import DerivativeAwareSolver

from tutorials.swe.beach_runup_swe_vs_gn_classical_v2 import SWEBeachTopoModel


def _load_reference_csv(path: str) -> Dict[str, np.ndarray]:
    cols = None
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                cols = [c.strip() for c in row]
            else:
                vals = []
                for v in row:
                    vv = v.strip()
                    vals.append(float(vv) if vv.lower() != "nan" else np.nan)
                rows.append(vals)
    assert cols is not None
    arr = np.asarray(rows, dtype=float)
    return {c: arr[:, j] for j, c in enumerate(cols)}


def _make_model(
    b_func,
    eta_func,
    u0_func,
    bc: BC.BoundaryConditions,
):
    def ic_q(x):
        X = float(x[0])
        b = float(b_func(X))
        eta = float(eta_func(X))
        h = max(eta - b, 1e-8)
        u = float(u0_func(X, h))
        Q = np.zeros(3, dtype=float)
        Q[0] = b
        Q[1] = h
        Q[2] = h * u
        return Q

    def ic_aux(x):
        X = float(x[0])
        b = float(b_func(X))
        eta = float(eta_func(X))
        h = max(eta - b, 1e-8)
        return np.array([1.0 / max(h, 1e-8)], dtype=float)

    return SWEBeachTopoModel(
        boundary_conditions=bc,
        initial_conditions=IC.UserFunction(ic_q),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )


class _SWEOperatorMixin:
    numerics_cls = PositiveNonconservativeRusanov

    def _field_map_from_symbolic_model(self, symbolic_model):
        var_keys = list(symbolic_model.variables.keys())
        aux_keys = list(symbolic_model.aux_variables.keys())
        return {
            "h": {"container": "q", "index": var_keys.index("h")},
            "b": {"container": "q", "index": var_keys.index("b")},
            "hinv": {"container": "qaux", "index": aux_keys.index("hinv")},
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


class SWEFluxHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = PositiveNonconservativeRusanov


class SWEFluxNoHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = NonconservativeRusanov


class SWEQuasiNoHRSolver(_SWEOperatorMixin, DerivativeAwareSolver):
    numerics_cls = QuasilinearRusanov


def _solve(model, mesh, time_end, cfl, solver_cls):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/benchmark_swashes_wetdry_compare_v2/tmp",
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
    QN, _ = solver.solve(mesh, model, write_output=False)
    return QN


def _error_norms(num: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    d = (num - ref)[mask]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    l1 = float(np.mean(np.abs(d)))
    l2 = float(np.sqrt(np.mean(d * d)))
    linf = float(np.max(np.abs(d)))
    return l1, l2, linf


def _smooth_region_mask(h_ref: np.ndarray, u_ref: np.ndarray, wet_thr: float = 1e-8) -> np.ndarray:
    n = h_ref.size
    grad_h = np.zeros_like(h_ref)
    grad_u = np.zeros_like(h_ref)
    grad_h[1:] = np.abs(np.diff(np.nan_to_num(h_ref, nan=0.0)))
    grad_u[1:] = np.abs(np.diff(np.nan_to_num(u_ref, nan=0.0)))
    jump = grad_h + 0.2 * grad_u
    pos = jump[jump > 0]
    if pos.size == 0:
        shock_idx = np.array([], dtype=int)
    else:
        thr = float(np.quantile(pos, 0.90))
        shock_idx = np.where(jump >= thr)[0]
    mask = np.ones(n, dtype=bool)
    # Exclude wet/dry fringe and discontinuity neighborhoods.
    mask &= np.nan_to_num(h_ref, nan=0.0) > max(wet_thr, 5e-4)
    for i in shock_idx:
        i0 = max(0, i - 4)
        i1 = min(n, i + 5)
        mask[i0:i1] = False
    return mask


def run():
    os.makedirs("outputs/benchmark_swashes_wetdry_compare_v2", exist_ok=True)

    cases = [
        {
            "label": "swashes_dam_break_dry_ritter",
            "ref_csv": "outputs/swashes_reference_v2/swashes_dam_break_dry_ritter.csv",
            "domain": (0.0, 10.0),
            "n_cells": 200,
            "time_end": 6.0,
            "cfl": 0.2,
            # SWASHES case defaults inferred from exported reference.
            "b_func": lambda x: 0.0,
            "eta_func": lambda x: 0.005 if x < 5.0 else 0.0,
            "u0_func": lambda x, h: 0.0,
            "bcs": BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
            # Quasilinear-noHR is not robust on advancing dry fronts.
            "variants": ["flux_hr", "flux_nohr"],
        },
        {
            "label": "swashes_dam_break_step",
            "ref_csv": "outputs/swashes_reference_v2/swashes_dam_break_step.csv",
            "domain": (0.0, 20.0),
            "n_cells": 200,
            "time_end": 1.0,
            "cfl": 0.2,
            "b_func": lambda x: 0.0 if x < 10.0 else 1.0,
            "eta_func": lambda x: 4.0 if x < 10.0 else 2.0,
            "u0_func": lambda x, h: 0.0,
            "bcs": BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
            "variants": ["flux_hr", "flux_nohr", "quasi_nohr"],
        },
    ]

    summary_rows = []
    n_levels = [100, 200, 400]
    variants = [
        ("flux_hr", SWEFluxHRSolver),
        ("flux_nohr", SWEFluxNoHRSolver),
        ("quasi_nohr", SWEQuasiNoHRSolver),
    ]
    variant_map = {name: cls for name, cls in variants}

    for cfg in cases:
        ref = _load_reference_csv(cfg["ref_csv"])
        for n_cells in n_levels:
            mesh = LSQMesh.create_1d(
                domain=cfg["domain"],
                n_inner_cells=n_cells,
                lsq_degree=1,
            )
            n = mesh.n_inner_cells
            x_num = mesh.cell_centers[0, :n]

            x_ref = ref["x"]
            h_ref = ref["h"]
            u_ref = ref["u"]
            eta_ref = ref["eta"]

            h_ref_i = np.interp(x_num, x_ref, h_ref)
            u_ref_i = np.interp(x_num, x_ref, np.nan_to_num(u_ref, nan=0.0))
            eta_ref_i = np.interp(x_num, x_ref, eta_ref)

            wet_mask = h_ref_i > 1e-8
            all_mask = np.isfinite(h_ref_i)
            smooth_mask = _smooth_region_mask(h_ref_i, u_ref_i)
            smooth_wet_mask = smooth_mask & wet_mask

            print(f"\n{cfg['label']} | n={n_cells}")
            for variant_name in cfg.get("variants", [n for n, _ in variants]):
                solver_cls = variant_map[variant_name]
                model = _make_model(
                    b_func=cfg["b_func"],
                    eta_func=cfg["eta_func"],
                    u0_func=cfg["u0_func"],
                    bc=cfg["bcs"],
                )
                QN = _solve(model, mesh, cfg["time_end"], cfg["cfl"], solver_cls)
                h_num = QN[1, :n]
                b_num = QN[0, :n]
                eta_num = h_num + b_num
                u_num = QN[2, :n] / np.maximum(h_num, 1e-10)

                # Global errors.
                h_l1, h_l2, h_linf = _error_norms(h_num, h_ref_i, all_mask)
                eta_l1, eta_l2, eta_linf = _error_norms(eta_num, eta_ref_i, all_mask)
                u_l1, u_l2, u_linf = _error_norms(u_num, u_ref_i, wet_mask)
                # Smooth-region errors.
                hs_l1, hs_l2, hs_linf = _error_norms(h_num, h_ref_i, smooth_mask)
                etas_l1, etas_l2, etas_linf = _error_norms(eta_num, eta_ref_i, smooth_mask)
                us_l1, us_l2, us_linf = _error_norms(u_num, u_ref_i, smooth_wet_mask)

                print(
                    f"  {variant_name:10s} | all: hL1={h_l1:.2e}, uL1={u_l1:.2e}, etaL1={eta_l1:.2e} "
                    f"| smooth: hL1={hs_l1:.2e}, uL1={us_l1:.2e}, etaL1={etas_l1:.2e}"
                )

                summary_rows.append(
                    {
                        "case": cfg["label"],
                        "variant": variant_name,
                        "n_cells": n_cells,
                        "h_L1_all": h_l1,
                        "h_L2_all": h_l2,
                        "h_Linf_all": h_linf,
                        "eta_L1_all": eta_l1,
                        "eta_L2_all": eta_l2,
                        "eta_Linf_all": eta_linf,
                        "u_L1_wet_all": u_l1,
                        "u_L2_wet_all": u_l2,
                        "u_Linf_wet_all": u_linf,
                        "h_L1_smooth": hs_l1,
                        "h_L2_smooth": hs_l2,
                        "h_Linf_smooth": hs_linf,
                        "eta_L1_smooth": etas_l1,
                        "eta_L2_smooth": etas_l2,
                        "eta_Linf_smooth": etas_linf,
                        "u_L1_wet_smooth": us_l1,
                        "u_L2_wet_smooth": us_l2,
                        "u_Linf_wet_smooth": us_linf,
                    }
                )

    out_csv = "outputs/benchmark_swashes_wetdry_compare_v2/summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\nsaved:", out_csv)


if __name__ == "__main__":
    run()
