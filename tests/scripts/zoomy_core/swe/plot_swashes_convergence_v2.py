import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.model.boundary_conditions as BC

from tests.scripts.zoomy_core.swe.benchmark_swashes_wetdry_compare_v2 import (
    SWEFluxHRSolver,
    SWEFluxNoHRSolver,
    SWEQuasiNoHRSolver,
    _error_norms,
    _load_reference_csv,
    _make_model,
    _solve,
)


def _case_definitions():
    return [
        {
            "label": "swashes_dam_break_dry_ritter",
            "ref_csv": "outputs/swashes_reference_v2/swashes_dam_break_dry_ritter.csv",
            "domain": (0.0, 10.0),
            "time_end": 6.0,
            "cfl": 0.2,
            "b_func": lambda x: 0.0,
            "eta_func": lambda x: 0.005 if x < 5.0 else 0.0,
            "u0_func": lambda x, h: 0.0,
            "bcs": BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
            "variants": [
                ("flux_hr", SWEFluxHRSolver),
                ("flux_nohr", SWEFluxNoHRSolver),
                ("quasi_nohr", SWEQuasiNoHRSolver),
            ],
        },
        {
            "label": "swashes_dam_break_step",
            "ref_csv": "outputs/swashes_reference_v2/swashes_dam_break_step.csv",
            "domain": (0.0, 20.0),
            "time_end": 1.0,
            "cfl": 0.2,
            "b_func": lambda x: 0.0 if x < 10.0 else 1.0,
            "eta_func": lambda x: 4.0 if x < 10.0 else 2.0,
            "u0_func": lambda x, h: 0.0,
            "bcs": BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
            "variants": [
                ("flux_hr", SWEFluxHRSolver),
                ("flux_nohr", SWEFluxNoHRSolver),
                ("quasi_nohr", SWEQuasiNoHRSolver),
            ],
        },
    ]


def run():
    os.makedirs("outputs/benchmark_swashes_wetdry_compare_v2", exist_ok=True)
    n_levels = [20, 40, 80, 120, 240]

    rows = []
    for cfg in _case_definitions():
        ref = _load_reference_csv(cfg["ref_csv"])
        x_ref = ref["x"]
        h_ref = ref["h"]
        u_ref = np.nan_to_num(ref["u"], nan=0.0)
        eta_ref = ref["eta"]

        for variant_name, solver_cls in cfg["variants"]:
            for n_cells in n_levels:
                mesh = petscMesh.Mesh.create_1d(
                    domain=cfg["domain"],
                    n_inner_cells=n_cells,
                    lsq_degree=1,
                )
                n = mesh.n_inner_cells
                x_num = mesh.cell_centers[0, :n]
                h_ref_i = np.interp(x_num, x_ref, h_ref)
                u_ref_i = np.interp(x_num, x_ref, u_ref)
                eta_ref_i = np.interp(x_num, x_ref, eta_ref)
                all_mask = np.isfinite(h_ref_i)
                wet_mask = h_ref_i > 1e-8

                model = _make_model(
                    b_func=cfg["b_func"],
                    eta_func=cfg["eta_func"],
                    u0_func=cfg["u0_func"],
                    bc=cfg["bcs"],
                )

                try:
                    QN = _solve(model, mesh, cfg["time_end"], cfg["cfl"], solver_cls)
                    h_num = QN[1, :n]
                    eta_num = QN[0, :n] + h_num
                    u_num = QN[2, :n] / np.maximum(h_num, 1e-10)

                    h_l1, _, _ = _error_norms(h_num, h_ref_i, all_mask)
                    u_l1, _, _ = _error_norms(u_num, u_ref_i, wet_mask)
                    eta_l1, _, _ = _error_norms(eta_num, eta_ref_i, all_mask)
                    status = "ok"
                except Exception:
                    h_l1 = np.nan
                    u_l1 = np.nan
                    eta_l1 = np.nan
                    status = "failed"

                rows.append(
                    {
                        "case": cfg["label"],
                        "variant": variant_name,
                        "n_cells": n_cells,
                        "h_L1": h_l1,
                        "u_L1_wet": u_l1,
                        "eta_L1": eta_l1,
                        "status": status,
                    }
                )
                print(f"{cfg['label']} | {variant_name} | n={n_cells}: {status}, hL1={h_l1}")

    # Save numeric table.
    csv_path = "outputs/benchmark_swashes_wetdry_compare_v2/convergence_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", csv_path)

    # Plot one log-error figure per case.
    for case in sorted(set(r["case"] for r in rows)):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        case_rows = [r for r in rows if r["case"] == case]
        for variant in sorted(set(r["variant"] for r in case_rows)):
            vals = sorted([r for r in case_rows if r["variant"] == variant], key=lambda d: d["n_cells"])
            xs = np.array([v["n_cells"] for v in vals], dtype=float)
            ys = np.array([v["h_L1"] for v in vals], dtype=float)
            m = np.isfinite(ys) & (ys > 0)
            if np.any(m):
                ax.loglog(xs[m], ys[m], marker="o", label=f"{variant} (h L1)")

        ax.set_title(f"{case} convergence")
        ax.set_xlabel("n_cells")
        ax.set_ylabel("L1 error in h")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend()
        out_path = f"outputs/benchmark_swashes_wetdry_compare_v2/{case}_log_error_hL1.png"
        fig.savefig(out_path, dpi=150)
        print("saved:", out_path)


if __name__ == "__main__":
    run()
