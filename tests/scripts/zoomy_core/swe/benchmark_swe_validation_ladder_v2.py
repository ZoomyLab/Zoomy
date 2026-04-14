import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.misc.misc import Zstruct

from tutorials.swe.beach_runup_swe_vs_gn_classical_v2 import (
    SWEBeachTopoModel,
    SymbolicNumericsBeachSolver,
    make_bathymetry,
    shoreline_position,
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


def _solve_swe_case(model, mesh, time_end, cfl):
    settings = Zstruct(
        output=Zstruct(
            directory="outputs/swe_validation_ladder_v2/tmp",
            filename="tmp",
            snapshots=2,
            clean_directory=False,
        )
    )
    solver = SymbolicNumericsBeachSolver(
        time_end=time_end,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=cfl),
    )
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    QN, _ = solver.solve(mesh, model, write_output=False)
    return Q0, QN


def _weighted_mass(h, cell_volumes):
    return float(np.sum(h * cell_volumes))


def _compute_case_metrics(name, x, volumes, Q0, QN, shore_threshold=2e-3):
    h0 = Q0[1, :]
    hN = QN[1, :]
    huN = QN[2, :]
    eta0 = Q0[0, :] + h0
    etaN = QN[0, :] + hN

    m0 = _weighted_mass(h0, volumes)
    mN = _weighted_mass(hN, volumes)
    rel_mass_drift = abs(mN - m0) / max(abs(m0), 1e-16)

    finite_ratio = float(np.isfinite(QN).sum() / QN.size)
    h_min = float(np.min(hN))
    h_max = float(np.max(hN))
    uN = huN / np.maximum(hN, 1e-10)
    u_max = float(np.max(np.abs(uN[np.isfinite(uN)])))
    eta_inf_drift = float(np.max(np.abs(etaN - eta0)))

    x_peak0 = float(x[np.argmax(h0)])
    x_peakN = float(x[np.argmax(hN)])
    x_shore0 = shoreline_position(x, h0, threshold=shore_threshold)
    x_shoreN = shoreline_position(x, hN, threshold=shore_threshold)
    shore_shift = float(x_shoreN - x_shore0) if np.isfinite(x_shore0) and np.isfinite(x_shoreN) else float("nan")

    return {
        "case": name,
        "finite_ratio": finite_ratio,
        "h_min": h_min,
        "h_max": h_max,
        "u_max": u_max,
        "mass0": m0,
        "massN": mN,
        "rel_mass_drift": rel_mass_drift,
        "eta_inf_drift": eta_inf_drift,
        "peak_shift": float(x_peakN - x_peak0),
        "shore_shift": shore_shift,
    }


def _print_case_summary(metrics, thresholds):
    ok_finite = metrics["finite_ratio"] >= 1.0
    ok_pos = metrics["h_min"] >= -thresholds["h_neg_tol"]
    ok_mass = metrics["rel_mass_drift"] <= thresholds["mass_rel_tol"]
    ok_eta = metrics["eta_inf_drift"] <= thresholds["eta_inf_tol"]
    status = "PASS" if all([ok_finite, ok_pos, ok_mass, ok_eta]) else "CHECK"

    print(
        f"{metrics['case']}: {status} | finite={metrics['finite_ratio']:.3f}, "
        f"h_min={metrics['h_min']:.3e}, h_max={metrics['h_max']:.3e}, "
        f"mass_drift={metrics['rel_mass_drift']:.3e}, eta_inf_drift={metrics['eta_inf_drift']:.3e}, "
        f"peak_shift={metrics['peak_shift']:.3f}, shore_shift={metrics['shore_shift']:.3f}"
    )


def _run_one_config(cfg, n_cells=None):
    n_inner_cells = int(cfg["n_cells"] if n_cells is None else n_cells)
    mesh = LSQMesh.create_1d(
        domain=cfg["domain"],
        n_inner_cells=n_inner_cells,
        lsq_degree=1,
    )
    n = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n]
    vols = mesh.cell_volumes[:n]
    model = _make_swe_model(
        b_func=cfg["b_func"],
        eta_func=cfg["eta_func"],
        u0_func=cfg["u0_func"],
        boundary_conditions=cfg["bcs"],
    )
    Q0, QN = _solve_swe_case(model, mesh, cfg["time_end"], cfg["cfl"])
    return x, vols, Q0[:, :n], QN[:, :n], n_inner_cells


def _run_lake_at_rest_test():
    cfg = dict(
        label="swe_lake_at_rest_slope",
        domain=(0.0, 70.0),
        n_cells=280,
        time_end=4.0,
        cfl=0.25,
        b_func=lambda x: make_bathymetry(x),
        eta_func=lambda x: 2.8,
        u0_func=lambda x, h: 0.0,
        bcs=BC.BoundaryConditions(
            [
                BC.Wall(tag="left", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
                BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
            ]
        ),
    )
    x, vols, Q0, QN, n_cells = _run_one_config(cfg)
    m = _compute_case_metrics(cfg["label"], x, vols, Q0, QN)
    m["n_cells"] = n_cells
    # For lake-at-rest, this is the primary well-balancing signal.
    wb_ok = (m["eta_inf_drift"] <= 5e-3) and (m["u_max"] <= 5e-3)
    wb_status = "PASS" if wb_ok else "CHECK"
    print(
        f"{cfg['label']}: {wb_status} | eta_inf_drift={m['eta_inf_drift']:.3e}, "
        f"u_max={m['u_max']:.3e}, mass_drift={m['rel_mass_drift']:.3e}"
    )
    return m


def _run_refinement_sweep(cases, n_cells_levels):
    rows = []
    for cfg in cases:
        print(f"\nrefinement sweep: {cfg['label']}")
        for n_cells in n_cells_levels:
            x, vols, Q0, QN, _ = _run_one_config(cfg, n_cells=n_cells)
            m = _compute_case_metrics(cfg["label"], x, vols, Q0, QN)
            m["n_cells"] = n_cells
            rows.append(m)
            print(
                f"  n={n_cells:4d} | mass_drift={m['rel_mass_drift']:.3e}, "
                f"eta_inf_drift={m['eta_inf_drift']:.3e}, h_min={m['h_min']:.3e}, h_max={m['h_max']:.3e}"
            )
    return rows


def _print_refinement_table(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["case"]].append(r)
    print("\nrefinement summary table:")
    for case, vals in grouped.items():
        vals = sorted(vals, key=lambda k: int(k["n_cells"]))
        print(f"  {case}")
        for v in vals:
            print(
                f"    n={int(v['n_cells']):4d} | mass={v['rel_mass_drift']:.3e} | "
                f"eta_inf={v['eta_inf_drift']:.3e} | peak_shift={v['peak_shift']:.3f} | shore_shift={v['shore_shift']:.3f}"
            )


def run():
    thresholds = {
        "h_neg_tol": 1e-10,
        "mass_rel_tol": 3e-3,
        "eta_inf_tol": 2e-1,
    }

    cases = [
        dict(
            label="swe_flat_fullwet",
            domain=(0.0, 40.0),
            n_cells=240,
            time_end=4.0,
            cfl=0.25,
            b_func=lambda x: 0.0,
            eta_func=lambda x: 1.0 + 0.03 * np.exp(-((x - 10.0) ** 2) / 6.0),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
        ),
        dict(
            label="swe_bathy_fullwet",
            domain=(0.0, 70.0),
            n_cells=280,
            time_end=5.0,
            cfl=0.22,
            b_func=lambda x: make_bathymetry(x),
            # Keep eta above max(b) so this case stays fully wet.
            eta_func=lambda x: 2.8 + 0.02 * np.exp(-((x - 12.0) ** 2) / 5.0),
            u0_func=lambda x, h: 0.0,
            bcs=BC.BoundaryConditions(
                [
                    BC.Extrapolation(tag="left"),
                    BC.Wall(tag="right", momentum_field_indices=[[2]], permeability=0.0, wall_slip=1.0),
                ]
            ),
        ),
        dict(
            label="swe_bathy_wetdry",
            domain=(0.0, 70.0),
            n_cells=280,
            time_end=6.0,
            cfl=0.20,
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

    os.makedirs("outputs/swe_validation_ladder_v2", exist_ok=True)
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)

    for i, cfg in enumerate(cases):
        x, vols, Q0i, QNi, _ = _run_one_config(cfg)
        m = _compute_case_metrics(cfg["label"], x, vols, Q0i, QNi)
        rows.append(m)
        _print_case_summary(m, thresholds)

        b = Q0i[0, :]
        eta0 = b + Q0i[1, :]
        etaN = b + QNi[1, :]
        uN = QNi[2, :] / np.maximum(QNi[1, :], 1e-10)

        axes[i, 0].plot(x, b, "k-", linewidth=1.0, label="b")
        axes[i, 0].plot(x, eta0, "k--", linewidth=1.0, label="eta0")
        axes[i, 0].plot(x, etaN, label="eta final")
        axes[i, 0].set_title(f"{cfg['label']} - free surface")
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].plot(x, uN, label="u final")
        axes[i, 1].set_title(f"{cfg['label']} - velocity")
        axes[i, 1].legend(fontsize=8)

    csv_path = "outputs/swe_validation_ladder_v2/metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_path = "outputs/swe_validation_ladder_v2/swe_validation_ladder_v2.png"
    fig.savefig(out_path, dpi=150)
    print("\nsaved:", out_path)
    print("saved:", csv_path)

    print("\nwell-balancing test:")
    wb_metrics = _run_lake_at_rest_test()
    wb_csv_path = "outputs/swe_validation_ladder_v2/well_balancing_metrics.csv"
    with open(wb_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wb_metrics.keys()))
        writer.writeheader()
        writer.writerow(wb_metrics)
    print("saved:", wb_csv_path)

    refinement_levels = [120, 240, 360]
    refinement_rows = _run_refinement_sweep(cases, refinement_levels)
    _print_refinement_table(refinement_rows)
    refinement_csv_path = "outputs/swe_validation_ladder_v2/refinement_metrics.csv"
    with open(refinement_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(refinement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(refinement_rows)
    print("saved:", refinement_csv_path)


if __name__ == "__main__":
    run()
