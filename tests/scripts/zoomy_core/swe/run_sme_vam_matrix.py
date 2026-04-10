"""
SME/VAM test matrix: L0/L1/L2 × 2D/3D derivation.

Dam-break on flat bed (no bathymetry bump) — tests regularization of 1/h
at the discontinuity in both Model and Numerics expressions.

Uses FreeSurfaceFlowSolver with PositiveNonconservativeRusanov.
"""

import os
import sys
import time
import traceback
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core", "zoomy_jax", "zoomy_tests"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC


def make_sme_model(level, ins_dim):
    from zoomy_core.model.models.sme_model import SMEInviscid
    class SME(SMEInviscid):
        ins_dimension = ins_dim
    return SME(level=level)


def make_vam_model(level, ins_dim):
    from zoomy_core.model.models.vam_model import VAMModel
    class VAM(VAMModel):
        ins_dimension = ins_dim
    return VAM(level=level, eigenvalue_mode="numerical")


def run_dam_break(model, label, n_cells=100, t_end=0.1):
    """Run a 1D dam-break and check for NaN / blowup."""
    nv = model.n_variables

    def ic(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = 2.0 if x[0] < 5.0 else 1.0  # h: dam-break
        return Q

    model.initial_conditions = IC.UserFunction(function=ic)
    model.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    mesh = BaseMesh.create_1d(domain=(0, 10), n_inner_cells=n_cells)
    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=0.3),
    )

    t0 = time.time()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.time() - t0

    nc = mesh.n_inner_cells
    h = Q[1, :nc]
    finite = np.isfinite(Q[:, :nc]).all()
    h_pos = (h > -1e-10).all()
    mass = np.sum(h) * 10.0 / nc
    mass_expected = 15.0  # (2*5 + 1*5) / 10 * 10
    mass_ok = abs(mass - mass_expected) / mass_expected < 0.05

    return {
        "label": label,
        "n_vars": nv,
        "finite": finite,
        "h_positive": h_pos,
        "mass_conserved": mass_ok,
        "mass": mass,
        "h_max": float(np.max(h)),
        "h_min": float(np.min(h)),
        "elapsed": elapsed,
    }


def main():
    results = []
    combos = []

    for model_type in ["SME", "VAM"]:
        for ins_dim in [2, 3]:
            dim_label = "2D" if ins_dim == 2 else "3D"
            for level in [0, 1, 2]:
                label = f"{model_type} {dim_label} L{level}"
                combos.append((model_type, ins_dim, level, label))

    print(f"{'Test':<16s} {'Vars':>5s} {'Finite':>7s} {'h>0':>5s} {'Mass':>5s} "
          f"{'h_max':>8s} {'h_min':>8s} {'Time':>6s} {'Status':>8s}")
    print("-" * 75)

    n_pass = 0
    n_fail = 0

    for model_type, ins_dim, level, label in combos:
        try:
            if model_type == "SME":
                model = make_sme_model(level, ins_dim)
            else:
                model = make_vam_model(level, ins_dim)

            r = run_dam_break(model, label)
            ok = r["finite"] and r["h_positive"] and r["mass_conserved"]
            status = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
            else:
                n_fail += 1

            print(f"{label:<16s} {r['n_vars']:5d} {str(r['finite']):>7s} "
                  f"{str(r['h_positive']):>5s} {str(r['mass_conserved']):>5s} "
                  f"{r['h_max']:8.4f} {r['h_min']:8.4f} {r['elapsed']:5.1f}s "
                  f"{'  ' + status:>8s}")
        except Exception as e:
            n_fail += 1
            print(f"{label:<16s}   --- ERROR: {str(e)[:50]}")
            traceback.print_exc()

    print("-" * 75)
    print(f"Results: {n_pass}/{n_pass + n_fail} passed")
    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
