#!/usr/bin/env python
"""Path-conservative Roe vs Rusanov on the Stoker wet-wet dam-break.

Validates :class:`PathConservativeRoe` (matrix |A|=R|Λ|L dissipation via a
runtime NUMERICAL eigendecomposition of the full quasilinear matrix — no
analytical eigenvectors) against :class:`PositiveNonconservativeRusanov`
(scalar s_max dissipation).  Both run at order-1 (constant reconstruction)
so the ONLY difference is the Riemann dissipation.

Asserts: Roe stays finite, conserves mass exactly, and has L1(h−Stoker) no
worse than Rusanov at every resolution (it is sharper at the shock).  Run via
the conda ``zoomy`` interpreter.  Produces roe_vs_rusanov.png next to itself.
"""
import sys
from pathlib import Path

import numpy as np
from sympy import Matrix

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "library/zoomy_foam/tools"))
from compare_stoker import stoker  # noqa: E402

import zoomy_core.fvm.timestepping as ts  # noqa: E402
from zoomy_core.mesh import BaseMesh  # noqa: E402
import zoomy_core.model.boundary_conditions as BC  # noqa: E402
import zoomy_core.model.initial_conditions as IC  # noqa: E402
from zoomy_core.misc.misc import ZArray  # noqa: E402
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel  # noqa: E402
from zoomy_core.fvm.solver_numpy import (  # noqa: E402
    FreeSurfaceFlowSolver, RoeFreeSurfaceFlowSolver,
)

X0, X1, XMID, G = 0.0, 10.0, 5.0, 9.81
H_L, H_R, T_END = 2.0, 1.0, 0.5
MASS_EXACT = H_L * (XMID - X0) + H_R * (X1 - XMID)


class SWE1D(StructuredDerivativeModel):
    """1D SWE [b, h, hu], flat bed; split flux + hydrostatic pressure."""
    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu
        F[2, 0] = hu * hu / h
        return ZArray(F)

    def hydrostatic_pressure(self):
        h, g = self.Q.h, self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = 0.5 * g * h * h
        return ZArray(P)

    def nonconservative_matrix(self):
        h, g = self.Q.h, self.params.g
        B = ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        B[2, 0, 0] = g * h
        return B


def _model():
    def ic(x):
        Q = np.zeros(3)
        Q[1] = H_L if x[0] < XMID else H_R
        return Q
    return SWE1D(
        boundary_conditions=BC.BoundaryConditions(
            [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
        initial_conditions=IC.UserFunction(function=ic),
    )


def run(solver_cls, n):
    mesh = BaseMesh.create_1d(domain=(X0, X1), n_inner_cells=n)
    solver = solver_cls(time_end=T_END, compute_dt=ts.adaptive(CFL=0.4))
    Q, _ = solver.solve(mesh, _model(), write_output=False)
    xc = np.asarray(mesh.cell_centers_computed())[0, :n]
    h = Q[1, :n]
    h_an, _ = stoker(xc, T_END, H_L, H_R, XMID, G)
    return dict(
        xc=xc, h=h, h_an=h_an,
        mass=float(np.sum(h) * (X1 - X0) / n),
        l1=float(np.mean(np.abs(h - h_an))),
        sharp=float(np.max(np.abs(np.diff(h)))),
        finite=bool(np.isfinite(Q[:, :n]).all()),
    )


def main():
    print(f"{'N':>5} {'scheme':>10} {'mass':>10} {'L1(h-Stoker)':>14} {'max|dh|':>10}")
    profiles = {}
    ok = True
    for n in (100, 200, 400):
        ru = run(FreeSurfaceFlowSolver, n)
        ro = run(RoeFreeSurfaceFlowSolver, n)
        if n == 200:
            profiles = {"rusanov": ru, "roe": ro}
        for name, r in (("Rusanov", ru), ("Roe", ro)):
            print(f"{n:>5} {name:>10} {r['mass']:>10.5f} "
                  f"{r['l1']:>14.4e} {r['sharp']:>10.4f}")
        # assertions: Roe finite, conservative, and no worse than Rusanov
        ok &= ro["finite"]
        ok &= abs(ro["mass"] - MASS_EXACT) / MASS_EXACT < 1e-4
        ok &= ro["l1"] <= ru["l1"] * 1.01
        print(f"      -> Roe L1/Rusanov L1 = {ro['l1']/ru['l1']:.3f}, "
              f"sharper x{ro['sharp']/ru['sharp']:.3f}")

    # plot (full profile + shock zoom) at N=200
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ru, ro = profiles["rusanov"], profiles["roe"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for a in ax:
            a.plot(ru["xc"], ru["h_an"], "k-", lw=1.2, label="Stoker (exact)")
            a.plot(ru["xc"], ru["h"], "C0o-", ms=3, lw=1, label="Rusanov")
            a.plot(ro["xc"], ro["h"], "C3s-", ms=3, lw=1, label="Roe |A|")
            a.set_xlabel("x"); a.set_ylabel("h")
        ax[0].set_title("Stoker dam-break, N=200, order 1"); ax[0].legend(fontsize=9)
        # zoom the shock
        ish = int(np.argmax(np.abs(np.diff(ro["h_an"]))))
        xs = ro["xc"][ish]
        ax[1].set_xlim(xs - 0.8, xs + 0.8)
        lo = min(H_R, ro["h_an"].min()); hi = ro["h_an"][ish] + 0.1
        ax[1].set_ylim(lo - 0.05, hi)
        ax[1].set_title("shock zoom — Roe is sharper"); ax[1].legend(fontsize=9)
        out = Path(__file__).resolve().parent / "roe_vs_rusanov.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\nfigure -> {out}")
    except Exception as e:  # plotting is optional
        print(f"[plot skipped] {type(e).__name__}: {e}")

    print(f"\n{'PASS' if ok else 'FAIL'}: Roe finite, mass-exact, L1 <= Rusanov")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
