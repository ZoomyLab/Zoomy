"""1-D reduction of test_radial_dry_dambreak: SME(0) walls, h=2 in |x-.5|<.15.
Per-step MOOD firing / positivity on the reference solver2 path."""
from common import *  # noqa
import numpy as np
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.solver2 import blocks as B

H0, RADIUS = 2.0, 0.15


def radial1d_nsm(order=2, limiter="minmod"):
    model = SME(level=0, dimension=2,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BoundaryConditions(
                    [FromModel(tag=t, definition="wall") for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    names = [str(s) for s in sm.state]
    hi = names.index("h")

    def _ic(x):
        q = np.zeros(len(names))
        q[hi] = H0 if abs(x[0] - 0.5) < RADIUS else 0.0
        return q
    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=limiter))


if __name__ == "__main__":
    for n, tend, lim in [(48, 0.02, "minmod"), (48, 0.05, "minmod"), (96, 0.05, "minmod"),
                         (48, 0.05, "venkatakrishnan")]:
        nsm = radial1d_nsm(2, lim)
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
        s = MarchSolver(time_end=tend, mood_redo=True)
        S = s.setup_march(mesh, nsm, CFL=CFL)
        nc = mesh.n_inner_cells
        p = s._rt_parameters
        dx = 1.0 / n
        m0 = float(np.asarray(S.Q)[1, :nc].sum() * dx)
        fired, log, hmin = 0, [], np.inf
        try:
            while float(S.time) < tend:
                lo, hi_, dt = s.step_head(S, p, tend - S.time, None)
                dt = B.assert_dt_admissible(dt, lo, hi_, s.MeshRT.inradius_f,
                                            time=S.time, iteration=S.iteration)
                Qc, tr = s.stage_loop(S.Q, S.Qaux, p, S.time, dt, False)
                Q1, tr1 = s.stage_loop(S.Q, S.Qaux, p, S.time, dt, True)
                S2, troubled = s.hyperbolic_step(S, dt, p)
                if bool(np.asarray(troubled).any()):
                    fired += 1
                    log.append((int(S.iteration), float(np.asarray(Qc)[1, :nc].min()),
                                float(np.asarray(Q1)[1, :nc].min()),
                                float(np.asarray(S2.Q)[1, :nc].min())))
                S = S2
                hmin = min(hmin, float(np.asarray(S.Q)[1, :nc].min()))
        except FloatingPointError as e:
            print("   FATAL", str(e)[:120])
        h = np.asarray(S.Q)[1, :nc]
        print(n, tend, lim, "steps", int(S.iteration), "mood fired", fired, "hmin(all steps)",
              hmin, "dry cells", int((h == 0).sum()), "mass drift", h.sum() * dx - m0)
        for r in log[:4]:
            print("   it", r[0], "o2 cand hmin", r[1], "o1 redo hmin", r[2], "accepted", r[3])
