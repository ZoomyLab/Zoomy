"""Minimal reproduction of the Malpasset clamp-saturated pits: SME(0) at rest in a Gaussian pit
(b = 1 - exp(-r^2/2) [+ tilt (x - 5)], filled to eta0) with the Malpasset thesis numerics
(PositiveNonconservativeHLL = HLL on the hydrostatically reconstructed states, order 2 on
(h, eta = b + h) with the bed derived and every row tied to eta, the front detector
front_tol 0.5, MOOD, CFL 0.4, the point-implicit source).  A lake at rest must stay at rest to
round-off.

    python pit_lake_at_rest.py [VARIANT ...]     (JAX_PLATFORMS=cpu; default: all)

Measured 2026-10-14 (this script, JAX CPU), max |u| at t = 1 s over the wet cells, quads
20 x 20 / triangles 20 x 20 x 2:

| variant | main | with reconstruction.shore_extension |
|---|---|---|
| venk, partially wet (eta0 0.5): quads / triangles | 0.40 / 0.54 m/s | 2.7e-15 / 1.1e-14 |
| venk, tilted pit: quads / triangles | 0.43 / 0.52 m/s | 5.0e-15 / 7.9e-15 |
| venk, fully wet (eta0 2) | 2.6e-15 | 2.6e-15 |
| minmod / Barth-Jespersen, partially wet | 6.1e-15 | 3.8e-15 |
| venk, no front detector | 4.1e-15 (MOOD redoes the inadmissible shore at order 1; without MOOD the march stalls) | 4.1e-15 |
| venk, order 1 | 4.1e-15 | 4.1e-15 |

foam (rt.H, 2026-10-14; no front detector on foam yet, so front_tol None): with the shore form
2.98e-15 (quads) / 1.35e-14 (triangles), 2 and 4 ranks the same; without it also at rest
(6.4e-15 / 4.9e-14): OpenFOAM's Venkatakrishnan has no eps and returns 0 at the shore minimum,
as Barth-Jespersen does.

Mechanism: a dry cell's eta = b enters the shore cells' LSQ stencil; a bank above the lake gives
the free surface a slope there.  Barth-Jespersen / minmod return phi = 0 (the shore cell is a
local minimum of eta), Venkatakrishnan (eps^2 = cell size^2) does not, so eta_f != eta_i at the
faces, the order-2 hydrostatic reconstruction is no longer balanced and the lake accelerates.
Without the front detector the order-2 candidate is inadmissible at the shore and MOOD's
order-1 redo hides the defect.
"""
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import zoomy_core.model.initial_conditions as IC  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel  # noqa: E402
from zoomy_core.model.models import SME  # noqa: E402
from zoomy_core.model.models.closures import Newtonian, StressFree  # noqa: E402
from zoomy_core.numerics import (Limiter, NumericalSystemModel,  # noqa: E402
                                 PositiveNonconservativeHLL, ReconstructionSpec,
                                 SourceTreatment)
from zoomy_core.systemmodel.system_model import SystemModel  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "matrix"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gates"))
from cases import tri_rect  # noqa: E402

TAGS = ("left", "right", "bottom", "top")


def build(eta0=0.5, limiter=Limiter.venkatakrishnan, front_tol=0.5, order=2, tilt=0.0,
          tri=False, nx=20):
    model = SME(level=0, dimension=3, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions(
                    [FromModel(tag=t, definition="wall") for t in TAGS]))
    sm = SystemModel.from_model(model)

    def ic(x):
        b = 1.0 - np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.3) ** 2) / 2.0) + tilt * (x[0] - 5.0)
        return np.array([b, max(eta0 - b, 0.0), 0.0, 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)
    names = [str(s) for s in sm.state]
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=order, limiter=limiter,
                                          front_tol=front_tol if order >= 2 else None,
                                          ties={n: "h" for n in names if n != "h"}),
        source_treatment=SourceTreatment.linearized)
    dom = (0, 10, 0, 10)
    return (tri_rect(dom, nx, nx, nsm) if tri
            else LSQMesh.create_2d(domain=dom, nx=nx, ny=nx)), nsm


VARIANTS = {
    "venk": {}, "venk_tri": dict(tri=True), "tilted": dict(tilt=0.05),
    "tilted_tri": dict(tilt=0.05, tri=True), "wet": dict(eta0=2.0),
    "minmod": dict(limiter=Limiter.minmod), "bj": dict(limiter=Limiter.barth_jespersen),
    "no_front": dict(front_tol=None), "order1": dict(order=1),
}


def run(name, mood=True, t_end=1.0, **kw):
    mesh, nsm = build(**kw)
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.4, mood=mood)
    Q, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q, A)
    Q = np.asarray(Q)
    h = Q[1]
    wet = h > 1e-3
    u = np.hypot(Q[2], Q[3])[wet] / h[wet]
    print(f"{name:12s} t={float(t):.3g} wet={wet.sum():4d} max|u|={u.max():.2e} "
          f"eta range={np.ptp((Q[0] + h)[wet]):.2e} h_min={h.min():.2e}", flush=True)
    return u.max()


if __name__ == "__main__":
    for v in sys.argv[1:] or VARIANTS:
        run(v, **VARIANTS[v])
