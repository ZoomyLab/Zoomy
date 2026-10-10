"""Lake at rest around an emerging island (Audusse et al. 2004, Fig. 1:
non-vertical shore): SME(0) O2 + MOOD, eta = 0.5, b = 0.8 exp(-x^2/0.02);
the island is dry.  Reconstructed pair (b, eta) vs Audusse's (h, eta).
Well-balanced: max|q| stays at round-off."""
import numpy as np
import common as C
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

import sys
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
for order, bed in ((1, "derived"), (2, "reconstructed"), (2, "derived"), (2, "old solver")):
    sm = SystemModel.from_model(C.sme(0, 2, ("left", "right"), "wall"))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        b = 0.8 * np.exp(-x[0] ** 2 / 0.02)
        q[names.index("b")], q[names.index("h")] = b, max(0.5 - b, 0.0)
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeRusanov, reconstruction=ReconstructionSpec(order=order, limiter="minmod",
                                              bed=bed if bed != "old solver" else "derived"))
    mesh = LSQMesh.create_1d(domain=(-1, 1), n_inner_cells=200)
    if bed == "old solver":
        import gate1_vs_solver_jax as G
        Q, _ = G.old_march(lambda: (mesh, nsm), 1.0, 0.45,
                           **G.eta_kw(lambda: (mesh, nsm)))
        t = 1.0
    else:
        s = SymbolicHyperbolicSolverJax(time_end=1.0, cfl=0.45)
        Q, A = s.setup(mesh, nsm)
        t, Q, A = s.run(Q, A)
    Q = np.asarray(Q)
    h, q = Q[names.index("h")], Q[[i for i, n in enumerate(names) if n not in ("b", "h")]]
    eta = Q[names.index("b")] + h
    wet = h > 1e-6
    print(names, "finite", np.isfinite(Q).all(), "h range", np.nanmin(h), np.nanmax(h), "wet", wet.sum())
    if not wet.any():
        continue
    xc = np.linspace(-1, 1, 201)[:-1] + 0.005
    i = int(np.argmax(np.abs(q).max(axis=0)))
    print(f"O{order} {bed:13s} t={float(t):.2f}  max|q| {np.abs(q).max():.2e} at x={xc[i]:+.3f} (h={h[i]:.2e})  "
          f"max|eta-0.5| (wet) {np.abs(eta[wet] - 0.5).max():.2e}  h_min {h.min():.2e}", flush=True)
