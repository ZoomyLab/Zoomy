"""Shared scratch setup: the 1-D SWE dam breaks on solver2 / EmittedMarchSolver.

Copied from zoomy_jax/tests/runtime/test_solver2_swashes.py (NOT edited)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
TESTS = os.path.expanduser("~/git/Zoomy/library/zoomy_jax/tests")
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import zoomy_jax  # noqa: E402,F401  (x64 on)

assert jax.config.jax_enable_x64

from zoomy_core.mesh import LSQMesh  # noqa: E402

DOMAIN, N_CELLS, T_END = (0.0, 10.0), 100, 1.0
CFL = 0.9


def nsm_for(ic_name, order):
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    from cases import ritter_ic, stoker_ic
    from models import swe
    ic = {"stoker": stoker_ic, "ritter": ritter_ic}[ic_name]
    sm = SystemModel.from_model(swe(2, "swashes"))
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    assert nsm.update_variables is None
    return nsm


def make_solver(cls, ic_name, order, *, n_cells=N_CELLS, t_end=T_END,
                recon=None):
    nsm = nsm_for(ic_name, order)
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=n_cells)
    if recon is None:
        recon = "eta" if (ic_name == "ritter" and order >= 2) else "conservative"
    kw = ({} if recon == "conservative" else
          dict(reconstruction_variables=recon, free_surface_h_index=1,
               free_surface_b_index=0))
    solver = cls(time_end=t_end, mood_redo=(order >= 2), **kw)
    S = solver.setup_march(mesh, nsm, CFL=CFL)
    return solver, S, nsm, mesh
