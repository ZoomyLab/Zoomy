"""REQ-151 DEFECT D (jax half): the Chorin solver must EVALUATE a plain-Symbol
aux rule, not only derivative-kind aux.

`hinv = KP(h)` is a plain `Symbol`, and every velocity scales by it
(`u = hu*hinv`).  `_refresh_aux_for_sm` historically walked only
`kinds=("derivative","limited_derivative")`, so `hinv` was never evaluated: it
sat at its init value 0 and silently zeroed every velocity — a WRONG-ANSWER
bug, not a crash.  This test pins the fix: in the predictor aux pool of the setup state,
`hinv` must equal `1/h` (to KP desingularisation), not 0.

⚠ Requires the CORE half too: `_pad_to_square`
(`zoomy_core/fvm/solver_chorin_vam_numpy.py`) must carry `update_aux_variables`
onto the padded predictor SystemModel, else the predictor runtime never
lowers the rule and there is nothing for this solver to apply.  While that is
outstanding the test SKIPS with a pointer rather than reporting a false pass —
it flips to a real assertion the moment core lands it.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
import sympy as sp
from loguru import logger
logger.remove()

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel import SystemModel
from zoomy_core.systemmodel.operations import desingularize_hinv
from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall, Dirichlet
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax


def _build():
    """2-D VAM(1,3) that ACTUALLY emits `hinv` (a plain VAM does not)."""
    m = VAM(level=1, dimension=3, closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(m)
    sm.apply(desingularize_hinv())                 # registers the `hinv` aux
    names = [str(s) for s in sm.state]
    aux = [str(a) for a in sm.aux_state]
    assert "hinv" in aux, "fixture no longer emits hinv — test would be vacuous"
    ih, ihinv = names.index("h"), aux.index("hinv")
    # Wall(tag) claims EVERY field and would conflict with the P pin -> on=momentum
    sm.attach_boundary_conditions(BoundaryConditions([
        Wall("left", on="momentum"), Wall("right", on="momentum"),
        Wall("bottom", on="momentum"), Wall("top", on="momentum"),
        # ⚠ These pins are NOT a rank fix — that comment was my misconception.
        # REQ-172 v3 measured the elliptic operator FULL-RANK with no pin at
        # all (sigma_min ~3e-5 and RISING under refinement; LU solves to 1e-14).
        # They're here only to fix P's otherwise-arbitrary additive constant.
        # Pre-REQ-174 they were silently ignored anyway (hardcoded Neumann);
        # they are now genuinely consumed — see test_elliptic_bc_jax_req174.py.
        Dirichlet("top", on="P_0", value=0.0),
        Dirichlet("top", on="P_1", value=0.0),
    ]))

    def ic(xv):
        o = np.zeros(len(names))
        o[ih] = 0.12 if float(xv[0]) < 0.5 else 0.061
        return o

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    split = m.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    mesh = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 0.25), nx=16, ny=4)
    solver = ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press, split.SM_corr,
                                     pressure_tol=1e-9, pressure_maxit=60)
    Q0 = solver.setup_simulation(mesh)
    return solver, Q0, ih, ihinv


@pytest.mark.jax
def test_chorin_recomputes_plain_symbol_aux_hinv():
    solver, Q0, ih, ihinv = _build()

    rt = solver._runtime_for_sm(solver.sm_pred)
    if getattr(rt, "update_aux_variables", None) is None:
        pytest.skip(
            "core half outstanding (REQ-151 DEFECT D): _pad_to_square drops "
            "update_aux_variables, so the predictor runtime never lowers the "
            "rule. jax applies it as soon as core carries it across.")

    nc = solver.nc
    qaux = np.asarray(solver._qaux_from_q(Q0, solver.parameters))
    h = np.asarray(Q0)[ih, :nc]
    hinv = qaux[ihinv, :nc]

    assert hinv.max() > 0, (
        "DEFECT D: hinv still 0 — every velocity u=hu*hinv silently vanishes")
    # KP-desingularised inverse depth == 1/h for these (well-wet) depths
    assert np.allclose(hinv, 1.0 / h, rtol=1e-6), (
        f"hinv != 1/h: got [{hinv.min():.6g},{hinv.max():.6g}], "
        f"expected [{(1.0/h).min():.6g},{(1.0/h).max():.6g}]")
