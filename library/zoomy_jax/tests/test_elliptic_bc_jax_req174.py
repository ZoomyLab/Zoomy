"""REQ-174 (jax half): the elliptic stage must CONSUME ``SM_press``'s declared
P boundary conditions.

The defect core fixed in `6ac880a` and this mirrors: the pressure solve
hardcoded homogeneous Neumann at every boundary face and never read
``SM_press.boundary_conditions``, so a user-declared Dirichlet P pin was
silently replaced by ``zeroGradient`` — **a well-posed solve of the WRONG
problem**. Not a singularity: the operator is full-rank without any pin
(measured, REQ-172 v3), which is why nothing ever crashed and the bug was
invisible.

⚠ THE JAX-SPECIFIC HAZARD core's equivalent test cannot catch — the two
backends' boundary arguments mean DIFFERENT THINGS:

* numpy ``compute_derivatives(..., u_boundary_face=<FACE values>)`` converts
  internally — ``_resolve_u_boundary_face``: ``ghost = 2·u_face − u_cell``.
* jax ``lsq_gradient_per_field(..., u_bf=<GHOST values>)`` uses the value
  DIRECTLY (``u_bf_delta = u_bf_i − u_i``). **No conversion.**

So the ``2·u_face − u_cell`` lift is the jax caller's job. Passing the bare face
value puts a face-valued sample at the ghost position and silently caps the
boundary gradient at 1st order — it does not raise, it just quietly degrades
(see ``tests/unit/zoomy_jax/test_mesh_derivatives_recent.py``, 4.16 vs 0.024).
``test_pin_is_exact_not_merely_close`` is what actually pins this: a bare-face
bug still moves P toward the pin, so only an exact-to-round-off check separates
"honoured" from "approximately nudged".
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

PIN_P0 = 0.037
PIN_P1 = 0.0


def _build(with_pin):
    m = VAM(level=1, dimension=3, closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(m)
    sm.apply(desingularize_hinv())
    names = [str(s) for s in sm.state]
    ih, iq = names.index("h"), names.index("q_x_0")
    bcs = [Wall("left", on="momentum"), Wall("right", on="momentum"),
           Wall("bottom", on="momentum"), Wall("top", on="momentum")]
    if with_pin:
        bcs += [Dirichlet("right", on="P_0", value=PIN_P0),
                Dirichlet("right", on="P_1", value=PIN_P1)]
    sm.attach_boundary_conditions(BoundaryConditions(bcs))

    def ic(xv):
        o = np.zeros(len(names))
        x = float(xv[0])
        o[ih] = 0.10 + 0.02 * np.cos(np.pi * x)
        o[iq] = 0.05 * np.sin(2 * np.pi * x)
        return o

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    split = m.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    mesh = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 0.25), nx=16, ny=4)
    s = ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press, split.SM_corr,
                                pressure_tol=1e-12, pressure_maxit=20000)
    s.Q0 = s.setup_simulation(mesh)
    return s


def _solve(s, dt=1e-3):
    p = s._sub_parameters(s._press_param_src, s.parameters, dt)
    Qaux = s._refresh_aux_for_sm(s._qaux_press0, s.sm_press, s.Q0, p)
    Q_new = s._step_pressure_pure(s.Q0, Qaux, dt, 0.0, p)
    return np.array(Q_new[s._press_state_idx, :], dtype=float)


# `setup_simulation` (symbolic lowering + jit) dominates at ~3 min a build, and
# every test wants one of only TWO solvers.  Module-scoped, which is safe here
# because `_step_pressure_pure` is pure — it returns new arrays and mutates
# nothing on the solver.  Naive per-test builds cost 5 lowerings / ~15 min.
@pytest.fixture(scope="module")
def solver_free():
    return _build(with_pin=False)


@pytest.fixture(scope="module")
def solver_pinned():
    return _build(with_pin=True)


@pytest.mark.jax
def test_no_declared_dirichlet_keeps_the_neumann_path(solver_free):
    """core: "default Neumann unchanged". jax must mean it literally — the
    no-BC case must take the *same* `u_bf=None` call as before REQ-174, not an
    equivalent-looking ghost array, so the old path cannot drift."""
    assert solver_free._press_dir is None
    assert solver_free._press_dir_j is None


@pytest.mark.jax
def test_declared_dirichlet_is_seen_by_the_elliptic_stage(solver_pinned):
    d = solver_pinned._press_dir
    assert d is not None, "declared P Dirichlet never reached the elliptic stage"
    # 16x4 grid, pin on "right" => 4 boundary faces, 4 cells, on BOTH modes.
    assert list(np.asarray(d["face_mask"]).sum(axis=1)) == [4, 4]
    assert list(np.asarray(d["cell_mask"]).sum(axis=1)) == [4, 4]


@pytest.mark.jax
def test_pin_is_exact_not_merely_close(solver_pinned):
    """THE gate. A bare-face/ghost mix-up still drags P toward the pin, so
    "close" proves nothing — only exactness separates a consumed BC from a
    coincidentally-nearby one."""
    P = _solve(solver_pinned)
    cm = np.asarray(solver_pinned._press_dir["cell_mask"])
    cv = np.asarray(solver_pinned._press_dir["cell_value"])
    err = np.abs(P[cm] - cv[cm])
    assert err.max() < 1e-12, f"pinned cells off by {err.max():.3e}"


@pytest.mark.jax
def test_pin_actually_changes_the_answer(solver_free, solver_pinned):
    """Guards the test above from passing vacuously: if P were ~the pin value
    at that boundary anyway, exactness would be meaningless."""
    P_free = _solve(solver_free)
    P_pin = _solve(solver_pinned)
    assert np.abs(P_pin - P_free).max() > 1e-3, \
        "the declared pin does not move the solution — gate is vacuous"
