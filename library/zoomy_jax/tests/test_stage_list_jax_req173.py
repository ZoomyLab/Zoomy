"""REQ-173 (jax half): ``stages=[...]`` construction ≡ the positional triple.

core landed the vocabulary in `zoomy_core@5e32610` (`SplitForPressureResult.stages`,
`Stage(label, kind, sm)`, binding BY KIND). This is the jax mirror's gate: the
stage-list constructor must select the SAME three sub-models as the legacy
positional call — same objects, same roles — so the refactor is a renaming of
the dispatch and not a change of numerics.

⚠ What this test canNOT catch, and core's equivalent can't either: a silently
unconverged `elliptic` stage. Both construction paths call the same pressure
primitive, so both reproduce each other bit-for-bit whether or not that solve
converged. Bit-for-bit proves the refactor is clean; it says nothing about
convergence. That is why the `elliptic` contract carries a residual (jax
surfaces it via `jax.debug.print` from inside jit — see `_step_pressure_pure`).
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
from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall, Dirichlet
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax


def _split():
    m = VAM(level=1, dimension=3, closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(m)
    names = [str(s) for s in sm.state]
    ih = names.index("h")
    sm.attach_boundary_conditions(BoundaryConditions([
        Wall("left", on="momentum"), Wall("right", on="momentum"),
        Wall("bottom", on="momentum"), Wall("top", on="momentum"),
        Dirichlet("top", on="P_0", value=0.0),
        Dirichlet("top", on="P_1", value=0.0),
    ]))

    def ic(xv):
        o = np.zeros(len(names))
        o[ih] = 0.12 if float(xv[0]) < 0.5 else 0.061
        return o

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return m.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)


@pytest.mark.jax
def test_stages_construction_selects_the_same_submodels():
    split = _split()
    positional = ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press, split.SM_corr)
    staged = ChorinSplitVAMSolverJax(stages=split.stages)
    # same OBJECTS, not merely equal ones — the stage list is a view of the
    # same three sub-models.
    assert staged.sm_pred is positional.sm_pred is split.SM_pred
    assert staged.sm_press is positional.sm_press is split.SM_press
    assert staged.sm_corr is positional.sm_corr is split.SM_corr


@pytest.mark.jax
def test_stages_bind_by_kind_not_position():
    """core binds by `kind`; a shuffled list must still bind correctly.
    If this ever passes by position, a reordered list would silently swap the
    predictor and the corrector."""
    split = _split()
    shuffled = [split.stages[2], split.stages[0], split.stages[1]]
    s = ChorinSplitVAMSolverJax(stages=shuffled)
    assert s.sm_pred is split.SM_pred
    assert s.sm_press is split.SM_press
    assert s.sm_corr is split.SM_corr


@pytest.mark.jax
def test_stage_list_rejects_malformed_input():
    split = _split()
    with pytest.raises(TypeError):                       # both forms at once
        ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press, split.SM_corr,
                                stages=split.stages)
    with pytest.raises(TypeError):                       # neither form
        ChorinSplitVAMSolverJax()
    with pytest.raises(ValueError):                      # duplicate kind
        ChorinSplitVAMSolverJax(stages=[split.stages[0], split.stages[0],
                                        split.stages[1], split.stages[2]])
    with pytest.raises(ValueError):                      # missing a kind
        ChorinSplitVAMSolverJax(stages=split.stages[:2])
    with pytest.raises(ValueError):                      # unknown kind
        bad = [("predictor", "parabolic", split.SM_pred)] + list(split.stages[1:])
        ChorinSplitVAMSolverJax(stages=bad)


@pytest.mark.jax
def test_stages_accepts_bare_triples_not_only_namedtuples():
    """core's `_bind_stages` documents bare `(label, kind, sm)` tuples as
    acceptable; a backend that only took the NamedTuple would silently diverge
    from the documented vocabulary."""
    split = _split()
    bare = [(s.label, s.kind, s.sm) for s in split.stages]
    s = ChorinSplitVAMSolverJax(stages=bare)
    assert s.sm_press is split.SM_press
