"""The free-surface ROW MAP — the test that would have caught the amrex
Audusse mis-indexing.

``SystemModel`` publishes two maps that every Audusse-type consumer reads:

* ``depth_scaled_state_indices`` — the rows a hydrostatic reconstruction must
  rescale by ``h*/h`` so the reconstructed pair is a STATE (``q* = h* u``);
* ``discharge_state_indices`` — per horizontal direction, the row whose value
  IS the mass flux there, i.e. the row that carries the bed-slope force.

The defect these replace was a STATE-ORDER guess: "the momentum rows are the
``n_dim`` rows after ``h``".  That is true only for the four-row SWE state
``[b, h, q_x_0, q_y_0]``.  ``SME(level=N, dimension=3)`` is
``[b, h, q_x_0..q_x_N, q_y_0..q_y_N]``, so the guess returns ``q_x_1`` where
``q_y_0`` is meant — which cost the LowerTriangle catchment ``h = -0.170 m``
through the amrex driver and built the numpy HLLC contact wave on the wrong
component.  A 1-D or level-0 test cannot see any of that: the two answers
coincide at four rows.  **Every case below is therefore ≥ 2 horizontal
directions AND ≥ 6 state rows, except the level-0/1-D controls that pin the
coincidence.**
"""
import pytest

from zoomy_core.model.models import SWE, SME, VAM, MLSME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.systemmodel import SystemModel

pytestmark = [pytest.mark.systemmodel, pytest.mark.small, pytest.mark.gate]


def _sm(build):
    return SystemModel.from_model(build())


def _names(sm):
    return [str(s) for s in sm.state]


# name -> (builder, expected depth-scaled names, expected discharge names)
CASES = {
    # -- controls: where the old state-ORDER guess is right -------------------
    "SWE 1-D": (lambda: SWE(dimension=1), ["hu"], ["hu"]),
    "SWE 2-D": (lambda: SWE(dimension=2), ["hu", "hv"], ["hu", "hv"]),
    "SME(0) 2 horiz": (
        lambda: SME(level=0, dimension=3,
                    closures=[Newtonian(), StressFree()]),
        ["q_x_0", "q_y_0"], ["q_x_0", "q_y_0"]),
    # -- the cases the guess gets WRONG --------------------------------------
    "SME(1) 2 horiz": (
        lambda: SME(level=1, dimension=3,
                    closures=[Newtonian(), StressFree()]),
        ["q_x_0", "q_x_1", "q_y_0", "q_y_1"], ["q_x_0", "q_y_0"]),
    "SME(2) 2 horiz": (
        lambda: SME(level=2, dimension=3,
                    closures=[Newtonian(), StressFree()]),
        ["q_x_0", "q_x_1", "q_x_2", "q_y_0", "q_y_1", "q_y_2"],
        ["q_x_0", "q_y_0"]),
    # -- pressure modes are NOT depth-weighted-momentum, but they ARE scaled --
    "VAM(1) 1 horiz": (
        lambda: VAM(level=1, dimension=2,
                    closures=[Newtonian(), StressFree()]),
        ["q_0", "q_1", "r_0", "r_1", "P_0", "P_1"], ["q_0"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_row_map_matches_the_named_state(name):
    build, want_scaled, want_discharge = CASES[name]
    sm = _sm(build)
    names = _names(sm)
    got_scaled = [names[i] for i in sm.depth_scaled_state_indices]
    got_discharge = [names[i] for i in sm.discharge_state_indices]
    assert got_scaled == want_scaled, f"{name}: state {names}"
    assert got_discharge == want_discharge, f"{name}: state {names}"


@pytest.mark.parametrize("name", list(CASES))
def test_depth_and_bed_rows_are_never_depth_scaled(name):
    """``h`` is set to ``h*`` and ``b`` to ``b_face``; neither may be MULTIPLIED
    by ``h*/h``.  Scaling ``b`` would move the bed at a face."""
    sm = _sm(CASES[name][0])
    names = _names(sm)
    scaled = {names[i] for i in sm.depth_scaled_state_indices}
    assert "h" not in scaled and "b" not in scaled


@pytest.mark.parametrize("name", list(CASES))
def test_discharge_rows_are_a_subset_of_the_depth_scaled_rows(name):
    """The direction-``d`` discharge is itself depth-weighted (``q = h u``), so
    the Audusse rescale must cover it.  The shipped amrex loop rescaled only
    the first ``n_dim`` rows after ``h`` and so MISSED ``q_y_0`` above level 0 —
    a face with ``h* = 0`` kept exporting mass."""
    sm = _sm(CASES[name][0])
    scaled = set(sm.depth_scaled_state_indices)
    assert set(sm.discharge_state_indices) <= scaled


def test_state_order_guess_and_the_map_disagree_above_four_rows():
    """Pin the DEFECT itself: "the first ``n_dim`` depth-scaled rows" is the
    x-moment block, not the momentum block, from level 1 up in 2 horizontals."""
    sm = _sm(lambda: SME(level=2, dimension=3,
                         closures=[Newtonian(), StressFree()]))
    names = _names(sm)
    guess = [names[i] for i in list(sm.depth_scaled_state_indices)[:sm.n_dim]]
    truth = [names[i] for i in sm.discharge_state_indices]
    assert guess == ["q_x_0", "q_x_1"]
    assert truth == ["q_x_0", "q_y_0"]
    assert guess != truth


def test_multilayer_has_no_unique_discharge_row_and_says_so():
    """A multilayer mass flux is the SUM of the layer discharges, so no single
    row carries it and no depth-averaged Audusse bed source is defined.  The
    map must answer -1 rather than name an arbitrary layer — that -1 is what
    makes the amrex driver refuse ``solver.well_balanced`` instead of feeding
    the whole bed force to layer 1."""
    sm = _sm(lambda: MLSME(n_layers=2, level=1, dimension=2,
                           closures=[Newtonian(), StressFree()]))
    assert sm.discharge_state_indices == [-1]
