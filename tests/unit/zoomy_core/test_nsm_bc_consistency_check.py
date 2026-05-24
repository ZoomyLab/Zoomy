"""NumericalSystemModel BC-consistency check.

After ``SystemModel.change_state_variables`` remaps state symbols in
``flux`` / ``source`` / ``NCP`` / ``mass_matrix`` but does NOT rebuild
the BC kernel's signature, the BC body silently reads wrong slots of
Q at runtime (Lambda-inflow values land on the wrong scaled state,
mass drifts).

The NumericalSystemModel constructor refuses to wrap such a
SystemModel — raises a ``ValueError`` with a clear message — so the
trap never reaches the numerical solver.

This test pins the validator on the smallest reproducer (SME model,
trivial CoV).
"""
from __future__ import annotations

import pytest
import sympy as sp

from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.models.sme_model import SMEModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel


def _build_sme_with_extrapolation_bc():
    m = SMEModel(
        level=0,
        boundary_conditions=BoundaryConditions([
            Extrapolation(tag="left"),
            Extrapolation(tag="right"),
        ]),
    )
    return SystemModel.from_model(m)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_nsm_accepts_consistent_sm():
    """Baseline: an SM whose BC kernels were built against its current
    state passes the consistency check."""
    sm = _build_sme_with_extrapolation_bc()
    nsm = NumericalSystemModel.from_system_model(sm)
    assert nsm.sm is sm  # no rejection


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_nsm_rejects_sm_with_stale_bc():
    """Simulates the post-CoV trap: ``sm.state`` was updated to use
    new state symbols, but the BC kernel's ``args.variables`` still
    holds the original symbols (the bug ``change_state_variables``
    introduces today by not remapping BC kernels).

    The NSM constructor must catch the mismatch and refuse with a
    descriptive message — the trap never reaches the solver."""
    sm = _build_sme_with_extrapolation_bc()
    # Replace sm.state with fresh symbols (simulating CoV) while
    # leaving sm.boundary_conditions.args.variables pointing at the
    # original symbols.  This is exactly the inconsistent state
    # change_state_variables leaves the SM in today.
    new_state = [sp.Symbol(f"{str(s)}_new", real=True) for s in sm.state]
    sm.state = new_state

    with pytest.raises(ValueError, match=r"is stale"):
        NumericalSystemModel.from_system_model(sm)
