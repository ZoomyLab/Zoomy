"""A structurally-zero diffusion tensor must not be lowered into the runtime.

``JaxRuntime`` used to gate the diffusion slots on SHAPE alone
(``_shape(_to_array_of_exprs(dm))``).  SWE carries a correctly shaped
``(4, 4, 2, 2)`` ``diffusion_matrix`` of literal ZEROS, so it passed, and
``solver_jax`` then built a ``DenseDiffusionOperatorJAX`` — whose constructor
walks every mesh face in Python doing a per-face device ``float()``.  Measured
on the LowerTriangle catchment (584,941 faces): **2287 s of a 2539 s run**
assembling an identically-zero operator.

The gate now also asks core's ``diffusion_matrix_is_zero``.  Skipping is safe
only because that boolean collapses UNDECIDABLE to False (see
``zoomy_core.numerics.numerical_system_model.operator_is_zero``).
"""
import pytest
import sympy as sp

import jax
jax.config.update("jax_enable_x64", True)

from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel, ReconstructionSpec)
from zoomy_jax.transformation.jax_runtime import (
    JaxRuntime, _slot_is_structurally_zero)

pytestmark = [pytest.mark.jax, pytest.mark.small]


def _nsm(model):
    return NumericalSystemModel.from_model(
        model, riemann=Rusanov, reconstruction=ReconstructionSpec(order=1))


def test_swe_zero_diffusion_slots_are_not_lowered():
    nsm = _nsm(SWE(dimension=2, parameters={"g": 9.81}))
    # The premise: the slot is PRESENT and shaped, so shape cannot decide.
    assert tuple(nsm.diffusion_matrix.shape) == (4, 4, 2, 2)
    rt = JaxRuntime(nsm)
    assert rt.diffusion_matrix is None
    assert rt.diffusion_matrix_explicit is None


def test_live_diffusion_is_still_lowered():
    """The gate must not swallow a real tensor: Malpasset's eddy-viscosity
    ``diffusion_matrix_explicit`` is live and has to reach the runtime."""
    nsm = _nsm(MalpassetSWE())
    assert nsm.diffusion_matrix_explicit_is_zero is False
    rt = JaxRuntime(nsm)
    assert rt.diffusion_matrix_explicit is not None


def test_fallback_is_todays_behaviour():
    """An SM that cannot answer keeps the shape-only path — the safe
    direction, since a spurious operator only costs time."""
    class _Mute:
        pass
    assert _slot_is_structurally_zero(_Mute(), "diffusion_matrix_is_zero") \
        is False
    # And an UNDECIDABLE entry must never license a skip.
    nsm = _nsm(SWE(dimension=2, parameters={"g": 9.81}))
    dm = nsm.diffusion_matrix
    entries = list(dm._array)
    entries[0] = sp.Symbol("nu_unknown")          # is_zero is None
    dm._array = entries
    assert _slot_is_structurally_zero(nsm, "diffusion_matrix_is_zero") is False
    assert JaxRuntime(nsm).diffusion_matrix is not None
