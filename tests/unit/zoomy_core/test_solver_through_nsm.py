"""HyperbolicSolver.setup_simulation consumes only NumericalSystemModel.

Contract:
1. A plain Model passed to ``setup_simulation`` is auto-promoted to an
   NSM with default specs (order=1, default limiter, default eps).
2. An NSM passed explicitly is honoured slot-for-slot; the solver
   exposes it as ``self.nsm`` for downstream reads.
3. The NSM's ``resolved_lsq_degree`` drives the mesh stencil — even
   when the source Model is *not* a StructuredDerivativeModel (the
   only signal lives in ``sm.aux_registry``).
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.derivative_workflow import (
    DerivativeSpec, StructuredDerivativeModel,
)
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.models.sme_model import SMEModel
from zoomy_core.numerics import (
    NumericalSystemModel, ReconstructionSpec, RegularizationSpec,
)


def _bcs():
    return BoundaryConditions(
        [Extrapolation(tag="left"), Extrapolation(tag="right")]
    )


def _ic(x):
    return np.array([0.0, 2.0 if x[0] < 5.0 else 1.0, 0.0])


def _sme():
    return SMEModel(
        level=0,
        boundary_conditions=_bcs(),
        initial_conditions=UserFunction(function=_ic),
    )


def _solver(**kw):
    return HyperbolicSolver(
        time_end=0.01,
        settings=Zstruct(output=Zstruct(
            directory="/tmp/_nsm_solver_test",
            filename="nsm_test",
            snapshots=1,
        )),
        **kw,
    )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_plain_model_auto_promotes_to_default_nsm():
    s = _solver()
    mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    s.setup_simulation(mesh, _sme(), write_output=False)
    assert isinstance(s.nsm, NumericalSystemModel)
    # NSM defaults: order=1, default limiter, default eps.
    assert s.nsm.reconstruction.order == 1
    assert s.nsm.reconstruction.limiter == "venkatakrishnan"
    assert s.nsm.regularization.eigenvalue_eps == 1e-8


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_explicit_nsm_slots_drive_solver():
    s = _solver()
    mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    nsm = NumericalSystemModel.from_system_model(
        _sme(),
        reconstruction=ReconstructionSpec(order=2, limiter="minmod"),
        regularization=RegularizationSpec(eigenvalue_eps=5e-7),
    )
    s.setup_simulation(mesh, nsm, write_output=False)
    # No setter on the solver: NSM is the source of truth.
    assert s.nsm is nsm
    assert s.nsm.reconstruction.order == 2
    assert s.nsm.reconstruction.limiter == "minmod"
    assert s.nsm.regularization.eigenvalue_eps == 5e-7


# ── LSQ-degree wiring ────────────────────────────────────────────────


class _DiffusiveAdvection(StructuredDerivativeModel):
    """Tiny 1D model that wants ``d^2 u / dx^2``."""
    dimension = 1
    variables = ["u"]
    parameters = {"nu": (0.01, "positive")}

    def requested_derivatives(self):
        return [DerivativeSpec(field="u", axes=("x", "x"))]

    def flux(self):
        return ZArray(sp.Matrix([[self.Q.u]]))

    def source(self):
        return ZArray.zeros(self.n_variables)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_nsm_lsq_degree_drives_mesh_stencil():
    s = _solver()
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    model = _DiffusiveAdvection(
        boundary_conditions=_bcs(),
        initial_conditions=UserFunction(
            function=lambda x: np.array([np.sin(np.pi * x[0])])),
    )
    s.setup_simulation(mesh, model, write_output=False)
    multi = s._sim_mesh.lsq_monomial_multi_index
    max_total_degree = max(sum(mi) for mi in multi)
    assert max_total_degree >= 2, (
        f"NSM declared 2nd derivative but mesh stencil only carries "
        f"degree-{max_total_degree} monomials: {list(multi)}"
    )
