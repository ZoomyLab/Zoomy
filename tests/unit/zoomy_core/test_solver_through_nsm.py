"""HyperbolicSolver.setup_simulation accepts and honours
NumericalSystemModel.

Three contracts:

1. Passing a plain Model auto-promotes to an NSM seeded from the
   solver's current kwargs (so legacy ``solver.reconstruction_order = 2``
   callers keep working).
2. Passing an explicit NSM overrides the solver attributes from the
   NSM's slots.
3. The NSM's ``resolved_lsq_degree`` drives the mesh stencil, even
   when the source Model is *not* a StructuredDerivativeModel (i.e.
   the only signal lives in ``sm.aux_registry``).
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
    s = HyperbolicSolver(
        time_end=0.01,
        settings=Zstruct(output=Zstruct(
            directory="/tmp/_nsm_solver_test",
            filename="nsm_test",
            snapshots=1,
        )),
        **kw,
    )
    return s


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_plain_model_path_preserves_solver_kwargs():
    s = _solver(reconstruction_order=2, limiter="minmod")
    mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    s.setup_simulation(mesh, _sme(), write_output=False)
    assert s.reconstruction_order == 2
    assert s.limiter == "minmod"
    assert isinstance(s.nsm, NumericalSystemModel)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_explicit_nsm_overrides_solver_kwargs():
    s = _solver(reconstruction_order=2, limiter="venkatakrishnan")
    mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    nsm = NumericalSystemModel.from_system_model(
        _sme(),
        reconstruction=ReconstructionSpec(order=1, limiter="minmod"),
        regularization=RegularizationSpec(eigenvalue_eps=5e-7),
    )
    s.setup_simulation(mesh, nsm, write_output=False)
    # NSM's slots override the solver kwargs.
    assert s.reconstruction_order == 1
    assert s.limiter == "minmod"
    assert s.eigenvalue_regularization == 5e-7
    assert s.nsm is nsm


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
    s = _solver(reconstruction_order=1)
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
