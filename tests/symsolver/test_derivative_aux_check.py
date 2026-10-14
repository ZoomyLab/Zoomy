"""A derivative aux the backend does not compute raises at setup (it would read 0).

`aux_registry` entries carry the target of the derivative: a state row (every backend), an aux
function (NumPy, JAX), or `unknown` -- a compound such as `h*zeta` no backend computes.  Before
2026-10-14 JAX read such an aux as 0 and the NumPy `derivative_aux` skipped it, silently.
"""
import warnings

import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver.discretization.residual import check_derivative_aux
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _nsm():
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")])))
    sm.initial_conditions = IC.Constant(constants=lambda n: [0.0, 1.0, 0.0][:n])
    return NumericalSystemModel.from_system_model(sm)


def _force(nsm, **entry):
    """Register a derivative aux of a fabricated target on an existing aux row."""
    base = dict(kind="derivative", name="h*zeta_x", row=0, atom=None, aux_symbol=None,
                target_name="h*zeta", multi_index=(1,), target_kind="unknown")
    nsm.aux_registry = list(nsm.aux_registry or []) + [{**base, **entry}]
    return nsm


def test_unknown_target_raises_at_setup():
    """NumPy: the setup names the aux and its target."""
    nsm = _force(_nsm())
    s = SymbolicHyperbolicSolverNumpy(time_end=0.01, dt=0.01)
    with pytest.raises(ValueError, match=r"numpy: the aux 'h\*zeta_x' is the derivative "
                                         r"\(1,\) of 'h\*zeta' \(target_kind 'unknown'\)"):
        s.setup(LSQMesh.create_1d(domain=(0, 1), n_inner_cells=8), nsm)


def test_limited_derivative_raises():
    """No symbolic solver computes a limited derivative."""
    nsm = _force(_nsm(), kind="limited_derivative", target_kind="state", target_name="h")
    with pytest.raises(ValueError, match="limited_derivative"):
        check_derivative_aux([nsm], ("state", "function"), "numpy")


def test_function_target_is_backend_specific():
    """A derivative of an aux function: NumPy / JAX compute it, foam / AMReX do not."""
    nsm = _force(_nsm(), target_kind="function", function_row=0, target_name="hinv")
    check_derivative_aux([nsm], ("state", "function"), "numpy")
    with pytest.raises(ValueError, match=r"foam: .*target_kind 'function'"):
        check_derivative_aux([nsm], ("state",), "foam")


def test_sigma3d_has_no_unknown_target():
    """Sigma3D's in-plane stress term takes d_x z = d_x b + zeta d_x h by the chain rule;
    before, the lifting registered d_x(h zeta) as an aux of the target `h*zeta`."""
    from zoomy_core.model.models.sigma3d import Sigma3D
    from zoomy_core.systemmodel.system_model import allow_unclosed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with allow_unclosed():                # tau~_xx: open by design (m16)
            sm = SystemModel.from_model(Sigma3D())
    targets = {e["name"]: e["target_kind"] for e in sm.aux_registry
               if e["kind"] == "derivative"}
    assert "unknown" not in targets.values()
    assert targets.get("h_x") == "state"
