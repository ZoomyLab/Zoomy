"""Typed options (`zoomy_core.misc.options`): a plain string where an `Option` member belongs
raises `TypeError` naming the member; "did you mean" for misspelt values and argument names;
backend-narrowed members.
"""
import pytest

from zoomy_core.model.models.closures import InterfaceQuantity, MeanInterface
from zoomy_core.numerics.numerical_system_model import (
    BedReconstruction, Limiter, ReconstructedVariables, ReconstructionSpec)
from zoomy_core.symsolver.discretization.time import Imex
from zoomy_core.symsolver.numpy_backend.solver import (
    SymbolicChorinSolverNumpy, SymbolicHyperbolicSolverNumpy)
from zoomy_core.symsolver.settings import CflWaveSpeed, LinearSolver, PressureSolve
from zoomy_core.symsolver.strategies.chorin import Projection

pytestmark = [pytest.mark.solver, pytest.mark.small]


def test_defaults_and_members():
    s = SymbolicChorinSolverNumpy()
    assert s.time_order == 1 and s.imex is False and s.projection is Projection.standard
    assert SymbolicHyperbolicSolverNumpy().time_order is None
    s = SymbolicChorinSolverNumpy(imex=Imex.ssp2, projection=Projection.increment,
                                  pressure_solve=PressureSolve.newton,
                                  cfl_wave_speed=CflWaveSpeed.cells)
    assert s.imex is Imex.ssp2 and s.projection is Projection.increment and s.imex == "ssp2"
    s.pressure_linear_solver = LinearSolver.pyamg
    assert s.pressure_linear_solver is LinearSolver.pyamg
    assert SymbolicChorinSolverNumpy(imex=False).imex is False       # the declared extra value


@pytest.mark.parametrize("kwargs, hint", [
    (dict(projection="increment"), "projection='increment': use Projection.increment"),
    (dict(imex="SSP2"), "imex='SSP2': use Imex.ssp2"),
    (dict(cfl_wave_speed="cells"), "use CflWaveSpeed.cells"),
    (dict(projection="incremnt"), r"did you mean Projection.increment\? \(one of: "
                                  r"Projection.standard, Projection.increment\)"),
    (dict(imex="ssp"), r"did you mean Imex.ssp2\? \(one of: False, Imex.lie, Imex.ssp2\)"),
    (dict(pressure_linear_solver="amgx"), "LinearSolver.amgx is not supported here"),
])
def test_string_value_is_a_type_error(kwargs, hint):
    with pytest.raises(TypeError, match=hint):
        SymbolicChorinSolverNumpy(**kwargs)


def test_string_assignment_is_a_type_error():
    s = SymbolicChorinSolverNumpy()
    with pytest.raises(TypeError, match="use LinearSolver.pyamg"):
        s.pressure_linear_solver = "pyamg"
    assert s.pressure_linear_solver is LinearSolver.auto


def test_unsupported_member_and_misspelt_name():
    with pytest.raises(ValueError, match="not supported here"):
        SymbolicChorinSolverNumpy(pressure_linear_solver=LinearSolver.amgx)
    with pytest.raises(TypeError, match="did you mean 'pressure_tol'"):
        SymbolicChorinSolverNumpy(pressur_tol=1e-3)
    with pytest.raises(TypeError, match="no setting 'settings'"):
        SymbolicChorinSolverNumpy(settings={"solver": {"cfl": 0.5}})


def test_reconstruction_spec_is_typed():
    r = ReconstructionSpec(order=2, limiter=Limiter.minmod, limiters={"b": Limiter.none},
                           variables=ReconstructedVariables.conservative,
                           bed=BedReconstruction.reconstructed)
    assert r.limiter is Limiter.minmod and r.limiters["b"] is Limiter.none
    for kwargs, hint in [(dict(limiter="minmod"), "limiter='minmod': use Limiter.minmod"),
                         (dict(limiters={"b": "none"}), r"limiters\['b'\]='none': use Limiter.none"),
                         (dict(variables="conservative"), "use ReconstructedVariables.conservative"),
                         (dict(bed="derivd"), "did you mean BedReconstruction.derived"),
                         (dict(limiter="venkat"), "did you mean Limiter.venkatakrishnan")]:
        with pytest.raises(TypeError, match=hint):
            ReconstructionSpec(**kwargs)


def test_closure_quantity_is_typed():
    assert MeanInterface(quantity=InterfaceQuantity.stress).quantity is InterfaceQuantity.stress
    with pytest.raises(TypeError, match="use InterfaceQuantity.stress"):
        MeanInterface(quantity="stress")


def test_model_names_are_checked():
    from zoomy_core.model.models import SME
    assert SME(level=1, dimension=2).level == 1
    with pytest.raises(TypeError, match="did you mean 'level'"):
        SME(levl=1)
    with pytest.raises(TypeError, match="no setting 'settings'"):
        SME(dimension=2, settings={"model": {"level": 1}})
