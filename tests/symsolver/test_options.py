"""Typed options and the settings hierarchy (`zoomy_core.misc.options`): defaults < settings file
< arguments, string coercion, "did you mean" for values and keys, backend-narrowed members.
"""
import json

import pytest

from zoomy_core.numerics.numerical_system_model import Limiter, ReconstructionSpec
from zoomy_core.symsolver.discretization.time import Imex
from zoomy_core.symsolver.numpy_backend.solver import (
    SymbolicChorinSolverNumpy, SymbolicHyperbolicSolverNumpy)
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_core.symsolver.strategies.chorin import Projection

pytestmark = [pytest.mark.solver, pytest.mark.small]


def test_defaults_and_coercion():
    s = SymbolicChorinSolverNumpy()
    assert s.time_order == 1 and s.imex is False and s.projection is Projection.standard
    assert SymbolicHyperbolicSolverNumpy().time_order is None
    s = SymbolicChorinSolverNumpy(imex="SSP2", projection="increment")
    assert s.imex is Imex.ssp2 and s.projection is Projection.increment and s.imex == "ssp2"
    s.pressure_linear_solver = "pyamg"
    assert s.pressure_linear_solver is LinearSolver.pyamg


@pytest.mark.parametrize("kwargs, exc, hint", [
    (dict(projection="incremnt"), ValueError, "did you mean 'increment'"),
    (dict(imex="ssp"), ValueError, "did you mean 'ssp2'"),
    (dict(pressur_tol=1e-3), TypeError, "did you mean 'pressure_tol'"),
    (dict(pressure_linear_solver="amgx"), ValueError, "not supported here"),
])
def test_typos_are_named(kwargs, exc, hint):
    with pytest.raises(exc, match=hint):
        SymbolicChorinSolverNumpy(**kwargs)


@pytest.mark.parametrize("suffix", [".json", ".yml"])
def test_settings_file_hierarchy(tmp_path, suffix):
    section = {"imex": "lie", "projection": "increment", "cfl": 0.5}
    path = tmp_path / f"run{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps({"solver": section, "model": {}}))
    else:
        path.write_text("solver:\n" + "".join(f"  {k}: {v}\n" for k, v in section.items()))
    s = SymbolicChorinSolverNumpy(settings=str(path), cfl=0.3)
    assert s.imex is Imex.lie and s.projection is Projection.increment
    assert s.cfl == 0.3                              # the argument wins over the file
    assert SymbolicChorinSolverNumpy(settings={"solver": section}).cfl == 0.5


def test_reconstruction_spec_is_typed():
    r = ReconstructionSpec(order=2, limiter="MinMod", limiters={"b": "none"})
    assert r.limiter is Limiter.minmod and r.limiters["b"] is Limiter.none
    with pytest.raises(ValueError, match="did you mean 'venkatakrishnan'"):
        ReconstructionSpec(limiter="venkat")
