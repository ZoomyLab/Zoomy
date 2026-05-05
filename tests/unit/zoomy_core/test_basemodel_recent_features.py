import pytest

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model


class _DummyModel(Model):
    dimension = 1
    variables = ["h", "hu"]
    aux_variables = {"hinv": (0.0, "positive")}
    parameters = {"g": (9.81, "positive"), "eps": 1e-9}

    def flux(self):
        return ZArray.zeros(self.n_variables, self.dimension)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.numpy
def test_model_parameter_parsing_and_defaults():
    model = _DummyModel()
    assert model.n_variables == 2
    assert model.n_aux_variables == 1
    assert model.n_parameters == 2
    assert model.parameters.g.is_positive is True
    assert model.parameter_defaults_map["g"] == 9.81
    assert model.parameter_defaults_map["eps"] == 1e-9
    assert len(model.parameters) == 2


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.numpy
def test_model_aux_boundary_conditions_auto_completed():
    bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
    aux_bcs = BC.BoundaryConditions([BC.Extrapolation(tag="left")])
    model = _DummyModel(boundary_conditions=bcs, aux_boundary_conditions=aux_bcs)
    aux_tags = {bc.tag for bc in model.aux_boundary_conditions.boundary_conditions_list}
    assert aux_tags == {"left", "right"}

