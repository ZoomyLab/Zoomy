import numpy as np
import pytest

from library.misc import *
from library.models.shallow_water import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *
from library.model import create_default_mesh_and_model


@pytest.mark.critical
@pytest.mark.parametrize(
    "dimension",
    ([1, 2]),
)
def test_model_initialization(dimension):
    parameters = {"g": 1.0, "ez": 1.0}
    class_list = ["ShallowWater", "ShallowWater2d"]
    momentum_eqns = [[1], [1, 2]]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(
        dimension,
        eval(class_list[dimension - 1]),
        dimension + 1,
        0,
        parameters,
        momentum_eqns[dimension - 1],
    )

    functions = model.get_runtime_model()
    if dimension == 1:
        assert (
            str(model.sympy_eigenvalues)
            == "Matrix([[n0*q1/q0 - n0*sqrt(ez*g*q0**5)/q0**2], [n0*q1/q0 + n0*sqrt(ez*g*q0**5)/q0**2]])"
        )
    elif dimension == 2:
        assert (
            str(model.sympy_eigenvalues)
            == "Matrix([[(n0*q1 + n1*q2)/q0], [(n0*q0*q1 + n1*q0*q2 + sqrt(ez*g*n0**2*q0**5 + ez*g*n1**2*q0**5))/q0**2], [(n0*q0*q1 + n1*q0*q2 - sqrt(ez*g*n0**2*q0**5 + ez*g*n1**2*q0**5))/q0**2]])"
        )
    else:
        assert False


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
