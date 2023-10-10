import numpy as np
import pytest

from sympy import powsimp

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
            str(powsimp(model.sympy_eigenvalues, combine="all", force=True))
            == "Matrix([[(n0*q1 + n1*q2)/q0], [(n0*q0*q1 + n1*q0*q2 + sqrt(ez*g*q0**5)*sqrt(n0**2 + n1**2))/q0**2], [(n0*q0*q1 + n1*q0*q2 - sqrt(ez*g*q0**5)*sqrt(n0**2 + n1**2))/q0**2]])"
        )
    else:
        assert False


@pytest.mark.critical
def test_topography_1d():
    parameters = {"g": 1.0, "ex": 0.0, "ez": 1.0, "nu": 1.0}
    settings = {"topography": True, "friction": ["manning", "newtonian"]}
    momentum_eqns = [1]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(
        dimension=1,
        cls=ShallowWater,
        fields=2,
        aux_fields=["dhdx"],
        parameters=parameters,
        momentum_eqns=momentum_eqns,
        settings=settings,
    )
    print(model.sympy_source)
    print(model.sympy_source_jacobian)

    functions = model.get_runtime_model()
    print(functions.source(Q, Qaux, parameters)[0])
    print(functions.source_jacobian(Q, Qaux, parameters)[0])


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
    test_topography_1d()
