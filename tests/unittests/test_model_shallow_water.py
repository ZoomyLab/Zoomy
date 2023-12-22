import numpy as np
import pytest

from library.misc import *
from library.models.shallow_water import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.fvm_mesh import *
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
            == "Matrix([[(n0*q1 + n1*q2)/q0], [(n0*q0*q1 + n1*q0*q2 + sqrt(ez*g*q0**5)*sqrt(n0**2 + n1**2))/q0**2], [(n0*q0*q1 + n1*q0*q2 - sqrt(ez*g*q0**5)*sqrt(n0**2 + n1**2))/q0**2]])"
        )
    else:
        assert False


@pytest.mark.critical
def test_source_1d():
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

    functions = model.get_runtime_model()
    assert (
        str(model.sympy_source)
        == "Matrix([[0], [-g*nu**2*q1*Abs(q1/q0)**2.33333333333333 + g*q0*(-dhdx*ez + ex) - nu*q1/q0]])"
    )
    assert (
        str(model.sympy_source_jacobian)
        == "Matrix([[0, 0], [2.33333333333333*g*nu**2*q1**2*Abs(q1/q0)**1.33333333333333*sign(q1/q0)/q0**2 + g*(-dhdx*ez + ex) + nu*q1/q0**2, -g*nu**2*Abs(q1/q0)**2.33333333333333 - 2.33333333333333*g*nu**2*q1*Abs(q1/q0)**1.33333333333333*sign(q1/q0)/q0 - nu/q0]])"
    )


@pytest.mark.critical
def test_source_2d():
    parameters = {"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0, "nu": 1.0}
    settings = {"topography": True, "friction": ["manning", "newtonian", "chezy"]}
    momentum_eqns = [1, 2]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(
        dimension=2,
        cls=ShallowWater2d,
        fields=3,
        aux_fields=["dhdx", "dhdy"],
        parameters=parameters,
        momentum_eqns=momentum_eqns,
        settings=settings,
    )
    assert (
        str(model.sympy_source)
        == "Matrix([[0], [-g*nu**2*q1*Abs(q1/q0)**2.33333333333333 + g*q0*(-dhdx*ez + ex) - nu*q1/q0], [-g*nu**2*q2*Abs(q2/q0)**2.33333333333333 + g*q0*(-dhdy*ez + ey) - nu*q2/q0]])"
    )
    assert (
        str(model.sympy_source_jacobian)
        == "Matrix([[0, 0, 0], [2.33333333333333*g*nu**2*q1**2*Abs(q1/q0)**1.33333333333333*sign(q1/q0)/q0**2 + g*(-dhdx*ez + ex) + nu*q1/q0**2, -g*nu**2*Abs(q1/q0)**2.33333333333333 - 2.33333333333333*g*nu**2*q1*Abs(q1/q0)**1.33333333333333*sign(q1/q0)/q0 - nu/q0, 0], [2.33333333333333*g*nu**2*q2**2*Abs(q2/q0)**1.33333333333333*sign(q2/q0)/q0**2 + g*(-dhdy*ez + ey) + nu*q2/q0**2, 0, -g*nu**2*Abs(q2/q0)**2.33333333333333 - 2.33333333333333*g*nu**2*q2*Abs(q2/q0)**1.33333333333333*sign(q2/q0)/q0 - nu/q0]])"
    )

    functions = model.get_runtime_model()


if __name__ == "__main__":
    test_model_initialization(1)
    #test_model_initialization(2)
    # test_source_1d()
    # test_source_2d()
