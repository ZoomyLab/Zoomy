import numpy as np
import pytest

from library.misc import *
from library.models.base import *
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
    parameters = [
        {"p0": 2.0},
        {"p0": 2.0, "p1": 1.5},
    ]
    momentum_eqns = [[0], [0, 1]]
    advection_speed = np.array(list(parameters[dimension - 1].values()))
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
        Advection,
        dimension,
        0,
        parameters[dimension - 1],
        momentum_eqns[dimension - 1],
    )

    functions = model.get_runtime_model()
    for d in range(dimension):
        assert np.allclose(
            functions.flux(Q, Qaux, parameters)[:, d, :][0], advection_speed * Q[0]
        )
    assert np.allclose(
        functions.flux_jacobian(Q, Qaux, parameters)[0],
        np.stack([np.diag((advection_speed)) for d in range(dimension)]),
    )
    assert np.allclose(functions.source(Q, Qaux, parameters)[0], np.zeros(dimension))
    assert np.allclose(
        functions.source_jacobian(Q, Qaux, parameters)[0],
        np.zeros((dimension, dimension)),
    )
    n_inner_elements = mesh.n_elements
    if dimension == 1:
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[0], parameters
            )[2],
            -np.diag(advection_speed),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[1], parameters
            )[2],
            np.array(advection_speed),
        )
    elif dimension == 2:
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[0], parameters
            )[2],
            -np.array(advection_speed),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[1], parameters
            )[2],
            np.array(advection_speed),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[2], parameters
            )[2],
            np.array(advection_speed),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[3], parameters
            )[2],
            -np.array(advection_speed),
        )
    else:
        assert False


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
