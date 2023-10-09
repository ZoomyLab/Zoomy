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
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0)

    functions = model.get_runtime_model()
    for d in range(dimension):
        assert np.allclose(functions.flux(Q, Qaux, parameters)[:, d, :], Q)
    assert np.allclose(
        functions.flux_jacobian(Q, Qaux, parameters)[0],
        np.stack([np.eye((dimension)) for d in range(dimension)]),
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
            -np.ones(dimension),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[1], parameters
            )[2],
            np.ones(dimension),
        )
    elif dimension == 2:
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[0], parameters
            )[2],
            -np.ones(dimension),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[1], parameters
            )[2],
            np.ones(dimension),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[2], parameters
            )[2],
            np.ones(dimension),
        )
        assert np.allclose(
            functions.eigenvalues(
                Q[:n_inner_elements], Qaux[:n_inner_elements], normals[3], parameters
            )[2],
            -np.ones(dimension),
        )
    else:
        assert False


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
