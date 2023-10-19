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
    momentum_eqns = [[0], [0, 1]]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0, momentum_eqns[dimension-1])

    functions = model.get_runtime_model()
    _ = model.create_c_interface()
    c_model = model.load_c_model()

    F = [np.zeros_like(Q) for i in range(model.dimension)]
    dF = [np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)]
    S = np.zeros_like(Q)
    dS = np.zeros((Q.shape[0], Q.shape[1], Q.shape[1]))
    NC = [np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)]
    A = [np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)]

    for i in range(Q.shape[0]):
        for d in range(model.dimension):
            c_model.flux[d](Q[i], Qaux[i], parameters, F[d][i])
            c_model.flux_jacobian[d](Q[i], Qaux[i], parameters, dF[d][i])
            c_model.nonconservative_matrix[d](Q[i], Qaux[i], parameters, NC[d][i])
            c_model.quasilinear_matrix[d](Q[i], Qaux[i], parameters, A[d][i])
        c_model.source(Q[i], Qaux[i], parameters, S[i])
        c_model.source_jacobian(Q[i], Qaux[i], parameters, dS[i])

    for d in range(dimension):
        assert np.allclose(F, Q)
        assert np.allclose([dF[d][0] for d in range(dimension)], [np.eye(dimension) for d in range(dimension)])
        assert np.allclose([NC[d][0] for d in range(dimension)], [np.zeros((dimension, dimension)) for d in range(dimension)])
        assert np.allclose([A[d][0] for d in range(dimension)], [np.eye(dimension) for d in range(dimension)])
        assert np.allclose(S, np.zeros_like(Q))
        assert np.allclose(dS, np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])))
    assert False

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
    # test_model_initialization(1)
    test_model_initialization(2)
