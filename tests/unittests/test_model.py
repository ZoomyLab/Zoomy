import numpy as np
import pytest

from library.python.misc.misc import *
from library.model.models.base import *
import library.model.boundary_conditions as BC
import library.model.initial_conditions as IC
from library.python.mesh.fvm_mesh import *
from library.model.model import create_default_mesh_and_model


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
    ) = create_default_mesh_and_model(
        dimension, Model, dimension, 0, 0, momentum_eqns[dimension - 1]
    )

    functions = model._get_pde()
    _ = model.create_c_interface()
    c_model = model.load_c_model()

    F = [np.zeros_like(Q) for i in range(model.dimension)]
    dF = [
        np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)
    ]
    S = np.zeros_like(Q)
    dS = np.zeros((Q.shape[0], Q.shape[1], Q.shape[1]))
    NC = [
        np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)
    ]
    A = [np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])) for i in range(model.dimension)]
    Evalues = np.zeros_like(Q)

    for i in range(Q.shape[0]):
        for d in range(model.dimension):
            c_model.flux[d](Q[i], Qaux[i], parameters, F[d][i])
            c_model.flux_jacobian[d](Q[i], Qaux[i], parameters, dF[d][i])
            c_model.nonconservative_matrix[d](Q[i], Qaux[i], parameters, NC[d][i])
            c_model.quasilinear_matrix[d](Q[i], Qaux[i], parameters, A[d][i])
        c_model.source(Q[i], Qaux[i], parameters, S[i])
        c_model.source_jacobian(Q[i], Qaux[i], parameters, dS[i])

    for d in range(dimension):
        assert np.allclose(F[d], Q)
        assert np.allclose(
            [dF[d][0] for d in range(dimension)],
            [np.eye(dimension) for d in range(dimension)],
        )
        assert np.allclose(
            [NC[d][0] for d in range(dimension)],
            [np.zeros((dimension, dimension)) for d in range(dimension)],
        )
        assert np.allclose(
            [A[d][0] for d in range(dimension)],
            [np.eye(dimension) for d in range(dimension)],
        )
    assert np.allclose(S, np.zeros_like(Q))
    assert np.allclose(dS, np.zeros((Q.shape[0], Q.shape[1], Q.shape[1])))

    if dimension == 1:
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[0][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            -np.ones(dimension),
        )
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[1][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            np.ones(dimension),
        )
    elif dimension == 2:
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[0][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            -np.ones(dimension),
        )
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[1][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            np.ones(dimension),
        )
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[2][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            np.ones(dimension),
        )
        for i in range(mesh.n_elements):
            c_model.eigenvalues(Q[i], Qaux[i], parameters, normals[3][i], Evalues[i])
        assert np.allclose(
            Evalues[2],
            -np.ones(dimension),
        )
    else:
        assert False


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
