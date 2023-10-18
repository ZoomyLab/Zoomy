import numpy as np
import pytest

from library.misc import *
from library.models.base import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *
from library.model import create_default_mesh_and_model

import jax.numpy as jnp
from functools import partial


@partial(jnp.vectorize, signature='(n,m),(n,k),(n,l),(n,m)->()')
def vectorized_function(Q, Qaux, parameters, out):
    out = Q



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
    c_functions = model.create_cython_interface()
    # flux = model.load_cython_model()
    flux = model.load_c_model()

    import numpy.ctypeslib as npct

    # define array prototypes
    array_2d_double = npct.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS')
    array_1d_double = npct.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
    
    # define function prototype
    flux[0].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
    flux[0].restype = None

    A = np.linspace(1,10,10, dtype=float).reshape((5,2))
    B = np.array([[]], dtype=float)
    C = np.linspace(1,10,10, dtype=float).reshape((5,2))
    a = np.array([1., 2.], dtype=float)
    b = np.array([], dtype=float)
    c = np.array([0., 0.], dtype=float)

    flux[0](a, b, b, c)
    
    @partial(jnp.vectorize, signature='(n,m),(n,k),(n,l),(n,m)->()')
    def vectorized_flux(Q, Qaux, parameters, out):
        flux[0](Q, Qaux, parameters, out)
    
    # vectorized_function(A, A, A, C)
    vectorized_flux(A, A, A, C)
    # C = vflux(A, A, A, C)



    print(c)
    print(C)
    assert False
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
    # test_model_initialization(1)
    test_model_initialization(2)
