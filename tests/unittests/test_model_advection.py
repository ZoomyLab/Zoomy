import numpy as np
import pytest

from library.misc import *
from library.models.advection import Advection
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *
from library.model import create_default_mesh_and_model


@pytest.mark.critical
def test_model_initialization_1d():
    dimension = 1
    parameters = {"p0": 2.0}
    momentum_eqns = [0]
    advection_speed = [parameters["p0"]]
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
        parameters,
        momentum_eqns,
    )

    # TODO make this one function call and replace others as well
    _ = model.get_runtime_model()
    _ = model.create_c_interface()
    c_model = model.load_c_model()

    num_elements = Q.shape[0]
    F = [np.zeros((model.n_fields)) for d in range(dimension)]
    JacF = [np.zeros((model.n_fields, model.n_fields)) for d in range(dimension)]
    S = np.zeros((model.n_fields))
    JacS = np.zeros((model.n_fields, model.n_fields))
    for i in range(num_elements):
        for d in range(dimension):
            c_model.flux[d](Q[i], Qaux[i], parameters, F[d])
            c_model.flux_jacobian[d](Q[i], Qaux[i], parameters, JacF[d])
        c_model.source(Q[i], Qaux[i], parameters, S)
        c_model.source_jacobian(Q[i], Qaux[i], parameters, JacS)
        assert np.allclose(F[0], advection_speed * Q[i])
        assert np.allclose(JacF[0], np.diag((advection_speed)))
        assert np.allclose(S, np.zeros_like(S))
        assert np.allclose(JacS, np.zeros_like(JacS))

    n_inner_elements = mesh.n_elements
    evalues = np.zeros((model.n_fields))
    for i_elem, i_edge in mesh.inner_edge_list:
        c_model.eigenvalues(Q[i_elem], Qaux[i_elem], parameters, mesh.element_edge_normal[i_elem, i_edge], evalues)
        assert np.allclose(
            evalues,
            np.diag(advection_speed),
        )
    for index, i_elem in enumerate(mesh.boundary_edge_elements):
        c_model.eigenvalues(Q[i_elem], Qaux[i_elem], parameters, mesh.boundary_edge_normal[index], evalues)
        assert np.allclose(
            evalues,
            np.diag(advection_speed) * mesh.boundary_edge_normal[index],
        )


@pytest.mark.critical
def test_model_initialization_2d():
    dimension = 2
    parameters = {"p0": 2.0, "p1": 1.5}
    momentum_eqns = [0, 1]
    advection_speed = np.array(list(parameters.values()))
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
        parameters,
        momentum_eqns,
    )

    # TODO make this one function call and replace others as well
    _ = model.get_runtime_model()
    _ = model.create_c_interface()
    c_model = model.load_c_model()

    num_elements = Q.shape[0]
    F = [np.zeros((model.n_fields)) for d in range(dimension)]
    JacF = [np.zeros((model.n_fields, model.n_fields)) for d in range(dimension)]
    S = np.zeros((model.n_fields))
    JacS = np.zeros((model.n_fields, model.n_fields))
    for i in range(num_elements):
        for d in range(dimension):
            c_model.flux[d](Q[i], Qaux[i], parameters, F[d])
            c_model.flux_jacobian[d](Q[i], Qaux[i], parameters, JacF[d])
        c_model.source(Q[i], Qaux[i], parameters, S)
        c_model.source_jacobian(Q[i], Qaux[i], parameters, JacS)
        assert np.allclose(F[0], np.dot(np.diag([advection_speed[0], 0]), Q[i]))
        assert np.allclose(F[1], np.dot(np.diag([0, advection_speed[1]]), Q[i]))
        assert np.allclose(JacF[0], np.diag(([advection_speed[0], 0])))
        assert np.allclose(JacF[1], np.diag(([0, advection_speed[1]])))
        assert np.allclose(S, np.zeros_like(S))
        assert np.allclose(JacS, np.zeros_like(JacS))
    n_inner_elements = mesh.n_elements
    evalues = np.zeros((model.n_fields))
    for i_elem, i_edge in mesh.inner_edge_list:
        c_model.eigenvalues(Q[i_elem], Qaux[i_elem], parameters, mesh.element_edge_normal[i_elem, i_edge], evalues)
        assert(np.allclose(evalues, np.dot(np.diag(advection_speed), mesh.element_edge_normal[i_elem, i_edge])))
    for index, i_elem in enumerate(mesh.boundary_edge_elements):
        c_model.eigenvalues(Q[i_elem], Qaux[i_elem], parameters, mesh.boundary_edge_normal[index], evalues)
        assert np.allclose(
            evalues,
            np.dot(np.diag(advection_speed), mesh.boundary_edge_normal[index]),
        )


if __name__ == "__main__":
    test_model_initialization_1d()
    test_model_initialization_2d()
