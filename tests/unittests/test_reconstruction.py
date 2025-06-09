import numpy as np
import pytest
from types import SimpleNamespace

from library.model.model import *
import library.pysolver.reconstruction as reconstruction


@pytest.mark.critical
def test_recon_const_1d():
    dimension = 1
    momentum_eqns = [0]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0, momentum_eqns)

    recon = reconstruction.constant

    Qi, Qj, Qauxi, Qauxj = recon(mesh, Q, Qaux)
    # test inner domain
    assert np.allclose(Q[:9], Qi[:9])
    assert np.allclose(Q[1:10], Qj[:9])
    # test boundary (Wall)
    assert np.allclose(-Q[0], Qj[9])
    assert np.allclose(-Q[9], Qj[10])


@pytest.mark.critical
@pytest.mark.unfinished
def test_recon_const_2d():
    dimension = 2
    momentum_eqns = [0, 1]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0, momentum_eqns)

    recon = reconstruction.constant

    Qi, Qj, Qauxi, Qauxj = recon(mesh, Q, Qaux)


@pytest.mark.critical
@pytest.mark.unfinished
def test_edges_to_elements_1d():
    dimension = 1
    momentum_eqns = [0]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0, momentum_eqns)

    recon = reconstruction.constant

    Qi, Qj, Qauxi, Qauxj = recon(mesh, Q, Qaux)

    map_elements_to_edges_plus, map_elements_to_edges_minus = (
        reconstruction.create_map_elements_to_edges(mesh)
    )

    Qnew = np.zeros_like(Q)
    Qnew2 = np.zeros_like(Q)

    Qnew[map_elements_to_edges_plus] += Qi
    Qnew2[map_elements_to_edges_minus] += Qj
    # # test inner domain
    # assert(np.allclose(Q[:9], Qi[:9]))
    # assert(np.allclose(Q[1:10], Qj[:9]))
    # # test boundary (Wall)
    # assert(np.allclose(-Q[0], Qj[9]))
    # assert(np.allclose(-Q[9], Qj[10]))


@pytest.mark.critical
@pytest.mark.future
def test_edges_to_elements_2d():
    pass


@pytest.mark.critical
@pytest.mark.unfinished
def test_get_edge_geometry_data():
    dimension = 1
    momentum_eqns = [0]
    (
        mesh,
        model,
        Q,
        Qaux,
        parameters,
        num_normals,
        normals,
    ) = create_default_mesh_and_model(dimension, Model, dimension, 0, 0, momentum_eqns)

    normals_ij, edge_length_ij = reconstruction.get_edge_geometry_data(mesh)


if __name__ == "__main__":
    test_recon_const_1d()
    test_recon_const_2d()
    test_edges_to_elements_1d()
    test_edges_to_elements_2d()
    test_get_edge_geometry_data()
