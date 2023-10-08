import numpy as np

from library.misc import *
from library.models.base import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *


def test_model_initialization():
    ic = IC.Constant()

    bc_tags = ["left", "right"]
    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag=tag, momentum_eqns=[1, 2]) for tag in bc_tags]
    )
    mesh = Mesh.create_1d((-1, 1), 10)
    model = Model(
        dimension=1,
        n_fields=3,
        n_aux_fields=0,
        n_parameters=0,
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 3 * n_all_elements, 3 * n_all_elements).reshape(
        n_all_elements, 3
    )
    Qaux = np.zeros((Q.shape[0], 0))
    parameters = np.array([], dtype=float)
    model.boundary_conditions.apply(Q)
    model.initial_conditions.apply(Q, mesh.element_centers)

    functions = model.get_runtime_model()
    assert np.allclose(functions.flux(Q, Qaux, parameters), Q)
    assert np.allclose(functions.flux_jacobian(Q, Qaux, parameters)[0], np.eye(3))
    assert np.allclose(functions.source(Q, Qaux, parameters)[0], np.zeros(3))
    assert np.allclose(
        functions.source_jacobian(Q, Qaux, parameters)[0], np.zeros((3, 3))
    )
    assert np.allclose(functions.eigenvalues(Q, Qaux, parameters)[0], np.ones((3, 3)))


def test_model_initialization_2d():
    main_dir = os.getenv("SMS")
    ic = IC.Constant()

    bc_tags = ["left", "right", "top", "bottom"]
    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag=tag, momentum_eqns=[1, 2]) for tag in bc_tags]
    )
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        bc_tags,
    )

    model = Model(
        dimension=2,
        n_fields=3,
        n_aux_fields=0,
        n_parameters=0,
        boundary_conditions=bcs,
        initial_conditions=ic,
    )
    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts

    Q = np.linspace(1, 3 * n_all_elements, 3 * n_all_elements).reshape(
        n_all_elements, 3
    )
    Qaux = np.zeros((Q.shape[0], 0))
    parameters = np.array([], dtype=float)

    model.initial_conditions.apply(Q, mesh.element_centers)
    model.boundary_conditions.apply(Q)

    functions = model.get_runtime_model()
    assert np.allclose(functions.flux(Q, Qaux, parameters), Q)
    assert np.allclose(functions.flux_jacobian(Q, Qaux, parameters)[0], np.eye(3))
    assert np.allclose(functions.source(Q, Qaux, parameters)[0], np.zeros(3))
    assert np.allclose(
        functions.source_jacobian(Q, Qaux, parameters)[0], np.zeros((3, 3))
    )
    assert np.allclose(functions.eigenvalues(Q, Qaux, parameters)[0], np.ones((3, 3)))


if __name__ == "__main__":
    test_model_initialization()
    test_model_initialization_2d()
