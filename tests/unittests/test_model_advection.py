import numpy as np
import pytest

from library.misc import *
from library.models.base import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *


@pytest.mark.parametrize(
    "dimension",
    ([1, 2]),
)
def test_model_initialization(
    dimension,
):
    main_dir = os.getenv("SMS")
    ic = IC.Constant()

    bc_tags = ["left", "right"]
    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag=tag, momentum_eqns=[1, 2]) for tag in bc_tags]
    )
    if dimension == 1:
        mesh = Mesh.create_1d((-1, 1), 10)
    elif dimension == 2:
        mesh = Mesh.load_mesh(
            os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
            "quad",
            2,
            bc_tags,
        )
    else:
        assert False
    model = Advection(
        dimension=dimension,
        n_fields=3,
        n_aux_fields=0,
        n_parameters=3,
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 3 * n_all_elements, 3 * n_all_elements).reshape(
        n_all_elements, 3
    )
    Qaux = np.zeros((Q.shape[0], 0))
    advection_speeds = np.array([1.0, 0.0, 2.0], dtype=float)
    parameters = advection_speeds
    model.boundary_conditions.apply(Q)
    model.initial_conditions.apply(Q, mesh.element_centers)

    functions = model.get_runtime_model()
    assert np.allclose(functions.flux(Q, Qaux, parameters), advection_speeds * Q)
    assert np.allclose(
        functions.flux_jacobian(Q, Qaux, parameters)[0], np.diag(advection_speeds)
    )
    assert np.allclose(functions.source(Q, Qaux, parameters)[0], np.zeros(3))
    assert np.allclose(
        functions.source_jacobian(Q, Qaux, parameters)[0], np.zeros((3, 3))
    )
    assert np.allclose(functions.eigenvalues(Q, Qaux, parameters)[0], advection_speeds)


if __name__ == "__main__":
    test_model_initialization(1)
    test_model_initialization(2)
