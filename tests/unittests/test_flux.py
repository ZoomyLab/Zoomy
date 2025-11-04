import numpy as np
import pytest
from types import SimpleNamespace

from library.zoomy_core.model.model import *
import library.pysolver.flux as num_flux
import library.pysolver.reconstruction as reconstruction


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize(
    "dimension",
    ([1, 2]),
)
def test_default(dimension):
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

    functions = model.get_runtime_model()
    _ = model.create_c_interface()
    c_model = model.load_c_model()

    # get Lax-Friedrichs flux
    F = num_flux.LF

    # Reconstruction
    Qi, Qj, Qauxi, Qauxj, normals_ij = reconstruction.constant(mesh, Q, Qaux)

    mesh_props = SimpleNamespace(dt_dx=0.0)
    flux = F(
        Qi, Qj, Qauxi, Qauxj, parameters, normals_ij, c_model, mesh_props=mesh_props
    )[0]

    # Add flux (edges) to elements
    (
        map_elements_to_edges_plus,
        map_elements_to_edges_minus,
    ) = reconstruction.create_map_elements_to_edges(mesh)
    Qnew = np.zeros_like(Q)
    for i, elem in enumerate(map_elements_to_edges_plus):
        Qnew[elem] += flux[i]
    for i, elem in enumerate(map_elements_to_edges_minus):
        Qnew[elem] -= flux[i]
    if dimension == 1:
        assert np.allclose(
            Qnew.flatten(),
            np.array(
                [-1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 9.5, 0.0, 0.0]
            ),
        )
    else:
        pass


if __name__ == "__main__":
    test_default(1)
    test_default(2)
