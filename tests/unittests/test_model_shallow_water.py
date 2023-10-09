import numpy as np
import pytest

from library.misc import *
from library.models.shallow_water import *
import library.boundary_conditions as BC
import library.initial_conditions as IC
from library.mesh import *


@pytest.mark.critical
def test_model_eigenvalues():
    main_dir = os.getenv("SMS")
    ic = IC.Constant()

    bc_tags = ["left", "right"]
    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag=tag, momentum_eqns=[1]) for tag in bc_tags]
    )
    mesh = Mesh.create_1d((-1, 1), 10)
    model = ShallowWater(
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    Qaux = np.zeros((Q.shape[0], 0))
    parameters = np.array([], dtype=float)
    model.boundary_conditions.apply(Q)
    model.initial_conditions.apply(Q, mesh.element_centers)

    functions = model.get_runtime_model()

    assert (
        str(model.sympy_eigenvalues)
        == 'Matrix([[n0*q1/q0 - n0*sqrt(ez*g*q0**5)/q0**2], [n0*q1/q0 + n0*sqrt(ez*g*q0**5)/q0**2]])'
    )


if __name__ == "__main__":
    test_model_eigenvalues()
