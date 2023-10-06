import numpy as np

from library.misc import *
from library.model import *
import library.boundary_conditions as bc
import library.initial_condition as ic
from library.mesh import *


def test_model_initialization():
    ic = ic.Default()

    bcs = [Wall(physical_tag=tag, momentum_eqns=[1, 2]) for tag in bc_tags]
    mesh = Mesh.create_1d((-1, 1), 10)
    n_ghosts = initialize(bcs, mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    apply_boundary_conditions(bcs, Q)

    model = Model(dimension=1, boundary_condition=bcs, initial_condition=ic)


if __name__ == "__main__":
    test_model_initialization()
