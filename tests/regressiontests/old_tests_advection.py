import os
import pytest
import matplotlib.pyplot as plt
import inspect
import numpy as np

from library.solver.controller import Controller
from library.solver.model import *
from library.solver.mesh import Mesh1D, Mesh2D
import library.solver.misc as misc
import library.gui.visualization.matplotlibstyle

main_dir = os.getenv("SMPYTHON")


@pytest.mark.critical
@pytest.mark.parametrize(
    "numerical_flux_func",
    [
        "rusanov",
        "lax_friedrichs",
        pytest.param("price_alphaC", marks=pytest.mark.broken),
    ],
)
def test_advection_1d(numerical_flux_func):
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = Advection(initial_conditions=ic)
    mesh = Mesh1D(number_of_elements=10, domain=[0, 10])
    controller = Controller(model=model, mesh=mesh)
    controller.time_end_sharp = True
    controller.time_end = 10.0
    controller.cfl = 1.0
    controller.solver.flux.scheme = numerical_flux_func
    Q, X, T, params = controller.solve_unsteady()
    # plt.plot(X[:, 0], Q[0, 0, :], "*")
    # plt.plot(X[:, 0], Q[-1, 0, :])
    # plt.show()
    assert np.allclose(Q[0, 0, :], Q[-1, 0, :])


# note: the cfl=1, even though the max limit for 2d is 0.5. However, this is quasi-1d
# allows for exact solutions in rusanov case
# this tc also test the funtinality of the periodic boundary conditions
@pytest.mark.critical
@pytest.mark.parametrize(
    "numerical_flux_func, mesh_filename, mesh_type",
    [
        ("rusanov", "meshes/quad_2d/mesh_coarse.msh", "quad"),
        ("lax_friedrichs", "meshes/quad_2d/mesh_coarse.msh", "quad"),
        ("rusanov", "meshes/tri_2d/mesh_coarse.msh", "triangle"),
        pytest.param(
            "price_alphaC",
            "meshes/quad_2d/mesh_coarse.msh",
            "quad",
            marks=pytest.mark.broken,
        ),
    ],
)
def test_advection_2d(numerical_flux_func, mesh_filename, mesh_type):
    ic = {"scheme": "func", "name": "double_riemann_2d"}
    model = Advection2d(initial_conditions=ic)
    model.advection_speed = [0.0, 1.0]
    model.boundary_conditions = [
        Periodic(physical_tag="left", periodic_to_physical_tag="right"),
        Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        Periodic(physical_tag="bottom", periodic_to_physical_tag="top"),
        Periodic(physical_tag="top", periodic_to_physical_tag="bottom"),
    ]
    mesh = Mesh2D(filename=mesh_filename, type=mesh_type)
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 50
    controller.time_end = 2.0
    controller.time_end_sharp = True
    controller.cfl = 1.0
    controller.output_write_vtk = True
    controller.solver.flux.scheme = numerical_flux_func
    Q, X, T, params = controller.solve_unsteady()
    if numerical_flux_func == "rusanov" and mesh_type == "quad":
        assert np.allclose(Q[0, 0, :], Q[-1, 0, :])
    else:
        function_name = inspect.currentframe().f_code.co_name
        # misc.write_field_to_npy(
        #     Q[-1],
        #     filepath=main_dir
        #     + "/library/tests/examples/referencedata/Advection/"
        #     + numerical_flux_func
        #     + "/"
        #     + mesh_filename
        #     + "/"
        #     + function_name
        #     + "/advection_in_y",
        #     filename="state.npy",
        # )
        Qref = misc.load_npy(
            filepath=main_dir
            + "/library/tests/examples/referencedata/Advection/"
            + numerical_flux_func
            + "/"
            + mesh_filename
            + "/"
            + function_name
            + "/advection_in_y",
            filename="state.npy",
        )
        assert np.allclose(Q[-1], Qref)
    controller.model.advection_speed = [-1.0, 0.0]
    Q, X, T, params = controller.solve_unsteady()
    if numerical_flux_func == "rusanov" and mesh_type == "quad":
        assert np.allclose(Q[0, 0, :], Q[-1, 0, :])
    else:
        function_name = inspect.currentframe().f_code.co_name
        # misc.write_field_to_npy(
        #     Q[-1],
        #     filepath=main_dir
        #     + "/library/tests/examples/referencedata/Advection/"
        #     + numerical_flux_func
        #     + "/"
        #     + mesh_filename
        #     + "/"
        #     + function_name
        #     + "/advection_in_x",
        #     filename="state.npy",
        # )
        Qref = misc.load_npy(
            filepath=main_dir
            + "/library/tests/examples/referencedata/Advection/"
            + numerical_flux_func
            + "/"
            + mesh_filename
            + "/"
            + function_name
            + "/advection_in_x",
            filename="state.npy",
        )
        assert np.allclose(Q[-1], Qref)
    controller.model.advection_speed = [-1.0, 0.0]


if __name__ == "__main__":
    test_advection_1d("rusanov")
    test_advection_1d("lax_friedrichs")
    # test_advection_1d("price_alphaC")
    test_advection_2d("rusanov", "meshes/quad_2d/mesh_coarse.msh", "quad")
    test_advection_2d("rusanov", "meshes/tri_2d/mesh_coarse.msh", "triangle")
    test_advection_2d("lax_friedrichs", "meshes/quad_2d/mesh_coarse.msh", "quad")
    # test_advection_2d("price_alphaC", "meshes/quad_2d/mesh_coarse.msh", "quad")
