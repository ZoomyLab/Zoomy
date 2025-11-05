import os
import pytest
import matplotlib.pyplot as plt
import inspect
import numpy as np

from solver.controller import Controller
from solver.model import *
from solver.mesh import Mesh1D, Mesh2D
import solver.misc as misc
import gui.visualization.matplotlibstyle

main_dir = os.getenv("SMPYTHON")


@pytest.mark.critical
def test_swe_basic_1d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWater(initial_conditions=ic)
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.broken
@pytest.mark.critical
def test_swe_sympy_basic_1d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterSympy(initial_conditions=ic)
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + "test_swe_basic_1d"
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.broken
@pytest.mark.critical
def test_swe_sympy_friction_1d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterSympy(initial_conditions=ic)
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 10.0}
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.solver.integrator.order = -1
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    # function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_swe_friction_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowWater(initial_conditions=ic)
    model.domain = [-1, 1]
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 10.0}
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_swe_inclination_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowWater(initial_conditions=ic)
    model.inclination_angle = 30.0
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 4.0
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)

    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


def test_swe_topography_1d():
    ic = {"scheme": "func", "name": "const_height"}
    model = ShallowWater(initial_conditions=ic)
    mesh = Mesh1D(number_of_elements=100)
    controller = Controller(model=model, mesh=mesh)
    controller.callback_list_init = ["controller_init_topo_bump"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)

    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_swe_timedependent_topography_1d():
    ic = {"scheme": "func", "name": "const_height"}
    model = ShallowWater(initial_conditions=ic)
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_timedependent_topography"]
    controller.callback_list_post_solvestep = ["controller_timedependent_topography"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q,
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q, Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label='H')
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.broken
@pytest.mark.critical
def test_sweb_wetdry_1d():
    # ic = {"scheme": "func", "name": "basin"}
    # ic = {"scheme": "func", "name": "riemann"}
    ic = {"scheme": "func", "name": "ramp_lake_at_rest"}
    # topo = {"scheme": "topo", "name": "basin"}
    model = ShallowWaterWithBottom(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
    ]
    mesh = Mesh1D(number_of_elements=31)
    controller = Controller(model=model, mesh=mesh)
    controller.dtmin = 10 ** (-7)
    controller.solver.flux.scheme = "roe_segmentpath"
    # controller.solver.flux.scheme = "rusanov_WB_swe_1d"
    # controller.solver.flux.scheme = "roe_osher_trapezoidal"
    # controller.solver.nc.scheme = 'swe_1d'
    controller.time_end = 10
    Q, X, T, params = controller.solve_unsteady()
    # function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q,
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q, Qref)

    plt.plot(X[:, 0], Q[0, -1, :], label="H")
    plt.plot(X[:, 0], Q[0, -1, :] + Q[0, 0, :], label="h(t=0)")
    plt.plot(X[:, 0], Q[-1, -1, :] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 0.025
    plt.legend()
    plt.show()


@pytest.mark.critical
def test_sweb_basin_wb_simple_1d():
    ic = {"scheme": "func", "name": "basin_lake_at_rest"}
    model = ShallowWaterWithBottom(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
    ]
    mesh = Mesh1D(number_of_elements=40)
    controller = Controller(model=model, mesh=mesh)
    controller = Controller(model=model)
    controller.solver.flux.scheme = "roe_segmentpath"
    controller.time_end = 3.0
    Q, X, T, params = controller.solve_unsteady()
    assert np.allclose(Q[-1], Q[0])

    # plt.plot(X[:, 0], Q[0, -1, :], "*", label="H")
    # plt.plot(X[:, 0], Q[0, -1, :] + Q[0, 0, :], "*", label="h(t=0)")
    # plt.plot(X[:, 0], Q[0, -1, :] + Q[-1, 0, :], "*", label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.broken
@pytest.mark.critical
def test_sweb_basin_wb_1d():
    ic = {"scheme": "func", "name": "basin"}
    model = ShallowWaterWithBottom(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
    ]
    # model.friction_models = ["newtonian"]
    # model.parameters = {"nu": 0.1}
    mesh = Mesh1D(number_of_elements=33)
    controller = Controller(model=model, mesh=mesh)
    controller = Controller(model=model)
    controller.solver.flux.scheme = "rusanov_WB_swe_1d"
    controller.time_end = 10.0
    Q, X, T, params = controller.solve_unsteady()
    # function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)

    plt.plot(X[:, 0], Q[0, -1, :], label="H")
    plt.plot(X[:, 0], Q[0, -1, :] + Q[0, 0, :], label="h(t=0)")
    plt.plot(X[:, 0], Q[0, -1, :] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.xlim(-0.25, 0.25)
    # plt.ylim(0, 0.025)
    plt.legend()
    plt.show()


@pytest.mark.broken
@pytest.mark.critical
def test_sweb_basin_wb_with_friction_1d():
    ic = {"scheme": "func", "name": "basin"}
    model = ShallowWaterWithBottom(initial_conditions=ic)
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 1.0}
    controller = Controller(model=model)
    controller.solver.integrator.order = -1
    controller.callback_list_pre = ["controller_template"]
    controller.solver.flux.scheme = "rusanov_WB_swe_1d"
    controller.time_end = 10.0
    Q, X, T, params = controller.solve_unsteady()
    # function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)

    plt.plot(X[:, 0], Q[0, -1, :], label="H")
    plt.plot(X[:, 0], Q[0, -1, :] + Q[0, 0, :], label="h(t=0)")
    plt.plot(X[:, 0], Q[-1, -1, :] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 0.025)
    plt.legend()
    plt.show()


@pytest.mark.broken
@pytest.mark.parallel
def test_sweb_parallel_1d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterWithBottom(initial_conditions=ic)
    mesh = Mesh1D(number_of_elements=1000)
    controller = Controller(model=model, mesh=mesh)
    controller.time_end = 10
    Q, X, T, params = controller.solve_unsteady()
    # function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], Q[0, -1, :], label="H")
    # plt.plot(X[:, 0], Q[0, -1, :] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], Q[-1, -1, :] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_sweb_custom_boundaries_1d():
    model = ShallowWaterWithBottom()
    model.boundary_conditions = [
        Custom_extrapolation(
            tag="left", bc_function_dict={0: "lambda t, Q, X: 1.0"}
        ),
        Custom_extrapolation(
            tag="right", bc_function_dict={1: "lambda t, Q, X: 1.0"}
        ),
    ]
    controller = Controller(model=model)
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_swe_basic_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWater2d(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
        Wall(tag="bottom"),
        Wall(tag="top"),
    ]
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.cfl = 0.45
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.output_write_vtk = False
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_swe_friction_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWater2d(initial_conditions=ic)
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 10.0}
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.solver.integrator.order = -1
    controller.cfl = 0.45
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.output_write_vtk = False
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_sweb_friction_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 10.0}
    mesh = Mesh2D(filename="meshes/quad_2d/mesh_mid.msh")
    controller = Controller(model=model, mesh=mesh)
    controller.solver.integrator.order = -1
    controller.cfl = 0.45
    controller.output_write_vtk = False
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_sweb_inflow_outflow_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.boundary_conditions = [
        Custom_extrapolation(
            tag="left", bc_function_dict={1: "lambda t, Q, X: 0.1"}
        ),
        Custom_extrapolation(
            tag="right", bc_function_dict={0: "lambda t, Q, X: 1.0"}
        ),
        Wall(tag="bottom"),
        Wall(tag="top"),
    ]
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.cfl = 0.45
    controller.output_write_vtk = False
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_sweb_wetdry_2d():
    ic = {"scheme": "func", "name": "riemann"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
        Wall(tag="top"),
        Wall(tag="bottom"),
    ]
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.output_write_vtk = False
    controller.cfl = 0.45
    controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_sweb_wb_simple_2d():
    ic = {"scheme": "func", "name": "basin_lake_at_rest"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
        Wall(tag="top"),
        Wall(tag="bottom"),
    ]
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 10
    controller.output_write_vtk = True
    controller.cfl = 0.45
    controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    assert np.allclose(Q[-1], Q[0])


@pytest.mark.broken
@pytest.mark.critical
def test_sweb_wb_2d():
    ic = {"scheme": "func", "name": "basin_lake_at_rest"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.boundary_conditions = [
        Wall(tag="left"),
        Wall(tag="right"),
        Wall(tag="top"),
        Wall(tag="bottom"),
    ]
    mesh = Mesh2D()
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 10
    controller.output_write_vtk = True
    controller.cfl = 0.45
    controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()


@pytest.mark.slow
@pytest.mark.parametrize(
    "mesh_file, mesh_type",
    [
        ("meshes/quad_2d_hole/mesh_coarse.msh", "quad"),
        pytest.param(
            "meshes/tri_2d_hole/mesh_coarse.msh", "triangle", marks=pytest.mark.broken
        ),
    ],
)
def test_sweb_mesh_with_hole(mesh_file, mesh_type):
    ic = {"scheme": "func", "name": "const_height"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.boundary_conditions = [
        Custom_extrapolation(
            tag="left",
            bc_function_dict={
                1: "lambda t, Q, X: 0.3 / Q[0]",
                2: "lambda t, Q, X: 0.0",
            },
        ),
        Extrapolation(tag="right"),
        Wall(tag="top"),
        Wall(tag="bottom"),
        Wall(tag="hole"),
    ]
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 1.0}
    mesh = Mesh2D(
        filename=mesh_file,
        type=mesh_type,
        boundary_tags=["left", "right", "bottom", "top", "hole"],
    )
    controller = Controller(model=model, mesh=mesh)
    controller.output_write_vtk = True
    controller.output_snapshots = 100
    controller.cfl = 0.45
    controller.solver.integrator.order = -1
    controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    controller.time_end = 5.0
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/"
    #     + mesh_type
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/"
        + mesh_type
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.slow
def test_swe_stiffler_2d():
    ic = {"scheme": "func", "name": "stiffler_2d"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.n_variables = 4
    # model.friction_models = ["chezy", "newtonian"]
    model.friction_models = ["chezy"]
    model.parameters = {"ChezyCoef": 16, "nu": 0.1}
    model.g = 9.81
    model.boundary_conditions = [
        Custom_extrapolation(
            tag="outflow",
            bc_function_dict={0: "lambda t, Q, X: 0.061"},
        ),
        Wall2D(tag="wall"),
        Custom_extrapolation(
            tag="inflow",
            bc_function_dict={
                1: "lambda t, Q, X: -0.36 * 0.061 ",
                2: "lambda t, Q, X: 0. ",
            },
        ),
    ]
    mesh = Mesh2D(
        filename="meshes/curved_open_channel/mesh_coarse.msh",
        type="quad",
        boundary_tags=["inflow", "outflow", "wall"],
    )
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 20
    controller.output_write_vtk = True
    controller.debug_output_level = 3
    controller.cfl = 0.45
    controller.solver.scheme = "explicit_split_source_coordinate_transform"
    # controller.solver.scheme = "explicit_split_source"
    controller.solver.integrator.order = -1
    controller.solver.flux.scheme = "rusanov"
    controller.solver.nc.scheme = "segmentpath"
    controller.time_end = 10.0
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SWE/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.old
def test_swe_complex_topography_2d():
    ic = {"scheme": "func", "name": "complex_topography"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.domain = [-1, 1, -1, 1]
    model.n_variables = 4
    model.friction_models = ["newtonian"]
    model.parameters = {"nu": 0.01}
    model.boundary_conditions = [
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
        # Extrapolation(tag="bottom"),
        Extrapolation(tag="top"),
        Custom_extrapolation(
            tag="bottom",
            bc_function_dict={
                # 0: "lambda t, Q, X: 1.0 + np.sin(5*np.pi*t)",
                # 0: "lambda t, Q, X: 1.0",
                1: "lambda t, Q, X: 1.1 * Q[0] * (X[0]> -0.1) * (X[0] < 0.1)",
                2: "lambda t, Q, X: 1.1 * Q[0] * (X[0]> -0.1) * (X[0] < 0.1)",
            },
        ),
    ]
    controller = Controller(model=model)
    controller.load_mesh = ["meshes/quad_2d/mesh_fine.msh", "quad"]
    controller.output_write_vtk = True
    controller.output_snapshots = 100
    controller.debug_output_level = 3
    controller.cfl = 0.45
    controller.scheme = "fully_explicit_lhs_split_rhs"
    controller.solver.scheme = "explicit_split_source_coordinate_transform"
    # controller.solver.flux.scheme = "rusanov"
    controller.solver.integrator.order = -1
    controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    # controller.solver.nc.scheme = "swe_1d"
    controller.solver.nc.scheme = "segmentpath"
    controller.time_end = 1.1
    Q, X, T, params = controller.solve_unsteady()


@pytest.mark.old
def test_swe_stiffler_dam_break_2d():
    ic = {"scheme": "func", "name": "stiffler_dam_break"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.domain = [-1, 1, -1, 1]
    model.n_variables = 4
    model.g = 9.81
    # model.friction_models = ["chezy", "newtonian"]
    # model.parameters = {"ChezyCoef": 16.0, "nu": 0.1}
    model.boundary_conditions = [
        Wall2D(tag="wall"),
        Wall2D(tag="inflow"),
        Wall2D(tag="outflow"),
    ]
    controller = Controller(model=model)
    controller.load_mesh = ["meshes/curved_open_channel/mesh_mid.msh", "quad"]
    controller.output_write_vtk = True
    controller.debug_output_level = 3
    controller.cfl = 0.45
    controller.scheme = "fully_explicit_lhs_split_rhs_coord_transform"
    # controller.scheme = "fully_explicit_lhs_split_rhs"
    controller.solver.flux.scheme = "rusanov"
    controller.solver.integrator.order = 1
    # controller.solver.flux.scheme = "rusanov_WB_swe_2d"
    # controller.solver.nc.scheme = "swe_1d"
    controller.solver.nc.scheme = "segmentpath"
    # controller.solver.nc.scheme = "none"
    controller.time_end = 500.0
    Q, X, T, params = controller.solve_unsteady()


@pytest.mark.old
def test_swe_nozzle_2d():
    # ic = {"scheme": "func", "name": "const_height"}
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowWaterWithBottom2d(initial_conditions=ic)
    model.n_variables = 4
    model.boundary_conditions = [
        # Custom_extrapolation(
        #     tag="left",
        #     bc_function_dict={
        #         1: "lambda t, Q, X: 0.1",
        #         2: "lambda t, Q, X: 0.0",
        #     },
        # ),
        # Custom_extrapolation(
        #     tag="right",
        #     bc_function_dict={
        #         0: "lambda t, Q, X: 1.0",
        #     },
        # ),
        Wall(tag="left"),
        Wall(tag="right"),
        Wall(tag="top"),
        Wall(tag="bottom"),
    ]
    model.g = 9.81
    controller = Controller(model=model)
    controller.load_mesh = ["meshes/quad_nozzle_2d/mesh_coarse.msh", "quad"]
    controller.output_write_vtk = True
    controller.debug_output_level = 3
    controller.cfl = 0.45
    controller.scheme = "fully_explicit_lhs_split_rhs_coord_transform"
    controller.solver.flux.scheme = "rusanov"
    controller.solver.integrator.order = -1
    # controller.solver.flux.scheme = "rusanov_WB_swe_1d"
    # controller.solver.nc.scheme = "swe_1d"
    controller.solver.nc.scheme = "segmentpath"
    controller.time_end = 3.0
    Q, X, T, params = controller.solve_unsteady()


@pytest.mark.future
def test_swe_exner_basic_1d():
    ic = {"scheme": "lambda", 0: "lambda x: 1.0*(x[:,0]<=0) + 0.05*(x[:,0]>0)"}
    model = ShallowWaterExner(initial_conditions=ic)
    model.n_variables = 3
    mesh = Mesh1D(number_of_elements=80)
    controller = Controller(model=model, mesh=mesh)
    # controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 3.0
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir + "/library/tests/examples/referencedata/SWE/" + function_name + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SWE/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], Q[-1, 0, :], label="h(t=t_end)")
    # plt.plot(X[:, 0], Q[-1, 1, :], label="h(t=t_end)")
    plt.plot(X[:, 0], Q[-1, 2, :], label="h(t=t_end)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_swe_basic_1d()
    # test_swe_sympy_basic_1d()
    # test_swe_sympy_stiff_rhs_1d()
    # test_swe_friction_1d()
    # test_swe_inclination_1d()
    # test_swe_topography_1d()
    # test_swe_timedependent_topography_1d()
    # test_sweb_wetdry_1d()
    # test_sweb_basin_wb_simple_1d()
    # test_sweb_basin_wb_1d()
    # test_sweb_basin_wb_with_friction_1d()
    # test_sweb_parallel_1d()
    # test_sweb_custom_boundaries_1d()

    test_swe_basic_2d()
    # test_swe_friction_2d()
    # test_sweb_friction_2d()
    # test_sweb_inflow_outflow_2d()
    # test_sweb_wetdry_2d()
    # test_sweb_wb_simple_2d()
    # test_sweb_wb_2d()
    # test_sweb_mesh_with_hole("meshes/quad_2d_hole/mesh_coarse.msh", "quad")
    # test_sweb_mesh_with_hole("meshes/tri_2d_hole/mesh_coarse.msh", "triangle")
    # test_swe_stiffler_2d()

    # test_swe_exner_basic_1d()

    # test_swe_complex_topography_2d()

    ## OLD STUFF
    # test_swe_stiff_topo_dry_2d()
    # test_swe_inflow_outflow_2d()
    # test_swe_nozzle_2d()
    # test_swe_wb_2d()
    # test_swe_stiffler_dam_break_2d()
    # @todo
    # test_swe_bc_wall_1d()
    # @todo
    # test_swe_bc_inflow_outflow_1d()
    # @todo
    # test_swe_cauchy_kowalewski_recon()
    # @todo
    # test_swe_fully_implicit()
    # @todo
    # test_swe_rhs_implicit()
    # @todo
    # test_swe_rhs_scipy_solver()
    # @todo
    # test_swe_inclination_1d()
