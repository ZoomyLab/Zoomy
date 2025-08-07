import os
import pytest
import matplotlib.pyplot as plt
import inspect
import numpy as np

from library.solver.controller import Controller
from library.solver.model import *
from library.solver.mesh import Mesh1D, Mesh2D
import library.solver.misc as misc
import library.visualization.matplotlibstyle

main_dir = os.getenv("SMPYTHON")


@pytest.mark.critical
def test_smm_basic_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowMoments(initial_conditions=ic)
    model.n_variables = 4
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SMM/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    print(np.linalg.norm(Q[-1] - Qref))
    assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], label="u")
    # plt.plot(X[:, 0], Q[-1, 2, :] / Q[-1, 0, :], label="a")
    # plt.plot(X[:, 0], Q[-1, 3, :] / Q[-1, 0, :], label="a2")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_smm_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowMoments(initial_conditions=ic)
    model.n_variables = 4
    model.models = ["newtonian"]
    model.parameters = {"nu": 0.1, "lamda": 1.0, "rho": 1.0}
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    controller.solver.integrator.order = -1
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SMM/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], label="u")
    # plt.plot(X[:, 0], Q[-1, 2, :] / Q[-1, 0, :], label="a")
    # plt.plot(X[:, 0], Q[-1, 3, :] / Q[-1, 0, :], label="a2")
    # plt.plot(X[:, 0], params["aux_variables"]["H"] + Qref[0, :], "*", label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_smmb_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowMomentsWithBottom(initial_conditions=ic)
    model.n_variables = 5
    model.models = ["newtonian"]
    model.parameters = {"nu": 0.1, "lamda": 1.0, "rho": 1.0}
    controller = Controller(model=model)
    controller.time_end = 0.3
    controller.solver.integrator.order = -1
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SMM/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)
    # plt.plot(X[:, 0], Q[-1, -1, :], label="H")
    # plt.plot(X[:, 0], Q[-1, -1, :] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], Q[-1, -1, :] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], label="u")
    # plt.plot(X[:, 0], Q[-1, 2, :] / Q[-1, 0, :], label="a")
    # plt.plot(X[:, 0], Q[-1, 3, :] / Q[-1, 0, :], label="a2")
    # plt.legend()
    # plt.show()


@pytest.mark.critical
def test_compare_smm_smmb_1d():
    Qsmm = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + "test_smm_1d"
        + "/",
        filename="state.npy",
    )
    Qsmmb = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + "test_smmb_1d"
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Qsmm, Qsmmb[:-1])


@pytest.mark.broken
@pytest.mark.critical
def test_smm_reproduce_paper_1d():
    ic = {"scheme": "func", "name": "paper_linear"}
    model = ShallowMoments(initial_conditions=ic)
    mesh = Mesh1D(number_of_elements=50, domain=[-1, 1])
    model.n_variables = 5
    model.models = ["newtonian"]
    model.parameters = {"nu": 0.1, "lamda": 1.0, "rho": 1.0}
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.cfl = 0.8
    controller.solver.flux.scheme = "rusanov"
    controller.time_end = 2.0
    controller.solver.integrator.order = 2
    controller.solver.reconstruction.order = 1
    controller.solver.reconstruction.limiter = "smm_paper_1"
    Q, X, T, params = controller.solve_unsteady()
    Qref = np.loadtxt(
        main_dir
        + "/library/tests/examples/referencedata/SMM/kowalski_mathematica/0p1/Data.csv",
        delimiter=",",
    )
    QrefProj = np.loadtxt(
        main_dir
        + "/library/tests/examples/referencedata/SMM/kowalski_mathematica/0p1/DataProj.csv",
        delimiter=",",
    )

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(X[:, 0], Q[-1, 0, :], "-")
    ax[0, 0].plot(QrefProj[:, 0], QrefProj[:, 1], ".", label="ref")
    ax[1, 0].plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], "-")
    ax[1, 0].plot(QrefProj[:, 0], QrefProj[:, 2], ".", label="ref")
    ax[0, 1].plot(X[:, 0], Q[-1, 2, :], ".")
    ax[0, 1].plot(QrefProj[:, 0], QrefProj[:, 3], "-", label="ref")
    ax[1, 1].plot(X[:, 0], Q[-1, 3, :], ".")
    ax[1, 1].plot(QrefProj[:, 0], QrefProj[:, 4], "-", label="ref")
    # ax[1].plot(Xold, Qold[1], '.')
    # index = 25
    # U, Z = kwargs['model'].matrices.recover_velocity_profile(Q[:,index])
    # ax[2].plot(U, Z)
    # ax[0].plot(X[:,0], Q2[0], '*')
    # ax[1].plot(X[:,0], Q2[1], '*')
    plt.show()
    assert False


@pytest.mark.broken
@pytest.mark.critical
def test_smmb_reproduce_paper_1d():
    ic = {"scheme": "func", "name": "paper_linear"}
    model = ShallowMomentsWithBottom(initial_conditions=ic)
    mesh = Mesh1D(number_of_elements=50, domain=[-1, 1])
    model.n_variables = 6
    model.models = ["newtonian"]
    model.parameters = {"nu": 0.1, "lamda": 1.0, "rho": 1.0}
    controller = Controller(model=model)
    controller.cfl = 0.8
    controller.solver.flux.scheme = "rusanov"
    controller.time_end = 2.0
    controller.solver.integrator.order = 2
    controller.solver.reconstruction.order = 1
    controller.solver.reconstruction.limiter = "smm_paper_1"
    Q, X, T, params = controller.solve_unsteady()
    Qref = np.loadtxt(
        main_dir
        + "/library/tests/examples/referencedata/SMM/kowalski_mathematica/0p1/Data.csv",
        delimiter=",",
    )
    QrefProj = np.loadtxt(
        main_dir
        + "/library/tests/examples/referencedata/SMM/kowalski_mathematica/0p1/DataProj.csv",
        delimiter=",",
    )

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(X[:, 0], Q[-1, 0, :], "-")
    ax[0, 0].plot(QrefProj[:, 0], QrefProj[:, 1], ".", label="ref")
    ax[1, 0].plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], "-")
    ax[1, 0].plot(QrefProj[:, 0], QrefProj[:, 2], ".", label="ref")
    ax[0, 1].plot(X[:, 0], Q[-1, 2, :], ".")
    ax[0, 1].plot(QrefProj[:, 0], QrefProj[:, 3], "-", label="ref")
    ax[1, 1].plot(X[:, 0], Q[-1, 3, :], ".")
    ax[1, 1].plot(QrefProj[:, 0], QrefProj[:, 4], "-", label="ref")
    # ax[1].plot(Xold, Qold[1], '.')
    # index = 25
    # U, Z = kwargs['model'].matrices.recover_velocity_profile(Q[:,index])
    # ax[2].plot(U, Z)
    # ax[0].plot(X[:,0], Q2[0], '*')
    # ax[1].plot(X[:,0], Q2[1], '*')
    plt.show()
    assert False


@pytest.mark.future
@pytest.mark.critical
def test_hsmm_basic_1d():
    ic = {"scheme": "func", "name": "smooth"}
    model = ShallowMomentsHyperbolic(initial_conditions=ic)
    model.n_variables = 4
    controller = Controller(model=model)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    # misc.write_field_to_npy(
    #     Q[-1],
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SMM/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    # Qref = misc.load_npy(
    #     filepath=main_dir
    #     + "/library/tests/examples/referencedata/SMM/"
    #     + function_name
    #     + "/",
    #     filename="state.npy",
    # )
    # assert np.allclose(Q[-1], Qref)
    plt.plot(X[:, 0], params["aux_variables"]["H"], label="H")
    plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[0, 0, :], label="h(t=0)")
    plt.plot(X[:, 0], params["aux_variables"]["H"] + Q[-1, 0, :], label="h(t=t_end)")
    plt.plot(X[:, 0], Q[-1, 1, :] / Q[-1, 0, :], label="u")
    plt.plot(X[:, 0], Q[-1, 2, :] / Q[-1, 0, :], label="a")
    plt.plot(X[:, 0], Q[-1, 3, :] / Q[-1, 0, :], label="a2")
    plt.legend()
    plt.show()


@pytest.mark.broken
@pytest.mark.critical
def test_smmb_basin_wb_simple_1d():
    ic = {"scheme": "func", "name": "basin_lake_at_rest"}
    model = ShallowMomentsWithBottom(initial_conditions=ic)
    model.n_variables = 3
    model.boundary_conditions = [
        Wall(physical_tag="left"),
        Wall(physical_tag="right"),
    ]
    mesh = Mesh1D(number_of_elements=40)
    controller = Controller(model=model, mesh=mesh)
    controller = Controller(model=model)
    controller.solver.flux.scheme = "rusanov_WB_swe_1d"
    controller.time_end = 3.0
    Q, X, T, params = controller.solve_unsteady()
    # assert np.allclose(Q[-1], Q[0])
    # plt.plot(X[:, 0], Q[-1][-1,:], label="H")
    # plt.plot(X[:, 0], Q[-1][-1,:] + Q[0, 0, :], label="h(t=0)")
    # plt.plot(X[:, 0], Q[-1][-1,:] + Q[-1, 0, :], label="h(t=t_end)")
    # plt.legend()
    # plt.show()


@pytest.mark.future
@pytest.mark.critical
def test_smm_scipy_rhs():
    return


@pytest.mark.future
@pytest.mark.critical
def test_smm_quasilinear_solver():
    return


@pytest.mark.future
@pytest.mark.critical
def test_smm_hyperbolic_fix():
    return


@pytest.mark.future
@pytest.mark.critical
def test_smm_strong_bc_coupling():
    return


@pytest.mark.critical
def test_smm_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowMoments2d(initial_conditions=ic)
    model.n_variables = 5
    model.boundary_conditions = [
        Extrapolation(physical_tag="left"),
        Extrapolation(physical_tag="right"),
        Extrapolation(physical_tag="bottom"),
        Extrapolation(physical_tag="top"),
    ]
    model.models = ["newtonian", "bc_slip"]
    model.parameters = {"nu": 0.1, "rho": 1.0, "sliplength": 1.0}
    mesh = Mesh2D(
        filename="meshes/quad_2d/mesh_coarse.msh",
    )
    controller = Controller(model=model, mesh=mesh)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.callback_list_post_solvestep = ["controller_clip_small_moments"]
    controller.output_write_vtk = False
    controller.cfl = 0.45
    controller.solver.integrator.order = -1
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    misc.write_field_to_npy(
        Q[-1],
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.critical
def test_smmb_2d():
    ic = {"scheme": "func", "name": "riemann_offset"}
    model = ShallowMomentsWithBottom2d(initial_conditions=ic)
    model.n_variables = 6
    model.boundary_conditions = [
        Extrapolation(physical_tag="left"),
        Extrapolation(physical_tag="right"),
        Extrapolation(physical_tag="bottom"),
        Extrapolation(physical_tag="top"),
    ]
    model.models = ["newtonian", "bc_slip"]
    model.parameters = {"nu": 0.1, "rho": 1.0, "sliplength": 1.0}
    mesh = Mesh2D(
        filename="meshes/quad_2d/mesh_coarse.msh",
    )
    controller = Controller(model=model, mesh=mesh)
    controller.callback_list_init = ["controller_init_topo_const"]
    controller.callback_list_post_solvestep = ["controller_clip_small_moments"]
    controller.output_write_vtk = False
    controller.cfl = 0.45
    controller.solver.integrator.order = -1
    controller.time_end = 0.3
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    misc.write_field_to_npy(
        Q[-1],
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.broken
@pytest.mark.critical
def test_compare_smm_smmb_2d():
    Qsmm = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + "test_smm_2d"
        + "/",
        filename="state.npy",
    )
    Qsmmb = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + "test_smmb_2d"
        + "/",
        filename="state.npy",
    )
    print(np.linalg.norm(Qsmm - Qsmmb[:-1]))
    assert np.allclose(Qsmm, Qsmmb[:-1])


@pytest.mark.broken
@pytest.mark.slow
def test_smm_with_hole_2d():
    ic = {"scheme": "func", "name": "riemann"}
    model = ShallowMomentsWithBottom2d(initial_conditions=ic)
    model.n_variables = 6
    model.boundary_conditions = [
        Custom_extrapolation(
            physical_tag="left",
            bc_function_dict={
                1: "lambda t, Q, X: 0.3 / Q[0]",
                2: "lambda t, Q, X: 0.0",
                3: "lambda t, Q, X: 0.0",
                4: "lambda t, Q, X: 0.0",
            },
        ),
        Extrapolation(physical_tag="right"),
        Wall2D(physical_tag="bottom"),
        Wall2D(physical_tag="top"),
        Wall2D(physical_tag="hole"),
    ]
    # model.models = ["newtonian", "bc_slip"]
    model.models = ["bc_slip"]
    model.parameters = {"nu": 0.1, "rho": 1.0, "sliplength": 1.0}
    mesh = Mesh2D(
        filename="meshes/quad_2d_hole/mesh_coarse.msh",
        boundary_tags=["left", "right", "bottom", "top", "hole"],
    )
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 50
    controller.output_write_vtk = True
    controller.cfl = 0.45
    controller.solver.integrator.order = -1
    controller.time_end = 1.5
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    misc.write_field_to_npy(
        Q[-1],
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


@pytest.mark.broken
@pytest.mark.slow
def test_smmb_basin_wb_2d():
    ic = {"scheme": "func", "name": "basin_2d"}
    model = ShallowMomentsWithBottom2d(initial_conditions=ic)
    model.n_variables = 6
    model.boundary_conditions = [
        Wall2D(physical_tag="left"),
        Wall2D(physical_tag="right"),
        Wall2D(physical_tag="bottom"),
        Wall2D(physical_tag="top"),
    ]
    # model.models = ["newtonian", "bc_slip"]
    model.models = []
    model.parameters = {"nu": 0.01, "rho": 1.0, "sliplength": 1.0}
    mesh = Mesh2D(
        filename="meshes/quad_2d/mesh_coarse.msh",
    )
    controller = Controller(model=model, mesh=mesh)
    controller.output_snapshots = 50
    controller.output_write_vtk = True
    controller.cfl = 0.45
    controller.solver.integrator.order = -1
    controller.solver.flux.scheme = "rusanov_WB_smm_2d"
    controller.time_end = 10.0
    Q, X, T, params = controller.solve_unsteady()
    function_name = inspect.currentframe().f_code.co_name
    misc.write_field_to_npy(
        Q[-1],
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    Qref = misc.load_npy(
        filepath=main_dir
        + "/library/tests/examples/referencedata/SMM/"
        + function_name
        + "/",
        filename="state.npy",
    )
    assert np.allclose(Q[-1], Qref)


if __name__ == "__main__":
    # test_smm_basic_1d()
    # test_smm_1d()
    # test_smmb_1d()
    # test_compare_smm_smmb_1d()
    # test_smm_reproduce_paper_1d()
    # test_smmb_reproduce_paper_1d()
    # test_smmb_basin_wb_simple_1d()
    test_hsmm_basic_1d()

    # test_smm_scipy_rhs()
    # test_smm_quasilinear_solver()
    # test_smm_hyperbolic_fix()
    # test_smm_strong_bc_coupling()

    # test_smm_2d()
    # test_smmb_2d()
    # test_compare_smm_smmb_2d()
    # test_smm_with_hole_2d()
    # test_smmb_basin_wb_2d()
