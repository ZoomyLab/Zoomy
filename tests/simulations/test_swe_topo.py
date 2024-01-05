import numpy as np
import pytest
from types import SimpleNamespace
import concurrent.futures
import os
from copy import deepcopy as copy

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import *
import library.misc.io as io


@pytest.mark.critical
@pytest.mark.unfinished
def test_swetopo_1d():
    settings = Settings(
        name="ShallowWaterTopo",
        momentum_eqns=[1],
        parameters={"g": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF_wb(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
        output_snapshots=100,
        output_dir="out",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )

    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = 1.0 - 0.1 * x[0]
        Q[2] = 0.1 * x[0]
        return Q

    ic = IC.UserFunction(ic_func)
    model = ShallowWaterTopo(
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_swetopo_2d(mesh_type):
    settings = Settings(
        name="ShallowWater2d",
        momentum_eqns=[1, 2],
        parameters={"g": 1.0, "C": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF_wb(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=1.0,
        output_snapshots=100,
    )

    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )

    def ic_func(x):
        Q = np.zeros(4, dtype=float)
        Q[0] = 1.0 - 0.1 * x[0]
        Q[3] = 0.1 * x[0]
        return Q

    ic = IC.UserFunction(ic_func)
    model = ShallowWaterTopo2d(
        dimension=2,
        fields=4,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"]},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

def test_calibration_1d(inputs):
    index=inputs[0]
    parameters = {"g": 9.81, "nm": 0.025}
    parameters['nm'] = inputs[1]

    mesh = Mesh.create_1d((-5, 5), 100)
    settings = Settings(
        name="Calibration",
        momentum_eqns=[1],
        parameters=parameters,
        reconstruction=recon.constant,
        num_flux=flux.LLF_wb(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=100.0,
        output_snapshots=100,
        output_dir = f'out_{index}'
    )

    hin = 0.2329
    huin = 0.155

    def bump(x):
        return 0.05 * np.exp(-(x[0]**2) * 10)

    def slope(x):
        return 0.
        return -(x[0]+5.)/100. + 0.1

    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = 0.10 - bump(x)
        Q[1] = 0.
        Q[2] = slope(x) + bump(x)
        return Q

    ic = IC.UserFunction(ic_func)

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    # inflow_dict = {0: hout, 1: vin*hout}
    # outflow_dict = {}

    # I want to extrapolate the bottom topography in order to avoid a reconstruction problem (wave generation)
    dx = 10./100.
    x_minus = copy(mesh.element_center[0])
    x_minus[0] -= dx
    x_plus = copy(mesh.element_center[-1])
    x_plus[0] += dx
    inflow_dict = {0: hin,1: huin, 2: slope(x_minus) }
    outflow_dict = { 2:slope(x_plus) }
    bcs = BC.BoundaryConditions(
        [
            BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
        ]
    )
    # bcs = BC.BoundaryConditions(
    #     [
    #         BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
    #         BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
    #     ]
    # )

    # ic = IC.RestartFromHdf5(
    #     path_to_old_mesh="out_restart/mesh.hdf5",
    #     path_to_fields="out_restart/fields.hdf5",
    #     mesh_new=mesh,
    #     mesh_identical=True,
    # )

    model = ShallowWaterTopo(
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ['manning']},
    )
    # model_functions = model.get_runtime_model()
    # _ = model.create_c_interface()
    runtime_model = model.load_c_model()


    fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_source=RKimplicit, runtime_model=runtime_model
    )
    logging.debug(f"Process {os.getpid()}: nm {settings.parameters['nm']}")
    io.generate_vtk(settings.output_dir)




if __name__ == "__main__":
    # test_swetopo_1d()
    # test_swetopo_2d("quad")
    # test_swetopo_2d("triangle")

    runs = 5
    samples_index = list(range(runs))
    samples_nm = list(np.linspace(0.00, 1.0, runs))
    # test_calibration_1d(list(zip(samples_index, samples_nm))[-1])
    # for inputs in list(zip(samples_index, samples_nm)):
    #     test_calibration_1d(inputs)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use executor.map to apply the function to each item in parallel
        executor.map(test_calibration_1d, list(zip(samples_index, samples_nm)))

