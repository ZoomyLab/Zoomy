import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.postprocessing.postprocessing as postprocessing
import argparse


@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d():
    level = 0
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 1.0, "C": 1.0, "nu": 0.1},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.constant(dt = 0.01),
        time_end=1.0,
        output_snapshots=100,
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array([2.0, 0.0] + [0.0 for l in range(level)]),
        right=lambda n_field: np.array([1.0, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["chezy", "newtonian"]},
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

def test_sindy_generate_reference_data():
    level = 0
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 20.0, "nu": 0.0016},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.constant(dt = 0.01),
        time_end=4.0,
        output_snapshots=100,
        output_dir=f"output_{str(level)}",
    )

    bc_tags = ["left", "right"]
    bc_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag=tag)
            for (tag, periodic_to) in zip(bc_tags, bc_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array([2.0, 0.0] + [0.0 for l in range(level)]),
        right=lambda n_field: np.array([1.0, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["chezy", "newtonian"]},
    )
    mesh = Mesh.create_1d((-1, 20), 200)

    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)
    # io.generate_vtk(settings.output_dir, filename_fields = 'fields_intermediate.hdf5', filename_out='out_intermediate')




@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_smm_2d(mesh_type):
    level = 2
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 1.0, "C": 1.0, "nu": 0.1},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
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
    ic = IC.RP(
        left=lambda n_field: np.array(
            [2.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
        right=lambda n_field: np.array(
            [1.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
    )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy", "newtonian"]},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
def test_inflowoutflow_2d():
    level = 0
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 1.0, "C": 1.0, "nu": 0.1},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=1.0,
        output_snapshots=100,
    )

    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[1] = 0.36
    outflow_dict = {0: 1.0}

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="right", prescribe_fields= outflow_dict),
        ]
    )
    ic = IC.Constant(
        constants=lambda n_fields: np.array(
            [1.0, 0.1, 0.1] + [0.0 for i in range(n_fields - 3)]
        )
    )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"), "quad"
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
def test_steffler():
    level = 2
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 16.0, "nu": 0.0016},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=30.0,
        output_snapshots=100,
    )

    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/curved_open_channel/mesh_mid.msh"), "quad"
    )

    h0 = 0.061
    vin = -0.36
    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[1] = h0 * vin
    outflow_dict = {0: h0}

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="wall"),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )
    # ic = IC.Constant(constants=lambda n_fields:np.array([h0, 0.0] + [0. for i in range(n_fields-2)]))
    folder = "./output_lvl1_friction"
    map_fields = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}
    ic = IC.RestartFromHdf5(
        path_to_old_mesh=folder + "/mesh.hdf5",
        path_to_fields=folder + "/fields.hdf5",
        mesh_new=mesh,
        mesh_identical=True,
        map_fields=map_fields,
    )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"]},
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
def test_channel_with_hole_2d():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="output/", help="Output folder path"
    )
    parser.add_argument("--vel", type=float, default=0.7, help="Velocity for inflow")
    parser.add_argument(
        "--nu", type=float, default=1.0 / 1000.0, help="kinematic viscosity"
    )
    parser.add_argument(
        "--C", type=float, default=100, help="Chezy friction coefficient"
    )
    args = parser.parse_args()
    level = 0
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": args.C, "nu": args.nu},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=2.0,
        output_snapshots=100,
        output_dir=args.path,
    )

    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/channel_2d_hole/mesh_fine.msh"), "triangle"
    )

    # inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    # inflow_dict[1] = 0.36
    # outflow_dict = {0: 1.0}

    inflow_dict = {i: 0.0 for i in range(0, 2 * (1 + level) + 1)}
    inflow_dict[0] = 1.0
    inflow_dict[1] = args.vel
    outflow_dict = {}

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="hole"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = 1.0
        Q[1] = args.vel
        if x[0] < 0.5:
            Q[0] += 0.1 * x[1]
            Q[1] += 0.1 * x[1] * args.vel
        return Q

    ic = IC.UserFunction(ic_func)
    # ic = IC.Constant(
    #     constants=lambda n_fields: np.array(
    #         [1.0, 0.7] + [0.0 for i in range(n_fields - 2)]
    #     )
    # )

    folder = "./output_channel_turbulent_restart_3"
    map_fields = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    # ic = IC.RestartFromHdf5(
    #     path_to_old_mesh=folder + "/mesh.hdf5",
    #     path_to_fields=folder + "/fields.hdf5",
    #     mesh_new=mesh,
    #     mesh_identical=True,
    #     map_fields=map_fields,
    # )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy", "newtonian"]},
    )

    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    fvm_c_unsteady_semidiscete(
        mesh, model, settings, ode_solver_flux="RK1", ode_solver_source="RK1"
    )
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_grad_2d():
    mesh_type = "triangle"
    level = 2
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 1.0, "C": 16.0, "nu": 0.1},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=30.0,
        output_snapshots=300,
    )

    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array(
            [2.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
        right=lambda n_field: np.array(
            [1.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
    )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"]},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )
    mesh = GradientMesh.fromMesh(mesh)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    # io.generate_vtk(settings.output_dir)
    postprocessing.recover_3d_from_smm_as_vtk(
        model,
        settings.output_dir,
        os.path.join(settings.output_dir, "mesh.hdf5"),
        os.path.join(settings.output_dir, "fields.hdf5"),
        Nz=10,
        start_at_time=1.0,
    )

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d_crazy_basis():
    level = 1
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 1.0, "C": 1.0, "nu": 0.001},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=10.00,
        output_snapshots=50,
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array([2.0, 0.0] + [0.0 for l in range(level)]),
        right=lambda n_field: np.array([1.0, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian"]},
        basis=Basis(basis=test_basis),
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_c_solver(mesh_type):
    level = 1
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 1.0, "C": 10.0, "nu": 0.001},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=1.2,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_c",
    )

    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array(
            [2.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
        right=lambda n_field: np.array(
            [1.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
    )
    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy", "newtonian"]},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(

        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_c_unsteady_semidiscete(
        mesh, model, settings, ode_solver_flux="RK1", ode_solver_source="RK1"
    )
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_c_turbulence():
    level = 0
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 30.0, "nu": 0.01},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        #compute_dt=timestepping.adaptive(CFL=0.15),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=0.20,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_c",
    )

    print(f"number of available cpus: {os.cpu_count()}")

    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[0] = 0.1
    inflow_dict[1] = 0.*10
    outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="hole"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = 0.1
        Q[1] = 0.
        Q[2] = 0.
        # Q[3] = 0.1
        # if x[0] < 0.5:
        #     Q[0] += 0.1 * x[1]
        #     Q[1] += 0.1 * x[1] * args.vel
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["chezy"]},
        settings={},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"),
        # os.path.join(main_dir, "meshes/channel_2d_hole/mesh_coarse.msh"),
        # os.path.join(main_dir, "meshes/channel_2d_hole/mesh_coarse.msh"),
        "triangle",
     )
    # mesh = Mesh.from_hdf5( os.path.join(os.path.join(main_dir, settings.output_dir), "mesh.hdf5"))

    fvm_c_unsteady_semidiscete(
        mesh,
        model,
        settings,
        ode_solver_flux="RK1",
        ode_solver_source="RK1",
        rebuild_model=True,
        rebuild_mesh=True,
        rebuild_c=True,
    )
    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)

    io.generate_vtk(settings.output_dir)


if __name__ == "__main__":
    # test_smm_1d()
    # test_sindy_generate_reference_data()
    # test_smm_2d("quad")
    # test_smm_2d("triangle")
    # test_inflowoutflow_2d()
    # test_steffler()
    # test_channel_with_hole_2d()
    # test_smm_grad_2d()
    # test_smm_1d_crazy_basis()
    # test_c_solver('quad')
    test_c_turbulence()