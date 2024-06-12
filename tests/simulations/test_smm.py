import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
# from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import argparse


@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_wave():
    level = 0
    settings = Settings(
        name="ShallowMomentsWave",
        parameters={"g": 1.0, "lamda": 1.0, "nu": 1., "ex": 0.0, "rho":1., "ez": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=10*2*np.pi,
        output_snapshots=100,
        output_dir = 'outputs/wave'
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    def custom_ic(x):
        Q = np.zeros(level+2, dtype=float)
        Q[0] = np.sin(x[0]) + 2
        Q[1] = Q[0] 
        return Q

    def custom_ic_aux(x):
        Q = np.zeros(1, dtype=float)
        Q[0] = np.cos(x[0])
        return Q

    # ic = IC.Constant(constants = lambda n_fields: [1., 4/3, -1/4, -1/12 ] + [0 for i in range(level-2)])
    ic = IC.UserFunction(custom_ic)
    icaux = IC.UserFunction(custom_ic_aux)

    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=['dhdx'],
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        aux_initial_conditions=icaux,
        # settings={"eigenvalue_mode": "symbolic", "friction": ["slip", "newtonian", "inclined_plane"]},
        settings={"eigenvalue_mode": "symbolic", "friction": [] , "topography": True},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    mesh = Mesh.create_1d((0, 2*np.pi), 30)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_analytical():
    level = 3
    settings = Settings(
        name="ShallowMomentsAnalytical",
        parameters={"g": 1.0, "lamda": 1.0, "nu": 1., "ex": 1.0, "rho":1., "ez": 0.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.1),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=5.0,
        output_snapshots=100,
        output_dir = 'outputs/analytical'
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    # ic = IC.Constant(constants = lambda n_fields: [1., 4/3, -1/4, -1/12 ] + [0 for i in range(level-2)])
    ic = IC.Constant(constants = lambda n_fields: [1., 0.1,] + [0 for i in range(level)])
    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"eigenvalue_mode": "symbolic", "friction": ["slip", "newtonian", "inclined_plane"]},
        settings={"eigenvalue_mode": "symbolic", "friction": ["slip", "newtonian", "inclined_plane"]},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

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
        output_dir = 'outputs/output_smm_1d'
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
        high=lambda n_field: np.array([2.0, 0.0] + [0.0 for l in range(level)]),
        low=lambda n_field: np.array([1.0, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["chezy", "newtonian"]},
    )

    mesh = petscMesh.Mesh.create_1d((-1, 1), 100)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

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
        output_dir='outputs/output_smm_2d'
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
        high=lambda n_field: np.array(
            [2.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
        ),
        low=lambda n_field: np.array(
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
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
    )

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


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
    offset = level+1
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 16.0, "nu": 0.0016},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=30.0,
        output_snapshots=100,
        output_dir = 'outputs/steffler',
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/curved_open_channel/mesh_mid.msh")
    )

    h0 = 0.061
    vin = -0.36
    inflow_dict = {i: "0.0" for i in range(0, 2 * (1 + level) + 1)}
    inflow_dict[0] = f"{h0}"
    inflow_dict[1] = f"{h0 * vin}"
    outflow_dict = {}

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="wall", momentum_field_indices=[[1 + i, 1+offset+i] for i in range(level+1)], wall_slip=0.),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )
    ic = IC.Constant(constants=lambda n_fields:np.array([h0, 0.0] + [0. for i in range(n_fields-2)]))
    # folder = "./output_lvl1_friction"
    # map_fields = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}
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
        settings={"friction": ["chezy"]},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


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

        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_c_unsteady_semidiscete(
        mesh, model, settings, ode_solver_flux="RK1", ode_solver_source="RK1"
    )
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_restart_from_openfoam(level=1):
    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6)},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        #compute_dt=timestepping.adaptive(CFL=0.15),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=2.00,
        output_snapshots=100,
        output_clean_dir=False,
        output_dir=os.path.join(main_dir, "outputs/output_c"),
        callbacks=['ComputeFoamDeltaDataSet', 'LoadOpenfoam' ]
        # callbacks=['LoadOpenfoam', 'ComputeFoamDeltaDataSet']
        # callbacks=['ComputeFoamDeltaDataSet']
        # callbacks=['LoadOpenfoam']
        # callbacks=[]
    )

    print(f"number of available cpus: {os.cpu_count()}")

    velocity = 1.
    height = 0.5
    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[0] = height
    inflow_dict[1] = velocity * height
    outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = height
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
        settings={"friction": ["newtonian"]},
        # settings={},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    main_dir = os.getenv("SMS")
    # mesh = Mesh.load_gmsh(
    # #     os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"),
    # #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finest.msh"),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_coarse.msh"),
    #     # os.path.join(main_dir, 'meshes/channel_openfoam/mesh_coarse_2d.msh'),
    #     os.path.join(main_dir, 'meshes/simple_openfoam/mesh_2d_mid.msh'),
    # #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_mid.msh"),
    #     "triangle",
    #  )
    mesh = Mesh.from_hdf5( os.path.join(os.path.join(main_dir, settings.output_dir), "mesh.hdf5"))

    fvm_c_unsteady_semidiscete(
        mesh,
        model,
        settings,
        ode_solver_flux="RK1",
        ode_solver_source="RK1",
        rebuild_model=True,
        rebuild_mesh=False,
        rebuild_c=True,
    )

    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_restart_from_openfoam_prediction(level=1, coefs=[1.957, -16.829, 3.119, 8.151, 0, -4.466, 0.061, 3.444]):
    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShallowMoments2d",
        parameters={**{"g": 9.81,  "nu": 1.034*10**(-6)}, **{f"C{i+1}": coef for i, coef in enumerate(coefs)}},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        #compute_dt=timestepping.adaptive(CFL=0.15),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=2.00,
        output_snapshots=100,
        output_clean_dir=False,
        output_dir=os.path.join(main_dir, "outputs/output_c_prediction"),
        callbacks=[]
    )

    print(f"number of available cpus: {os.cpu_count()}")

    velocity = 1.
    height = 0.5
    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[0] = height
    inflow_dict[1] = velocity * height
    outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = height
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
        settings={"friction": ["newtonian", "sindy"]},
        # settings={},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
    #     os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finest.msh"),
        # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_coarse.msh"),
        # os.path.join(main_dir, 'meshes/channel_openfoam/mesh_coarse_2d.msh'),
        os.path.join(main_dir, 'meshes/simple_openfoam/mesh_2d_mid.msh'),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_mid.msh"),
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

    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_spline_strongbc_1d():
    level = 2
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 1., "C": 1.0, "nu": 0.001},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
        output_snapshots=100,
        output_dir='outputs/output_spline'
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
        basis=Basis(basis=Legendre_shifted(order=level)),
        # basis=Basis(basis=Spline()),
        # basis=Basis(basis=OrthogonalSplineWithConstant(degree=2, knots=[0, 0.01, 0.01, 0.5, 0.9, 1])),
        # basis=Basis(basis=OrthogonalSplineWithConstant(degree=2, knots=[0,0.1, 0.3, 0.5, 1, 1])),
        # basis=Basis(basis=OrthogonalSplineWithConstant(degree=1, knots=[0, 0.0, 0.1, 1])),
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)
    Q, Qaux, time = io.load_fields_from_hdf5(os.path.join(settings.output_dir, "fields.hdf5"))
    list_U, list_means, list_of_positions, Z, list_h = generate_velocity_profiles(Q, mesh.element_center, model, [np.array([x]) for x in np.arange(-0.9, 0.9, 0.2)])
    fig, ax = plt.subplots()
    dim = 0
    ax.plot(mesh.element_center[:,0], Q[:,0], label='h')
    def rescale(U, scale=0.5):
        return (U)/(np.max(U)-np.min(U)) * scale * np.max(np.abs(U))
        
    for i, pos in enumerate(list_of_positions):
        if i == 0:
            ax.plot(pos + rescale(list_U[i][dim]), list_h[i] * Z, color='green', label='velocity profile')
            # ax.plot(pos + list_means[i][dim] * np.ones_like(Z), list_h[i] * Z, color='red', label='mean profile')
            ax.plot(pos * np.ones_like(Z), list_h[i] * Z, color='black', label='zero velocity reference')
        else:
            ax.plot(pos + rescale(list_U[i][dim]), list_h[i] * Z, color='green')
            # ax.plot(pos + list_means[i][dim] * np.ones_like(Z), list_h[i] * Z, color='red')
            ax.plot(pos * np.ones_like(Z), list_h[i] * Z, color='black')
    ax.set_xlabel('x/velocity-profile')
    ax.set_ylabel('h/z')
    plt.title('Dam-break with spline basis')
    plt.legend()
    plt.show()

@pytest.mark.critical
@pytest.mark.unfinished
def test_restart_from_openfoam_plotter(level=1):
    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6)},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        #compute_dt=timestepping.adaptive(CFL=0.15),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=2.00,
        output_snapshots=100,
        output_clean_dir=False,
        output_dir=os.path.join(main_dir, "outputs/output_c"),
        callbacks=['ComputeFoamDeltaDataSet', 'LoadOpenfoam' ]
        # callbacks=['LoadOpenfoam', 'ComputeFoamDeltaDataSet']
        # callbacks=['ComputeFoamDeltaDataSet']
        # callbacks=['LoadOpenfoam']
        # callbacks=[]
    )

    print(f"number of available cpus: {os.cpu_count()}")

    velocity = 30.*1000./3600.
    height = 0.5
    inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict[0] = height
    inflow_dict[1] = velocity * height
    outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = height
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
        settings={"friction": ["newtonian"]},
        # settings={},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    main_dir = os.getenv("SMS")
    # mesh = Mesh.load_gmsh(
    # #     os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"),
    # #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finest.msh"),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_coarse.msh"),
    #     # os.path.join(main_dir, 'meshes/channel_openfoam/mesh_coarse_2d.msh'),
    #     os.path.join(main_dir, 'meshes/simple_openfoam/mesh_2d_mid.msh'),
    # #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_mid.msh"),
    #     "triangle",
    #  )
    mesh = Mesh.from_hdf5( os.path.join(os.path.join(main_dir, settings.output_dir), "mesh.hdf5"))

    # fvm_c_unsteady_semidiscete(
    #     mesh,
    #     model,
    #     settings,
    #     ode_solver_flux="RK1",
    #     ode_solver_source="RK1",
    #     rebuild_model=False,
    #     rebuild_mesh=False,
    #     rebuild_c=True,
    # )

    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_petsc(mesh_type):
    level = 1
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 1.0, "C": 10.0, "nu": 0.001},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=2.0,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_jax",
    )

    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    # ic = IC.RadialDambreak(
    #     high=lambda n_field: np.array(
    #         [2.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
    #     ),
    #     low=lambda n_field: np.array(
    #         [1.0, 0.0, 0.0] + [0.0 for l in range(2 * level)]
    #     ),
    # )

    def custom_ic(x):
        Q = np.zeros(1+2*(level+1), dtype=float)
        Q[0] = 2 * x[1] * (x[0] < 0.5) + 2
        return Q

    ic = IC.UserFunction(function = custom_ic)
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
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)))
    mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)))

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

@pytest.mark.critical
@pytest.mark.unfinished
def test_enforce_w_bc():
    level = 1
    settings = Settings(
        name="ShallowMoments2d",
        parameters={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6)},
        # parameters={"g": 9.81, "C": 3.0, "nu": 1.034*10**(-1)},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=.45),
        time_end=10.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_jax2",
    )

    # inflow_dict = {i: "0.0" for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict = {}
    # inflow_dict[0] = "0.1*sympy.sin(2*3.14*time)+ 1.0 + 0.1*X[1]"
    h0 = "Q[0]"
    hu0 = 0.1
    w0 = 0.0
    dxh = f"(Q[0] - {h0})/(2*dX)"
    # dxh = f"0"
    dxhu = f"(Q[1] - {hu0})/(2*dX)"
    nominator = f"-({w0} + Q[2] - 3/2 * dX * {dxhu} + 3/2 * dX * {hu0}/{h0} * {dxh})"
    denominator = f"1/2 * dX / {h0} * {dxh} + 1"
    inflow_dict[0] = f"{1.0}"
    # inflow_dict[0] = f"{h0}"
    inflow_dict[1] = f"{hu0}"
    # inflow_dict[2] = f'{nominator}/{denominator}'
    inflow_dict.update({i: "0.0" for i in range(2, 3+2*level)})
    # inflow_dict[2] = f'{nominator}/{denominator}'
    # outflow_dict = {0: f"{h0}"}
    # outflow_dict = {3: "0.0"}
    # outflow_dict = {4: "0.0"}
    outflow_dict = {}

    # data_dict = {}
    # data_dict[0] = np.linspace(1, 2, 10)
    # data_dict[1] = np.linspace(0.1, 0.1, 10)
    # data_dict[2] = np.linspace(0, 0, 10)
    # data_dict[3] = np.linspace(0, 0, 10)
    # data_dict[4] = np.linspace(0, 0, 10)
    # timeline = np.linspace(0, 10, 10)

    offset = level+1
    bcs = BC.BoundaryConditions(
        [
            # BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="outflow", prescribe_fields= outflow_dict),
            BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.FromData(physical_tag='left', prescribe_fields=data_dict, timeline=timeline),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields= outflow_dict),
            # BC.Wall(physical_tag="top", momentum_field_indices=[[1 + i*offset, 1+offset+i*offset] for i in range(level)], wall_slip=0.),
            # BC.Wall(physical_tag="bottom", momentum_field_indices=[[1 + i*offset, 1+offset+i*offset] for i in range(level)], wall_slip=0.),
            # BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
            # BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            BC.Extrapolation(physical_tag="top"),
            BC.Extrapolation(physical_tag="bottom"),
            BC.Extrapolation(physical_tag="right"),
            # BC.Extrapolation(physical_tag="left"),
            # BC.Extrapolation(physical_tag="right"),
        ]
    )


    ic = IC.Constant(
        constants=lambda n_fields: np.array(
            [1.0, 0.0, 0.0] + [0.0 for i in range(n_fields - 3)]
        )
    )

    def custom_ic(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = 2*np.ones_like(x[0])
        Q[1] = x[0]
        Q[2] = x[1]
        Q[3] = x[0]**2 
        Q[4] = x[1]**2 
        return Q

    # ic = IC.UserFunction(custom_ic)


    model = ShallowMoments2d(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": []},
        settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )


    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/quad_2d/mesh_fine.msh"))
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/simple_openfoam/mesh_2d_mid.msh"))
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/simple_openfoam/mesh_2d_finest.msh"))
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"))
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finer.msh"))
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finest.msh"))

    print(mesh.boundary_conditions_sorted_names)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    # io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'), field_names=[f'Q_{i}' for i in range(model.n_fields)], aux_field_names=['dQdx', 'dQdy'] + [f'phi_{i}' for i in range(model.n_fields)])
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))
    # postprocessing.recover_3d_from_smm_as_vtk(
    #     model,
    #     settings.output_dir,
    #     os.path.join(settings.output_dir, "mesh.hdf5"),
    #     os.path.join(settings.output_dir, "fields.hdf5"),
    #     Nz=10,
    #     start_at_time=1.0,
    # )

# @pytest.mark.critical
@pytest.mark.unfinished
def test_ijshs24():
    level = 1
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6)},
        # parameters={"g": 9.81, "C": 3.0, "nu": 1.034*10**(-1)},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=.9),
        time_end=10.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_ijshs24",
    )

    # inflow_dict = {i: "0.0" for i in range(1, 2 * (1 + level) + 1)}
    inflow_dict = {}
    # inflow_dict[0] = "0.1*sympy.sin(2*3.14*time)+ 1.0 + 0.1*X[1]"
    h0 = "Q[0]"
    hu0 = 0.1
    w0 = 0.0
    dxh = f"(Q[0] - {h0})/(2*dX)"
    # dxh = f"0"
    dxhu = f"(Q[1] - {hu0})/(2*dX)"
    nominator = f"-({w0} + Q[2] - 3/2 * dX * {dxhu} + 3/2 * dX * {hu0}/{h0} * {dxh})"
    denominator = f"1/2 * dX / {h0} * {dxh} + 1"
    inflow_dict[0] = f"{1.0}"
    # inflow_dict[0] = f"{h0}"
    inflow_dict[1] = f"{hu0}"
    # inflow_dict[2] = f'{nominator}/{denominator}'
    inflow_dict.update({i: "0.0" for i in range(2, 2+level)})
    # inflow_dict[2] = f'{nominator}/{denominator}'
    # outflow_dict = {0: f"{h0}"}
    # outflow_dict = {3: "0.0"}
    # outflow_dict = {4: "0.0"}
    outflow_dict = {}

    data_dict = {}
    data_dict[0] = np.linspace(1, 2, 10)
    data_dict[1] = np.linspace(0.1, 0.1, 10)
    data_dict[2] = np.linspace(0, 0, 10)
    timeline = np.linspace(0, 10, 10)

    offset = level+1
    bcs = BC.BoundaryConditions(
        [
            # BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="outflow", prescribe_fields= outflow_dict),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            BC.FromData(physical_tag='left', prescribe_fields=data_dict, timeline=timeline),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields= outflow_dict),
            # BC.Wall(physical_tag="top", momentum_field_indices=[[1 + i*offset, 1+offset+i*offset] for i in range(level)], wall_slip=0.),
            # BC.Wall(physical_tag="bottom", momentum_field_indices=[[1 + i*offset, 1+offset+i*offset] for i in range(level)], wall_slip=0.),
            # BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
            # BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            # BC.Extrapolation(physical_tag="top"),
            # BC.Extrapolation(physical_tag="bottom"),
            BC.Extrapolation(physical_tag="right"),
            # BC.Extrapolation(physical_tag="left"),
            # BC.Extrapolation(physical_tag="right"),
        ]
    )


    ic = IC.Constant(
        constants=lambda n_fields: np.array(
            [1.0, 0.0] + [0.0 for i in range(n_fields - 2)]
        )
    )

    def custom_ic(x):
        Q = np.zeros(3+2*level, dtype=float)
        Q[0] = 2*np.ones_like(x[0])
        Q[1] = x[0]
        Q[3] = x[0]**2 
        return Q

    # ic = IC.UserFunction(custom_ic)


    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": []},
        settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )


    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((0, 10), 500)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    # io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'), field_names=[f'Q_{i}' for i in range(model.n_fields)], aux_field_names=['dQdx', 'dQdy'] + [f'phi_{i}' for i in range(model.n_fields)])
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))
    # postprocessing.recover_3d_from_smm_as_vtk(
    #     model,
    #     settings.output_dir,
    #     os.path.join(settings.output_dir, "mesh.hdf5"),
    #     os.path.join(settings.output_dir, "fields.hdf5"),
    #     Nz=10,
    #     start_at_time=1.0,
    # )

# @pytest.mark.critical
@pytest.mark.unfinished
def test_eccomas_hyperbolicity():
    level = 2
    Cf = 1/200.
    Cr = 0.00
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.1),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=10,
        output_snapshots=100,
        output_dir = 'outputs/eccomas/SMMWE'
    )

    offset = level+1
    bcs = BC.BoundaryConditions(
        [
            # BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            # BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
            BC.Extrapolation(physical_tag="left"),
            BC.Extrapolation(physical_tag="right"),
        ]
    )


    P11_0 = 0.00
    # a1_high = np.sqrt(0.02 * P11_0 * 3 )
    # a1_low = np.sqrt(0.01 * P11_0* 3 )
    a1_high = 0
    a1_low = 0
    # a2_high = np.sqrt(0.02 * P11_0 * 5 )
    # a2_low = np.sqrt(0.01 * P11_0* 5 )
    a2_high = 0
    a2_low = 0
    # a1_high = np.sqrt( P11_0 * 3 )
    # a1_low = np.sqrt( P11_0* 3 )
    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    #     low=lambda n_field: np.array([.01, 0.0] + [0.0 for l in range(level)]),
    # )
    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0, a1_high] + [0.0 for l in range(level-1)]),
    #     low=lambda n_field: np.array([.01, 0.0, a1_low] + [0.0 for l in range(level-1)]),
    # )
    ic = IC.RP(
        high=lambda n_field: np.array([0.02, 0.0, a1_high, a2_high] + [0.0 for l in range(level-2)]),
        low=lambda n_field: np.array([.01, 0.0, a1_low, a2_high] + [0.0 for l in range(level-2)]),
    )

    # model_SMMWS = ShallowMomentsSSFEnergy(
    model_SMMWS = ShallowMomentsSSF(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ['chezy']},
        basis=BasisNoHOM(basis=Legendre_shifted(order=level)),
    )
    # print(f"Eigenvalues SSF:")
    # print(model_SSF.sympy_eigenvalues)
    # print("\n")
    
    # model_SMM = ShallowMoments(
    #     dimension=1,
    #     fields=2 + level,
    #     aux_fields=0,
    #     parameters=settings.parameters,
    #     boundary_conditions=bcs,
    #     initial_conditions=ic,
    #     settings={"friction": ['chezy']},
    #     basis=Basis(basis=Legendre_shifted(order=level)),
    # )


    def ic_func(x):
        Q = np.zeros(6, dtype=float)
        Q[0] = np.where(x[0]<0.0, 0.02, 0.01)
        P11 = P11_0
        # P11 = 0.
        P22 = P11
        P12 = 0
        R11 = Q[0] * P11
        R12 = Q[0] * P12
        R22 = Q[0] * P22
        u = 0.
        v = 0.
        Q[3] =  (1/2 * R11 + 1/2 * Q[0] * u * u)
        Q[4] =  (1/2 * R12 + 1/2 * Q[0] * u * v)
        Q[5] =  (1/2 * R22 + 1/2 * Q[0] * v * v)
        return Q

    ic = IC.UserFunction(ic_func)


    model_SSF = ShearShallowFlowPathconservative(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"]},
        # settings={"friction": ["newtonian"]},
        # settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    model_SSF.name = 'ShearShallowFlowPathconservative'

    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = np.where(x[0]<0.0, 0.02, 0.01)
        Q[2] =  P11_0
        return Q

    ic = IC.UserFunction(ic_func)

    model_SSF_orig = ShearShallowFlow(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        settings={"friction": ["chezy"]},
        # settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    model_SSF_orig.name = 'ShearShallowFlow'

    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = np.where(x[0]<0.0, 0.02, 0.01)
        return Q

    ic = IC.UserFunction(ic_func)
    
    model_SSF_energy = ShearShallowFlowEnergy(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        settings={"friction": ["chezy"]},
        # settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    model_SSF_energy.name = 'ShearShallowFlowEnergy'

    
    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    print('SMM-WS')
    settings.output_dir = 'outputs/eccomas/SMMWE'
    jax_fvm_unsteady_semidiscrete(
        mesh, model_SMMWS, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

    # print('SMM')
    # settings.output_dir = 'outputs/eccomas/SMM'
    # jax_fvm_unsteady_semidiscrete(
    #     mesh, model_SMM, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    # )
    # io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

    print('SSF')
    settings.output_dir = 'outputs/eccomas/SSF'
    jax_fvm_unsteady_semidiscrete(
        mesh, model_SSF, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

    print('SSF orig')
    settings.output_dir = 'outputs/eccomas/SSF_orig'
    jax_fvm_unsteady_semidiscrete(
        mesh, model_SSF_orig, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

    print('SSF energy')
    settings.output_dir = 'outputs/eccomas/SSF_energy'
    jax_fvm_unsteady_semidiscrete(
        mesh, model_SSF_energy, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))



if __name__ == "__main__":

    ### TESTED

    # test_smm_1d()
    # test_smm_2d("quad")
    # test_smm_2d("triangle")
    # test_steffler()

    # test_ijshs24()
    test_eccomas_hyperbolicity()


    ### UNTESTED
    # test_sindy_generate_reference_data()
    # test_inflowoutflow_2d()
    # test_steffler()
    # test_channel_with_hole_2d()
    # test_smm_grad_2d()
    # test_smm_1d_crazy_basis()
    # test_c_solver('quad')
    # test_restart_from_openfoam()
    # test_restart_from_openfoam_prediction()
    # test_restart_from_openfoam_plotter()
    # test_spline_strongbc_1d()
    # test_smm_analytical()
    # test_smm_wave()
    # test_petsc('quad')
    # test_petsc('triangle')
    # test_enforce_w_bc()