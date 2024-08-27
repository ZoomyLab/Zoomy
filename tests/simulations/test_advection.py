import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
import library.mesh.mesh as petscMesh


@pytest.mark.critical
@pytest.mark.unfinished
def test_advection_1d():
    settings = Settings(name = "Advection", parameters = {'p0':-1.0}, reconstruction = recon.constant, num_flux = flux.LF(), compute_dt = timestepping.adaptive(CFL=0.9), time_end = 1., output_snapshots = 100)


    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP()
    model = Advection(
        dimension=1,
        fields=1,
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
def test_advection_2d(mesh_type):
    settings = Settings(name = "Advection",  parameters = {'px':1.0, 'py':1.0}, reconstruction = recon.constant, num_flux = flux.LF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 1.0, output_snapshots = 100)


    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP2d()
    model = Advection(
        dimension=2,
        fields=2,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["tetra"])
def test_advection_3d(mesh_type):
    settings = Settings(name = "Advection",  parameters = {'px':0.0, 'py':0.0, 'pz':1.0}, reconstruction = recon.constant, num_flux = flux.LF(), compute_dt = timestepping.constant(dt=0.01), time_end = .1, output_snapshots = 10)


    bc_tags = ["left", "right", "top", "bottom", "front", "back"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top", "back", "front"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP()
    model = Advection(
        dimension=3,
        fields=2,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_3d/mesh_coarse.msh".format(mesh_type)),
        mesh_type
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_periodic_bc(mesh_type):
    settings = Settings(
        name="Advection",
        parameters={"px": -2./8, "py": -0./8.,},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.constant(dt=1.),
        time_end=10.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_advection",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
            BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
        ]
    )

    ic = IC.RP2d(
        low=lambda n_fields: np.array(
            [0.1] + [0.0 for i in range(n_fields - 3)]
        ),
        high=lambda n_fields: np.array(
            [0.2] + [0.0 for i in range(n_fields - 3)]
        ),
        jump_position_x = 0,
        jump_position_y = 1.
    )

    model = Advection(
        dimension=2,
        fields=1,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
    )


    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)))

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_reconstruction(mesh_type):
    settings = Settings(
        name="Advection",
        parameters={"px": -1./10, "py": -1./10.,},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=1.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/output_advection",
    )

    bcs = BC.BoundaryConditions(
        [
            # BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            # BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
            BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
            # BC.Extrapolation(physical_tag="bottom"),
            # BC.Extrapolation(physical_tag="top"),
        ]
    )


    # ic = IC.RP2d(
    #     low=lambda n_fields: np.array(
    #         [0.1] + [0.0 for i in range(n_fields - 3)]
    #     ),
    #     high=lambda n_fields: np.array(
    #         [0.2] + [0.0 for i in range(n_fields - 3)]
    #     ),
    #     jump_position_x = 0,
    #     jump_position_y = 1.
    # )

    # ic = IC.RadialDambreak(
    #     low=lambda n_fields: np.array(
    #         [0.1] + [0.0 for i in range(n_fields - 3)]
    #     ),
    #     high=lambda n_fields: np.array(
    #         [0.2] + [0.0 for i in range(n_fields - 3)]
    #     ),
    #     radius = 0.2,
    # )

    def custom_ic(x):
        Q = np.zeros(1, dtype=float)
        Q[0] = 2.*x[1]
        return Q

    ic = IC.UserFunction(custom_ic)

    model = Advection(
        dimension=2,
        fields=1,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
    )


    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/{}_2d/mesh.msh".format(mesh_type)))

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'), field_names=['Q'], aux_field_names=['dQdx', 'dQdy', 'phi'])



if __name__ == "__main__":
    # test_advection_1d()
    # test_advection_2d("quad")
    # test_advection_2d("triangle")
    # test_advection_3d("tetra")
    # test_periodic_bc("quad")
    test_reconstruction("quad")