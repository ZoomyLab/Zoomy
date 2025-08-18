import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io


@pytest.mark.critical
@pytest.mark.unfinished
def test_advection_1d():
    settings = Settings(
        name="Advection",
        momentum_eqns=[0],
        parameters={"p0": -1.0},
        reconstruction=recon.constant,
        num_flux=flux.LF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
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
    ic = IC.RP()
    model = Advection(
        dimension=1,
        fields=1,
        aux_variables=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output.directory)


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_advection_2d(mesh_type):
    settings = Settings(
        name="Advection",
        momentum_eqns=[0, 1],
        parameters={"px": 1.0, "py": 1.0},
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
    ic = IC.RP2d()
    model = Advection(
        dimension=2,
        fields=2,
        aux_variables=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    main_dir = os.getenv("ZOOMY_DIR")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output.directory)


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["tetra"])
def test_advection_3d(mesh_type):
    settings = Settings(
        name="Advection",
        momentum_eqns=[0, 1],
        parameters={"px": 0.0, "py": 0.0, "pz": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LF(),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=0.1,
        output_snapshots=10,
    )

    bc_tags = ["left", "right", "top", "bottom", "front", "back"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top", "back", "front"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP()
    model = Advection(
        dimension=3,
        fields=2,
        aux_variables=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    main_dir = os.getenv("ZOOMY_DIR")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_3d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output.directory)


if __name__ == "__main__":
    test_advection_1d()
    # test_advection_2d("quad")
    # test_advection_2d("triangle")
    # test_advection_3d("tetra")
