import numpy as np
import pytest
from types import SimpleNamespace

from library.solver import *
from library.model import *
import library.initial_conditions as IC
import library.boundary_conditions as BC
from library.ode import RK1

@pytest.mark.critical
@pytest.mark.unfinished
def test_advection_1d():
    settings = Settings(name = "Advection", momentum_eqns = [0], parameters = {'p0':-1.0}, reconstruction = recon.constant, num_flux = flux.LF, compute_dt = timestepping.constant(dt=0.1), time_end = .1, output_timesteps = 10)


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
    mesh = Mesh.create_1d((-1, 1), 10)


    output, mesh = fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    print(output[0][0])
    print(output[-1][0])

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_advection_2d(mesh_type):
    settings = Settings(name = "Advection", momentum_eqns = [0, 1], parameters = {'px':0.0, 'py':0.0}, reconstruction = recon.constant, num_flux = flux.LF, compute_dt = timestepping.constant(dt=0.01), time_end = .1, output_timesteps = 10)


    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP()
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


    output, mesh = fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    print(output[0][0])
    print(output[-1][0])


if __name__ == "__main__":
    test_advection_1d()
    # test_advection_2d("quad")
    # test_advection_2d("triangle")