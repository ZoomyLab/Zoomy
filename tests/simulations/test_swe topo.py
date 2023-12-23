import numpy as np
import pytest
from types import SimpleNamespace

from library.solver import *
from library.model import *
import library.initial_conditions as IC
import library.boundary_conditions as BC
from library.ode import RK1
import library.io as io

@pytest.mark.critical
@pytest.mark.unfinished
def test_swetopo_1d():
    settings = Settings(name = "ShallowWaterTopo", momentum_eqns = [1], parameters = {'g':1.0}, reconstruction = recon.constant, num_flux = flux.LLF_wb(), compute_dt = timestepping.adaptive(CFL=0.9), time_end = 1., output_snapshots = 100)


    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = 1. -0.1*x[0]
        Q[2] = 0.1*x[0]
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
    settings = Settings(name = "ShallowWater2d", momentum_eqns = [1, 2], parameters = {'g':1.0, 'C':1.0}, reconstruction = recon.constant, num_flux = flux.LLF(), nc_flux=nonconservative_flux.zero(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 1., output_snapshots = 100)


    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    def ic_func(x):
        Q = np.zeros(4, dtype=float)
        Q[0] = 1. -0.1*x[0]
        Q[3] = 0.1*x[0]
        return Q
    ic = IC.UserFunction(ic_func)
    model = ShallowWaterTopo2d(
        dimension=2,
        fields=4,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': ['chezy']},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
        mesh_type
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)




if __name__ == "__main__":
    # test_swetopo_1d()
    test_swetopo_2d("quad")
    # test_swetopo_2d("triangle")