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
def test_smm_1d():
    level = 4
    settings = Settings(name = "ShallowMoments", momentum_eqns = [1] + [2+l for l in range(level)], parameters = {'g':1.0, 'C':1.0, 'nu':0.1}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.9), time_end = 1., output_snapshots = 100)


    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP(left=lambda n_field: np.array([2., 0.] + [0. for l in range(level)]), right=lambda n_field: np.array([1., 0.]+ [0. for l in range(level)]) )
    model = ShallowMoments(
        dimension=1,
        fields=2+level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'eigenvalue_mode': 'symbolic', 'friction': ['chezy', 'newtonian']},
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_smm_2d(mesh_type):
    level = 2
    settings = Settings(name = "ShallowMoments2d", momentum_eqns = [1, 2] + [3+l  for l in range(2*level)], parameters = {'g':1.0, 'C': 1., 'nu': 0.1}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 1., output_snapshots = 100)


    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to) for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)]
    )
    ic = IC.RP(left=lambda n_field: np.array([2., 0., 0.] + [0. for l in range(2*level)]), right=lambda n_field: np.array([1., 0., 0.]+ [0. for l in range(2*level)]) )
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': ['chezy', 'newtonian']},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_fine.msh".format(mesh_type)),
        mesh_type
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_inflowoutflow_2d():
    level = 0
    settings = Settings(name = "ShallowMoments2d", momentum_eqns = [1, 2] + [3+l  for l in range(2*level)], parameters = {'g':1.0, 'C': 1., 'nu': 0.1}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 1., output_snapshots = 100)


    inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    inflow_dict[1] = 0.36
    outflow_dict = {0: 1.0}

    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag='top'), BC.Wall(physical_tag='bottom'), BC.InflowOutflow(physical_tag='left', prescribe_fields=inflow_dict), BC.InflowOutflow(physical_tag='right', prescribe_fields=outflow_dict)]
    )
    ic = IC.Constant(constants=lambda n_fields:np.array([1.0, 0.36] + [0. for i in range(n_fields-2)]))
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': []},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        'quad'
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_steffler():
    level = 2
    settings = Settings(name = "ShallowMoments2d", momentum_eqns = [1, 2] + [3+l  for l in range(2*level)], parameters = {'g':9.81, 'C': 16., 'nu': 0.0016}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 30., output_snapshots = 100)

    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/curved_open_channel/mesh_mid.msh"),
        'quad'
    )


    h0 = 0.061
    vin = -0.36
    inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    inflow_dict[1] = h0 * vin
    outflow_dict = {0: h0}

    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag='wall'), BC.InflowOutflow(physical_tag='inflow', prescribe_fields=inflow_dict), BC.InflowOutflow(physical_tag='outflow', prescribe_fields=outflow_dict)]
    )
    # ic = IC.Constant(constants=lambda n_fields:np.array([h0, 0.0] + [0. for i in range(n_fields-2)]))
    folder = './output_lvl1_friction'
    map_fields = {0:0, 1:1, 2:2, 3:4, 4:5}
    ic = IC.RestartFromHdf5(path_to_old_mesh=folder + '/mesh.hdf5', path_to_fields= folder + '/fields.hdf5', mesh_new = mesh, mesh_identical=True, map_fields=map_fields )
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': ['chezy']},
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
def test_inflowoutflow_2d():
    level = 0
    settings = Settings(name = "ShallowMoments2d", momentum_eqns = [1, 2] + [3+l  for l in range(2*level)], parameters = {'g':1.0, 'C': 1., 'nu': 0.1}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 1., output_snapshots = 100)


    inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    inflow_dict[1] = 0.36
    outflow_dict = {0: 1.0}

    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag='top'), BC.Wall(physical_tag='bottom'), BC.InflowOutflow(physical_tag='left', prescribe_fields=inflow_dict), BC.InflowOutflow(physical_tag='right', prescribe_fields=outflow_dict)]
    )
    ic = IC.Constant(constants=lambda n_fields:np.array([1.0, 0.36] + [0. for i in range(n_fields-2)]))
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': []},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        'quad'
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)

@pytest.mark.critical
@pytest.mark.unfinished
def test_steffler():
    level = 2
    settings = Settings(name = "ShallowMoments2d", momentum_eqns = [1, 2] + [3+l  for l in range(2*level)], parameters = {'g':9.81, 'C': 16., 'nu': 0.0016}, reconstruction = recon.constant, num_flux = flux.LLF(), compute_dt = timestepping.adaptive(CFL=0.45), time_end = 30., output_snapshots = 100)


    inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    inflow_dict[1] = 0.36
    outflow_dict = {0: 1.0}

    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag='wall'), BC.InflowOutflow(physical_tag='inflow', prescribe_fields=inflow_dict), BC.InflowOutflow(physical_tag='outflow', prescribe_fields=outflow_dict)]
    )
    ic = IC.Constant(constants=lambda n_fields:np.array([1.0, 0.0] + [0. for i in range(n_fields-2)]))
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': []},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/curved_open_channel/mesh_mid.msh"),
        'quad'
    )


    h0 = 0.061
    vin = -0.36
    inflow_dict = {i: 0. for i in range(1, 2*(1+level)+1)}
    inflow_dict[1] = h0 * vin
    outflow_dict = {0: h0}

    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag='wall'), BC.InflowOutflow(physical_tag='inflow', prescribe_fields=inflow_dict), BC.InflowOutflow(physical_tag='outflow', prescribe_fields=outflow_dict)]
    )
    # ic = IC.Constant(constants=lambda n_fields:np.array([h0, 0.0] + [0. for i in range(n_fields-2)]))
    folder = './output_lvl1_friction'
    map_fields = {0:0, 1:1, 2:2, 3:4, 4:5}
    ic = IC.RestartFromHdf5(path_to_old_mesh=folder + '/mesh.hdf5', path_to_fields= folder + '/fields.hdf5', mesh_new = mesh, mesh_identical=True, map_fields=map_fields )
    model = ShallowMoments2d(
        dimension=2,
        fields=3+2*level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={'friction': ['chezy']},
    )


    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


if __name__ == "__main__":
    # test_smm_1d()
    # test_smm_2d("quad")
    # test_smm_2d("triangle")
    # test_inflowoutflow_2d()
    # test_steffler()
    test_channel_with_hole_2d()