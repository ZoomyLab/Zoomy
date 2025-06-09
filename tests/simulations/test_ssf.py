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
import library.mesh.mesh as petscMesh
import argparse


@pytest.mark.critical
@pytest.mark.unfinished
def test_ssf():
    Lx = 1.3
    Cf = 0.0036
    h0 = 7.98 * 10 ** (-3)
    a = 0.05
    g = 9.81
    theta = 0.05011
    phi = 22.76
    Cr = 0.00035

    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShearShallowFlow",
        parameters={"g": g, "Cr": Cr, "theta": theta, "phi": phi},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.01),
        time_end=25.00,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=os.path.join(main_dir, "outputs/output_ssf"),
        callbacks=[],
        debug=True,
        profiling=False,
    )

    # velocity = 1.*1000./3600.
    # height = 0.5
    # inflow_dict = {i: 0.0 for i in range(0, 3)}
    # inflow_dict[0] = height
    # inflow_dict[1] = velocity * height
    # outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            BC.Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]
    )

    def ic_func(x):
        Q = np.zeros(3, dtype=float)
        # Q[0] = height
        Q[0] = h0 * (1 + a * np.sin(2 * np.pi * x[0] / Lx))
        Q[1] = h0 * np.sqrt(g * h0 * np.tan(theta) / Cf)
        Q[2] = phi * h0**2 / 2
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShearShallowFlow(
        dimension=1,
        fields=3,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        # settings={"friction": ["friction_paper"]},
    )
    main_dir = os.getenv("SMS")

    # mesh, _ = petscMesh.Mesh.create_1d([0, Lx], 100)
    mesh = petscMesh.Mesh.create_1d((0, Lx), 100)

    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )

    # io.generate_vtk(settings.output_dir)
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


def test_ssf_energy():
    Lx = 1.3
    Cf = 0.0036
    h0 = 7.98 * 10 ** (-3)
    a = 0.05
    g = 9.81
    theta = 0.05011
    phi = 22.76
    Cr = 0.00035

    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShearShallowFlowEnergy",
        parameters={"g": g, "Cr": Cr, "theta": theta, "phi": phi},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.01),
        time_end=25.00,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=os.path.join(main_dir, "outputs/output_ssf"),
        callbacks=[],
        debug=True,
        profiling=False,
    )

    # velocity = 1.*1000./3600.
    # height = 0.5
    # inflow_dict = {i: 0.0 for i in range(0, 3)}
    # inflow_dict[0] = height
    # inflow_dict[1] = velocity * height
    # outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            BC.Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]
    )

    def ic_func(x):
        Q = np.zeros(4, dtype=float)
        # Q[0] = height
        Q[0] = h0 * (1 + a * np.sin(2 * np.pi * x[0] / Lx))
        Q[1] = h0 * np.sqrt(g * h0 * np.tan(theta) / Cf)
        Q[2] = phi * h0**2 / 2
        h = Q[0]
        u = Q[1] / h
        P11 = Q[2]
        Q[3] = h * (u**2 + g * h + P11)
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShearShallowFlowEnergy(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        settings={"friction": ["friction_paper"]},
    )
    main_dir = os.getenv("SMS")

    # mesh, _ = petscMesh.Mesh.create_1d([0, Lx], 100)
    mesh = petscMesh.Mesh.create_1d((0, Lx), 100)

    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RKimplicit
    )

    # io.generate_vtk(settings.output_dir)
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


def test_ssf_pathconservative():
    Lx = 1.3
    Cf = 0.0036
    h0 = 7.98 * 10 ** (-3)
    a = 0.05
    g = 9.81
    theta = 0.05011
    phi = 22.76
    Cr = 0.00035

    main_dir = os.getenv("SMS")
    settings = Settings(
        parameters={"g": g, "Cf": Cf, "theta": theta, "phi": phi, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        # nc_flux = nonconservative_flux.zero(),
        nc_flux=nonconservative_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.01),
        time_end=26.99,
        output_snapshots=300,
        output_clean_dir=True,
        output_dir=os.path.join(main_dir, "outputs/output_ssf"),
        callbacks=[],
        debug=True,
        profiling=False,
    )

    # velocity = 1.*1000./3600.
    # height = 0.5
    # inflow_dict = {i: 0.0 for i in range(0, 3)}
    # inflow_dict[0] = height
    # inflow_dict[1] = velocity * height
    # outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            BC.Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]
    )

    def ic_func(x):
        Q = np.zeros(6, dtype=float)
        # Q[0] = height
        Q[0] = h0 * (1 + a * np.sin(2 * np.pi * x[0] / Lx))
        Q[1] = Q[0] * np.sqrt(g * h0 * np.tan(theta) / Cf)
        Q[2] = 0
        P11 = 1 / 2 * phi * Q[0] ** 2
        P22 = P11
        P12 = 0
        R11 = Q[0] * P11
        R12 = Q[0] * P12
        R22 = Q[0] * P22
        u = Q[1] / Q[0]
        v = Q[2] / Q[0]
        Q[3] = 1 / 2 * R11 + 1 / 2 * Q[0] * u * u
        Q[4] = 1 / 2 * R12 + 1 / 2 * Q[0] * u * v
        Q[5] = 1 / 2 * R22 + 1 / 2 * Q[0] * v * v
        return Q

    def ic_func(x):
        Q = np.zeros(6, dtype=float)
        Q[0] = np.where(x[0] < 0.5, 0.02, 0.01)
        # P11 = 10**(-1)
        P11 = 0.0
        P22 = P11
        P12 = 0
        R11 = Q[0] * P11
        R12 = Q[0] * P12
        R22 = Q[0] * P22
        u = 0.0
        v = 0.0
        Q[3] = 1 / 2 * R11 + 1 / 2 * Q[0] * u * u
        Q[4] = 1 / 2 * R12 + 1 / 2 * Q[0] * u * v
        Q[5] = 1 / 2 * R22 + 1 / 2 * Q[0] * v * v
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShearShallowFlowPathconservative(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        # settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    main_dir = os.getenv("SMS")

    # mesh, _ = petscMesh.Mesh.create_1d([0, Lx], 100)
    mesh = petscMesh.Mesh.create_1d((0, Lx), 100)

    # fvm_unsteady_semidiscrete(mesh, model, settings, RK1)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )

    # io.generate_vtk(settings.output_dir)
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


@pytest.mark.critical
@pytest.mark.unfinished
def test_ssf_2d():
    theta = 0.05011
    phi = 22.76
    Cr = 0.00035

    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShearShallowFlow2d",
        parameters={"g": g, "Cr": Cr, "theta": theta, "phi": phi},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        # compute_dt=timestepping.constant(dt=0.01),
        time_end=2.00,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=os.path.join(main_dir, "outputs/output_ssf"),
        callbacks=[],
        debug=False,
        profiling=False,
    )

    velocity = 36.0 * 1000.0 / 3600.0
    height = 0.5
    inflow_dict = {i: "0.0" for i in range(0, 6)}
    inflow_dict[0] = f"{height}"
    inflow_dict[1] = f"{velocity} * {height}"
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
            # BC.Wall(physical_tag="inflow"),
            # BC.Wall(physical_tag="outflow"),
        ]
    )

    def ic_func(x):
        Q = np.zeros(6, dtype=float)
        Q[0] = height
        Q[1] = 0.0
        Q[2] = 0.0
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShearShallowFlow2d(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        settings={"friction": ["friction_paper"]},
        # settings={},
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
    # mesh = Mesh.from_hdf5( os.path.join(os.path.join(main_dir, settings.output_dir), "mesh.hdf5"))

    # fvm_c_unsteady_semidiscete(
    #     mesh,
    #     model,
    #     settings,
    #     ode_solver_flux="RK1",
    #     ode_solver_source="RK1",
    #     rebuild_model=True,
    #     rebuild_mesh=True,
    #     rebuild_c=True,
    # )

    # io.generate_vtk(settings.output_dir)

    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/simple_openfoam/mesh_2d_mid.msh")
    )
    # mesh = petscMesh.Mesh.from_gmsh( os.path.join(main_dir, "meshes/simple_openfoam/mesh_2d_finest.msh"))

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


if __name__ == "__main__":
    # test_ssf()
    # test_ssf_energy()
    test_ssf_pathconservative()
    # test_ssf_2d()
