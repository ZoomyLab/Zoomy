import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from types import SimpleNamespace
from sympy import cos, pi

# from library.pysolver.solver import *
# import library.pysolver.flux as flux
# import library.pysolver.nonconservative_flux as nc_flux
# from library.pysolver.ode import RK1
# import library.pysolver.reconstruction as recon
# import library.pysolver.timestepping as timestepping

from library.fvm.solver import Solver, Settings
from library.fvm.ode import RK1
import library.fvm.reconstruction as recon
import library.fvm.timestepping as timestepping
import library.fvm.flux as flux
import library.fvm.nonconservative_flux as nc_flux
from library.model.boundary_conditions import BoundaryCondition

from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
import library.misc.io as io
from library.mesh.mesh import compute_gradient


import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
from library.mesh.mesh import convert_mesh_to_jax
import argparse


@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d():
    level = 1
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=1.0,
        output_snapshots=100,
        output_dir="outputs/output_test",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag=tag, momentum_field_indices=[[i] for i in range(1, level+1)])
            BC.Extrapolation(physical_tag=tag)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        high=lambda n_field: np.array([0.2, 0.0] + [0.0 for l in range(level)]),
        low=lambda n_field: np.array([0.1, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip_mod"]},
    )

    mesh = petscMesh.Mesh.create_1d((-1, 30), 100)

    solver = Solver()
    solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))

def test_smm_2d():
    level = 0
    n_fields = 3 + 2 * level
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=100,
        output_dir="outputs/test",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.Wall(physical_tag="left"),
            BC.Wall(physical_tag="right"),
        ]
    )

    def custom_ic(x):
        Q = np.zeros(3 + 2 * level, dtype=float)
        Q[0] = np.where(x[0] < 0.5, 1.0, 1.2) + np.where(x[1] < 0.5, 1.0, 1.2)
        return Q

    ic = IC.UserFunction(custom_ic)

    model = ShallowMoments2d(
        fields=3 + 2 * level,
        aux_fields=2,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip"]},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    )

    mesh = convert_mesh_to_jax(mesh)
    solver = Solver()
    Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)



def test_jax_jit_grad():
    level = 1
    n_fields = 3 + 2 * level
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=100,
        output_dir="outputs/test",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.Wall(physical_tag="left"),
            BC.Wall(physical_tag="right"),
        ]
    )

    def custom_ic(x):
        Q = np.zeros(3 + 2 * level, dtype=float)
        Q[0] = np.where(x[0] < 0.5, 1.0, 1.2) + np.where(x[1] < 0.5, 1.0, 1.2)
        return Q

    ic = IC.UserFunction(custom_ic)

    model = ShallowMoments2d(
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip"]},
        # settings={"eigenvalue_mode": "symbolic", "friction": []},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    )
    # print(settings.parameters)

    mesh = convert_mesh_to_jax(mesh)
    solver = Solver()
    # Qaux, Qnew = solver.jax_fvm_unsteady_semidiscrete(
    #     mesh, model, settings
    # )

    # Automatic differentiation example
    def full(params):
        model.parameter_values = params
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    def single(g):
        # print(model.parameter_values)
        param = jax.numpy.array(model.parameter_values)
        param = param.at[0].set(g)
        model.parameter_values = param
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    def no_ad():
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    # jax.config.update("jax_enable_compilation_cache", False)

    params_orig = model.parameter_values.copy()
    params = model.parameter_values
    gradient = jax.jacfwd(full)(params)
    jax.debug.print("param: {params}", params=params)
    jax.debug.print("grad: {gradient}", gradient=gradient)
    # jax.clear_caches()

    model.parameter_values = params_orig
    g = 9.81
    gradient = jax.jacfwd(single)(g)
    jax.debug.print("param: {g}", g=g)
    jax.debug.print("grad: {gradient}", gradient=gradient)
    # jax.clear_caches()
    model.parameter_values = params_orig
    no_ad()

    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


def test_jax_jit_grad_minimal():
    level = 1
    n_fields = 3 + 2 * level
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=100,
        output_dir="outputs/test",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.Wall(physical_tag="left"),
            BC.Wall(physical_tag="right"),
        ]
    )

    def custom_ic(x):
        Q = np.zeros(3 + 2 * level, dtype=float)
        Q[0] = np.where(x[0] < 0.5, 1.0, 1.2) + np.where(x[1] < 0.5, 1.0, 1.2)
        return Q

    ic = IC.UserFunction(custom_ic)

    model = ShallowMoments2d(
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip"]},
        # settings={"eigenvalue_mode": "symbolic", "friction": []},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    )
    # print(settings.parameters)

    mesh = convert_mesh_to_jax(mesh)
    solver = Solver()
    # Qaux, Qnew = solver.jax_fvm_unsteady_semidiscrete(
    #     mesh, model, settings
    # )

    # Automatic differentiation example
    def full(params):
        model.parameter_values = params
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    def single(g):
        # print(model.parameter_values)
        param = jax.numpy.array(model.parameter_values)
        param = param.at[0].set(g)
        model.parameter_values = param
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    def no_ad():
        Qnew, Qaux = solver.jax_fvm_unsteady_semidiscrete(mesh, model, settings)
        return jax.numpy.sum(Qnew)

    # jax.config.update("jax_enable_compilation_cache", False)

    params_orig = model.parameter_values.copy()
    # params = model.parameter_values
    # gradient = jax.jacfwd(full)(params)
    # jax.debug.print("param: {params}", params=params)
    # jax.debug.print("grad: {gradient}", gradient=gradient)
    # jax.clear_caches()

    model.parameter_values = params_orig
    g = 9.81
    gradient = jax.jacfwd(single)(g)
    # jax.debug.print("param: {g}", g=g)
    # jax.debug.print("grad: {gradient}", gradient=gradient)
    # jax.clear_caches()
    # model.parameter_values = params_orig
    # no_ad()

    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


def test_reconstruction():
    level = 0
    n_fields = 3 + 2 * level
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=10,
        output_dir="outputs/test/reconstruction",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
            BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            # BC.Extrapolation(physical_tag="top"),
            # BC.Extrapolation(physical_tag="bottom"),
            BC.Lambda(physical_tag="left", prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 11.0}),
            BC.Lambda(physical_tag="right", prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 12.0}),

        ]
    )

    def custom_ic(x):
        Q = np.zeros(3 + 2 * level, dtype=float)
        Q[0] = 10.0
        return Q

    ic = IC.UserFunction(custom_ic)

    model = ShallowMoments2d(
        fields=3 + 2 * level,
        aux_fields=["dhdx", "dhdy"],
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip"]},
        # settings={"eigenvalue_mode": "symbolic", "friction": []},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    )

    # mesh = convert_mesh_to_jax(mesh)
    solver = Solver()
    Qaux, Qnew = solver.jax_reconstruction(mesh, model, settings)

    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))

def test_reconstruction_faces():
    level = 0
    n_fields = 3 + 2 * level
    settings = Settings(
        name="ShallowMoments",
        parameters={
            "g": 9.81,
            "C": 1.0,
            "nu": 0.000001,
            "lamda": 7,
            "rho": 1,
            "eta": 1,
            "c_slipmod": 1 / 70.0,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=10,
        output_dir="outputs/test/reconstruction",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="top"),
            BC.Extrapolation(physical_tag="bottom"),
            BC.Extrapolation(physical_tag="left"),
            BC.Extrapolation(physical_tag="right"),
        ]
    )

    def custom_ic(x):
        Q = np.zeros(3 + 2 * level, dtype=float)
        Q[0] = x[0] * 2 - x[1] * 1.0 + 10.0
        return Q

    ic = IC.UserFunction(custom_ic)

    model = ShallowMoments2d(
        fields=3 + 2 * level,
        aux_fields=2,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip"]},
        # settings={"eigenvalue_mode": "symbolic", "friction": []},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    )

    mesh = convert_mesh_to_jax(mesh)
    solver = Solver()
    Qaux, Qnew = solver.jax_reconstruction_faces(mesh, model, settings)

    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))

class MySolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        grad_u = compute_gradient(Q[0], mesh)
        # jax.debug.print("{}", grad_u)
        grad_p = compute_gradient(Q[1], mesh)
        Qaux = Qaux.at[0].set((Q[0]-Qold[0])/dt)
        Qaux = Qaux.at[1].set(grad_u[:,0])
        Qaux = Qaux.at[2].set(grad_p[:,0])
        x = mesh.cell_centers[0,:]
        Qaux = Qaux.at[3].set(-jnp.sin(x * jnp.pi))
        return Qaux

def test_implicit():
    settings = Settings(
        name="CoupledConstrained",
        parameters={},
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=0.1,
        output_snapshots=10,
        output_dir="outputs/test/CoupledConstrained",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="top", periodic_to_physical_tag='bottom'),
            BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
            BC.Lambda(physical_tag="left", prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: -1./pi + t, 1: lambda t, x, dx, q, qaux, p, n: 1.}),
            BC.Lambda(physical_tag="right", prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: -1./pi + t, 1: lambda t, x, dx, q, qaux, p, n: -1.}),
        ]
    )

    def custom_ic(x):
        Q = np.zeros(2, dtype=float)
        #Q[0] = -np.cos(x[0]*np.pi)/np.pi
        #Q[1] = - x[0]
        Q[0] = -np.cos(x[0]*np.pi)/np.pi
        Q[1] = 0.
        return Q

    ic = IC.UserFunction(custom_ic)

    model = CoupledConstrained(
        fields=2,
        aux_fields=["dudt", "dudx", "dpdx", "f"],
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.from_gmsh(
        # os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
        os.path.join(main_dir, "meshes/quad_2d/mesh_fine.msh")
    )
    # mesh = convert_mesh_to_jax(mesh)


    solver = MySolver()
    Qaux, Qnew = solver.jax_implicit(mesh, model, settings)

    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))


if __name__ == "__main__":
    #test_smm_1d()
    #test_smm_2d()
    # test_jax_jit_grad()
    #test_jax_jit_grad_minimal()
    # test_reconstruction()
    #test_reconstruction_faces()
    test_implicit()
