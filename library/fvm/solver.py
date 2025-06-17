# from __future__ import division, print_function

import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.sparse.linalg import gmres
from jaxopt import Broyden

from types import SimpleNamespace
import pyprog
from attr import define, field
from typing import Callable, Optional, Type, Any

# from copy import deepcopy
from time import time as gettime
import os
import sys
import argparse
import shutil
from functools import partial


# WARNING: I get a segmentation fault if I do not include petsc4py before precice
try:
    from petsc4py import PETSc
except ModuleNotFoundError as err:
    print(err)

try:
    import precice
except ModuleNotFoundError as err:
    print(err)


from library.model.model import *
from library.model.models.base import register_parameter_defaults
from library.model.models.shallow_moments import reconstruct_uvw
import library.fvm.reconstruction as recon
import library.fvm.flux as flux
import library.fvm.nonconservative_flux as nonconservative_flux
import library.fvm.ader_flux as ader_flux
import library.fvm.timestepping as timestepping
import library.misc.io as io
from library.mesh.mesh import convert_mesh_to_jax, compute_gradient, compute_face_gradient
from library.fvm.ode import *
from library.misc.static_class import register_static_pytree


@register_static_pytree
@define(slots=True, frozen=True, kw_only=True)
class Settings:
    name: str = "Simulation"
    parameters: dict = {}
    reconstruction: Callable = recon.constant
    reconstruction_edge: Callable = recon.constant_edge
    num_flux: Callable = flux.LF()
    nc_flux: Callable = nonconservative_flux.segmentpath()
    compute_dt: Callable = timestepping.constant(dt=0.1)
    time_end: float = 1.0
    truncate_last_time_step: bool = True
    output_snapshots: int = 10
    output_write_all: bool = False
    output_dir: str = "outputs/output"
    output_clean_dir: bool = True
    solver_code_base: str = "python"
    callbacks: [str] = []
    debug: bool = False
    profiling: bool = False
    compute_gradient: bool = False
    precice_config_path: str = "/home/ingo/Desktop/precice-tutorial/partitioned-backwards-facing-step/precice-config.xml"


@define(frozen=True)
class Solver:
    def initialize(self, model, mesh, settings=None):
        model.boundary_conditions.initialize(
            mesh,
            model.time,
            model.position,
            model.distance,
            model.variables,
            model.aux_variables,
            model.parameters,
            model.sympy_normal,
        )

        n_fields = model.n_fields
        n_cells = mesh.n_cells
        n_aux_fields = model.aux_variables.length()
        if settings:
            if settings.compute_gradient:
                n_aux_fields += n_fields * mesh.dimension

        Q = np.empty((n_fields, n_cells), dtype=float)
        Qaux = np.zeros((n_aux_fields, n_cells), dtype=float)

        Q = model.initial_conditions.apply(mesh.cell_centers, Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
        return jnp.array(Q), jnp.array(Qaux)

    # @partial(jax.jit, static_argnames=['self'])
    def get_compute_max_abs_eigenvalue(self, mesh, pde, settings):
        @jax.jit
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            max_abs_eigenvalue = -jnp.inf
            eigenvalues_i = jnp.empty(Q.shape[1], dtype=float)
            eigenvalues_j = jnp.empty(Q.shape[1], dtype=float)
            i_cellA = mesh.face_cells[0]
            i_cellB = mesh.face_cells[1]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]

            normal = mesh.face_normals

            evA = pde.eigenvalues(qA, qauxA, parameters, normal)
            evB = pde.eigenvalues(qB, qauxB, parameters, normal)
            # max_abs_eigenvalue = max(jnp.abs(evA).max(), jnp.abs(evB).max())
            max_abs_eigenvalue = jnp.maximum(jnp.abs(evA).max(), jnp.abs(evB).max())

            # if not max_abs_eigenvalue > 10 ** (-8):
            #     iA = jnp.abs(evA).argmax()
            #     iB = jnp.abs(evB).argmax()
            #     print(Q[:, iA])
            #     print(Q[:, iB])
            #     assert False

            return max_abs_eigenvalue

        return compute_max_abs_eigenvalue

    def get_compute_source(self, mesh, pde, settings):
        @jax.jit
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = dQ.at[:, : mesh.n_inner_cells].set(
                pde.source(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            )
            return dQ

        return compute_source

    @partial(jax.jit, static_argnames=["self"])
    def compute_source_jac(self, dt, Q, Qaux, dQ, parameters, mesh, pde, settings):
        # Loop over the inner elements
        for i_cell in range(mesh.n_inner_cells):
            dQ[:, :, i_cell] = pde.source_jacobian(
                Q[:, i_cell], Qaux[:, i_cell], parameters
            )
        return dQ

        return source_jac

    def get_space_solution_operator(self, mesh, pde, bcs, settings):
        @jax.jit
        def space_solution_operator(dt, Q, Qaux, parameters, dQ):
            compute_num_flux = settings.num_flux
            compute_nc_flux = settings.nc_flux
            # Initialize dQ as zeros using jax.numpy
            dQ = jnp.zeros_like(dQ)

            iA = mesh.face_cells[0]
            iB = mesh.face_cells[1]

            qA = Q[:, iA]
            qB = Q[:, iB]
            qauxA = Qaux[:, iA]
            qauxB = Qaux[:, iB]
            normals = mesh.face_normals
            face_volumes = mesh.face_volumes
            cell_volumesA = mesh.cell_volumes[iA]
            cell_volumesB = mesh.cell_volumes[iB]
            svA = mesh.face_subvolumes[:, 0]
            svB = mesh.face_subvolumes[:, 1]

            # Compute non-conservative fluxes
            nc_fluxA, failedA = compute_nc_flux(
                qA,
                qB,
                qauxA,
                qauxB,
                parameters,
                normals,
                svA,
                svB,
                face_volumes,
                dt,
                pde,
            )
            # Ensure no failure
            assert not failedA

            nc_fluxB, failedB = compute_nc_flux(
                qB,
                qA,
                qauxB,
                qauxA,
                parameters,
                -normals,
                svB,
                svA,
                face_volumes,
                dt,
                pde,
            )
            assert not failedB

            # Vectorized operations to handle cell_faces
            # Assuming mesh.cell_faces is a 2D array where each row corresponds to a cell and contains its face indices
            # Example shape: (n_cells, faces_per_cell)

            # Create a mask for inner cells
            inner_cells = jnp.arange(mesh.n_inner_cells)

            # Extract faces for inner cells
            # Shape of faces: (n_cells, faces_per_cell)
            # faces = mesh.cell_faces  # Assuming mesh.cell_faces is a JAX array

            def update_dQ_body(
                loop_idx,
                dQ,
                mesh,
                iA,
                iB,
                nc_fluxA,
                nc_fluxB,
                face_volumes,
                cell_volumesA,
                cell_volumesB,
            ):
                faces = mesh.cell_faces[loop_idx]
                inner_range = jnp.arange(mesh.n_inner_cells)

                iA_faces = iA[faces]
                iB_faces = iB[faces]

                dim = nc_fluxA.shape[0]

                zeros = jnp.zeros(mesh.n_inner_cells)

                iA_masked = iA_faces == inner_range
                iA_masked = jnp.repeat(iA_masked[jnp.newaxis], repeats=dim, axis=0)
                iB_masked = iB_faces == inner_range
                iB_masked = jnp.repeat(iB_masked[jnp.newaxis], repeats=dim, axis=0)

                fluxA_contribution = jnp.where(
                    iA_masked,
                    (nc_fluxA * face_volumes / cell_volumesA)[:, faces],
                    zeros,
                )
                fluxB_contribution = jnp.where(
                    iB_masked,
                    (nc_fluxB * face_volumes / cell_volumesB)[:, faces],
                    zeros,
                )

                # slice_mask = jnp.arange(mesh.n_cells) < mesh.n_inner_cells

                # iA_slice_masked = slice_mask & iA_masked
                # iB_slice_masked = slice_mask & iB_masked

                # dQ = dQ.at[:, :mesh.n_inner_cells].subtract(0.5*(fluxA_contribution + fluxB_contribution))

                fA = fluxA_contribution
                fB = fluxB_contribution
                # fA = fluxA_contribution[:, iA_masked]
                # fB = fluxB_contribution[:, iB_masked]
                # dQ = dQ.at[:, iA_masked].subtract(fA)
                # dQ = dQ.at[:, iA_masked].subtract(fB)
                dQ = dQ.at[:, inner_range].subtract(fA)
                dQ = dQ.at[:, inner_range].subtract(fB)

                # fluxA_contribution_exp = fluxA_contribution[:, jnp.newaxis, :]
                # fluxB_contribution_exp = fluxB_contribution[:, jnp.newaxis, :]

                # iA_masked_exp = iA_masked[jnp.newaxis, :, :]
                # iB_masked_exp = iB_masked[jnp.newaxis, :, :]

                # masked_fluxA = fluxA_contribution_exp * iA_masked_exp
                # masked_fluxB = fluxB_contribution_exp * iB_masked_exp

                # summed_fluxA = jnp.sum(masked_fluxA, axis=2)
                # summed_fluxB = jnp.sum(masked_fluxB, axis=2)

                # dQ = dQ.at[:, :mesh.n_inner_cells].subtract(summed_fluxA + summed_fluxB)

                return dQ

            def loop_body(loop_idx, dQ):
                return update_dQ_body(
                    loop_idx,
                    dQ,
                    mesh,
                    iA,
                    iB,
                    nc_fluxA,
                    nc_fluxB,
                    face_volumes,
                    cell_volumesA,
                    cell_volumesB,
                )

            dQ = jax.lax.fori_loop(0, mesh.cell_faces.shape[0], loop_body, dQ)
            # # Gather iA and iB for all faces
            # iA_faces = iA[faces]  # Shape: (n_cells, faces_per_cell)
            # iB_faces = iB[faces]  # Shape: (n_cells, faces_per_cell)

            # # Create masks where iA or iB is an inner cell
            # # Broadcasting inner_cells over faces
            # # Compare each iA_face and iB_face with inner_cells
            # # This results in boolean masks
            # iA_masked = (iA_faces[..., None] == inner_cells).any(axis=-1)  # Shape: (n_cells, faces_per_cell)
            # iB_masked = (iB_faces[..., None] == inner_cells).any(axis=-1)  # Shape: (n_cells, faces_per_cell)

            # # Compute the contributions for all cells and faces at once
            # # Compute flux contributions
            # fluxA_contribution = (nc_fluxA * face_volumes / cell_volumesA)[:, faces]  # Shape: (Q_dim, n_cells, faces_per_cell)
            # fluxB_contribution = (nc_fluxB * face_volumes / cell_volumesB)[:, faces]  # Shape: (Q_dim, n_cells, faces_per_cell)

            # # Apply masks and sum contributions
            # # We need to sum over faces, but only where masks are True
            # # Reshape masks to match flux dimensions
            # # Expand dims to align with flux coordinates
            # iA_masked_expanded = iA_masked[jnp.newaxis, :, :]  # Shape: (1, n_cells, n_faces_per_cell)
            # iB_masked_expanded = iB_masked[jnp.newaxis, :, :]  # Sh

            # # Apply masks
            # fluxA_masked = jnp.where(iA_masked_expanded, fluxA_contribution, 0.0)
            # fluxB_masked = jnp.where(iB_masked_expanded, fluxB_contribution, 0.0)

            # # Sum over faces
            # total_fluxA = jnp.sum(fluxA_masked, axis=1)  # Shape: (Q_dim, n_cells)
            # total_fluxB = jnp.sum(fluxB_masked, axis=1)  # Shape: (Q_dim, n_cells)

            # # Update dQ
            # total_flux = total_fluxA + total_fluxB
            # dQ = dQ.at[:, :mesh.n_inner_cells].subtract(total_flux)

            return dQ

        return space_solution_operator

    def _load_runtime_model(self, model):
        runtime_pde = model.get_pde(printer="jax")
        # runtime_bcs = model.create_python_boundary_interface(printer='numpy')
        runtime_bcs = model.get_boundary_conditions(printer="jax")
        # runtime_bc = model.get_boundary_conditions()
        # model.boundary_conditions.runtime_bc = runtime_bcs
        return runtime_pde, runtime_bcs

    def save_model_to_C(self, model, settings):
        _ = model.create_c_interface(
            path=os.path.join(settings.output_dir, "c_interface")
        )
        _ = model.create_c_boundary_interface(
            path=os.path.join(settings.output_dir, "c_interface")
        )

    # @partial(jax.jit, static_argnames=['self'])
    def get_apply_boundary_conditions(self, _mesh, _runtime_bcs):
        # mesh = convert_mesh_to_jax(_mesh)
        mesh = _mesh
        # mesh = _mesh
        # mesh = jax.tree_map(jnp.array, _mesh)
        runtime_bcs = tuple(_runtime_bcs)

        @jax.jit
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            """
            Applies boundary conditions to the solution arrays Q and Qaux using JAX's functional updates.

            Parameters:
                mesh: Mesh object containing boundary face information.
                time: Current simulation time.
                Q: JAX array of shape (Q_dim, n_cells), solution variables.
                Qaux: JAX array of auxiliary solution variables.
                parameters: Dictionary of simulation parameters.
                runtime_bcs: List of JAX-compatible boundary condition functions.

            Returns:
                Updated Q array with boundary conditions applied to ghost cells.
            """

            # mesh = _mesh
            # mesh = jnp.array(_mesh)

            def loop_body(i, Q):
                """
                Body function for jax.lax.fori_loop to apply boundary conditions iteratively.

                Parameters:
                    i: Current index in the loop.
                    Q: Current state of the solution array.

                Returns:
                    Updated Q after applying the boundary condition for face i.
                """
                # Extract boundary face index and corresponding BC function
                i = jnp.asarray(i, dtype=jnp.int32)
                i_face = mesh.boundary_face_face_indices[i]
                # TODO make this numpy indiced!
                i_bc_func = mesh.boundary_face_function_numbers[i]
                # bc_func = runtime_bcs[i_bc_func]

                # Extract solution variables for the boundary cell
                q_cell = Q[:, mesh.boundary_face_cells[i]]  # Shape: (Q_dim,)
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]

                # Get geometric information
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]

                # Compute distance between face and ghost cell
                distance = jnp.linalg.norm(position - position_ghost)  # Scalar

                # Apply boundary condition function to compute ghost cell values
                # Ensure bc_func returns a JAX-compatible array with shape (Q_dim,)
                # q_ghost = bc_func(
                #     time, position, distance, q_cell, qaux_cell, parameters, normal
                # )
                # q_ghost = runtime_bcs[0](time, position, distance, q_cell, qaux_cell, parameters, normal)
                q_ghost = jax.lax.switch(
                    i_bc_func,
                    runtime_bcs,
                    time,
                    position,
                    distance,
                    q_cell,
                    qaux_cell,
                    parameters,
                    normal,
                )

                # Update Q at ghost cell using functional update
                # mesh.boundary_face_ghosts[i] is the index of the ghost cell
                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)

                return Q

            # Initialize Q_updated as Q and apply boundary conditions using fori_loop
            Q_updated = jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)

            return Q_updated

        return apply_boundary_conditions

    def jax_fvm_unsteady_semidiscrete(
        self, mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        Q, Qaux = self.initialize(model, mesh)

        parameters = model.parameter_values

        #mesh = convert_mesh_to_jax(mesh)
        parameters = jnp.asarray(parameters)

        pde, bcs = self._load_runtime_model(model)
        # Q = self._apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
        save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

        def run(Q, Qaux, parameters, pde, bcs):
            iteration = 0.0
            time = 0.0
            assert model.dimension == mesh.dimension

            i_snapshot = 0.0
            dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
            io.init_output_directory(settings.output_dir, settings.output_clean_dir)
            mesh.write_to_hdf5(output_hdf5_path)
            i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

            # Qnew = deepcopy(Q)
            Qnew = Q

            min_inradius = jnp.min(mesh.cell_inradius)

            def enforce_boundary_conditions(Q):
                return Q

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(
                mesh, pde, settings
            )
            space_solution_operator = self.get_space_solution_operator(
                mesh, pde, bcs, settings
            )
            source_operator = self.get_compute_source(mesh, pde, settings)
            boundary_operator = self.get_apply_boundary_conditions(mesh, bcs)

            time_start = gettime()

            @jax.jit
            def time_loop(time, iteration, i_snapshot, Qnew, Qaux):
                loop_val = (time, iteration, i_snapshot, Qnew, Qaux)

                def loop_body(init_value):
                    time, iteration, i_snapshot, Qnew, Qaux = init_value
                    Q = Qnew

                    dt = settings.compute_dt(
                        Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
                    )
                    # assert dt > 10 ** (-6)
                    # assert not jnp.isnan(dt) and jnp.isfinite(dt)

                    # if settings.truncate_last_time_step:
                    #    if time + dt * 1.001 > settings.time_end:
                    #        dt = settings.time_end - time + 10 ** (-10)

                    Q1 = ode_solver_flux(
                        space_solution_operator, Q, Qaux, parameters, dt
                    )

                    Q2 = ode_solver_source(
                        source_operator,
                        Q1,
                        Qaux,
                        parameters,
                        dt,
                        func_jac=self.compute_source_jac,
                    )

                    Q3 = boundary_operator(time, Q2, Qaux, parameters)

                    # Update solution and time
                    time += dt
                    iteration += 1

                    jax.debug.print(
                        "iteration: {iteration}, time: {time}, dt: {dt}",
                        iteration=iteration,
                        time=time,
                        dt=dt,
                    )

                    time_stamp = (i_snapshot + 1) * dt_snapshot

                    # i_snapshot = jax.pure_callback(save_fields, jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32), time, time_stamp , i_snapshot, Qnew, Qaux)
                    i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qaux)

                    return (time, iteration, i_snapshot, Q3, Qaux)

                def proceed(loop_val):
                    time, iteration, i_snapshot, Qnew, Qaux = loop_val
                    return time < settings.time_end

                (time, iteration, i_snapshot, Qnew, Qaux) = jax.lax.while_loop(
                    proceed, loop_body, loop_val
                )

                return Qnew

            Qnew = time_loop(time, iteration, i_snapshot, Qnew, Qaux)
            return Qnew, Qaux

        time_start = gettime()
        Qnew, Qaux = run(Q, Qaux, parameters, pde, bcs)

        print(f"Runtime: {gettime() - time_start}")

        return Qnew, Qaux

    def jax_test(
        self, mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        Q, Qaux = self.initialize(model, mesh)

        parameters = model.parameter_values

        #mesh = convert_mesh_to_jax(mesh)
        parameters = jnp.asarray(parameters)

        pde, bcs = self._load_runtime_model(model)
        # Q = self._apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
        save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

        def run(Q, Qaux, parameters, pde, bcs):
            iteration = 0.0
            time = 0.0
            assert model.dimension == mesh.dimension

            i_snapshot = 0.0
            dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
            io.init_output_directory(settings.output_dir, settings.output_clean_dir)
            mesh.write_to_hdf5(output_hdf5_path)
            i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

            # Qnew = deepcopy(Q)
            Qnew = Q

            min_inradius = jnp.min(mesh.cell_inradius)

            def enforce_boundary_conditions(Q):
                return Q

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(
                mesh, pde, settings
            )
            space_solution_operator = self.get_space_solution_operator(
                mesh, pde, bcs, settings
            )
            source_operator = self.get_compute_source(mesh, pde, settings)
            boundary_operator = self.get_apply_boundary_conditions(mesh, bcs)

            time_start = gettime()

            @jax.jit
            def time_loop(time, iteration, i_snapshot, Qnew, Qaux):
                loop_val = (time, iteration, i_snapshot, Qnew, Qaux)

                def loop_body(init_value):
                    time, iteration, i_snapshot, Qnew, Qaux = init_value
                    Q = Qnew

                    dt = settings.compute_dt(
                        Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
                    )
                    # assert dt > 10 ** (-6)
                    # assert not jnp.isnan(dt) and jnp.isfinite(dt)

                    # if settings.truncate_last_time_step:
                    #    if time + dt * 1.001 > settings.time_end:
                    #        dt = settings.time_end - time + 10 ** (-10)

                    Q1 = ode_solver_flux(
                        space_solution_operator, Q, Qaux, parameters, dt
                    )

                    Q2 = ode_solver_source(
                        source_operator,
                        Q1,
                        Qaux,
                        parameters,
                        dt,
                        func_jac=self.compute_source_jac,
                    )

                    Q3 = boundary_operator(time, Q2, Qaux, parameters)

                    # Update solution and time
                    time += dt
                    iteration += 1

                    jax.debug.print(
                        "iteration: {iteration}, time: {time}, dt: {dt}",
                        iteration=iteration,
                        time=time,
                        dt=dt,
                    )

                    time_stamp = (i_snapshot + 1) * dt_snapshot

                    # i_snapshot = jax.pure_callback(save_fields, jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32), time, time_stamp , i_snapshot, Qnew, Qaux)
                    i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qaux)

                    return (time, iteration, i_snapshot, Q3, Qaux)

                def proceed(loop_val):
                    time, iteration, i_snapshot, Qnew, Qaux = loop_val
                    return time < settings.time_end

                (time, iteration, i_snapshot, Qnew, Qaux) = jax.lax.while_loop(
                    proceed, loop_body, loop_val
                )

                return Qnew

            Qnew = time_loop(time, iteration, i_snapshot, Qnew, Qaux)
            return Qnew, Qaux

        time_start = gettime()
        Qnew, Qaux = run(Q, Qaux, parameters, pde, bcs)

        print(f"Runtime: {gettime() - time_start}")

        return Qnew, Qaux

    def jax_reconstruction(
        self, mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        Q, Qaux = self.initialize(model, mesh)

        parameters = model.parameter_values

        mesh = convert_mesh_to_jax(mesh)
        parameters = jnp.asarray(parameters)

        pde, bcs = self._load_runtime_model(model)
        
        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
        mesh.write_to_hdf5(output_hdf5_path)
        save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)
        boundary_operator = self.get_apply_boundary_conditions(mesh, bcs)
        mesh = convert_mesh_to_jax(mesh)


        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        time = 0.0
        i_snapshot = 0.0
        time_stamp = 0.0

        jax.debug.print("Reconstruction")

        time_start = gettime()


        Q = boundary_operator(time, Q, Qaux, parameters)
        grad = compute_gradient(Q[0], mesh)
        div = compute_gradient(grad[:, 0], mesh)
        Qaux = Qaux.at[0].set(grad[:, 0])
        Qaux = Qaux.at[1].set(div[:, 0])
        i_snapshot = save_fields(time, time_stamp, i_snapshot, Q, Qaux)

        print(f"Runtime: {gettime() - time_start}")

        return Q, Qaux

    def jax_reconstruction_faces(
        self, mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        Q, Qaux = self.initialize(model, mesh)

        parameters = model.parameter_values

        # mesh = convert_mesh_to_jax(mesh)
        parameters = jnp.asarray(parameters)

        pde, bcs = self._load_runtime_model(model)
        # Q = self._apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
        save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        time = 0.0
        i_snapshot = 0.0
        time_stamp = 0.0

        jax.debug.print("Reconstruction")

        time_start = gettime()

        grad = compute_face_gradient(Q[0], mesh)
        # jax.debug.print("{}", grad[:, 0])
        # jax.debug.print("{}", grad[:, 1])
        i_snapshot = save_fields(time, time_stamp, i_snapshot, Q, Qaux)

        print(f"Runtime: {gettime() - time_start}")

        return Q, Qaux

    def jax_implicit(
        self, mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        Q, Qaux = self.initialize(model, mesh)

        Qold = Q
        Qauxold = Qaux

        parameters = model.parameter_values

        mesh = convert_mesh_to_jax(mesh)
        parameters = jnp.asarray(parameters)

        pde, bcs = self._load_runtime_model(model)
        # Q = self._apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)

        time = 0.0
        dt = 0.1
        time_next_snapshot = 0.0
        Qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
        jax.debug.print("IMPLICIT SOLVER")

        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
        mesh.write_to_hdf5(output_hdf5_path)
        save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

        #mesh = convert_mesh_to_jax(mesh)
        boundary_operator = self.get_apply_boundary_conditions(mesh, bcs)

        i_snapshot = 0.0

        Q = boundary_operator(time, Q, Qaux, parameters)
        Qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
        i_snapshot = save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux)
        #jax.debug.print("{}", Qaux[0])
        #jax.debug.print("{}", Qaux[1])
        #jax.debug.print("{}", model.sympy_source_implicit)

        time_start = gettime()

        Q = self.implicit_solve(Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator)
        Qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
        #jax.debug.print("constraint")
        #jax.debug.print("{}", constraint)

        time =+ dt
        time_next_snapshot += dt
        i_snapshot = save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux)
       


        print(f"Runtime: {gettime() - time_start}")

        return Q, Qaux
    

    def implicit_solve_alt(self, Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator, debug=[False, False]):
    
        def residual(Q):
            qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
            q = boundary_operator(time, Q, qaux, parameters)
            res = pde.source_implicit(Q, qaux, parameters)
            res = res.at[:, mesh.n_inner_cells:].set(0.0)
            return res

        def fun(Q):
            return residual(Q)
    
        # Initialize the solver
        solver = Broyden(fun=fun)
    
        # Run the solver
        result = solver.run(Q)
    
        if debug[0]:
            jax.debug.print("Broyden finished with norm = {res:.3e}", res=jnp.linalg.norm(fun(result.params)))
    
        return result.params


    def implicit_solve(self, Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator, debug=[False, False], user_residual=None):

        def default_residual(Q):
            qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
            q = boundary_operator(time, Q, qaux, parameters)
            res = pde.source_implicit(q, qaux, parameters)
            res = res.at[:, mesh.n_inner_cells:].set(0.)
            #hp0 = q[0]
            #hp1 = q[1]
            #delta = 0.
            #res = res.at[0].add(delta * jnp.sum(hp0 + hp1))
            #res = res.at[1].add(delta * jnp.sum(hp0 + hp1))
            return res

        if user_residual is None:
            residual = default_residual
        else:
            residual = user_residual

        def Jv(Q, U):
            return jax.jvp(lambda q: residual(q), (Q,), (U,))[1]

        @jax.jit
        def compute_diagonal_of_jacobian(Q):
            ndof, N = Q.shape

            def compute_entry(i, j):
                e = jnp.zeros_like(Q).at[i, j].set(1.0)
                J_e = Jv(Q, e)
                return J_e[i, j]

            def outer_loop(i, diag):
                def inner_loop(j, d):
                    val = compute_entry(i, j)
                    return d.at[i, j].set(val)
                return jax.lax.fori_loop(0, N, inner_loop, diag)

            diag_init = jnp.zeros_like(Q)
            return jax.lax.fori_loop(0, ndof, outer_loop, diag_init)

        def newton_solve(Q):
            def cond_fun(state):
                _, r, i = state
                maxiter = 20
                return jnp.logical_and(jnp.linalg.norm(r) > 1e-6, i < maxiter)

            def body_fun(state):
                Q, r, i = state

                if debug[0]:
                    jax.debug.print("Newton Iter {i}: residual norm = {res:.3e}", i=i, res=jnp.linalg.norm(r))

                def lin_op(v):
                    return Jv(Q, v)

                # Preconditioner
                #diag_J = compute_diagonal_of_jacobian(Q)
                ## regularize diagonal to avoid division by zero
                #diag_J = jnp.where(jnp.abs(diag_J) > 1e-12, diag_J, 1.0)
                #def preconditioner(v):
                #    return v / diag_J

                delta, info = gmres(
                    lin_op,
                    -r,
                    x0=jnp.zeros_like(Q),
                    #x0=Qold,
                    maxiter=100,
                    solve_method="incremental",
                    restart = 20,
                    tol=1e-6,
                )

                def backtrack(alpha, Q, delta, r):
                    def cond(val):
                        alpha, _, _ = val
                        return alpha > 1e-3

                    def body(val):
                        alpha, Q_curr, r_curr = val
                        Qnew = Q_curr + alpha * delta
                        r_new = residual(Qnew)
                        improved = jnp.linalg.norm(r_new) < jnp.linalg.norm(r_curr)

                        if debug[1]:
                            jax.debug.print("  Line search α = {alpha:.2e}, new residual norm = {res:.3e}", alpha=alpha, res=jnp.linalg.norm(r_new))

                        return jax.lax.cond(
                            improved,
                            lambda _: (0.0, Qnew, r_new),  # Accept and stop
                            lambda _: (alpha * 0.5, Q_curr, r_curr),  # Retry
                            operand=None
                        )

                    return jax.lax.while_loop(cond, body, (alpha, Q, r))[1:]

                Q_new, r_new = backtrack(1.0, Q, delta, r)

                return (Q_new, r_new, i + 1)

            r0 = residual(Q)
            init_state = (Q, r0, 0)
            Q_final, res, i = jax.lax.while_loop(cond_fun, body_fun, init_state)
            if debug[0]:
                jax.debug.print("Newton Iter {i}: residual norm = {res:.3e}", i=i, res=jnp.linalg.norm(res))

            return Q_final

        return jax.jit(newton_solve)(Q) if not debug else newton_solve(Q)


    # def implicit_solve(self, Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator):


    #     def residual(Q):
    #         qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
    #         q = boundary_operator(time, Q, qaux, parameters)
    #         res = pde.source_implicit(Q, qaux, parameters)
    #         res = res.at[:, mesh.n_inner_cells:].set(0.)
    #         # res = res.at[1, 0].set(Q[1, 0] -0)
    #         return res
        
        
    #     # Jacobian-vector product helper
    #     def Jv(Q, U):
    #         return jax.jvp(lambda q: residual(q), (Q, ), (U, ))[1]
        
                
    #     def compute_diagonal_of_jacobian(Q):
    #         ndof, N = Q.shape
    #         diag = jnp.zeros_like(Q)
    #         for i in range(ndof):
    #             for j in range(N):
    #                 e = jnp.zeros_like(Q).at[i, j].set(1.0)
    #                 J_e = Jv(Q, e)
    #                 diag = diag.at[i, j].set(J_e[i, j])
    #         return diag  # shape (ndof, N)
        
        
    #     # Newton solver using CG for linear solve
    #     def newton_solve(Q, tol=1e-6, maxiter=8):
    #         for i in range(maxiter):
    #             r = residual(Q)
    #             res_norm = jnp.linalg.norm(r)
    #             jax.debug.print("Iter {i} , residual norm = {res_norm:.3e}", i=i, res_norm=res_norm)
    #             if res_norm < tol:
    #                 break
        
    #             def lin_op(v):
    #                 return Jv(Q, v)
                
    #             # # Preconditioner
    #             # diag_J = compute_diagonal_of_jacobian(Q)
    #             # # regularize diagonal to avoid division by zero
    #             # diag_J = jnp.where(jnp.abs(diag_J) > 1e-12, diag_J, 1.0)
    #             # def preconditioner(v):
    #             #     return v / diag_J
        
    #             delta, info = gmres(
    #                 lin_op,
    #                 -r,
    #                 x0=jnp.zeros_like(Q),
    #                 maxiter=100,
    #                 solve_method="batched",
    #                 tol=10 ** (-6),
    #                 # M=preconditioner,
    #             )
        
    #             alpha = 1.0
    #             for _ in range(10):
    #                 Qnew = Q + alpha * delta
    #                 r_new = residual(Qnew)
    #                 if jnp.linalg.norm(r_new) < jnp.linalg.norm(r):
    #                     Q = Qnew
    #                     break
    #                 alpha *= 0.5

        
    #         return Q
        
    #     return newton_solve(Q)


    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        return Qaux


