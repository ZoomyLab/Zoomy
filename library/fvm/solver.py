
#from __future__ import division, print_function

import numpy as np
import jax.numpy as jnp
import jax

from types import SimpleNamespace
import pyprog
from attr import define, field
from typing import Callable, Optional, Type, Any
#from copy import deepcopy
from time import time as gettime
import os
import sys
import argparse
import shutil
from functools import partial


#WARNING: I get a segmentation fault if I do not include petsc4py before precice
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
from library.mesh.mesh import convert_mesh_to_jax
from library.fvm.ode import *
from library.solver import python_c_interface as c_interface
from library.solver import modify_sympy_c_code as modify_sympy_c_code
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
class Solver():
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
            dQ = dQ.at[:, : mesh.n_inner_cells].set(pde.source(
                Q[:, : mesh.n_inner_cells], Qaux[:, : mesh.n_inner_cells], parameters
            ))
            return dQ
        return compute_source
    
    
    
    @partial(jax.jit, static_argnames=['self'])
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
                qA, qB, qauxA, qauxB, parameters, normals, svA, svB, face_volumes, dt, pde
            )
            # Ensure no failure
            assert not failedA

            nc_fluxB, failedB = compute_nc_flux(
                qB, qA, qauxB, qauxA, parameters, -normals, svB, svA, face_volumes, dt, pde
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
            
            
            def update_dQ_body(loop_idx, dQ, mesh, iA, iB, nc_fluxA, nc_fluxB, face_volumes, cell_volumesA, cell_volumesB):
                faces = mesh.cell_faces[loop_idx]
                inner_range = jnp.arange(mesh.n_inner_cells)  

                iA_faces = iA[faces]
                iB_faces = iB[faces] 

                iA_masked = (iA_faces == inner_range) 
                iB_masked = (iB_faces == inner_range) 

                fluxA_contribution = (nc_fluxA * face_volumes / cell_volumesA)[:, faces] 
                fluxB_contribution = (nc_fluxB * face_volumes / cell_volumesB)[:, faces]  
                
                # slice_mask = jnp.arange(mesh.n_cells) < mesh.n_inner_cells
                
                # iA_slice_masked = slice_mask & iA_masked
                # iB_slice_masked = slice_mask & iB_masked

                
                # dQ = dQ.at[:, :mesh.n_inner_cells].subtract(0.5*(fluxA_contribution + fluxB_contribution))
                
                fA = fluxA_contribution[:, iA_masked]
                fB = fluxB_contribution[:, iB_masked]
                dQ = dQ.at[:, iA_masked].subtract(fA)
                dQ = dQ.at[:, iB_masked].subtract(fB)



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
                return update_dQ_body(loop_idx, dQ, mesh, iA, iB, nc_fluxA, nc_fluxB, face_volumes, cell_volumesA, cell_volumesB)

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
        runtime_pde = model.get_pde(printer='jax')
        # runtime_bcs = model.create_python_boundary_interface(printer='numpy')
        runtime_bcs = model.get_boundary_conditions(printer='jax')
        # runtime_bc = model.get_boundary_conditions()
        # model.boundary_conditions.runtime_bc = runtime_bcs
        return runtime_pde, runtime_bcs
    
    
    def save_model_to_C(self, model, settings):
        _ = model.create_c_interface(path=os.path.join(settings.output_dir, "c_interface"))
        _ = model.create_c_boundary_interface(
            path=os.path.join(settings.output_dir, "c_interface")
        )

    #@partial(jax.jit, static_argnames=['self'])
    def get_apply_boundary_conditions(self, _mesh, _runtime_bcs):
        # mesh = convert_mesh_to_jax(_mesh)
        mesh = _mesh
        #mesh = _mesh
        #mesh = jax.tree_map(jnp.array, _mesh)
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

            #mesh = _mesh
            #mesh = jnp.array(_mesh)

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
                #TODO make this numpy indiced!
                i_bc_func = mesh.boundary_face_function_numbers[i]
                #bc_func = runtime_bcs[i_bc_func]

                # Extract solution variables for the boundary cell
                q_cell = Q[:, mesh.boundary_face_cells[i]]                            # Shape: (Q_dim,)
                qaux_cell = Qaux[:1, mesh.boundary_face_cells[i]]                    # Shape: (1,)

                # Get geometric information
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]

                # Compute distance between face and ghost cell
                distance = jnp.linalg.norm(position - position_ghost)                # Scalar

                # Apply boundary condition function to compute ghost cell values
                # Ensure bc_func returns a JAX-compatible array with shape (Q_dim,)
                # q_ghost = bc_func(
                #     time, position, distance, q_cell, qaux_cell, parameters, normal
                # )
                #q_ghost = runtime_bcs[0](time, position, distance, q_cell, qaux_cell, parameters, normal)
                q_ghost = jax.lax.switch(i_bc_func, runtime_bcs, time, position, distance, q_cell, qaux_cell, parameters, normal)

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
        #Q = self._apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)

        def run(Q, Qaux, parameters, pde, bcs):
    

            iteration = 0
            time = 0.0
    
            output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
    
            assert model.dimension == mesh.dimension
    
    
    
            i_snapshot = 0
            dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
            io.init_output_directory(settings.output_dir, settings.output_clean_dir)
            mesh.write_to_hdf5(output_hdf5_path)
            i_snapshot = io.save_fields(
                output_hdf5_path, time, 0, i_snapshot, Q, Qaux, settings.output_write_all
            )
    
            #Qnew = deepcopy(Q)
            Qnew = Q
    
    
            min_inradius = jnp.min(mesh.cell_inradius)
    
            enforce_boundary_conditions = lambda Q: Q

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, pde, settings)
            space_solution_operator = self.get_space_solution_operator(mesh, pde, bcs, settings)
            source_operator = self.get_compute_source(mesh, pde, settings)
            boundary_operator = self.get_apply_boundary_conditions(mesh, bcs)
    
            time_start = gettime()
            while time < settings.time_end:
                #     # print(f'in loop from process {os.getpid()}')
                #Q = deepcopy(Qnew)
                Q = Qnew

                dt = settings.compute_dt(
                    Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
                )
                assert dt > 10 ** (-6)
                assert not jnp.isnan(dt) and jnp.isfinite(dt)
    
                if settings.truncate_last_time_step:
                    if time + dt * 1.001 > settings.time_end:
                        dt = settings.time_end - time + 10 ** (-10)

    
                Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)

                Qnew = ode_solver_source(
                    source_operator, Qnew, Qaux, parameters, dt, func_jac=self.compute_source_jac
                )
    
                Qnew = boundary_operator(time, Qnew, Qaux, parameters)
                #Qnew = enforce_boundary_conditions(Qnew)
                # Update solution and time
                time += dt
                iteration += 1
                print(iteration, time, dt)

                ## AD EXAMPLE
                #dQ = jnp.zeros_like(Qnew)
                #def f(parameters):
                #    return jnp.sum(source_operator(dt, Qnew, Qaux, parameters, dQ))

                #source_jac =  jax.grad(f)
                #gradient = source_jac(parameters)  
                #print(settings.parameters)
                #print(gradient)
    
                #     # for callback in self.callback_function_list_post_solvestep:
                #     #     Qnew, kwargs = callback(self, Qnew, **kwargs)
    
                i_snapshot = io.save_fields(
                    output_hdf5_path,
                    time,
                    (i_snapshot + 1) * dt_snapshot,
                    i_snapshot,
                    Qnew,
                    Qaux,
                    settings.output_write_all,
                )
            return Qnew
    
        #def f(parameters):
        #    return jnp.sum(run(Q, Qaux, parameters, pde, bcs))
        time_start = gettime()
        Qnew = run(Q, Qaux, parameters, pde, bcs)

        #time_loop_jac =  jax.grad(f)
        #gradient = time_loop_jac(parameters)  
        #print(settings.parameters)
        #print(gradient)
        print(f"Runtime: {gettime() - time_start}")
    
        return settings
    
    
