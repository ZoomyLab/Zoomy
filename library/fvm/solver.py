import os
from time import time as gettime

import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from attr import define
from jax.scipy.sparse.linalg import gmres
from jaxopt import Broyden

from typing import Callable
from attrs import define, field


from library.misc.logger_config import logger


# WARNING: I get a segmentation fault if I do not include petsc4py before precice
try:
    from petsc4py import PETSc
except ModuleNotFoundError as err:
    logger.warning(err)

try:
    import precice
except (ModuleNotFoundError, Exception) as err:
    logger.warning(err)


import library.fvm.flux as flux
import library.fvm.nonconservative_flux as nonconservative_flux
import library.misc.io as io
from library.mesh.mesh import convert_mesh_to_jax
from library.misc.misc import Zstruct, Settings
import library.misc.transformation as transformation
import library.fvm.ode as ode
import library.fvm.timestepping as timestepping
from library.model.models.base import JaxRuntimeModel


def log_callback_hyperbolic(iteration, time, dt, time_stamp, log_every=10):
    if iteration % log_every == 0:
        logger.info(
            f"iteration: {int(iteration)}, time: {float(time):.6f}, "
            f"dt: {float(dt):.6f}, next write at time: {float(time_stamp):.6f}"
        )
    return None    


def log_callback_poisson(iteration, res):
    logger.debug(f"Newton iterations: {iteration}, final residual norm: {jnp.linalg.norm(res):.3e}")
    return None    

def log_callback_execution_time(time):
    logger.info(f"Finished simulation with in {time:.3f} seconds")
    return None   


def newton_solver(residual):

    def Jv(Q, U):
        return jax.jvp(lambda q: residual(q), (Q,), (U,))[1]

    @jax.jit
    @partial(jax.named_call, name="preconditioner")
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

    @jax.jit
    @partial(jax.named_call, name="newton_solver")
    def newton_solve(Q):
        def cond_fun(state):
            _, r, i = state
            maxiter = 10
            return jnp.logical_and(jnp.linalg.norm(r) > 1e-6, i < maxiter)

        def body_fun(state):
            Q, r, i = state

            def lin_op(v):
                return Jv(Q, v)

            # Preconditioner
            # diag_J = compute_diagonal_of_jacobian(Q)
            # # regularize diagonal to avoid division by zero
            # diag_J = jnp.where(jnp.abs(diag_J) > 1e-12, diag_J, 1.0)
            # def preconditioner(v):
            #    return v / diag_J

            delta, info = gmres(
                lin_op,
                -r,
                x0=jnp.zeros_like(Q),
                # x0=Qold,
                maxiter=10,
                solve_method="incremental",
                # solve_method="batched",
                restart = 100,
                tol=1e-6,
                # M=preconditioner,
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


        jax.experimental.io_callback(
            log_callback_poisson,                 
            None,                          
            i, res
        )


        return Q_final

    return newton_solve



@define(frozen=True, slots=True, kw_only=True)            
class Solver():
    settings: Zstruct = field(factory=lambda: Settings.default())

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, 'settings', defaults)
        

    def initialize(self, mesh, model):
        model.boundary_conditions.initialize(
            mesh,
            model.time,
            model.position,
            model.distance,
            model.variables,
            model.aux_variables,
            model.parameters,
            model.normal,
        )

        n_variables = model.n_variables
        n_cells = mesh.n_cells
        n_aux_variables = model.aux_variables.length()

        Q = np.empty((n_variables, n_cells), dtype=float)
        Qaux = np.empty((n_aux_variables, n_cells), dtype=float)
        return Q, Qaux
        
    def create_runtime(self, Q, Qaux, mesh, model):      
        jax_mesh = convert_mesh_to_jax(mesh)
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        parameters = jnp.asarray(model.parameter_values)
        runtime_model = JaxRuntimeModel.from_model(model)        
        return Q, Qaux, parameters, jax_mesh, runtime_model

    def get_compute_source(self, mesh, model):
        @jax.jit
        @partial(jax.named_call, name="source")
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = dQ.at[:, : mesh.n_inner_cells].set(
                model.source(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            )
            return dQ

        return compute_source

    def get_compute_source_jacobian(self, mesh, model):
        @jax.jit
        @partial(jax.named_call, name="source_jacobian")
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = dQ.at[:, : mesh.n_inner_cells].set(
                model.source_jacobian(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            )
            return dQ

        return compute_source
    
    def get_apply_boundary_conditions(self, mesh, model):
        runtime_bcs = tuple(model.bcs)

        @jax.jit
        @partial(jax.named_call, name="boudnary_conditions")
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
                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)

                return Q

            # Initialize Q_updated as Q and apply boundary conditions using fori_loop
            Q_updated = jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)

            return Q_updated

        return apply_boundary_conditions
    
    def update_q(self, Q, Qaux, mesh, model, parameters):
        """
        Update variables before the solve step.
        """
        # This is a placeholder implementation. Replace with actual logic as needed.
        return Q
    
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        """
        Update auxiliary variables
        """
        # This is a placeholder implementation. Replace with actual logic as needed.
        return Qaux
    
    def solve(self, mesh, model):
        logger.error(
            "Solver.solve() is not implemented. Please implement this method in the derived class."
        )
        raise NotImplementedError("Solver.solve() must be implemented in derived classes.")


    
@define(frozen=True, slots=True, kw_only=True)            
class HyperbolicSolver(Solver):
    settings: Zstruct = field(factory=lambda: Settings.default())
    compute_dt: Callable = field(factory=lambda: timestepping.adaptive(CFL=0.45))
    num_flux: Callable = field(factory=lambda: flux.Zero())
    nc_flux: Callable = field(factory=lambda: nonconservative_flux.segmentpath())
    time_end: float = 0.1


    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        object.__setattr__(self, 'settings', defaults)
        

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        Q = model.initial_conditions.apply(mesh.cell_centers, Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
        return Q, Qaux

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        
        @jax.jit
        @partial(jax.named_call, name="max_abs_eigenvalue")
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            max_abs_eigenvalue = -jnp.inf
            i_cellA = mesh.face_cells[0]
            i_cellB = mesh.face_cells[1]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]

            normal = mesh.face_normals

            evA = model.eigenvalues(qA, qauxA, parameters, normal)
            evB = model.eigenvalues(qB, qauxB, parameters, normal)
            
            # filterA = jnp.where(qA[1] < 10**(-4), 0., 1.)
            # filterA = jnp.stack(filterA*evA.shape[0], axis=0)
            # filterB = jnp.where(qB[1] < 10**(-4), 0., 1.)
            # filterB = jnp.stack([filterB]*evB.shape[0], axis=0)
            # evA *= filterA
            # evB *= filterB



            max_abs_eigenvalue = jnp.maximum(jnp.abs(evA).max(), jnp.abs(evB).max())

            return max_abs_eigenvalue

        return compute_max_abs_eigenvalue

    def get_flux_operator(self, mesh, model):
        @jax.jit
        @partial(jax.named_call, name="Flux")
        def flux_operator(dt, Q, Qaux, parameters, dQ):
            compute_num_flux = self.num_flux
            compute_nc_flux = self.nc_flux
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
                -normals,
                svA,
                svB,
                face_volumes,
                dt,
                model,
            )

            nc_fluxB, failedB = compute_nc_flux(
                qB,
                qA,
                qauxB,
                qauxA,
                parameters,
                normals,
                svB,
                svA,
                face_volumes,
                dt,
                model,
            )

            @partial(jax.named_call, name="update_dQ_body")
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
                    # (nc_fluxA)[:, faces],
                    zeros,
                )
                fluxB_contribution = jnp.where(
                    iB_masked,
                    (nc_fluxB * face_volumes / cell_volumesB)[:, faces],
                    # (nc_fluxB)[:, faces],
                    zeros,
                )

                fA = fluxA_contribution
                fB = fluxB_contribution

                dQ = dQ.at[:, inner_range].subtract(fA)
                dQ = dQ.at[:, inner_range].subtract(fB)

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
            return dQ

        return flux_operator

    

    def solve(self, mesh, model, write_output=True):
        Q, Qaux = self.initialize(mesh, model)
        
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        
        # init once with dummy values for dt
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_field(time, time_stamp, i_snapshot, Q, Qaux):
                return i_snapshot
            
        Q = jax.device_put(Q)
        Qaux = jax.device_put(Qaux)
        mesh = jax.device_put(mesh)

        def run(Q, Qaux, parameters, model):
            iteration = 0.0
            time = 0.0

            i_snapshot = 0.0
            dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
            if write_output:
                io.init_output_directory(
                    self.settings.output.directory, self.settings.output.clean_directory
                )
                mesh.write_to_hdf5(output_hdf5_path)
                io.save_settings(self.settings)
            i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

            Qnew = Q

            min_inradius = jnp.min(mesh.cell_inradius)

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
            flux_operator = self.get_flux_operator(mesh, model)
            source_operator = self.get_compute_source(mesh, model)
            boundary_operator = self.get_apply_boundary_conditions(mesh, model)

            @jax.jit
            @partial(jax.named_call, name="time loop")
            def time_loop(time, iteration, i_snapshot, Qnew, Qaux):
                loop_val = (time, iteration, i_snapshot, Qnew, Qaux)

                @partial(jax.named_call, name="time_step")
                def loop_body(init_value):
                    time, iteration, i_snapshot, Qnew, Qauxnew = init_value
                    
                    Q = Qnew
                    Qaux = Qauxnew
                
                    dt = self.compute_dt(
                        Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
                    )

                    Q1 = ode.RK1(flux_operator, Q, Qaux, parameters, dt)
                    # Q1 = Q

                    Q2 = ode.RK1(
                        source_operator,
                        Q1,
                        Qaux,
                        parameters,
                        dt,
                    )
                    # Q2 = Q1

                    Q3 = boundary_operator(time, Q2, Qaux, parameters)
                    # Q3 = Q2
                    
                    # Update solution and time
                    time += dt
                    iteration += 1

                    time_stamp = (i_snapshot) * dt_snapshot
                    
                    Qnew = self.update_q(Q3, Qaux, mesh, model, parameters)
                    Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model, parameters, time, dt)


                    i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qauxnew)

                    
                    jax.experimental.io_callback(
                        log_callback_hyperbolic,                 
                        None,                          
                        iteration, time, dt, time_stamp 
                    )
                    
                    return (time, iteration, i_snapshot, Qnew, Qauxnew)

                def proceed(loop_val):
                    time, iteration, i_snapshot, Qnew, Qaux = loop_val
                    return time < self.time_end

                (time, iteration, i_snapshot, Qnew, Qauxnew) = jax.lax.while_loop(
                    proceed, loop_body, loop_val
                )

                return Qnew

            Qnew = time_loop(time, iteration, i_snapshot, Qnew, Qaux)
            return Qnew, Qaux

        time_start = gettime()
        Qnew, Qaux = run(Q, Qaux, parameters, model)
        jax.experimental.io_callback(
            log_callback_execution_time,                 
            None,                          
            gettime() - time_start 
        )
        return Qnew, Qaux


@define(frozen=True, slots=True, kw_only=True)
class PoissonSolver(Solver):
    
    def get_residual(self, Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt):
        def residual(Q):
            qaux = self.update_qaux(Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
            q = boundary_operator(time, Q, qaux, parameters)
            res = model.residual(q, qaux, parameters)
            # res = res.at[:, mesh.n_inner_cells:].set((10*(q-Q)**2)[:, mesh.n_inner_cells:])
            res = res.at[:, mesh.n_inner_cells:].set(0.0)
            return res
        return residual
    


    @jax.jit
    @partial(jax.named_call, name="poission_solver")
    def solve(self, mesh, model, write_output=True):
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        
        # dummy values for a consistent interface
        i_snapshot = 0.0
        time = 0.0
        time_next_snapshot = 0.0
        dt = 0.0
        
        Qold = Q
        Qauxold = Qaux

        boundary_operator = self.get_apply_boundary_conditions(mesh, model)


        Q = boundary_operator(time, Q, Qaux, parameters)
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        if write_output:
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            mesh.write_to_hdf5(output_hdf5_path)
            save_fields = io.get_save_fields(
                output_hdf5_path, True
            )
        else:
            def save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux):
                return i_snapshot


        

        residual = self.get_residual(Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt)
        newton_solve = newton_solver(residual)

        time_start = gettime()
        
        Q = newton_solve(Q)
        
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        i_snapshot = save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux)

        jax.experimental.io_callback(
            log_callback_execution_time,                 
            None,                          
            gettime() - time_start 
        )

        return Q, Qaux
    