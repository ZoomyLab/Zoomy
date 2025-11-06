import os
from time import time as gettime

import jax
from functools import partial
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
# from jaxopt import Broyden

from typing import Callable
from attrs import define, field


from zoomy_core.misc.logger_config import logger

import zoomy_jax.fvm.flux as fvmflux
import zoomy_jax.fvm.nonconservative_flux as nonconservative_flux
import zoomy_core.misc.io as io
import zoomy_jax.misc.io as jax_io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_jax.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_core.transformation.to_jax import JaxRuntimeModel
from zoomy_jax.mesh.mesh import convert_mesh_to_jax

def log_callback_hyperbolic(iteration, time, dt, time_stamp, log_every=10):
    if iteration % log_every == 0:
        logger.info(
            f"iteration: {int(iteration)}, time: {float(time):.6f}, "
            f"dt: {float(dt):.6f}, next write at time: {float(time_stamp):.6f}"
        )
    return None


def log_callback_poisson(iteration, res):
    logger.debug(f"Newton iterations: {iteration}, final residual norm: {
                 jnp.linalg.norm(res):.3e}"
                )
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
                restart=100,
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
class HyperbolicSolver(HyperbolicSolverNumpy):
    flux: fvmflux.Flux = field(factory=lambda: fvmflux.Zero())
    nc_flux: nonconservative_flux.NonconservativeFlux = field(
        factory=lambda: nonconservative_flux.Rusanov()
    )
    
    def create_runtime(self, Q, Qaux, mesh, model):
        jax_mesh = convert_mesh_to_jax(mesh)
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        parameters = jnp.asarray(model.parameter_values)
        runtime_model = JaxRuntimeModel(model)
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
        """
        Returns a JAX-compatible function that applies boundary
        conditions using the new unified `model.boundary_conditions`.
        """

        @jax.jit
        @partial(jax.named_call, name="apply_boundary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            """
            JAX version of the boundary condition application.

            Mirrors the NumPy logic:
                - Loops over boundary faces
                - Computes distance, position, normal, etc.
                - Calls `model.boundary_conditions(...)`
                - Updates ghost cells in Q

            Args:
                time: scalar
                Q: (Q_dim, n_cells)
                Qaux: (Qaux_dim, n_cells)
                parameters: model parameters (pytree)

            Returns:
                Updated Q with ghost cells set by boundary conditions.
            """

            def loop_body(i, Q):
                # Extract face and BC info
                i = jnp.asarray(i, dtype=jnp.int32)
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]

                # Local state
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]

                # Geometry
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = jnp.linalg.norm(position - position_ghost)

                # Call the unified boundary condition function
                q_ghost = model.boundary_conditions(
                    i_bc_func,
                    time,
                    position,
                    distance,
                    q_cell,
                    qaux_cell,
                    parameters,
                    normal,
                )

                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)
                return Q

            # Loop over boundary faces
            Q_updated = jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)
            return Q_updated

        return apply_boundary_conditions


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
            max_abs_eigenvalue = jnp.maximum(
                jnp.abs(evA).max(), jnp.abs(evB).max())
            return max_abs_eigenvalue
        return compute_max_abs_eigenvalue

    def get_flux_operator(self, mesh, model):
        compute_num_flux = self.flux.get_flux_operator(model)
        compute_nc_flux = self.nc_flux.get_flux_operator(model)
        
        @jax.jit
        @partial(jax.named_call, name="Flux")
        def flux_operator(dt, Q, Qaux, parameters, dQ):

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

            Dp, Dm = compute_nc_flux(
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
            )
            flux_out = Dm * face_volumes / cell_volumesA
            flux_in = Dp * face_volumes / cell_volumesB

            dQ = dQ.at[:, iA].subtract(flux_out)
            dQ = dQ.at[:, iB].subtract(flux_in)
            return dQ
        return flux_operator

    # @jax.jit
    # @partial(jax.named_call, name="hyperbolic solver")
    def solve(self, mesh, model, write_output=True):
        Q, Qaux = self.initialize(mesh, model)

        Q, Qaux, parameters, mesh, model = self.create_runtime(
            Q, Qaux, mesh, model)

        # init once with dummy values for dt
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh,
                                model, parameters, 0.0, 1.0)

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{
                    self.settings.output.filename}.h5"
            )
            save_fields = jax_io.get_save_fields(output_hdf5_path, write_all=False)
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

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(
                mesh, model)
            flux_operator = self.get_flux_operator(mesh, model)
            source_operator = self.get_compute_source(mesh, model)
            boundary_operator = self.get_apply_boundary_conditions(mesh, model)
            Qnew = boundary_operator(time, Qnew, Qaux, parameters)

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
                    Q2 = ode.RK1(
                        source_operator,
                        Q1,
                        Qaux,
                        parameters,
                        dt,
                    )

                    Q3 = boundary_operator(time, Q2, Qaux, parameters)

                    # Update solution and time
                    time += dt
                    iteration += 1

                    time_stamp = (i_snapshot) * dt_snapshot

                    Qnew = self.update_q(Q3, Qaux, mesh, model, parameters)
                    Qauxnew = self.update_qaux(
                        Qnew, Qaux, Q, Qaux, mesh, model, parameters, time, dt)

                    i_snapshot = save_fields(
                        time, time_stamp, i_snapshot, Qnew, Qauxnew)

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
