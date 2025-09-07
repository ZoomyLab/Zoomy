import os
from time import time as gettime

import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import attrs
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
import library.fvm.solver as solver


def precice_callback_read(interface, meshName, vertexIDs, dataName, dt):
    velocity = interface.read_data(meshName, dataName, vertexIDs, float(dt))
    return np.asarray(velocity, dtype=float)    

def precice_callback_write(interface, meshName, vertexIDs, dataName, data):
    interface.write_data(meshName, dataName, vertexIDs, data)
    return None





@define(frozen=True, slots=True, kw_only=True)            
class PreciceHyperbolicSolver(solver.HyperbolicSolver):
    settings: Zstruct      = field(factory=lambda: Settings.default())
    compute_dt: Callable   = field(factory=lambda: timestepping.adaptive(CFL=0.45))
    num_flux: Callable     = field(factory=lambda: flux.Zero())
    nc_flux: Callable      = field(factory=lambda: nonconservative_flux.segmentpath())
    time_end: float        = 0.1
    config_path: str       = attrs.field(
        factory=lambda: "library/precice_configs/precice-config.xml"
    )

    # -----------------------------------------------------------------------
    # initialisation
    # -----------------------------------------------------------------------
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        Q       = model.initial_conditions.apply(mesh.cell_centers, Q)
        Qaux    = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
        return Q, Qaux
    
    
    @staticmethod
    def get_precice_callback_reader(interface, mesh, ids, name):
        def _cb(dt_scalar):
            arr = interface.read_data(mesh, name, ids, float(dt_scalar))
            return jnp.asarray(arr, dtype=jnp.float_)
        return _cb

    @staticmethod
    def get_precice_callback_writer_pressure(interface, mesh, ids, name):
        def _cb(Q, pr_back):
            if not interface.is_coupling_ongoing():
                return None
            N = 243
            pressure = np.zeros(N+1)  
            z = np.linspace(0., 1., N+1)
            idx_ghost = -2
            h = float(Q[0,idx_ghost])
            pressure = 1000. * (9.81 * h*(1. - z * (h / 0.12)))
            interface.write_data(mesh, name, ids,
                                    pr_back)
            return None
        return _cb
    
    @staticmethod
    def get_precice_callback_writer_alpha(interface, mesh, ids, name):
        def _cb(Q, al):
            if not interface.is_coupling_ongoing():
                return None
            N = 243
            z = np.linspace(0., 1., N+1)
            idx_ghost = -2
            h = float(Q[0,0])
            dz = z[1]-z[0]
            idx = int(h/dz)
            alpha = np.zeros(N+1)
            alpha[:idx] = 1.0
            alpha[idx] = 1.0 - (h - idx*dz)/dz
            interface.write_data(mesh, name, ids,
                                    alpha)
            return None
        return _cb
    
    @staticmethod
    def get_precice_callback_writer_velocity(interface, mesh, ids, name):
        def _cb(Q, al, vel):
            if not interface.is_coupling_ongoing():
                return None
            N = 243
            z = np.linspace(0., 1., N+1)
            idx_ghost = -2
            h = float(Q[0,0])
            q = float(Q[1, 0])


            u_mean     = np.trapz(al * vel[:,0], z) / h         # depth average
            eps = 1e-8
            factor     = (q / h) / (u_mean + eps)                        # target / current
            u_profile  = vel[:,0] * factor                         # scale entire profile
            
            dz = z[1]-z[0]
            idx = int(h/dz)
            vel = np.array(vel)
            vel[:idx, 0] = 2 *  u_profile[:idx]
            # vel[idx] = u * (1.0 - (h - idx*dz)/dz)
            interface.write_data(mesh, name, ids,
                                    vel)
            return None
        return _cb


    @staticmethod
    def get_precice_callback_advance_dt(interface):
        def _cb(dt):
            interface.advance(float(dt))
            return None
        return _cb

    @staticmethod
    def get_precice_callback_get_dt(interface):
        def _cb():
            return interface.get_max_time_step_size()
        return _cb

    @staticmethod
    def get_precice_callback_advance_or_read_checkpoint(interface, checkpoint_path, output_hdf5_path, dt_snapshot, write_all):
        save_fields = io.get_save_fields(output_hdf5_path, write_all=True)

        def _cb(Q, Qaux, time, dt, i_snapshot, iteration):
            if interface.requires_reading_checkpoint():
                Q, Qaux, time = io.load_fields_from_hdf5(checkpoint_path, 0)
            else:
                time += dt
                iteration += 1
                i_snapshot = save_fields(time, dt, i_snapshot, Q, Qaux)

            return jnp.array(Q), jnp.array(Qaux), jnp.float_(time), jnp.float_(iteration), jnp.float_(i_snapshot)
        return _cb

    @staticmethod
    def get_precice_callback_write_checkpoint(interface, checkpoint_path):
        def _cb(Q, Qaux, time):
            if interface.requires_writing_checkpoint():
                io._save_fields_to_hdf5(checkpoint_path, 0, np.array(time), np.array(Q), np.array(Qaux))
            return None
        return _cb

    @staticmethod
    def get_is_coupling_ongoing(interface):
        def _cb():
            return jnp.asarray(interface.is_coupling_ongoing(), dtype=jnp.bool)
        return _cb

    # -----------------------------------------------------------------------
    # run-time boundary conditions
    # -----------------------------------------------------------------------
    def get_apply_boundary_conditions(self, mesh, model, model_orig):

        bc_hyp      = super().get_apply_boundary_conditions(mesh, model)
        bc_precice  = self.get_apply_boundary_conditions_precice(mesh, model, model_orig)


        @partial(jax.named_call, name="boundary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters,
                                      z, ve, al, pr):
            Q = bc_hyp(time, Q, Qaux, parameters)
            q_shape_dtype = jax.ShapeDtypeStruct(
                Q.shape,  
                jnp.asarray(0.0, Q.dtype).dtype
                )
            Q = jax.experimental.io_callback(
                bc_precice,
                q_shape_dtype,
                Q, z, ve, al, pr)         
            return Q

        return apply_boundary_conditions

    # -----------------------------------------------------------------------
    #   Boundary condition that uses the coupling data
    # -----------------------------------------------------------------------
    def get_apply_boundary_conditions_precice(self, mesh, model, model_orig):

        def apply_bc_numpy(Qnew, z, ve, al, pr, threshold=0.55):
            Qnew = np.array(Qnew)   
            z = np.array(z)
            ve = np.array(ve)
            al = np.array(al)
            pr = np.array(pr)
            
            idx       = Qnew.shape[1] - 2
            q_ghost   = Qnew[:, idx]
            h         = q_ghost[0]

            sort_order = z.argsort()
            al_sorted  = al[sort_order]

            mask      = al_sorted < threshold
            if mask.all():
                # assert False
                return Qnew                                        # no water

            i_water   = mask.argmax()

            h_a, h_b  = z[i_water],  z[i_water - 1]
            a_a, a_b  = al[i_water], al[i_water - 1]
            h_lin     = h_b + (h_a - h_b) / (a_a - a_b) * (threshold - a_b)
            h_of      = np.trapz(al, z)
            ve_water  = ve[:i_water, 0]
            z_water = z[:i_water]
            z_water = (z_water-z_water.min()) / (z_water.max()-z_water.min())


            alpha_coeffs = model_orig.basismatrices.basisfunctions.reconstruct_alpha(
                ve_water, z_water
            )
            # q_ghost[0] = h_of
            q_ghost[0] = Qnew[0,0]
            levels   = model_orig.level
            q_ghost[1 : 2 + levels] = h_of * alpha_coeffs[: levels + 1]
            Qnew[:, idx] = q_ghost
            return jnp.array(Qnew)

        return apply_bc_numpy




    


    # -----------------------------------------------------------------------
    #  main driver
    # -----------------------------------------------------------------------
    def solve(self, mesh, model, write_output=True):

        # ----------------- initial state -----------------------------------
        Q, Qaux                  = self.initialize(mesh, model)
        model_orig = model
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)

        # ----------------- preCICE bootstrap -------------------------------
        main_dir      = os.getenv("ZOOMY_DIR")
        interface     = precice.Participant(
            "Fluid2",
            os.path.join(main_dir, self.config_path),
            0, 1
        )
        meshName      = "Fluid2-Mesh"
        velocityName  = "Velocity"
        pressureName  = "Pressure"
        alphaName     = "Alpha"
        
        checkpoint_path = os.path.join(main_dir, self.settings.output.directory, 'checkpoint.h5')
        output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
        io.init_output_directory(
            self.settings.output.directory, self.settings.output.clean_directory
        )
        mesh.write_to_hdf5(output_hdf5_path)
        io.save_settings(self.settings)


        N             = 243
        z             = np.linspace(0.0, 0.12, N + 1)

        # convert grid to numpy here (preCICE wants np)
        grid = np.zeros((N + 1, 3))
        grid[:, 0] = 0.5
        grid[:, 1] = z
        grid[:, 2] = 0.5
        vertexIDs   = interface.set_mesh_vertices(meshName, grid)

        interface.initialize()                           # handshake

        # --------------- place objects on JAX device -----------------------
        Q      = jax.device_put(Q)
        Qaux   = jax.device_put(Qaux)
        mesh   = jax.device_put(mesh)
        z_dev  = jax.device_put(z)                       # constant over time

        # ------------------ callbacks --------------------------------------
        cb_read_velocity  = self.get_precice_callback_reader(
            interface, meshName, vertexIDs, velocityName
        )
        cb_read_alpha     = self.get_precice_callback_reader(
            interface, meshName, vertexIDs, alphaName
        )
        cb_write_pressure = self.get_precice_callback_writer_pressure(
            interface, meshName, vertexIDs, pressureName
        )
        cb_write_velocity = self.get_precice_callback_writer_velocity(
            interface, meshName, vertexIDs, velocityName
        )
        cb_write_alpha = self.get_precice_callback_writer_alpha(
            interface, meshName, vertexIDs, alphaName
        )
        cb_advance_dt     = self.get_precice_callback_advance_dt(interface)
        cb_get_dt         = self.get_precice_callback_get_dt(interface)
        
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)

        cb_advance_or_read_checkpoint = self.get_precice_callback_advance_or_read_checkpoint(interface, checkpoint_path, output_hdf5_path, dt_snapshot, False)
        cb_write_checkpoint = self.get_precice_callback_write_checkpoint(interface, checkpoint_path)
        cb_is_coupling_ongoing = self.get_is_coupling_ongoing(interface)

        # shape descriptors for io_callback
        shape_vel = jax.ShapeDtypeStruct((N + 1, 3), jnp.float_)
        shape_al  = jax.ShapeDtypeStruct((N + 1,),  jnp.float_)
        


        # ------------------ run loop (jit-compiled) ------------------------
        @partial(jax.named_call, name="time_loop")
        def run(Q, Qaux, parameters):

            # local constants for JIT
            min_inradius = jnp.min(mesh.cell_inradius)
            flux_operator      = self.get_flux_operator(mesh, model)
            source_operator    = self.get_compute_source(mesh, model)
            boundary_operator  = self.get_apply_boundary_conditions(mesh, model, model_orig)
            compute_max_eval   = self.get_compute_max_abs_eigenvalue(mesh, model)


            
            i_snapshot = jnp.float_(0.) 
            iteration = jnp.float_(0.)
            time = np.float_(0.)
            i_snapshot = jnp.float_(0.)
            q_desc          = jax.ShapeDtypeStruct(Q.shape,    Q.dtype)
            qaux_desc       = jax.ShapeDtypeStruct(Qaux.shape, Qaux.dtype)
            time_desc       = jax.ShapeDtypeStruct((),          time.dtype)
            iteration_desc  = jax.ShapeDtypeStruct((),          iteration.dtype)
            

            def body(carry):
                time, Q, Qaux, i_snapshot, iteration = carry
                
                jax.experimental.io_callback(cb_write_checkpoint,
                                    None,
                
                                    Q, Qaux, time)
                

                dt_solver  = self.compute_dt(
                    Q, Qaux, parameters, min_inradius, compute_max_eval
                )
                dt_precice = jax.experimental.io_callback(cb_get_dt,
                                                        jax.ShapeDtypeStruct((), jnp.float_),
                                                        )
                
                dt = jnp.minimum(dt_solver, dt_precice)

                vel = jax.experimental.io_callback(cb_read_velocity,
                                                   shape_vel, 0.0)
                al  = jax.experimental.io_callback(cb_read_alpha,
                                                   shape_al,  0.0)

                Q = boundary_operator(time, Q, Qaux, parameters,
                                      z_dev, vel, al, None)

                Q1 = ode.RK1(flux_operator, Q, Qaux, parameters, dt)
                Q2 = ode.RK1(source_operator, Q1, Qaux, parameters, dt)
                Q3 = boundary_operator(time, Q2, Qaux, parameters,
                                       z_dev, vel, al, None)
               
                
                # pressure = pressure.at
                jax.experimental.io_callback(cb_write_pressure,
                                                  None,
                                                  Q3)
                jax.experimental.io_callback(cb_advance_dt,
                                    None,
                                    dt)
                # interface.advance(float(dt)) 
                

                Q, Qaux, time, iteration, i_snapshot = jax.experimental.io_callback(cb_advance_or_read_checkpoint,
                    (q_desc, qaux_desc, time_desc, iteration_desc, iteration_desc),
                    Q3, Qaux, time, dt, i_snapshot, iteration)
                


                return (time, Q, Qaux, i_snapshot, iteration)

            # run until coupling finished
            (time_end, Q_final, Qaux_final, i_snapshot, iteration) = jax.lax.while_loop(
                lambda c: jax.experimental.io_callback(cb_is_coupling_ongoing,
                                                        jax.ShapeDtypeStruct((), jnp.bool),
                                                        ),
                body,
                (jnp.array(0.0), Q, Qaux, i_snapshot, iteration)
            )
            return Q_final, Qaux_final

        # ------------------ wall-clock measurement -------------------------
        t0 = gettime()
        Q, Qaux = run(Q, Qaux, parameters=parameters)
        solver.log_callback_execution_time(gettime() - t0)

        return Q, Qaux



@define(frozen=True, slots=True, kw_only=True)            
class PreciceHyperbolicSolverBidirectional(PreciceHyperbolicSolver):
    
    def solve(self, mesh, model, write_output=True):

        # ----------------- initial state -----------------------------------
        Q, Qaux                  = self.initialize(mesh, model)
        model_orig = model
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)

        # ----------------- preCICE bootstrap -------------------------------
        main_dir      = os.getenv("ZOOMY_DIR")
        interface     = precice.Participant(
            "Fluid2",
            os.path.join(main_dir, self.config_path),
            0, 1
        )
        meshName      = "Fluid2-Mesh"
        velocityName  = "Velocity"
        pressureName  = "Pressure"
        alphaName     = "Alpha"
        velocityBackName = "VelocityBack"
        pressureBackName = "PressureBack"
        alphaBackName    = "AlphaBack"
        
        checkpoint_path = os.path.join(main_dir, self.settings.output.directory, 'checkpoint.h5')
        output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
        io.init_output_directory(
            self.settings.output.directory, self.settings.output.clean_directory
        )
        mesh.write_to_hdf5(output_hdf5_path)
        io.save_settings(self.settings)


        N             = 243
        z             = np.linspace(0.0, 0.12, N + 1)

        # convert grid to numpy here (preCICE wants np)
        grid = np.zeros((N + 1, 3))
        grid[:, 0] = 0.5
        grid[:, 1] = z
        vertexIDs   = interface.set_mesh_vertices(meshName, grid)

        interface.initialize()                           # handshake

        # --------------- place objects on JAX device -----------------------
        Q      = jax.device_put(Q)
        Qaux   = jax.device_put(Qaux)
        mesh   = jax.device_put(mesh)
        z_dev  = jax.device_put(z)                       # constant over time

        # ------------------ callbacks --------------------------------------
        cb_read_velocity  = self.get_precice_callback_reader(
            interface, meshName, vertexIDs, velocityName
        )
        # cb_read_pressureBack  = self.get_precice_callback_reader(
        #     interface, meshName, vertexIDs, pressureBackName
        # )
        cb_read_alpha     = self.get_precice_callback_reader(
            interface, meshName, vertexIDs, alphaName
        )
        # cb_write_pressure = self.get_precice_callback_writer_pressure(
        #     interface, meshName, vertexIDs, pressureName
        # )
        cb_write_alpha = self.get_precice_callback_writer_alpha(
            interface, meshName, vertexIDs, alphaBackName
        )
        cb_write_velocity = self.get_precice_callback_writer_velocity(
            interface, meshName, vertexIDs, velocityBackName
        )
        
        cb_advance_dt     = self.get_precice_callback_advance_dt(interface)
        cb_get_dt         = self.get_precice_callback_get_dt(interface)
        
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)

        cb_advance_or_read_checkpoint = self.get_precice_callback_advance_or_read_checkpoint(interface, checkpoint_path, output_hdf5_path, dt_snapshot, False)
        cb_write_checkpoint = self.get_precice_callback_write_checkpoint(interface, checkpoint_path)
        cb_is_coupling_ongoing = self.get_is_coupling_ongoing(interface)

        # shape descriptors for io_callback
        shape_vel = jax.ShapeDtypeStruct((N + 1, 3), jnp.float_)
        shape_al  = jax.ShapeDtypeStruct((N + 1,),  jnp.float_)
        


        # ------------------ run loop (jit-compiled) ------------------------
        @partial(jax.named_call, name="time_loop")
        def run(Q, Qaux, parameters):

            # local constants for JIT
            min_inradius = jnp.min(mesh.cell_inradius)
            flux_operator      = self.get_flux_operator(mesh, model)
            source_operator    = self.get_compute_source(mesh, model)
            boundary_operator  = self.get_apply_boundary_conditions(mesh, model, model_orig)
            compute_max_eval   = self.get_compute_max_abs_eigenvalue(mesh, model)


            
            i_snapshot = jnp.float_(0.) 
            iteration = jnp.float_(0.)
            time = np.float_(0.)
            i_snapshot = jnp.float_(0.)
            q_desc          = jax.ShapeDtypeStruct(Q.shape,    Q.dtype)
            qaux_desc       = jax.ShapeDtypeStruct(Qaux.shape, Qaux.dtype)
            time_desc       = jax.ShapeDtypeStruct((),          time.dtype)
            iteration_desc  = jax.ShapeDtypeStruct((),          iteration.dtype)
            

            def body(carry):
                time, Q, Qaux, i_snapshot, iteration = carry
                
                jax.experimental.io_callback(cb_write_checkpoint,
                                    None,
                
                                    Q, Qaux, time)
                

                dt_solver  = self.compute_dt(
                    Q, Qaux, parameters, min_inradius, compute_max_eval
                )
                dt_precice = jax.experimental.io_callback(cb_get_dt,
                                                        jax.ShapeDtypeStruct((), jnp.float_),
                                                        )
                
                dt = jnp.minimum(dt_solver, dt_precice)

                vel = jax.experimental.io_callback(cb_read_velocity,
                                                   shape_vel, 0.0)
                al  = jax.experimental.io_callback(cb_read_alpha,
                                                   shape_al,  0.0)
                # pr_back = jax.experimental.io_callback(cb_read_pressureBack,
                #                                    shape_al, 0.0)

                Q = boundary_operator(time, Q, Qaux, parameters,
                                      z_dev, vel, al, None)

                Q1 = ode.RK1(flux_operator, Q, Qaux, parameters, dt)
                Q2 = ode.RK1(source_operator, Q1, Qaux, parameters, dt)
                Q3 = boundary_operator(time, Q2, Qaux, parameters,
                                       z_dev, vel, al, None)

                # _ = jax.experimental.io_callback(cb_write_pressure,
                #                                   None,
                #                                   Q3, pr_back)
                _ = jax.experimental.io_callback(cb_write_alpha,
                                                  None,
                                                  Q3, al)
                _ = jax.experimental.io_callback(cb_write_velocity,
                                                  None,
                                                  Q3, al, vel)
                jax.experimental.io_callback(cb_advance_dt,
                                    None,
                                    dt)
                # interface.advance(float(dt)) 
                

                Q, Qaux, time, iteration, i_snapshot = jax.experimental.io_callback(cb_advance_or_read_checkpoint,
                    (q_desc, qaux_desc, time_desc, iteration_desc, iteration_desc),
                    Q3, Qaux, time, dt, i_snapshot, iteration)
                


                return (time, Q, Qaux, i_snapshot, iteration)

            # run until coupling finished
            (time_end, Q_final, Qaux_final, i_snapshot, iteration) = jax.lax.while_loop(
                lambda c: jax.experimental.io_callback(cb_is_coupling_ongoing,
                                                        jax.ShapeDtypeStruct((), jnp.bool),
                                                        ),
                body,
                (jnp.array(0.0), Q, Qaux, i_snapshot, iteration)
            )
            return Q_final, Qaux_final

        # ------------------ wall-clock measurement -------------------------
        t0 = gettime()
        Q, Qaux = run(Q, Qaux, parameters=parameters)
        solver.log_callback_execution_time(gettime() - t0)

        return Q, Qaux