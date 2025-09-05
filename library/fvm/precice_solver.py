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
import library.fvm.solver as solver

@define(frozen=True, slots=True, kw_only=True)            
class PreciceHyperbolicSolver(solver.HyperbolicSolver):
    settings: Zstruct = field(factory=lambda: Settings.default())
    compute_dt: Callable = field(factory=lambda: timestepping.adaptive(CFL=0.45))
    num_flux: Callable = field(factory=lambda: flux.Zero())
    nc_flux: Callable = field(factory=lambda: nonconservative_flux.segmentpath())
    time_end: float = 0.1
    config_dir: str = attrs.field(factory=lambda: "of_coupling/precice-config.xml")


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

    def get_apply_boundary_conditions(self, mesh, model):
        runtime_bcs = tuple(model.bcs)

        bc_hyp = super().get_apply_boundary_conditions(mesh, model)
        bc_precice = self.get_apply_boundary_conditions_precice(mesh, model)
        

        @jax.jit
        @partial(jax.named_call, name="boudnary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters, z, ve, al, pr):
            Q = bc_hyp(time, Q, Qaux, parameters)
            Q = bc_precice(Q, z, ve, al, pr)


    def get_apply_boundary_conditions_precice(self, mesh, model):

        @jax.jit
        @partial(jax.named_call, name="boudnary_conditions")
        def apply_bc(Qnew, z, ve, al, pr):
            idx = Qnew.shape[1]-2
            q_ghost = Qnew[:, idx]
            h = q_ghost[0]
    
            # hydrostatic pressure
            #p_hyd_sms = 1000 * 9.81 * (h-z) * (z<=h)
    
            # filter water velocities
            threshold = 0.55
            sort_order = z.argsort()
            al_sorted = al[sort_order]
            i_water = 0
            for i, alpha in enumerate(al_sorted):
                if alpha < threshold:
                    i_water = i
                    break
            if not i_water > 0:
                #print(al)
                return Qnew, pr, np.zeros_like(al), np.zeros_like(ve)
                #assert False
            #i_water = (z*(al > threshold)).argmax()
    
    
    
            h_a = z[i_water]
            h_b = z[i_water-1]
            a_a = al[i_water]
            a_b = al[i_water-1]
            h_lin = h_b + (h_a - h_b)/(a_a - a_b) * (threshold - a_b)
            h_of = h_lin
    
            h_of = np.trapz(al, z)
    
            ve_water = ve[:i_water, 0]
            ve_of_water = np.zeros_like(pr)
            ve_of_water[:i_water] = ve_water
            z_water = z[:i_water]
            z_water = (z_water-z_water.min()) / (z_water.max()-z_water.min())
    
    
    
            # project velocities to moments
            alpha = model.basismatrices.basisfunctions.reconstruct_alpha(ve_water, z_water)
            levels = model.levels
            q_ghost[1:2+levels] = h_of*alpha[:levels+1]
            Qnew[:, idx] = q_ghost
    
            rho = 1000. * al + 1. * (1-al)
            p_hyd_of_abs = rho * 9.81 * h
            pressure = p_hyd_of_abs
    
            dx = (5-0.5)/(Qnew.shape[1]-2)
            h0 = Qnew[0, 0]
            alphaGradientTop = (h0-h_of)/dx
            alphaGradient = np.zeros_like(al)
            #alphaGradient = np.where(al < 0.1 , 0., np.where(al > 0.9, 0., alphaGradientTop))
            #alphaGradient *= 0.
    
    
    
            velocityGradient = np.zeros_like(ve)
            #uG = model.basismatrices.basisfunctions.reconstruct_velocity_profile(Qnew[1:,idx]/Qnew[0, idx])
            #u0 = model.basismatrices.basisfunctions.reconstruct_velocity_profile(Qnew[1:,0 ]/Qnew[0, 0])
            #dudx = (u0-uG)/dx
            #velocityGradient[:i_water, 0] = np.interp(z_water, np.linspace(0, 1, 100), dudx)
            #velocityGradient = velocityGradient.at[:i_water, 0].set(np.interp(z_water, np.linspace(0, 1, 100), dudx))
    
            #velocityGradient *= 0.
            #alphaGradient *= 0.
    
    
    
            #pressuregradient = 1000 * 9.81 * (h0-h_of)/dx
            #pressure = pressuregradient
    
            #print('--------------------')
            #print(i_water, h_of)
            #print(pressure)
            #print(al)
            #print('--------------------')
            return Qnew, pressure, alphaGradient, velocityGradient


    def precice_read_checkpoint(self, output_path):
        Q, Qaux, time = io.load_fields_from_hdf5(output_path, 0)
        return Q, Qaux, time

    def solve(self, mesh, model, write_output=True):


        Q, Qaux = self.initialize(mesh, model)
        
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)

        print("Configure preCICE...")
        main_dir = os.getenv("ZOOMY_DIR")
        interface = precice.Participant("Fluid2", os.path.join(main_dir, self.config_path), 0, 1)
        meshName = "Fluid2-Mesh"
        velocityName = "Velocity"
        pressureName = "Pressure"
        alphaName = "Alpha"
        dimensions = interface.get_mesh_dimensions(meshName)

        grid = np.zeros([N + 1, dimensions])
        grid[:, 0] = 0.5
        grid[:, 1] = np.linspace(0.0, 0.12, N + 1)  # x component
        grid[:, 2] = 0.
        vertexIDs = interface.set_mesh_vertices(meshName, grid)
        if interface.requires_initial_data():
            interface.write_data(meshName, pressureName + "In", vertexIDs, pressure)
            #interface.write_data(meshName, alphaName+ "In", vertexIDs, alpha)
            interface.write_data(meshName, velocityName+ "In", vertexIDs, velocity)
        
        # preCICE defines timestep size of solver via precice-config.xml
        interface.initialize()
        velocity = interface.read_data(
            meshName, velocityName+"Out", vertexIDs, 0)
        pressure = interface.read_data(
            meshName, pressureName+"Out", vertexIDs, 0)
        alpha = interface.read_data(
            meshName, alphaName+"Out", vertexIDs, 0)
        
        # init once with dummy values for dt
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        output_checkpoint_path = os.path.join(settings.output_dir, "precice_checkpoints.h5")

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


    def precice_fvm(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    ):
        iteration = 0
        time = 0.0
    
        output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
    
        assert model.dimension == mesh.dimension
    
        progressbar = pyprog.ProgressBar(
            "{}: ".format(settings.name),
            "\r",
            settings.time_end,
            complete_symbol="█",
            not_complete_symbol="-",
        )
        progressbar.progress_explain = ""
        progressbar.update()
    
        parser = argparse.ArgumentParser()
        #parser.add_argument("configurationFileName", help="Name of the xml precice configuration file.", nargs='?', type=str, default=f"{os.path.join(main_dir, 'tests/simulations/precice_configurations/precice-config.xml')}")
        parser.add_argument("configurationFileName", help="Name of the xml precice configuration file.", nargs='?', type=str, default=settings.precice_config_path)
    
        try:
            args = parser.parse_args()
        except SystemExit:
            print("")
            print("Did you forget adding the precice configuration file as an argument?")
            print("Try '$ python FluidSolver.py precice-config.xml'")
            quit()
        
        print("Configure preCICE...")
        interface = precice.Participant("Fluid2", args.configurationFileName, 0, 1)
        
        meshName = "Fluid2-Mesh"
        velocityName = "Velocity"
        pressureName = "Pressure"
        alphaName = "Alpha"
        
        dimensions = interface.get_mesh_dimensions(meshName)
        
        N = 100
        velocity = 0.* np.ones((N + 1, 3))
        z = np.linspace(0, 0.12, N+1)
        alpha = np.where(z <= 0.02, 1., 0.)
        alpha0 = alpha.copy()
        pressure = 995.21 * 9.81 * 0.02 * alpha + (alpha < 0.3) * (1-alpha) * (-10.)
        alphaGradient = 0. * np.ones(N+1)
        velocityGradient = 0. * np.ones((N+1, 3))
        
        grid = np.zeros([N + 1, dimensions])
        grid[:, 0] = 0.5
        grid[:, 1] = np.linspace(0.0, 0.12, N + 1)  # x component
        grid[:, 2] = 0.
        
        vertexIDs = interface.set_mesh_vertices(meshName, grid)
        
        if interface.requires_initial_data():
            interface.write_data(meshName, pressureName + "In", vertexIDs, pressure)
            #interface.write_data(meshName, alphaName+ "In", vertexIDs, alpha)
            interface.write_data(meshName, velocityName+ "In", vertexIDs, velocity)
        
        # preCICE defines timestep size of solver via precice-config.xml
        interface.initialize()
        
        velocity = interface.read_data(
            meshName, velocityName+"Out", vertexIDs, 0)
        pressure = interface.read_data(
            meshName, pressureName+"Out", vertexIDs, 0)
        alpha = interface.read_data(
            meshName, alphaName+"Out", vertexIDs, 0)
        
        #velocity_old = np.copy(velocity)
    
        Q, Qaux = _initialize_problem(model, mesh)
    
        parameters = model.parameter_values
        pde, bcs = load_runtime_model(model)
        Q = apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)
        Qnew, pressure, alphaGradient, velocityGradient = apply_boundary_conditions_precice(model, Q, z, velocity, alpha, pressure)
    
        i_snapshot = 0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        mesh.write_to_hdf5(output_checkpoint_path)
        i_snapshot = io.save_fields(
            output_hdf5_path, time, 0, i_snapshot, Q, Qaux, settings.output_write_all
        )
    
        Qnew = deepcopy(Q)
    
        space_solution_operator = _get_semidiscrete_solution_operator(
            mesh, pde, bcs, settings
        )
        compute_source = _get_source(mesh, pde, settings)
        compute_source_jac = _get_source_jac(mesh, pde, settings)
    
        compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(mesh, pde, settings)
        min_inradius = np.min(mesh.cell_inradius)
    
        time_start = gettime()
    
        
        time_it = 0
        while interface.is_coupling_ongoing():
            # When an implicit coupling scheme is used, checkpointing is required
            if interface.requires_writing_checkpoint():
                io._save_fields_to_hdf5(output_checkpoint_path, 0, time, Qnew, Qaux)
        
            precice_dt = interface.get_max_time_step_size()
        
    
            solver_dt = settings.compute_dt(
                Qnew, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
            )
    
            #if settings.truncate_last_time_step:
            #    if time + solver_dt * 1.001 > settings.time_end:
            #        solver_dt = settings.time_end - time + 10 ** (-10)
    
            assert solver_dt > 10 ** (-6)
            assert not np.isnan(solver_dt) and np.isfinite(solver_dt)
    
            dt = min(solver_dt, precice_dt)
    
    
            velocity = interface.read_data(
                meshName, velocityName+"Out", vertexIDs, dt)
            pressure = interface.read_data(
                meshName, pressureName+"Out", vertexIDs, dt)
            alpha = interface.read_data(
            meshName, alphaName+"Out", vertexIDs, dt)
    
            if alpha[0] == 0:
                alpha = alpha0
    
        
            Q = deepcopy(Qnew)
            Q, pressure, alphaGradient, velocityGradient = apply_boundary_conditions_precice(model, Q, z, velocity, alpha, pressure)
    
    
            Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)
    
            Qnew = ode_solver_source(
                compute_source, Qnew, Qaux, parameters, dt, func_jac=compute_source_jac
            )
    
            Qnew = apply_boundary_conditions(mesh, time, Qnew, Qaux, parameters, bcs)
    
            Qnew = strong_bc(model, Qnew)
    
            interface.write_data(meshName, pressureName + "In", vertexIDs, pressure)
            #interface.write_data(meshName, alphaName+ "In", vertexIDs, alpha)
            interface.write_data(meshName, velocityName+ "In", vertexIDs, velocity)
        # 
            interface.advance(dt)
        
            # i.e. not yet converged
            if interface.requires_reading_checkpoint():
                Qnew, Qaux, time  = precice_read_checkpoint(output_checkpoint_path)
    
            else:  # converged, timestep complete
                # Update solution and time
                time += dt
                iteration += 1
                print(iteration, time, dt)
    
                i_snapshot = io.save_fields(
                    output_hdf5_path,
                    time,
                    (i_snapshot + 1) * dt_snapshot,
                    i_snapshot,
                    Qnew,
                    Qaux,
                    settings.output_write_all,
                )
    
        print(f"Runtime: {gettime() - time_start}")
    
        progressbar.end()
        return settings
